"""
Three-term error decomposition (Section from neurips.tex, §11.03.2026-3).

For each AR step t, compute and verify:
    ε²(t+1) = ||x̂_{t+1} - x_{t+1}||²          (total trajectory error, squared)
    d_TD²(t) = ||x_{t+1} - GT(x̂_t)||²          (trajectory drift, squared)
    d_OS²(t) = ||GT(x̂_t) - NN(x̃_t)||²         (one-step error, squared)
    d_EB²(t) = ||NN(x̃_t) - NN(x̂_t)||²         (exposure bias, squared)

In high dimensions, vectors are near-orthogonal, so:
    ε²(t+1) ≈ d_TD²(t) + d_OS²(t) + d_EB²(t)  (Pythagorean approx.)
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import exponax as ex
import apebench
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────────────────────────────
SCENARIO_NAME   = "norm_ks"
NET_CONFIG      = 'Dil;2;32;2;relu' #"UNet;12;2;relu"
TRAIN_CONFIGS   = ["one", "sup;2", "sup;5"]
OPTIM_CONFIG    = "adam;20_000;warmup_cosine;0.0;1e-3;2_000"
NUM_SEEDS       = 1
AR_STEPS        = 20              # Autoregressive rollout length
NUM_TRAJECTORIES = 20              # Number of trajectories to evaluate

K               = 5               # GT steps in proxy optimization: find x_IC s.t. GT^K(x_IC) ≈ x̂_t
N_OPT_STEPS     = 1000            # Gradient steps per IC optimization
OPT_LR          = 1e-3            # Learning rate for IC optimization

# ──────────────────────────────────────────────────────────────────────
# Setup: scenario, steppers, shared ICs
# ──────────────────────────────────────────────────────────────────────
scenario    = apebench.scenarios.normalized.KuramotoSivashinsky(
    diffusion_alpha=-0.000025,
    hyp_diffusion_alpha=-3.0e-9,
)
ref_stepper = scenario.get_ref_stepper()
k_stepper   = ex.repeat(ref_stepper, K) if K > 1 else ref_stepper

# Shared initial conditions
ic_set = ex.build_ic_set(
    scenario.get_ic_generator(),
    num_points=scenario.num_points,
    num_samples=NUM_TRAJECTORIES,
    key=jax.random.PRNGKey(42),
)
warmup_fn = ex.repeat(ref_stepper, scenario.num_warmup_steps)
ic_set    = jax.vmap(warmup_fn)(ic_set)
print(f"Warmed up {NUM_TRAJECTORIES} ICs, shape: {ic_set.shape}")

# GT reference trajectory
gt_trajectories = jax.vmap(
    ex.rollout(ref_stepper, AR_STEPS, include_init=True)
)(ic_set)
print(f"GT trajectory shape: {gt_trajectories.shape}")

# ──────────────────────────────────────────────────────────────────────
# IC optimization: find x_IC s.t. GT^K(x_IC) ≈ target
# ──────────────────────────────────────────────────────────────────────
ic_opt = optax.adam(OPT_LR)

def optimize_single_ic(target, init_state):
    """target: (C, X). Returns proxy = GT^K(x_IC_opt), final loss."""
    x_ic      = init_state
    opt_state = ic_opt.init(x_ic)

    def step(carry, _):
        x_ic, opt_state = carry
        loss, grads = jax.value_and_grad(
            lambda x: jnp.mean((k_stepper(x) - target) ** 2)
        )(x_ic)
        updates, new_opt_state = ic_opt.update(grads, opt_state)
        x_ic = optax.apply_updates(x_ic, updates)
        return (x_ic, new_opt_state), loss

    (x_ic_opt, _), loss_hist = jax.lax.scan(
        step, (x_ic, opt_state), None, length=N_OPT_STEPS
    )
    proxy = k_stepper(x_ic_opt)
    return proxy, loss_hist[-1]

optimize_batch = jax.jit(jax.vmap(optimize_single_ic))

# ──────────────────────────────────────────────────────────────────────
# Train/load all emulators
# ──────────────────────────────────────────────────────────────────────
configs = [
    {
        "scenario": SCENARIO_NAME, "task": "predict", "net": NET_CONFIG,
        "train": tc, "start_seed": 0, "num_seeds": NUM_SEEDS,
        "diffusion_alpha": -0.000025, "hyp_diffusion_alpha": -3.0e-9,
        "optim_config": OPTIM_CONFIG,
    }
    for tc in TRAIN_CONFIGS
]
_, _, _, weight_paths = apebench.run_study_convenience(configs=configs, base_path="results")

# ──────────────────────────────────────────────────────────────────────
# Main loop: compute three-term decomposition per train config
# ──────────────────────────────────────────────────────────────────────
results = {}

for train_config, weight_path in zip(TRAIN_CONFIGS, weight_paths):
    print("\n" + "=" * 60)
    print(f"Train config: {train_config}")
    print("=" * 60)

    # Load emulator
    neural_stepper = scenario.get_neural_stepper(
        task_config="predict", network_config=NET_CONFIG, key=jax.random.PRNGKey(0)
    )
    neural_stepper = eqx.tree_deserialise_leaves(str(weight_path), neural_stepper)

    # NN rollout: shape (N, AR_STEPS+1, C, X)
    nn_trajectories = jax.vmap(
        ex.rollout(neural_stepper, AR_STEPS, include_init=True)
    )(ic_set)
    print(f"  NN trajectory shape: {nn_trajectories.shape}")

    # Per-timestep decomposition
    total_error = np.zeros(AR_STEPS)    # ε²(t+1) = ||x̂_{t+1} - x_{t+1}||²
    d_td        = np.zeros(AR_STEPS)    # d_TD²(t) = ||x_{t+1} - GT(x̂_t)||²
    d_os        = np.zeros(AR_STEPS)    # d_OS²(t) = ||GT(x̂_t) - NN(x̃_t)||²
    d_eb        = np.zeros(AR_STEPS)    # d_EB²(t) = ||NN(x̃_t) - NN(x̂_t)||²

    for t in range(AR_STEPS):
        nn_states_t = nn_trajectories[:, t]      # x̂_t     (N, C, X)
        nn_next_t   = nn_trajectories[:, t + 1]  # x̂_{t+1} = NN(x̂_t)
        gt_next_t   = gt_trajectories[:, t + 1]  # x_{t+1}

        # GT applied to NN state: GT(x̂_t)
        gt_on_nn_t = jax.vmap(ref_stepper)(nn_states_t)

        # Proxy optimization: find x̃_t ≈ x̂_t on a physical trajectory
        proxy_t, opt_loss = optimize_batch(nn_states_t, nn_next_t)

        # NN applied to proxy: NN(x̃_t)
        nn_on_proxy_t = jax.vmap(neural_stepper)(proxy_t)

        # Three-term decomposition
        total_error[t] = float(jnp.mean(jnp.mean((nn_next_t - gt_next_t) ** 2, axis=(-2, -1))))
        d_td[t]        = float(jnp.mean(jnp.mean((gt_next_t - gt_on_nn_t) ** 2, axis=(-2, -1))))
        d_os[t]        = float(jnp.mean(jnp.mean((gt_on_nn_t - nn_on_proxy_t) ** 2, axis=(-2, -1))))
        d_eb[t]        = float(jnp.mean(jnp.mean((nn_on_proxy_t - nn_next_t) ** 2, axis=(-2, -1))))

        if t % 10 == 0:
            pyth_sum = d_td[t] + d_os[t] + d_eb[t]
            print(f"  t={t:>3d}  ε²={total_error[t]:.3e}  "
                  f"d_TD²={d_td[t]:.3e}  d_OS²={d_os[t]:.3e}  d_EB²={d_eb[t]:.3e}  "
                  f"sum={pyth_sum:.3e}  gap={pyth_sum - total_error[t]:.3e}")

    results[train_config] = {
        "total_error": total_error,
        "d_td": d_td,
        "d_os": d_os,
        "d_eb": d_eb,
    }

# ──────────────────────────────────────────────────────────────────────
# Ranking benchmark (§11.03.2026): does d_TD²+d_OS²+d_EB² rank models
# the same way as ε² at each timestep?
# ──────────────────────────────────────────────────────────────────────
config_names = list(results.keys())

# Stack arrays: shape (n_models, AR_STEPS)
total_error_mat = np.stack([results[c]["total_error"] for c in config_names])
pyth_sum_mat    = np.stack([results[c]["d_td"] + results[c]["d_os"] + results[c]["d_eb"]
                            for c in config_names])

# Per-timestep: do rankings match?
rank_eps  = np.argsort(np.argsort(total_error_mat, axis=0), axis=0)  # (n_models, AR_STEPS)
rank_pyth = np.argsort(np.argsort(pyth_sum_mat,    axis=0), axis=0)
rank_match = (rank_eps == rank_pyth).all(axis=0)  # True if all models correctly ranked at t

print("\n" + "=" * 60)
print("Ranking benchmark (§11.03.2026): does Σd²_i rank models like ε²?")
print(f"{'t':>4}  {'ranking matches':>16}  {'ε²':>10}  {'Σd²':>10}")
for t in range(AR_STEPS):
    vals_eps  = "  ".join(f"{total_error_mat[i,t]:.2e}" for i in range(len(config_names)))
    vals_pyth = "  ".join(f"{pyth_sum_mat[i,t]:.2e}"    for i in range(len(config_names)))
    print(f"  t={t+1:>2d}  {'YES' if rank_match[t] else 'NO':>5}  ε²=[{vals_eps}]  Σd²=[{vals_pyth}]")
print(f"\nRanking agreement: {rank_match.mean()*100:.1f}% of timesteps ({rank_match.sum()}/{AR_STEPS})")

# ──────────────────────────────────────────────────────────────────────
# Plots: one subplot per train config
# ──────────────────────────────────────────────────────────────────────
timesteps = np.arange(1, AR_STEPS + 1)
n_configs = len(results)

fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 5))
if n_configs == 1:
    axes = [axes]
fig.suptitle(f"Three-term decomposition (squared norms)  |  K={K}  |  {NET_CONFIG}  |  {SCENARIO_NAME}", fontsize=12)

for ax, (train_config, res) in zip(axes, results.items()):
    pyth_sum = res["d_td"] + res["d_os"] + res["d_eb"]

    ax.plot(timesteps, res["total_error"], "k-",  lw=2, label=r"$\epsilon^2(t{+}1)$ (total)")
    ax.plot(timesteps, pyth_sum,            "k--", lw=1, label=r"$d_{TD}^2+d_{OS}^2+d_{EB}^2$")
    ax.plot(timesteps, res["d_td"], label=r"$d_{TD}^2$ (traj. drift)")
    ax.plot(timesteps, res["d_os"], label=r"$d_{OS}^2$ (one-step)")
    ax.plot(timesteps, res["d_eb"], label=r"$d_{EB}^2$ (exposure bias)")

    ax.set_yscale("log")
    ax.set_xlabel("AR step t")
    ax.set_ylabel("Error (squared L2 norm)")
    ax.legend(fontsize=8)
    ax.set_title(f"train: {train_config}")

plt.tight_layout()
plt.savefig("proxy_model_evaluation.pdf", dpi=150)
plt.savefig("proxy_model_evaluation.png", dpi=150)
print("\nSaved proxy_model_evaluation.pdf / .png")

# ── Ranking benchmark plot ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
fig2.suptitle("Ranking benchmark (§11.03.2026): $\\Sigma d_i^2$ vs $\\epsilon^2$ per model", fontsize=11)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, name in enumerate(config_names):
    ax2.plot(timesteps, total_error_mat[i], color=colors[i], lw=2,    label=f"{name}  ε²")
    ax2.plot(timesteps, pyth_sum_mat[i],    color=colors[i], lw=1, ls="--", label=f"{name}  Σd²")

# Shade timesteps where ranking disagrees
for t in range(AR_STEPS):
    if not rank_match[t]:
        ax2.axvspan(t + 0.5, t + 1.5, color="red", alpha=0.15)

ax2.set_yscale("log")
ax2.set_xlabel("AR step t")
ax2.set_ylabel("Squared L2 norm")
ax2.legend(fontsize=7, ncol=2)
ax2.set_title(f"Ranking agreement: {rank_match.mean()*100:.1f}% of timesteps  |  red = mismatch")

plt.tight_layout()
plt.savefig("ranking_benchmark.pdf", dpi=150)
plt.savefig("ranking_benchmark.png", dpi=150)
print("Saved ranking_benchmark.pdf / .png")
