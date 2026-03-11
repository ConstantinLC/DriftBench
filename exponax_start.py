"""
Three-term error decomposition (Section from neurips.tex, §11.03.2026-3).

For each AR step t, compute and verify:
    ε²(t+1)  = ||x̂_{t+1} - x_{t+1}||²               (total trajectory error, squared)
    d_TD²(t) = ||x_{t+1} - GT(x̂_t^ID)||²            (trajectory drift, squared)
    d_OS²(t) = ||GT(x̂_t^ID) - NN(x̂_t^ID)||²        (one-step error, squared)
    d_EB²(t) = ||NN(x̂_t^ID) - NN(x̂_t)||²           (exposure bias, squared)

"Another possibility" definition (neurips.tex 11/03/2026):
    pivot is GT(x̂_t^ID) — GT applied to the proxy — instead of GT(x̂_t).

In high dimensions, vectors are near-orthogonal, so:
    ε²(t+1) ≈ d_TD²(t) + d_OS²(t) + d_EB²(t)  (Pythagorean approx.)
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
NET_CONFIG      = 'Dil;2;32;2;relu' #"UNet;12;2;relu" # #
TRAIN_CONFIGS   = ["one", "sup;2", "sup;5"]
OPTIM_CONFIG    = "adam;20_000;warmup_cosine;0.0;1e-3;2_000"
NUM_SEEDS       = 1
AR_STEPS        = 20              # Autoregressive rollout length
NUM_TRAJECTORIES = 20              # Number of trajectories to evaluate

K               = 5               # GT steps in proxy optimization: find x_IC s.t. GT^K(x_IC) ≈ x̂_t
N_OPT_STEPS     = 1000            # Gradient steps per IC optimization
TWO_TERM        = True            # If True: merge d_TD+d_OS → d_TDOS = ||x_{t+1} - NN(x̂_t^ID)||
                                  # Only one cross dot-product remains: 2⟨d_TDOS, d_EB⟩
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
    total_error = np.zeros(AR_STEPS)    # ε²(t+1)  = ||x̂_{t+1} - x_{t+1}||²
    d_eb        = np.zeros(AR_STEPS)    # d_EB²(t) = ||NN(x̂_t^ID) - NN(x̂_t)||²
    if TWO_TERM:
        d_tdos    = np.zeros(AR_STEPS)  # d_TDOS²(t) = ||x_{t+1} - NN(x̂_t^ID)||²  (merged TD+OS)
        dot_tdos_eb = np.zeros(AR_STEPS)  # 2⟨d_TDOS, d_EB⟩
    else:
        d_td      = np.zeros(AR_STEPS)  # d_TD²(t) = ||x_{t+1} - GT(x̂_t^ID)||²
        d_os      = np.zeros(AR_STEPS)  # d_OS²(t) = ||GT(x̂_t^ID) - NN(x̂_t^ID)||²
        dot_eb_os = np.zeros(AR_STEPS)  # 2⟨d_EB, d_OS⟩
        dot_eb_td = np.zeros(AR_STEPS)  # 2⟨d_EB, d_TD⟩
        dot_os_td = np.zeros(AR_STEPS)  # 2⟨d_OS, d_TD⟩

    def dot(a, b):
        """Mean over batch of spatial dot products: E[⟨a_i, b_i⟩]."""
        return jnp.mean(jnp.sum(a * b, axis=(-2, -1)))

    for t in range(AR_STEPS):
        nn_states_t = nn_trajectories[:, t]      # x̂_t     (N, C, X)
        nn_next_t   = nn_trajectories[:, t + 1]  # x̂_{t+1} = NN(x̂_t)
        gt_next_t   = gt_trajectories[:, t + 1]  # x_{t+1}

        # Proxy optimization: find x̂_t^ID s.t. GT^K(x̂_t^ID) ≈ x̂_t
        proxy_t, opt_loss = optimize_batch(nn_states_t, nn_next_t)

        # NN applied to proxy: NN(x̂_t^ID)
        nn_on_proxy_t = jax.vmap(neural_stepper)(proxy_t)

        # EB vector (common to both modes)
        vec_eb = nn_on_proxy_t - nn_next_t        # NN(x̂_t^ID) - NN(x̂_t)

        total_error[t] = float(jnp.mean(jnp.sum((nn_next_t - gt_next_t) ** 2, axis=(-2, -1))))
        d_eb[t]        = float(jnp.mean(jnp.sum(vec_eb ** 2, axis=(-2, -1))))

        if TWO_TERM:
            vec_tdos     = gt_next_t - nn_on_proxy_t  # x_{t+1} - NN(x̂_t^ID)
            d_tdos[t]    = float(jnp.mean(jnp.sum(vec_tdos ** 2, axis=(-2, -1))))
            dot_tdos_eb[t] = float(2 * dot(vec_tdos, vec_eb))

            if t % 5 == 0:
                pyth_sum  = d_tdos[t] + d_eb[t]
                exact_sum = pyth_sum + dot_tdos_eb[t]
                cross_frac = dot_tdos_eb[t] / (total_error[t] + 1e-30)
                print(f"  t={t+1:>3d}  ε²={total_error[t]:.3e}  "
                      f"d_TDOS²={d_tdos[t]:.3e}  d_EB²={d_eb[t]:.3e}  "
                      f"exact={exact_sum:.3e}  cross/ε²={cross_frac:+.3f}")
        else:
            gt_on_proxy_t = jax.vmap(ref_stepper)(proxy_t)
            vec_td = gt_next_t     - gt_on_proxy_t   # x_{t+1} - GT(x̂_t^ID)
            vec_os = gt_on_proxy_t - nn_on_proxy_t   # GT(x̂_t^ID) - NN(x̂_t^ID)

            d_td[t]        = float(jnp.mean(jnp.sum(vec_td ** 2, axis=(-2, -1))))
            d_os[t]        = float(jnp.mean(jnp.sum(vec_os ** 2, axis=(-2, -1))))
            dot_eb_os[t]   = float(2 * dot(vec_eb, vec_os))
            dot_eb_td[t]   = float(2 * dot(vec_eb, vec_td))
            dot_os_td[t]   = float(2 * dot(vec_os, vec_td))

            if t % 5 == 0:
                pyth_sum  = d_td[t] + d_os[t] + d_eb[t]
                exact_sum = pyth_sum + dot_eb_os[t] + dot_eb_td[t] + dot_os_td[t]
                cross_frac = (dot_eb_os[t] + dot_eb_td[t] + dot_os_td[t]) / (total_error[t] + 1e-30)
                print(f"  t={t+1:>3d}  ε²={total_error[t]:.3e}  "
                      f"Σd²={pyth_sum:.3e}  exact={exact_sum:.3e}  "
                      f"cross/ε²={cross_frac:+.3f}")

    if TWO_TERM:
        results[train_config] = {
            "total_error": total_error,
            "d_tdos": d_tdos,
            "d_eb": d_eb,
            "dot_tdos_eb": dot_tdos_eb,
        }
    else:
        results[train_config] = {
            "total_error": total_error,
            "d_td": d_td,
            "d_os": d_os,
            "d_eb": d_eb,
            "dot_eb_os": dot_eb_os,
            "dot_eb_td": dot_eb_td,
            "dot_os_td": dot_os_td,
        }

# ──────────────────────────────────────────────────────────────────────
# Ranking benchmark (§11.03.2026)
# ──────────────────────────────────────────────────────────────────────
config_names = list(results.keys())

# Stack arrays: shape (n_models, AR_STEPS)
total_error_mat = np.stack([results[c]["total_error"] for c in config_names])
if TWO_TERM:
    pyth_sum_mat  = np.stack([results[c]["d_tdos"] + results[c]["d_eb"] for c in config_names])
    exact_sum_mat = np.stack([results[c]["d_tdos"] + results[c]["d_eb"] + results[c]["dot_tdos_eb"]
                              for c in config_names])
else:
    pyth_sum_mat  = np.stack([results[c]["d_td"] + results[c]["d_os"] + results[c]["d_eb"]
                              for c in config_names])
    exact_sum_mat = np.stack([results[c]["d_td"] + results[c]["d_os"] + results[c]["d_eb"]
                              + results[c]["dot_eb_os"] + results[c]["dot_eb_td"] + results[c]["dot_os_td"]
                              for c in config_names])

# Per-timestep: do rankings match?
rank_eps   = np.argsort(np.argsort(total_error_mat, axis=0), axis=0)  # (n_models, AR_STEPS)
rank_pyth  = np.argsort(np.argsort(pyth_sum_mat,    axis=0), axis=0)
rank_exact = np.argsort(np.argsort(exact_sum_mat,   axis=0), axis=0)
rank_match_pyth  = (rank_eps == rank_pyth).all(axis=0)
rank_match_exact = (rank_eps == rank_exact).all(axis=0)

print("\n" + "=" * 60)
print("Ranking benchmark (§11.03.2026)")
print(f"{'t':>4}  {'Σd² matches':>12}  {'Σd²+cross matches':>18}")
for t in range(AR_STEPS):
    print(f"  t={t+1:>2d}  {'YES' if rank_match_pyth[t] else 'NO':>11}  {'YES' if rank_match_exact[t] else 'NO':>17}")
print(f"\nΣd²       agreement: {rank_match_pyth.mean()*100:.1f}% ({rank_match_pyth.sum()}/{AR_STEPS})")
print(f"Σd²+cross agreement: {rank_match_exact.mean()*100:.1f}% ({rank_match_exact.sum()}/{AR_STEPS})")

# ──────────────────────────────────────────────────────────────────────
# Plots: one subplot per train config
# ──────────────────────────────────────────────────────────────────────
timesteps = np.arange(1, AR_STEPS + 1)
n_configs = len(results)

n_terms = "Two" if TWO_TERM else "Three"
fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 5))
if n_configs == 1:
    axes = [axes]
fig.suptitle(f"{n_terms}-term decomposition (squared norms)  |  K={K}  |  {NET_CONFIG}  |  {SCENARIO_NAME}", fontsize=12)

for ax, (train_config, res) in zip(axes, results.items()):
    if TWO_TERM:
        pyth_sum  = res["d_tdos"] + res["d_eb"]
        exact_sum = pyth_sum + res["dot_tdos_eb"]
        ax.plot(timesteps, res["total_error"], "k-",  lw=2,   label=r"$\epsilon^2$ (total)")
        ax.plot(timesteps, exact_sum,          "k--", lw=1.5, label=r"$d_{TDOS}^2+d_{EB}^2+2\langle d_{TDOS},d_{EB}\rangle$ (exact)")
        ax.plot(timesteps, pyth_sum,           "k:",  lw=1,   label=r"$d_{TDOS}^2+d_{EB}^2$ (Pythagorean)")
        ax.plot(timesteps, res["d_tdos"], label=r"$d_{TDOS}^2$ (drift+one-step)")
        ax.plot(timesteps, res["d_eb"],   label=r"$d_{EB}^2$ (exposure bias)")
        vals = res["dot_tdos_eb"]
        line, = ax.plot(timesteps, np.abs(vals), ls="--", alpha=0.6, label=r"$|2\langle d_{TDOS},d_{EB}\rangle|$")
        neg_mask = vals < 0
        if neg_mask.any():
            ax.scatter(timesteps[neg_mask], np.abs(vals[neg_mask]),
                       color=line.get_color(), marker="x", s=40, zorder=5)
    else:
        pyth_sum  = res["d_td"] + res["d_os"] + res["d_eb"]
        cross_sum = res["dot_eb_os"] + res["dot_eb_td"] + res["dot_os_td"]
        exact_sum = pyth_sum + cross_sum
        ax.plot(timesteps, res["total_error"], "k-",  lw=2,   label=r"$\epsilon^2$ (total)")
        ax.plot(timesteps, exact_sum,          "k--", lw=1.5, label=r"$\Sigma d_i^2 + 2\Sigma\langle d_i,d_j\rangle$ (exact)")
        ax.plot(timesteps, pyth_sum,           "k:",  lw=1,   label=r"$\Sigma d_i^2$ (Pythagorean)")
        ax.plot(timesteps, res["d_td"], label=r"$d_{TD}^2$")
        ax.plot(timesteps, res["d_os"], label=r"$d_{OS}^2$")
        ax.plot(timesteps, res["d_eb"], label=r"$d_{EB}^2$")
        for key, label in [
            ("dot_eb_os", r"$|2\langle d_{EB},d_{OS}\rangle|$"),
            ("dot_eb_td", r"$|2\langle d_{EB},d_{TD}\rangle|$"),
            ("dot_os_td", r"$|2\langle d_{OS},d_{TD}\rangle|$"),
        ]:
            vals = res[key]
            line, = ax.plot(timesteps, np.abs(vals), ls="--", alpha=0.6, label=label)
            neg_mask = vals < 0
            if neg_mask.any():
                ax.scatter(timesteps[neg_mask], np.abs(vals[neg_mask]),
                           color=line.get_color(), marker="x", s=40, zorder=5)

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
    markers = ["o", "s", "^", "D", "v"]
    mk = markers[i % len(markers)]
    ax2.plot(timesteps, total_error_mat[i], color=colors[i],    lw=2,                            label=f"{name}  ε²")
    ax2.plot(timesteps, pyth_sum_mat[i],    color="tab:orange", lw=1, ls="--", marker=mk, ms=4,  label=f"{name}  Σd²")
    ax2.plot(timesteps, exact_sum_mat[i],   color="tab:green",  lw=1, ls=":",  marker=mk, ms=4,  label=f"{name}  Σd²+cross")

# Shade timesteps where Σd² ranking disagrees (red) or exact disagrees (orange)
for t in range(AR_STEPS):
    if not rank_match_pyth[t]:
        ax2.axvspan(t + 0.5, t + 1.5, color="red", alpha=0.15)
    elif not rank_match_exact[t]:
        ax2.axvspan(t + 0.5, t + 1.5, color="orange", alpha=0.15)

ax2.set_yscale("log")
ax2.set_xlabel("AR step t")
ax2.set_ylabel("Squared L2 norm")
ax2.legend(fontsize=7, ncol=2)
ax2.set_title(f"Ranking agreement: Σd²={rank_match_pyth.mean()*100:.1f}%  exact={rank_match_exact.mean()*100:.1f}%  |  red=Σd² mismatch  orange=exact mismatch")

plt.tight_layout()
plt.savefig("ranking_benchmark.pdf", dpi=150)
plt.savefig("ranking_benchmark.png", dpi=150)
print("Saved ranking_benchmark.pdf / .png")
