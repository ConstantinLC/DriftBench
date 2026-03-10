"""
Proxy GT evaluation via IC optimization (Section 4.3 of neurips.tex).

For each NN state x̂_t, we solve the inverse problem:
    find x_IC such that GT^K(x_IC) ≈ x̂_t
The proxy state is x̃_t = GT^K(x_IC), which lies on a physical trajectory.
The proxy next step is GT(x̃_t) = GT^{K+1}(x_IC).

Metrics computed per timestep:
    - ||NN(x̂_t) - GT¹(x̂_t)||²   : direct GT¹ applied to NN state (kept as baseline)
    - ||x̃_t - x̂_t||²             : proxy optimization quality
    - ||NN(x̂_t) - GT(x̃_t)||²     : exposure bias (Element 1 from paper)
    - ||x̂_{t+1} - x_{t+1}||²      : total trajectory error

Compares multiple emulator training configs (e.g. "one" vs "sup;2").
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
NET_CONFIG      = "UNet;12;2;relu"
TRAIN_CONFIGS   = ["one", "sup;2"]
OPTIM_CONFIG    = "adam;20_000;warmup_cosine;0.0;1e-3;2_000"
NUM_SEEDS       = 1
AR_STEPS        = 300              # Autoregressive rollout length
NUM_TRAJECTORIES = 20              # Number of trajectories to evaluate

K               = 1               # GT steps in the proxy optimization: find x_IC s.t. GT^K(x_IC) ≈ x̂_t
N_OPT_STEPS     = 1000              # Gradient steps per IC optimization
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

# Shared initial conditions (same across all train configs for fair comparison)
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
# IC optimization: for a batch of targets, find x_IC s.t. GT^K(x_IC) ≈ target
# ──────────────────────────────────────────────────────────────────────
ic_opt = optax.adam(OPT_LR)

def optimize_single_ic(target, init_state):
    """target: (C, X). Returns proxy = GT^K(x_IC_opt), final loss."""
    x_ic      = init_state  # warm-start from the state at t+1
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

# Vmapped over a batch of (target, init_state) → proxies (N, C, X), losses (N,)
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
# Main loop: one pass per train config
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

    # Per-timestep metrics
    mse_nn_vs_gt1      = np.zeros(AR_STEPS)  # ||NN(x̂_t) - GT¹(x̂_t)||²
    mse_proxy_match    = np.zeros(AR_STEPS)  # ||x̃_t - x̂_t||²  (optimization quality)
    mse_exposure_bias  = np.zeros(AR_STEPS)  # ||NN(x̂_t) - GT(x̃_t)||²  (Element 1)
    mse_total          = np.zeros(AR_STEPS)  # ||x̂_{t+1} - x_{t+1}||²

    for t in range(AR_STEPS):
        nn_states_t  = nn_trajectories[:, t]     # x̂_t,   (N, C, X)
        nn_next_t    = nn_trajectories[:, t + 1] # x̂_{t+1} = NN(x̂_t)
        gt_next_t    = gt_trajectories[:, t + 1] # x_{t+1} (GT reference)

        # Baseline: GT¹ applied directly to NN state
        gt1_on_nn_t  = jax.vmap(ref_stepper)(nn_states_t)
        mse_nn_vs_gt1[t] = float(jnp.mean((nn_next_t - gt1_on_nn_t) ** 2))

        # Proxy: optimize x_IC s.t. GT^K(x_IC) ≈ x̂_t, warm-started from x̂_{t+1}
        proxy_t, opt_losses = optimize_batch(nn_states_t, nn_next_t)  # (N, C, X), (N,)

        # Proxy next step: GT(x̃_t)
        proxy_next_t = jax.vmap(ref_stepper)(proxy_t)

        mse_proxy_match[t]   = float(jnp.mean((proxy_t - nn_states_t) ** 2))
        mse_exposure_bias[t] = float(jnp.mean((nn_next_t - proxy_next_t) ** 2))
        mse_total[t]         = float(jnp.mean((nn_next_t - gt_next_t) ** 2))

        if t % 10 == 0:
            print(f"  t={t:>3d}  opt_loss={float(opt_losses.mean()):.3e}"
                  f"  proxy_match={mse_proxy_match[t]:.3e}"
                  f"  EB={mse_exposure_bias[t]:.3e}")

    results[train_config] = {
        "nn_trajectories":   nn_trajectories,
        "mse_nn_vs_gt1":     mse_nn_vs_gt1,
        "mse_proxy_match":   mse_proxy_match,
        "mse_exposure_bias": mse_exposure_bias,
        "mse_total":         mse_total,
    }

# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────
colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
timesteps = np.arange(1, AR_STEPS + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"K={K}  |  N_OPT={N_OPT_STEPS}  |  {NET_CONFIG}  |  {SCENARIO_NAME}", fontsize=12)

for i, (train_config, res) in enumerate(results.items()):
    c = colors[i]

    axes[0, 0].plot(timesteps, res["mse_nn_vs_gt1"],     color=c, label=train_config)
    axes[0, 1].plot(timesteps, res["mse_proxy_match"],   color=c, label=train_config)
    axes[1, 0].plot(timesteps, res["mse_exposure_bias"]/res["mse_nn_vs_gt1"], color=c, label=train_config)
    axes[1, 1].plot(timesteps, res["mse_total"],         color=c, label=train_config)

for ax in axes.flat:
    ax.set_yscale("log")
    ax.set_xlabel("AR step t")
    ax.set_ylabel("MSE")
    ax.legend()

axes[0, 0].set_title(r"$\|NN(\hat{x}_t) - GT^1(\hat{x}_t)\|^2$" + "  (baseline: GT¹ on NN states)")
axes[0, 1].set_title(r"$\|\tilde{x}_t - \hat{x}_t\|^2$" + "  (proxy optimization quality)")
axes[1, 0].set_title(r"$\|NN(\hat{x}_t) - GT(\tilde{x}_t)\|^2$" + "  (exposure bias, Element 1)")
axes[1, 1].set_title(r"$\|\hat{x}_{t+1} - x_{t+1}\|^2$" + "  (total trajectory error)")

plt.tight_layout()
plt.savefig("proxy_model_evaluation.pdf", dpi=150)
plt.savefig("proxy_model_evaluation.png", dpi=150)
print("\nSaved proxy_model_evaluation.pdf / .png")

# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Summary  |  K={K}  |  N_OPT={N_OPT_STEPS}  |  {NET_CONFIG}")
print("=" * 60)
for train_config, res in results.items():
    print(f"\n  [{train_config}]")
    print(f"    NN vs GT¹ @ t=1:   {res['mse_nn_vs_gt1'][0]:.4e}")
    print(f"    NN vs GT¹ @ t={AR_STEPS}: {res['mse_nn_vs_gt1'][-1]:.4e}")
    print(f"    Proxy match @ t=1: {res['mse_proxy_match'][0]:.4e}")
    print(f"    Exposure bias @ t=1:   {res['mse_exposure_bias'][0]:.4e}")
    print(f"    Exposure bias @ t={AR_STEPS}: {res['mse_exposure_bias'][-1]:.4e}")
    print(f"    Total error @ t={AR_STEPS}:   {res['mse_total'][-1]:.4e}")
