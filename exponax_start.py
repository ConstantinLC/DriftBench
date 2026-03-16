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
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────────────────────────────
SCENARIO_NAME   = "norm_ks"      # key into SCENARIO_MAP below
NET_CONFIG      = 'Dil;2;32;2;relu' #"UNet;12;2;relu"
TRAIN_CONFIGS   = ["one", "sup;2", "sup;5", "sup;8"]
OPTIM_CONFIG    = "adam;20_000;warmup_cosine;0.0;1e-3;2_000"
NUM_SEEDS       = 1
AR_STEPS        = 10
NUM_TRAJECTORIES = 20

K               = 5
N_OPT_STEPS     = 1000
TWO_TERM        = True
NO_PROXY        = True
SQUARED         = True
OPT_LR          = 1e-3

# Scenario registry — extend to support new scenarios
SCENARIO_MAP = {
    "norm_kdv":  apebench.scenarios.normalized.KortewegDeVries,
    "norm_ks":   apebench.scenarios.normalized.KuramotoSivashinsky,
    "norm_burgers": apebench.scenarios.normalized.Burgers,
    "norm_adv":  apebench.scenarios.normalized.Advection,
}

if SCENARIO_NAME not in SCENARIO_MAP:
    raise ValueError(f"Unknown scenario '{SCENARIO_NAME}'. Choose from: {list(SCENARIO_MAP)}")

OUTPUT_DIR = Path("outputs") / SCENARIO_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Setup: scenario, steppers, shared ICs
# ──────────────────────────────────────────────────────────────────────
scenario = SCENARIO_MAP[SCENARIO_NAME]()
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
        "optim_config": OPTIM_CONFIG,
    }
    for tc in TRAIN_CONFIGS
]
_, _, _, weight_paths = apebench.run_study_convenience(
    configs=configs, base_path=str(OUTPUT_DIR / "weights")
)

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
    if NO_PROXY:
        d_td      = np.zeros(AR_STEPS)  # d_TD²(t)   = ||x_{t+1} - GT(x̂_t)||²
        d_oseb    = np.zeros(AR_STEPS)  # d_OSEB²(t) = ||GT(x̂_t) - NN(x̂_t)||²
        dot_td_oseb = np.zeros(AR_STEPS)  # 2⟨d_TD, d_OSEB⟩
    elif TWO_TERM:
        d_tdos      = np.zeros(AR_STEPS)  # d_TDOS²(t) = ||x_{t+1} - NN(x̂_t^ID)||²
        d_eb        = np.zeros(AR_STEPS)  # d_EB²(t)   = ||NN(x̂_t^ID) - NN(x̂_t)||²
        dot_tdos_eb = np.zeros(AR_STEPS)  # 2⟨d_TDOS, d_EB⟩
    else:
        d_td      = np.zeros(AR_STEPS)  # d_TD²(t) = ||x_{t+1} - GT(x̂_t^ID)||²
        d_os      = np.zeros(AR_STEPS)  # d_OS²(t) = ||GT(x̂_t^ID) - NN(x̂_t^ID)||²
        d_eb      = np.zeros(AR_STEPS)  # d_EB²(t) = ||NN(x̂_t^ID) - NN(x̂_t)||²
        dot_eb_os = np.zeros(AR_STEPS)  # 2⟨d_EB, d_OS⟩
        dot_eb_td = np.zeros(AR_STEPS)  # 2⟨d_EB, d_TD⟩
        dot_os_td = np.zeros(AR_STEPS)  # 2⟨d_OS, d_TD⟩

    def norm(v):
        """Mean over batch of L2 norms (or squared norms)."""
        sq = jnp.sum(v ** 2, axis=(-2, -1))
        return jnp.mean(jnp.sqrt(sq) if not SQUARED else sq)

    def dot(a, b):
        """Mean over batch of spatial dot products: E[⟨a_i, b_i⟩]. Only used when SQUARED=True."""
        return jnp.mean(jnp.sum(a * b, axis=(-2, -1)))

    for t in range(AR_STEPS):
        nn_states_t = nn_trajectories[:, t]      # x̂_t     (N, C, X)
        nn_next_t   = nn_trajectories[:, t + 1]  # x̂_{t+1} = NN(x̂_t)
        gt_next_t   = gt_trajectories[:, t + 1]  # x_{t+1}

        total_error[t] = float(norm(nn_next_t - gt_next_t))
        sym = "²" if SQUARED else ""

        if NO_PROXY:
            gt_on_nn_t = jax.vmap(ref_stepper)(nn_states_t)  # GT(x̂_t)
            vec_td   = gt_next_t  - gt_on_nn_t
            vec_oseb = gt_on_nn_t - nn_next_t

            d_td[t]   = float(norm(vec_td))
            d_oseb[t] = float(norm(vec_oseb))
            if SQUARED:
                dot_td_oseb[t] = float(2 * dot(vec_td, vec_oseb))

            if t % 5 == 0:
                tri_sum   = d_td[t] + d_oseb[t]
                exact_sum = tri_sum + (dot_td_oseb[t] if SQUARED else 0)
                print(f"  t={t+1:>3d}  ε{sym}={total_error[t]:.3e}  "
                      f"d_TD{sym}={d_td[t]:.3e}  d_OSEB{sym}={d_oseb[t]:.3e}  "
                      f"sum={tri_sum:.3e}" +
                      (f"  exact={exact_sum:.3e}  cross/ε²={dot_td_oseb[t]/(total_error[t]+1e-30):+.3f}" if SQUARED else ""))
        else:
            proxy_t, opt_loss = optimize_batch(nn_states_t, nn_next_t)
            nn_on_proxy_t = jax.vmap(neural_stepper)(proxy_t)
            vec_eb  = nn_on_proxy_t - nn_next_t
            d_eb[t] = float(norm(vec_eb))

            if TWO_TERM:
                vec_tdos  = gt_next_t - nn_on_proxy_t
                d_tdos[t] = float(norm(vec_tdos))
                if SQUARED:
                    dot_tdos_eb[t] = float(2 * dot(vec_tdos, vec_eb))

                if t % 5 == 0:
                    tri_sum   = d_tdos[t] + d_eb[t]
                    exact_sum = tri_sum + (dot_tdos_eb[t] if SQUARED else 0)
                    print(f"  t={t+1:>3d}  ε{sym}={total_error[t]:.3e}  "
                          f"d_TDOS{sym}={d_tdos[t]:.3e}  d_EB{sym}={d_eb[t]:.3e}  "
                          f"sum={tri_sum:.3e}" +
                          (f"  exact={exact_sum:.3e}  cross/ε²={dot_tdos_eb[t]/(total_error[t]+1e-30):+.3f}" if SQUARED else ""))
            else:
                gt_on_proxy_t = jax.vmap(ref_stepper)(proxy_t)
                vec_td = gt_next_t     - gt_on_proxy_t
                vec_os = gt_on_proxy_t - nn_on_proxy_t

                d_td[t] = float(norm(vec_td))
                d_os[t] = float(norm(vec_os))
                if SQUARED:
                    dot_eb_os[t] = float(2 * dot(vec_eb, vec_os))
                    dot_eb_td[t] = float(2 * dot(vec_eb, vec_td))
                    dot_os_td[t] = float(2 * dot(vec_os, vec_td))

                if t % 5 == 0:
                    tri_sum   = d_td[t] + d_os[t] + d_eb[t]
                    exact_sum = tri_sum + (dot_eb_os[t] + dot_eb_td[t] + dot_os_td[t] if SQUARED else 0)
                    print(f"  t={t+1:>3d}  ε{sym}={total_error[t]:.3e}  "
                          f"Σd{sym}={tri_sum:.3e}" +
                          (f"  exact={exact_sum:.3e}  cross/ε²={(dot_eb_os[t]+dot_eb_td[t]+dot_os_td[t])/(total_error[t]+1e-30):+.3f}" if SQUARED else ""))

    if NO_PROXY:
        results[train_config] = {
            "total_error": total_error,
            "d_td": d_td,
            "d_oseb": d_oseb,
            "dot_td_oseb": dot_td_oseb,
        }
    elif TWO_TERM:
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
if NO_PROXY:
    pyth_sum_mat  = np.stack([results[c]["d_td"] + results[c]["d_oseb"] for c in config_names])
    exact_sum_mat = np.stack([results[c]["d_td"] + results[c]["d_oseb"] + results[c]["dot_td_oseb"]
                              for c in config_names])
elif TWO_TERM:
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

n_terms = "No-proxy" if NO_PROXY else ("Two" if TWO_TERM else "Three")
fig, axes_grid = plt.subplots(2, n_configs, figsize=(6 * n_configs, 8),
                              gridspec_kw={"height_ratios": [2, 1]})
if n_configs == 1:
    axes_grid = axes_grid.reshape(2, 1)
axes     = axes_grid[0]   # top row: main terms
axes_bot = axes_grid[1]   # bottom row: secondary terms (d_OSEB / cross)
fig.suptitle(f"{n_terms}-term decomposition (squared norms)  |  K={K}  |  {NET_CONFIG}  |  {SCENARIO_NAME}", fontsize=12)

def _plot_cross(ax, vals, label):
    t, v = timesteps[1:], vals[1:]  # skip t=0 where dot product is ill-defined
    line, = ax.plot(t, np.abs(v), ls="--", alpha=0.6, label=label)
    neg_mask = v < 0
    if neg_mask.any():
        ax.scatter(t[neg_mask], np.abs(v[neg_mask]),
                   color=line.get_color(), marker="x", s=40, zorder=5)

for ax, ax_bot, (train_config, res) in zip(axes, axes_bot, results.items()):
    s = "^2" if SQUARED else ""
    sym = "²" if SQUARED else ""
    ax.plot(timesteps, res["total_error"], "k-", lw=2, label=rf"$\epsilon{sym}$ (total)")

    if NO_PROXY:
        tri_sum = res["d_td"] + res["d_oseb"]
        ax.plot(timesteps, tri_sum, "k:", lw=1, label=rf"$d_{{TD}}{s}+d_{{OSEB}}{s}$ ({'Pythagorean' if SQUARED else 'triangle ineq.'})")
        if SQUARED:
            exact_sum = tri_sum + res["dot_td_oseb"]
            ax.plot(timesteps, exact_sum, "k--", lw=1.5, label=r"$d_{TD}^2+d_{OSEB}^2+2\langle d_{TD},d_{OSEB}\rangle$ (exact)")
        ax.plot(timesteps[1:], res["d_td"][1:], label=rf"$d_{{TD}}{s}$ (trajectory drift)")
        # bottom: d_OSEB, cross term, and their sum
        ax_bot.plot(timesteps, res["d_oseb"], label=rf"$d_{{OSEB}}{s}$ (one-step+EB)")
        if SQUARED:
            _plot_cross(ax_bot, res["dot_td_oseb"], r"$|2\langle d_{TD},d_{OSEB}\rangle|$")
            oseb_plus_cross = res["d_oseb"] + res["dot_td_oseb"]
            ax_bot.plot(timesteps, oseb_plus_cross, "k-", lw=1.5, label=r"$d_{OSEB}^2+2\langle d_{TD},d_{OSEB}\rangle$")
    elif TWO_TERM:
        tri_sum = res["d_tdos"] + res["d_eb"]
        ax.plot(timesteps, tri_sum, "k:", lw=1, label=rf"$d_{{TDOS}}{s}+d_{{EB}}{s}$ ({'Pythagorean' if SQUARED else 'triangle ineq.'})")
        if SQUARED:
            exact_sum = tri_sum + res["dot_tdos_eb"]
            ax.plot(timesteps, exact_sum, "k--", lw=1.5, label=r"$d_{TDOS}^2+d_{EB}^2+2\langle d_{TDOS},d_{EB}\rangle$ (exact)")
        ax.plot(timesteps, res["d_tdos"], label=rf"$d_{{TDOS}}{s}$ (drift+one-step)")
        # bottom: d_EB, cross term, and their sum
        ax_bot.plot(timesteps, res["d_eb"], label=rf"$d_{{EB}}{s}$ (exposure bias)")
        if SQUARED:
            _plot_cross(ax_bot, res["dot_tdos_eb"], r"$|2\langle d_{TDOS},d_{EB}\rangle|$")
            eb_plus_cross = res["d_eb"] + res["dot_tdos_eb"]
            ax_bot.plot(timesteps, eb_plus_cross, "k-", lw=1.5, label=r"$d_{EB}^2+2\langle d_{TDOS},d_{EB}\rangle$")
    else:
        tri_sum = res["d_td"] + res["d_os"] + res["d_eb"]
        ax.plot(timesteps, tri_sum, "k:", lw=1, label=rf"$\Sigma d_i{s}$ ({'Pythagorean' if SQUARED else 'triangle ineq.'})")
        if SQUARED:
            exact_sum = tri_sum + res["dot_eb_os"] + res["dot_eb_td"] + res["dot_os_td"]
            ax.plot(timesteps, exact_sum, "k--", lw=1.5, label=r"$\Sigma d_i^2 + 2\Sigma\langle d_i,d_j\rangle$ (exact)")
        ax.plot(timesteps, res["d_td"], label=rf"$d_{{TD}}{s}$")
        ax.plot(timesteps, res["d_os"], label=rf"$d_{{OS}}{s}$")
        # bottom: d_EB, cross terms, and sum
        ax_bot.plot(timesteps, res["d_eb"], label=rf"$d_{{EB}}{s}$ (exposure bias)")
        if SQUARED:
            for key, lbl in [
                ("dot_eb_os", r"$|2\langle d_{EB},d_{OS}\rangle|$"),
                ("dot_eb_td", r"$|2\langle d_{EB},d_{TD}\rangle|$"),
                ("dot_os_td", r"$|2\langle d_{OS},d_{TD}\rangle|$"),
            ]:
                _plot_cross(ax_bot, res[key], lbl)
            eb_plus_cross = res["d_eb"] + res["dot_eb_os"] + res["dot_eb_td"] + res["dot_os_td"]
            ax_bot.plot(timesteps, eb_plus_cross, "k-", lw=1.5, label=r"$d_{EB}^2+\Sigma 2\langle d_i,d_j\rangle$")

    for a, ylabel in [(ax, "Squared L2 norm" if SQUARED else "L2 norm"),
                      (ax_bot, "Squared L2 norm" if SQUARED else "L2 norm")]:
        a.set_yscale("log")
        a.set_xlabel("AR step t")
        a.set_ylabel(ylabel)
        a.legend(fontsize=8)
    ax.set_title(f"train: {train_config}")
    ax.set_xlabel("")  # remove x-label from top row

# Synchronize y-axis limits across all top plots
if n_configs > 1:
    ylims = [a.get_ylim() for a in axes]
    ymin = min(yl[0] for yl in ylims)
    ymax = max(yl[1] for yl in ylims)
    for a in axes:
        a.set_ylim(ymin, ymax)

# Synchronize y-axis limits across all bottom plots
if n_configs > 1:
    ylims = [ax_b.get_ylim() for ax_b in axes_bot]
    ymin = min(yl[0] for yl in ylims)
    ymax = max(yl[1] for yl in ylims)
    for ax_b in axes_bot:
        ax_b.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "proxy_model_evaluation.pdf", dpi=150)
plt.savefig(OUTPUT_DIR / "proxy_model_evaluation.png", dpi=150)
print("\nSaved proxy_model_evaluation.pdf / .png")

# ── Ranking benchmark plot ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
fig2.suptitle("Ranking benchmark (§11.03.2026): $\\Sigma d_i^2$ vs $\\epsilon^2$ per model", fontsize=11)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, name in enumerate(config_names):
    markers = ["o", "s", "^", "D", "v"]
    mk = markers[i % len(markers)]
    sym = "²" if SQUARED else ""
    ax2.plot(timesteps, total_error_mat[i], color=colors[i],    lw=2,                           label=f"{name}  ε{sym}")
    ax2.plot(timesteps, pyth_sum_mat[i],    color="tab:orange", lw=1, ls="--", marker=mk, ms=4, label=f"{name}  Σd{sym}")
    if SQUARED:
        ax2.plot(timesteps, exact_sum_mat[i], color="tab:green", lw=1, ls=":", marker=mk, ms=4, label=f"{name}  Σd²+cross")

for t in range(AR_STEPS):
    if not rank_match_pyth[t]:
        ax2.axvspan(t + 0.5, t + 1.5, color="red", alpha=0.15)
    elif not rank_match_exact[t]:
        ax2.axvspan(t + 0.5, t + 1.5, color="orange", alpha=0.15)

ax2.set_yscale("log")
ax2.set_xlabel("AR step t")
ax2.set_ylabel("Squared L2 norm" if SQUARED else "L2 norm")
ax2.legend(fontsize=7, ncol=2)
mode_str = "Pythagorean" if SQUARED else "triangle ineq."
ax2.set_title(f"Ranking ({mode_str}): Σd{sym}={rank_match_pyth.mean()*100:.1f}%" +
              (f"  exact={rank_match_exact.mean()*100:.1f}%  |  red=Σd² mismatch  orange=exact mismatch" if SQUARED else "  |  red=mismatch"))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ranking_benchmark.pdf", dpi=150)
plt.savefig(OUTPUT_DIR / "ranking_benchmark.png", dpi=150)
print("Saved ranking_benchmark.pdf / .png")

# ──────────────────────────────────────────────────────────────────────
# Barplot: percentage of each component in total error, per AR step & training method
# ──────────────────────────────────────────────────────────────────────
n_methods = len(config_names)

# Stacking order: bottom=d_TD, middle=dot_product, top=d_OSEB
if NO_PROXY:
    comp_labels = [r"$d_{TD}^2$", r"$2\langle d_{TD}, d_{OSEB}\rangle$", r"$d_{OSEB}^2$"]
    comp_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    comp_keys   = ["d_td", "dot_td_oseb", "d_oseb"]
elif TWO_TERM:
    comp_labels = [r"$d_{TDOS}^2$", r"$2\langle d_{TDOS}, d_{EB}\rangle$", r"$d_{EB}^2$"]
    comp_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    comp_keys   = ["d_tdos", "dot_tdos_eb", "d_eb"]
else:
    comp_labels = [r"$d_{TD}^2$", r"$d_{OS}^2$",
                   r"$2\langle d_{OS},d_{TD}\rangle$", r"$2\langle d_{EB},d_{TD}\rangle$", r"$2\langle d_{EB},d_{OS}\rangle$",
                   r"$d_{EB}^2$"]
    comp_colors = ["#2196F3", "#FF9800", "#795548", "#F44336", "#9C27B0", "#4CAF50"]
    comp_keys   = ["d_td", "d_os", "dot_os_td", "dot_eb_td", "dot_eb_os", "d_eb"]

n_comps = len(comp_keys)

# pct_arr: (n_methods, AR_STEPS, n_comps)
pct_arr = np.zeros((n_methods, AR_STEPS, n_comps))
for i, c in enumerate(config_names):
    r = results[c]
    eps = r["total_error"]
    for j, key in enumerate(comp_keys):
        pct_arr[i, :, j] = r[key] / (eps + 1e-30) * 100

# One subplot per training method, sharing y-axis
fig3, axes3 = plt.subplots(1, n_methods, figsize=(4 * n_methods, 5), sharey=True)
if n_methods == 1:
    axes3 = [axes3]
fig3.suptitle(f"Error decomposition ratios per AR step  |  K={K}  |  {NET_CONFIG}  |  {SCENARIO_NAME}", fontsize=11)

bar_width = 0.7
for i, (ax, cname) in enumerate(zip(axes3, config_names)):
    x_pos = np.arange(1, AR_STEPS + 1)
    bottoms_pos = np.zeros(AR_STEPS)
    bottoms_neg = np.zeros(AR_STEPS)
    for j in range(n_comps):
        vals = pct_arr[i, :, j]
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        ax.bar(x_pos, pos_vals, bar_width, bottom=bottoms_pos,
               label=comp_labels[j], color=comp_colors[j], edgecolor="white", linewidth=0.5)
        ax.bar(x_pos, neg_vals, bar_width, bottom=bottoms_neg,
               color=comp_colors[j], edgecolor="white", linewidth=0.5)
        bottoms_pos += pos_vals
        bottoms_neg += neg_vals

    # Percentage labels on bars
    for t_idx in range(AR_STEPS):
        cum_pos = 0
        cum_neg = 0
        for j in range(n_comps):
            val = pct_arr[i, t_idx, j]
            if abs(val) > 8:  # label only if large enough
                if val > 0:
                    y_center = cum_pos + val / 2
                    cum_pos += val
                else:
                    y_center = cum_neg + val / 2
                    cum_neg += val
                ax.text(x_pos[t_idx], y_center, f"{val:.0f}%",
                        ha="center", va="center", fontsize=6, fontweight="bold")
            else:
                if val > 0:
                    cum_pos += val
                else:
                    cum_neg += val

    ax.axhline(100, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("AR step t")
    ax.set_title(f"train: {cname}", fontsize=10)
    ax.set_xticks(x_pos)

axes3[0].set_ylabel("% of total error $\\epsilon^2$")
# Place legend inside the last subplot (most space typically)
axes3[-1].legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "component_ratios.pdf", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "component_ratios.png", dpi=150, bbox_inches="tight")
print("Saved component_ratios.pdf / .png")
