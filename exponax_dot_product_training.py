"""
Dot-product penalty training (§15.03.2026 in neurips.tex).

Observation: Unrolling training reduces |⟨d_OSEB, d_TD⟩|.
Idea: Explicitly penalize this dot product to achieve a similar effect
without long unrolling.

Loss per step t:
    L_θ(t) = α · A_θ(t) + β · B_θ(t)
where:
    A_θ(t) = ||x̂_t - x_t||²                    (standard one-step MSE)
    B_θ(t) = |⟨x̂_{t+1}, x_{t+1} - GT(x̂_t)⟩|  (dot-product penalty)

Total training loss: L_θ^UT = Σ_{t=1}^{K} L_θ(t), evaluated for K=2.

Then compare with standard training methods using the same
three-term decomposition as exponax_start.py.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import jax
import jax.numpy as jnp
import equinox as eqx
import exponax as ex
import apebench
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from trainax._general_trainer import GeneralTrainer
from trainax._mixer import TrajectorySubStacker
from trainax.configuration import DotProductPenalty, Supervised
from trainax.trainer import SupervisedTrainer

# ──────────────────────────────────────────────────────────────────────
# 0. Configuration
# ──────────────────────────────────────────────────────────────────────
SCENARIO_NAME   = "norm_kdv"
NET_CONFIG      = "UNet;12;2;relu"
TRAIN_CONFIGS   = ["sup;2"] #"one", , "sup;5", ]  # Standard baselines
OPTIM_CONFIG    = "adam;20_000;warmup_cosine;0.0;1e-3;2_000"
NUM_SEEDS       = 1
AR_STEPS        = 10
NUM_TRAJECTORIES = 20

# Dot-product penalty training config
DOT_K           = 2       # Unrolling depth for the new loss
ALPHA           = 1.0     # Weight on MSE term
BETA            = 0.5     # Weight on dot-product penalty
DOT_NUM_STEPS   = 20_000  # Training steps
DOT_LR_MAX      = 1e-3
DOT_WARMUP      = 2_000
DOT_BATCH_SIZE  = 20

# Evaluation decomposition config
NO_PROXY        = True
SQUARED         = True

# Output directory
OUTPUT_DIR = Path("outputs/2026-03-15-dot-product")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. Setup: scenario, steppers, shared ICs
# ──────────────────────────────────────────────────────────────────────
scenario = apebench.scenarios.normalized.KortewegDeVries()
ref_stepper = scenario.get_ref_stepper()

# Shared initial conditions for evaluation
ic_set = ex.build_ic_set(
    scenario.get_ic_generator(),
    num_points=scenario.num_points,
    num_samples=NUM_TRAJECTORIES,
    key=jax.random.PRNGKey(42),
)
warmup_fn = ex.repeat(ref_stepper, scenario.num_warmup_steps)
ic_set    = jax.vmap(warmup_fn)(ic_set)
print(f"Warmed up {NUM_TRAJECTORIES} ICs, shape: {ic_set.shape}")

# GT reference trajectories
gt_trajectories = jax.vmap(
    ex.rollout(ref_stepper, AR_STEPS, include_init=True)
)(ic_set)
print(f"GT trajectory shape: {gt_trajectories.shape}")

# ──────────────────────────────────────────────────────────────────────
# 2. Train all models via trainax directly
# ──────────────────────────────────────────────────────────────────────
train_data = scenario.get_train_data()
print(f"Training data shape: {train_data.shape}")

# Build optimizer from OPTIM_CONFIG so the LR schedule matches DOT_NUM_STEPS.
# (scenario.get_optimizer() uses the scenario's default 10k-step config.)
import optax
_lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=DOT_LR_MAX,
    warmup_steps=DOT_WARMUP,
    decay_steps=DOT_NUM_STEPS,
    end_value=0.0,
)
optimizer = optax.adam(_lr_schedule)

def train_supervised(train_config, seed=0):
    """Train with standard supervised rollout ('one' or 'sup;N')."""
    args = train_config.split(";")
    num_rollout_steps = 1 if args[0] == "one" else int(args[1])
    model = scenario.get_neural_stepper(
        task_config="predict", network_config=NET_CONFIG,
        key=jax.random.PRNGKey(seed),
    )
    trainer = SupervisedTrainer(
        train_data,
        optimizer=optimizer,
        num_training_steps=DOT_NUM_STEPS,
        batch_size=DOT_BATCH_SIZE,
        num_rollout_steps=num_rollout_steps,
    )
    print(f"\n{'='*60}\nTraining: {train_config}  (rollout={num_rollout_steps})\n{'='*60}")
    model, loss_hist = trainer(model, jax.random.PRNGKey(seed))
    return model, loss_hist

def train_dot_product(seed=0):
    """Train with dot-product penalty (§15.03.2026)."""
    # sub_trajectory_len: K+2 when β>0 (needs x_{K+1} for B term), else K+1
    sub_trajectory_len = DOT_K + 2 if BETA > 0 else DOT_K + 1
    model = scenario.get_neural_stepper(
        task_config="predict", network_config=NET_CONFIG,
        key=jax.random.PRNGKey(seed),
    )
    stacker = TrajectorySubStacker(
        train_data, sub_trajectory_len=sub_trajectory_len, do_sub_stacking=True,
    )
    trainer = GeneralTrainer(
        stacker,
        DotProductPenalty(num_rollout_steps=DOT_K, alpha=ALPHA, beta=BETA),
        ref_stepper=ref_stepper,
        optimizer=optimizer,
        num_minibatches=DOT_NUM_STEPS,
        batch_size=DOT_BATCH_SIZE,
    )
    print(f"\n{'='*60}\nTraining: dot;K={DOT_K};β={BETA}\n{'='*60}")
    model, loss_hist = trainer(model, jax.random.PRNGKey(seed))
    return model, loss_hist

# Train everything
all_configs = TRAIN_CONFIGS + [f"dot;K={DOT_K};β={BETA}"]
all_models  = []
loss_histories = {}

for tc in TRAIN_CONFIGS:
    model, loss_hist = train_supervised(tc)
    all_models.append(model)
    loss_histories[tc] = loss_hist

dot_model, loss_hist = train_dot_product()
all_models.append(dot_model)
loss_histories[f"dot;K={DOT_K};β={BETA}"] = loss_hist

# ──────────────────────────────────────────────────────────────────────
# 3. Evaluate all models
# ──────────────────────────────────────────────────────────────────────

# ── Decomposition loop ──────────────────────────────────────────────
results = {}

def norm(v):
    sq = jnp.sum(v ** 2, axis=(-2, -1))
    return jnp.mean(jnp.sqrt(sq) if not SQUARED else sq)

def dot(a, b):
    return jnp.mean(jnp.sum(a * b, axis=(-2, -1)))

for config_name, neural_stepper in zip(all_configs, all_models):
    print("\n" + "=" * 60)
    print(f"Evaluating: {config_name}")
    print("=" * 60)

    nn_trajectories = jax.vmap(
        ex.rollout(neural_stepper, AR_STEPS, include_init=True)
    )(ic_set)

    total_error  = np.zeros(AR_STEPS)
    d_td         = np.zeros(AR_STEPS)
    d_oseb       = np.zeros(AR_STEPS)
    dot_td_oseb  = np.zeros(AR_STEPS)

    for t in range(AR_STEPS):
        nn_states_t = nn_trajectories[:, t]
        nn_next_t   = nn_trajectories[:, t + 1]
        gt_next_t   = gt_trajectories[:, t + 1]

        total_error[t] = float(norm(nn_next_t - gt_next_t))

        gt_on_nn_t = jax.vmap(ref_stepper)(nn_states_t)
        vec_td   = gt_next_t  - gt_on_nn_t
        vec_oseb = gt_on_nn_t - nn_next_t

        d_td[t]   = float(norm(vec_td))
        d_oseb[t] = float(norm(vec_oseb))
        if SQUARED:
            dot_td_oseb[t] = float(2 * dot(vec_td, vec_oseb))

        if t % 5 == 0:
            sym = "²" if SQUARED else ""
            tri_sum   = d_td[t] + d_oseb[t]
            exact_sum = tri_sum + dot_td_oseb[t]
            print(f"  t={t+1:>3d}  ε²={total_error[t]:.3e}  "
                  f"d_TD²={d_td[t]:.3e}  d_OSEB²={d_oseb[t]:.3e}  "
                  f"sum={tri_sum:.3e}  exact={exact_sum:.3e}  "
                  f"cross/ε²={dot_td_oseb[t]/(total_error[t]+1e-30):+.3f}")

    results[config_name] = {
        "total_error": total_error,
        "d_td": d_td,
        "d_oseb": d_oseb,
        "dot_td_oseb": dot_td_oseb,
    }

# ──────────────────────────────────────────────────────────────────────
# 5. Plots
# ──────────────────────────────────────────────────────────────────────
config_names = list(results.keys())
n_configs = len(config_names)
timesteps = np.arange(1, AR_STEPS + 1)

# ── Plot 1: Decomposition per model ──────────────────────────────────
fig, axes_grid = plt.subplots(2, n_configs, figsize=(5 * n_configs, 8),
                              gridspec_kw={"height_ratios": [2, 1]})
if n_configs == 1:
    axes_grid = axes_grid.reshape(2, 1)
axes     = axes_grid[0]
axes_bot = axes_grid[1]
fig.suptitle(f"No-proxy decomposition  |  {NET_CONFIG}  |  {SCENARIO_NAME}\n"
             f"Dot-product penalty: K={DOT_K}, α={ALPHA}, β={BETA}",
             fontsize=12)

def _plot_cross(ax, vals, label):
    t, v = timesteps[1:], vals[1:]
    line, = ax.plot(t, np.abs(v), ls="--", alpha=0.6, label=label)
    neg_mask = v < 0
    if neg_mask.any():
        ax.scatter(t[neg_mask], np.abs(v[neg_mask]),
                   color=line.get_color(), marker="x", s=40, zorder=5)

for ax, ax_bot, (train_config, res) in zip(axes, axes_bot, results.items()):
    ax.plot(timesteps, res["total_error"], "k-", lw=2, label=r"$\epsilon^2$ (total)")
    tri_sum = res["d_td"] + res["d_oseb"]
    ax.plot(timesteps, tri_sum, "k:", lw=1,
            label=r"$d_{TD}^2+d_{OSEB}^2$ (Pythagorean)")
    exact_sum = tri_sum + res["dot_td_oseb"]
    ax.plot(timesteps, exact_sum, "k--", lw=1.5,
            label=r"$+2\langle d_{TD},d_{OSEB}\rangle$ (exact)")
    ax.plot(timesteps, res["d_td"], label=r"$d_{TD}^2$")

    ax_bot.plot(timesteps, res["d_oseb"], label=r"$d_{OSEB}^2$")
    _plot_cross(ax_bot, res["dot_td_oseb"], r"$|2\langle d_{TD},d_{OSEB}\rangle|$")
    oseb_plus_cross = res["d_oseb"] + res["dot_td_oseb"]
    ax_bot.plot(timesteps, oseb_plus_cross, "k-", lw=1.5,
                label=r"$d_{OSEB}^2+2\langle\rangle$")

    for a, ylabel in [(ax, "Squared L2 norm"), (ax_bot, "Squared L2 norm")]:
        a.set_yscale("log")
        a.set_xlabel("AR step t")
        a.set_ylabel(ylabel)
        a.legend(fontsize=7)
    ax.set_title(f"train: {train_config}")
    ax.set_xlabel("")

# Sync y-axes
if n_configs > 1:
    for ax_row in [axes, axes_bot]:
        ylims = [a.get_ylim() for a in ax_row]
        ymin = min(yl[0] for yl in ylims)
        ymax = max(yl[1] for yl in ylims)
        for a in ax_row:
            a.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "decomposition.pdf", dpi=150)
plt.savefig(OUTPUT_DIR / "decomposition.png", dpi=150)
print(f"\nSaved decomposition plots to {OUTPUT_DIR}")

# ── Plot 2: Dot-product comparison across methods ────────────────────
fig2, (ax_dot, ax_err) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle(f"Dot-product penalty effect  |  {NET_CONFIG}  |  {SCENARIO_NAME}\n"
              f"New method: K={DOT_K}, α={ALPHA}, β={BETA}", fontsize=12)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
markers = ["o", "s", "^", "D", "v", "P", "X"]

for i, name in enumerate(config_names):
    res = results[name]
    mk = markers[i % len(markers)]
    c = colors[i % len(colors)]
    lw = 2.5 if "dot" in name else 1.5
    ls = "-" if "dot" in name else "--"

    # Absolute dot product
    ax_dot.plot(timesteps, np.abs(res["dot_td_oseb"]),
                color=c, marker=mk, ms=4, lw=lw, ls=ls, label=name)

    # Total error
    ax_err.plot(timesteps, res["total_error"],
                color=c, marker=mk, ms=4, lw=lw, ls=ls, label=name)

ax_dot.set_yscale("log")
ax_dot.set_xlabel("AR step t")
ax_dot.set_ylabel(r"$|2\langle d_{TD}, d_{OSEB}\rangle|$")
ax_dot.set_title("Dot product magnitude")
ax_dot.legend(fontsize=8)

ax_err.set_yscale("log")
ax_err.set_xlabel("AR step t")
ax_err.set_ylabel(r"$\epsilon^2$ (total error)")
ax_err.set_title("Total trajectory error")
ax_err.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dot_product_comparison.pdf", dpi=150)
plt.savefig(OUTPUT_DIR / "dot_product_comparison.png", dpi=150)
print(f"Saved dot-product comparison to {OUTPUT_DIR}")

# ── Plot 3: Component ratios (barplot) ───────────────────────────────
comp_labels = [r"$d_{TD}^2$", r"$2\langle d_{TD}, d_{OSEB}\rangle$", r"$d_{OSEB}^2$"]
comp_colors = ["#2196F3", "#4CAF50", "#FF9800"]
comp_keys   = ["d_td", "dot_td_oseb", "d_oseb"]
n_comps = len(comp_keys)

pct_arr = np.zeros((n_configs, AR_STEPS, n_comps))
for i, c in enumerate(config_names):
    r = results[c]
    eps = r["total_error"]
    for j, key in enumerate(comp_keys):
        pct_arr[i, :, j] = r[key] / (eps + 1e-30) * 100

fig3, axes3 = plt.subplots(1, n_configs, figsize=(4 * n_configs, 5), sharey=True)
if n_configs == 1:
    axes3 = [axes3]
fig3.suptitle(f"Error decomposition ratios  |  {NET_CONFIG}  |  {SCENARIO_NAME}\n"
              f"Dot-product penalty: K={DOT_K}, α={ALPHA}, β={BETA}", fontsize=11)

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
               label=comp_labels[j], color=comp_colors[j],
               edgecolor="white", linewidth=0.5)
        ax.bar(x_pos, neg_vals, bar_width, bottom=bottoms_neg,
               color=comp_colors[j], edgecolor="white", linewidth=0.5)
        bottoms_pos += pos_vals
        bottoms_neg += neg_vals

    for t_idx in range(AR_STEPS):
        cum_pos = 0
        cum_neg = 0
        for j in range(n_comps):
            val = pct_arr[i, t_idx, j]
            if abs(val) > 8:
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

axes3[0].set_ylabel(r"% of total error $\epsilon^2$")
axes3[-1].legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "component_ratios.pdf", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "component_ratios.png", dpi=150, bbox_inches="tight")
print(f"Saved component ratios to {OUTPUT_DIR}")

# ── Plot 4: Training loss curves ─────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 4))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, (name, lh) in enumerate(loss_histories.items()):
    lh = np.array(lh)
    ax4.plot(lh, lw=0.5, alpha=0.2, color=colors[i % len(colors)])
    window = min(500, len(lh) // 10)
    if window > 1:
        smoothed = np.convolve(lh, np.ones(window)/window, mode="valid")
        ax4.plot(np.arange(window-1, len(lh)), smoothed, lw=1.5,
                 color=colors[i % len(colors)], label=name)
ax4.set_xlabel("Training step")
ax4.set_ylabel("Loss")
ax4.set_title(f"Training loss curves  |  K={DOT_K}, α={ALPHA}, β={BETA}")
ax4.legend(fontsize=8)
ax4.set_yscale("log")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.pdf", dpi=150)
plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
print(f"Saved training loss to {OUTPUT_DIR}")

# ── Summary table ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary: |⟨d_TD, d_OSEB⟩| / ε² at each timestep")
print("=" * 60)
header = f"{'t':>4}"
for name in config_names:
    header += f"  {name:>16}"
print(header)
for t in range(AR_STEPS):
    row = f"  {t+1:>2}"
    for name in config_names:
        r = results[name]
        ratio = abs(r["dot_td_oseb"][t]) / (r["total_error"][t] + 1e-30)
        row += f"  {ratio:>16.3f}"
    print(row)
