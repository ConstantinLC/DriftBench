"""
Three-term error decomposition (Section from neurips.tex, §11.03.2026-3).

For each AR step t, compute and verify:
    ε²(t+1)  = ||x̂_{t+1} - x_{t+1}||²               (total trajectory error, squared)
    TD²(t) = ||x_{t+1} - GT(x̂_t^ID)||²            (trajectory drift, squared)
    OS²(t) = ||GT(x̂_t^ID) - NN(x̂_t^ID)||²        (one-step error, squared)
    EB²(t) = ||NN(x̂_t^ID) - NN(x̂_t)||²           (exposure bias, squared)

"Another possibility" definition (neurips.tex 11/03/2026):
    pivot is GT(x̂_t^ID) — GT applied to the proxy — instead of GT(x̂_t).

In high dimensions, vectors are near-orthogonal, so:
    ε²(t+1) ≈ TD²(t) + OS²(t) + EB²(t)  (Pythagorean approx.)
"""

import os
import argparse
import mix_chain_training
mix_chain_training.patch_get_trainer()

def parse_args():
    parser = argparse.ArgumentParser(description="Three-term error decomposition for neural emulators")
    parser.add_argument("--gpu_id", type=str, default="4", help="CUDA device ID")
    parser.add_argument("--scenario_name", type=str, default="phy_kolm_flow",
                        help="Scenario key (e.g. phy_kolm_flow, norm_ks)")
    parser.add_argument("--net_configs", type=str, nargs="+", default=["UNet;12;2;relu"],
                        help="Network configs (e.g. 'UNet;12;2;relu')")
    parser.add_argument("--train_configs", type=str, nargs="+",
                        default=["one", "sup;2", "sup;3", "sup;4", "sup;5", "sup;6", "sup;7", "sup;8"], #, "tdcrossos;2;true", "tdcrossos;2;true;1.0;0.0"], #"tdcross;2;false", "tdcross;2;true",, , "tdcrossos;3;true"], #, "sup;8;true;1", "fg;8", "fg;8;true;1"], #"one@40000->mix;2;6;false;false@2000"],
                        help="Training configs (e.g. 'one' 'sup;2')")
    parser.add_argument("--n_epochs", type=str, default='40_000',
                        help="Number of training epochs (overrides value in OPTIM_CONFIGS)")
    parser.add_argument("--lrs", type=str, nargs="+", default=["1e-3"],
                        help="Peak learning rate(s) for the warmup_cosine schedule. Pass several "
                             "to sweep, e.g. --lrs 3e-4 1e-3 3e-3; each LR is trained + evaluated "
                             "as a separate model so rankings can be compared across LR.")
    parser.add_argument("--num_seeds", type=int, default=1, help="Number of random seeds")
    parser.add_argument("--ar_steps", type=int, default=5, help="Autoregressive evaluation steps")
    parser.add_argument("--predict_steps", type=int, default=1,
                         help="Number of reference-simulator steps the model must predict in a single "
                              "forward call (e.g. 2 -> model predicts x_2 from x_0). The simulator's own "
                              "dt/integration is unchanged; only the gap between supervised snapshots grows.")
    return parser.parse_args()


args = parse_args()

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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
SCENARIO_NAME    = args.scenario_name
NET_CONFIGS      = args.net_configs
TRAIN_CONFIGS    = args.train_configs
# One optim config per requested peak LR. Grammar (see apebench
# components/_optimization.py): adam;<num_steps>;warmup_cosine;<init_lr>;<peak_lr>;<warmup_steps>.
# The peak LR is the 5th field. Different optim configs are kept as separate
# models throughout the pipeline (apebench records the config in each run's
# identity), so listing several LRs here performs an LR sweep in one run.
OPTIM_CONFIGS    = [f"adam;{str(args.n_epochs)};warmup_cosine;0.0;{lr};2_000" for lr in args.lrs]
NUM_SEEDS        = args.num_seeds
AR_STEPS         = args.ar_steps
PREDICT_STEPS    = args.predict_steps
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
    "phy_kolm_flow": apebench.scenarios.physical.KolmogorovFlow,
    "phy_burgers": apebench.scenarios.physical.BurgersSingleChannel,
    "phy_kdv": apebench.scenarios.physical.KortewegDeVries,
    "phy_ks": apebench.scenarios.physical.KuramotoSivashinsky,
}

if SCENARIO_NAME not in SCENARIO_MAP:
    raise ValueError(f"Unknown scenario '{SCENARIO_NAME}'. Choose from: {list(SCENARIO_MAP)}")

if PREDICT_STEPS > 1:
    # Wrap the reference stepper so the model has to predict PREDICT_STEPS
    # simulator calls ahead in one forward pass (e.g. x_2 from x_0), without
    # touching the simulator's own dt/integration scheme.
    _base_scenario_cls = SCENARIO_MAP[SCENARIO_NAME]

    class _MultiStepScenario(_base_scenario_cls):
        def get_ref_stepper(self):
            return ex.repeat(super().get_ref_stepper(), PREDICT_STEPS)

    _MultiStepScenario.__name__ = f"{_base_scenario_cls.__name__}_x{PREDICT_STEPS}"
    SCENARIO_MAP[SCENARIO_NAME] = _MultiStepScenario
    # apebench's regular (non-chained) training path resolves scenarios by
    # name through its own internal registry, so patch that entry too.
    apebench.scenarios.scenario_dict[SCENARIO_NAME] = _MultiStepScenario

_output_name = SCENARIO_NAME if PREDICT_STEPS == 1 else f"{SCENARIO_NAME}_x{PREDICT_STEPS}"
OUTPUT_DIR = Path("outputs") / _output_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Per-config figure directory. FIG_DIR = OUTPUT_DIR/<hash-of-run-params>,
# so every figure produced below lands in a folder keyed by the run's
# parameters — different configs get their own folder instead of
# overwriting each other's figures; re-running the same config reuses it.
# (Trained weights are NOT nested under FIG_DIR: they're cached in
# OUTPUT_DIR/weights keyed by their own net/train/optim identity so
# unrelated config changes, e.g. AR_STEPS, don't force retraining.)
# ──────────────────────────────────────────────────────────────────────
import hashlib
import json
from datetime import datetime

_run_config = {
    "scenario_name": SCENARIO_NAME,
    "net_configs": NET_CONFIGS,
    "train_configs": TRAIN_CONFIGS,
    "optim_configs": OPTIM_CONFIGS,
    "num_seeds": NUM_SEEDS,
    "ar_steps": AR_STEPS,
    "predict_steps": PREDICT_STEPS,
    "num_trajectories": NUM_TRAJECTORIES,
    "K": K,
    "n_opt_steps": N_OPT_STEPS,
    "two_term": TWO_TERM,
    "no_proxy": NO_PROXY,
    "squared": SQUARED,
    "opt_lr": OPT_LR,
}
_config_hash = hashlib.sha1(
    json.dumps(_run_config, sort_keys=True).encode()
).hexdigest()[:8]

FIG_DIR = OUTPUT_DIR / _config_hash
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(FIG_DIR / "config.json", "w") as f:
    json.dump(_run_config, f, indent=2)

_config_fig_path = FIG_DIR / "config_summary.png"
if not _config_fig_path.exists():
    _cfg_lines = [f"Run config  (hash={_config_hash})",
                  f"saved: {datetime.now().isoformat(timespec='seconds')}", ""]
    _cfg_lines += [f"{k}: {v}" for k, v in _run_config.items()]
    fig_cfg, ax_cfg = plt.subplots(figsize=(7, 0.35 * len(_run_config) + 1.4))
    ax_cfg.axis("off")
    ax_cfg.text(0.02, 0.98, "\n".join(_cfg_lines), va="top", ha="left",
                fontsize=9, family="monospace", transform=ax_cfg.transAxes)
    fig_cfg.savefig(_config_fig_path, dpi=150, bbox_inches="tight")
    fig_cfg.savefig(FIG_DIR / "config_summary.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig_cfg)
    print(f"Saved config summary -> {_config_fig_path}")
else:
    print(f"Config summary already exists for hash {_config_hash} -> {_config_fig_path}")

# ──────────────────────────────────────────────────────────────────────
# Setup: scenario, steppers, shared ICs
# ──────────────────────────────────────────────────────────────────────
_scenario_kwargs = {"num_spatial_dims": 2} if SCENARIO_NAME == "phy_kolm_flow" else {}
scenario = SCENARIO_MAP[SCENARIO_NAME](**_scenario_kwargs)
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
_scenario_extra = _scenario_kwargs

# Chained train_configs (syntax "cfg1@epochs1->cfg2@epochs2->...") are trained
# stage-by-stage, carrying weights forward, instead of via apebench's
# from-scratch run_study_convenience.
chain_train_configs   = [tc for tc in TRAIN_CONFIGS if mix_chain_training.is_chain_train_config(tc)]
regular_train_configs = [tc for tc in TRAIN_CONFIGS if not mix_chain_training.is_chain_train_config(tc)]

chain_weight_paths = {}
for net_config in NET_CONFIGS:
    for tc in chain_train_configs:
        for oc in OPTIM_CONFIGS:
            weight_path = (
                OUTPUT_DIR / "weights" / "chained"
                / f"{SCENARIO_NAME}__{net_config}__{tc}__{oc}__seeds{NUM_SEEDS}.eqx"
            )
            mix_chain_training.train_chain(
                scenario_cls=SCENARIO_MAP[SCENARIO_NAME],
                scenario_kwargs=_scenario_extra,
                network_config=net_config,
                train_config=tc,
                base_optim_config=oc,
                start_seed=0,
                num_seeds=NUM_SEEDS,
                weight_path=weight_path,
            )
            chain_weight_paths[(net_config, tc, oc)] = weight_path

all_run_configs = [
    {
        "scenario": SCENARIO_NAME, "task": "predict", "net": net_config,
        "train": tc, "start_seed": 0, "num_seeds": NUM_SEEDS,
        "optim_config": oc,
        **_scenario_extra,
    }
    for net_config in NET_CONFIGS
    for tc in regular_train_configs
    for oc in OPTIM_CONFIGS
]
if all_run_configs:
    _, loss_df, _, regular_weight_paths = apebench.run_study_convenience(
        configs=all_run_configs, base_path=str(OUTPUT_DIR / "weights"),
        do_loss=True,
    )
else:
    loss_df, regular_weight_paths = None, []
regular_run_keys = [(net, tc, oc) for net in NET_CONFIGS for tc in regular_train_configs for oc in OPTIM_CONFIGS]
regular_weight_paths = dict(zip(regular_run_keys, regular_weight_paths))

# zip run order: (net0,train0,optim0), (net0,train0,optim1), ..., (net0,train1,optim0), ...
run_keys = [(net, tc, oc) for net in NET_CONFIGS for tc in TRAIN_CONFIGS for oc in OPTIM_CONFIGS]
all_weight_paths = [
    chain_weight_paths[k] if mix_chain_training.is_chain_train_config(k[1]) else regular_weight_paths[k]
    for k in run_keys
]

# ──────────────────────────────────────────────────────────────────────
# Main loop: compute three-term decomposition per train config
# ──────────────────────────────────────────────────────────────────────
results = {}

for (net_config, train_config, optim_config), weight_path in zip(run_keys, all_weight_paths):
    result_key = f"{net_config}|{train_config}|{optim_config}"
    print("\n" + "=" * 60)
    print(f"Net: {net_config}  |  Train: {train_config}  |  Optim: {optim_config}")
    print("=" * 60)

    neural_stepper_single = scenario.get_neural_stepper(
        task_config="predict", network_config=net_config,
        key=jax.random.PRNGKey(0),
    )
    if NUM_SEEDS > 1:
        arr, static = eqx.partition(neural_stepper_single, eqx.is_array)
        arr_stacked = jax.tree_util.tree_map(lambda x: jnp.stack([x] * NUM_SEEDS), arr)
        neural_stepper_like = eqx.combine(arr_stacked, static)
    else:
        neural_stepper_like = neural_stepper_single
    neural_stepper_all = eqx.tree_deserialise_leaves(str(weight_path), neural_stepper_like)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(neural_stepper_single, eqx.is_array)))
    print(f"  Parameters: {n_params:,}")
    # Accumulators — summed over seeds, divided after the loop
    total_error = np.zeros(AR_STEPS)
    # Per-seed values (NUM_SEEDS, AR_STEPS) kept so we can report the
    # across-seed spread (std / SEM) and check whether ranking differences
    # between unrolling methods exceed seed-to-seed noise.
    total_error_seeds = np.zeros((NUM_SEEDS, AR_STEPS))
    if NO_PROXY:
        td        = np.zeros(AR_STEPS)
        oseb      = np.zeros(AR_STEPS)
        oseb_gt   = np.zeros(AR_STEPS)
        cross_td_oseb = np.zeros(AR_STEPS)
        cos_td_oseb   = np.zeros(AR_STEPS)
        td_seeds   = np.zeros((NUM_SEEDS, AR_STEPS))
        oseb_seeds = np.zeros((NUM_SEEDS, AR_STEPS))
    elif TWO_TERM:
        tdos      = np.zeros(AR_STEPS)
        eb        = np.zeros(AR_STEPS)
        dot_tdos_eb = np.zeros(AR_STEPS)
        cos_tdos_eb = np.zeros(AR_STEPS)
    else:
        td      = np.zeros(AR_STEPS)
        os_err      = np.zeros(AR_STEPS)
        eb      = np.zeros(AR_STEPS)
        dot_eb_os = np.zeros(AR_STEPS)
        dot_eb_td = np.zeros(AR_STEPS)
        dot_os_td = np.zeros(AR_STEPS)
        cos_eb_os = np.zeros(AR_STEPS)
        cos_eb_td = np.zeros(AR_STEPS)
        cos_os_td = np.zeros(AR_STEPS)

    def norm(v):
        sq = jnp.sum(v ** 2, axis=(-2, -1))
        return jnp.mean(jnp.sqrt(sq) if not SQUARED else sq)

    def dot(a, b):
        return jnp.mean(jnp.sum(a * b, axis=(-2, -1)))

    def cosine_sim(a, b):
        """Per-sample cosine similarity, averaged over the batch. Unlike
        `dot`/`norm` (which are scale-dependent), this isolates whether error
        *directions* are becoming more orthogonal, vs. just growing in
        magnitude."""
        dot_per_sample = jnp.sum(a * b, axis=(-2, -1))
        norm_a = jnp.sqrt(jnp.sum(a ** 2, axis=(-2, -1)))
        norm_b = jnp.sqrt(jnp.sum(b ** 2, axis=(-2, -1)))
        return jnp.mean(dot_per_sample / (norm_a * norm_b + 1e-30))

    for s in range(NUM_SEEDS):
        if NUM_SEEDS > 1:
            arr_all, static_all = eqx.partition(neural_stepper_all, eqx.is_array)
            arr_s = jax.tree_util.tree_map(lambda x: x[s], arr_all)
            neural_stepper = eqx.combine(arr_s, static_all)
        else:
            neural_stepper = neural_stepper_all
        nn_trajectories = jax.vmap(
            ex.rollout(neural_stepper, AR_STEPS, include_init=True)
        )(ic_set)
        if s == 0:
            print(f"  NN trajectory shape: {nn_trajectories.shape}")

        for t in range(AR_STEPS):
            nn_states_t = nn_trajectories[:, t]
            nn_next_t   = nn_trajectories[:, t + 1]
            gt_next_t   = gt_trajectories[:, t + 1]

            _te = float(norm(nn_next_t - gt_next_t))
            total_error[t] += _te
            total_error_seeds[s, t] = _te

            if NO_PROXY:
                gt_on_nn_t = jax.vmap(ref_stepper)(nn_states_t)
                vec_td   = gt_next_t  - gt_on_nn_t
                vec_oseb = gt_on_nn_t - nn_next_t
                _td = float(norm(vec_td)); _os = float(norm(vec_oseb))
                td[t]   += _td;  td_seeds[s, t]   = _td
                oseb[t] += _os;  oseb_seeds[s, t] = _os
                if SQUARED:
                    cross_td_oseb[t] += float(2 * dot(vec_td, vec_oseb))
                cos_td_oseb[t] += float(cosine_sim(vec_td, vec_oseb))
                gt_states_t   = gt_trajectories[:, t]
                nn_on_gt_t    = jax.vmap(neural_stepper)(gt_states_t)
                vec_oseb_gt   = gt_next_t - nn_on_gt_t
                oseb_gt[t]   += float(norm(vec_oseb_gt))
            else:
                proxy_t, _ = optimize_batch(nn_states_t, nn_next_t)
                nn_on_proxy_t = jax.vmap(neural_stepper)(proxy_t)
                vec_eb = nn_on_proxy_t - nn_next_t
                eb[t] += float(norm(vec_eb))
                if TWO_TERM:
                    vec_tdos  = gt_next_t - nn_on_proxy_t
                    tdos[t] += float(norm(vec_tdos))
                    if SQUARED:
                        dot_tdos_eb[t] += float(2 * dot(vec_tdos, vec_eb))
                    cos_tdos_eb[t] += float(cosine_sim(vec_tdos, vec_eb))
                else:
                    gt_on_proxy_t = jax.vmap(ref_stepper)(proxy_t)
                    vec_td = gt_next_t     - gt_on_proxy_t
                    vec_os = gt_on_proxy_t - nn_on_proxy_t
                    td[t] += float(norm(vec_td))
                    os_err[t] += float(norm(vec_os))
                    if SQUARED:
                        dot_eb_os[t] += float(2 * dot(vec_eb, vec_os))
                        dot_eb_td[t] += float(2 * dot(vec_eb, vec_td))
                        dot_os_td[t] += float(2 * dot(vec_os, vec_td))
                    cos_eb_os[t] += float(cosine_sim(vec_eb, vec_os))
                    cos_eb_td[t] += float(cosine_sim(vec_eb, vec_td))
                    cos_os_td[t] += float(cosine_sim(vec_os, vec_td))

    # Average over seeds
    total_error /= NUM_SEEDS
    # Across-seed sample std (ddof=1 when we have >1 seed) for error bars.
    _ddof = 1 if NUM_SEEDS > 1 else 0
    total_error_std = total_error_seeds.std(axis=0, ddof=_ddof)
    if NO_PROXY:
        td /= NUM_SEEDS; oseb /= NUM_SEEDS; oseb_gt /= NUM_SEEDS; cross_td_oseb /= NUM_SEEDS
        cos_td_oseb /= NUM_SEEDS
        td_std   = td_seeds.std(axis=0, ddof=_ddof)
        oseb_std = oseb_seeds.std(axis=0, ddof=_ddof)
    elif TWO_TERM:
        tdos /= NUM_SEEDS; eb /= NUM_SEEDS; dot_tdos_eb /= NUM_SEEDS
        cos_tdos_eb /= NUM_SEEDS
    else:
        td /= NUM_SEEDS; os_err /= NUM_SEEDS; eb /= NUM_SEEDS
        dot_eb_os /= NUM_SEEDS; dot_eb_td /= NUM_SEEDS; dot_os_td /= NUM_SEEDS
        cos_eb_os /= NUM_SEEDS; cos_eb_td /= NUM_SEEDS; cos_os_td /= NUM_SEEDS

    sym = "²" if SQUARED else ""
    for t in range(AR_STEPS):
        if t % 5 == 0:
            if NO_PROXY:
                tri_sum   = td[t] + oseb[t]
                exact_sum = tri_sum + (cross_td_oseb[t] if SQUARED else 0)
                print(f"  t={t+1:>3d}  ε{sym}={total_error[t]:.3e}  "
                      f"TD{sym}={td[t]:.3e}  OSEB{sym}={oseb[t]:.3e}  "
                      f"sum={tri_sum:.3e}" +
                      (f"  exact={exact_sum:.3e}  cross/ε²={cross_td_oseb[t]/(total_error[t]+1e-30):+.3f}" if SQUARED else "") +
                      f"  cos(TD,OSEB)={cos_td_oseb[t]:+.3f}")
            elif TWO_TERM:
                tri_sum   = tdos[t] + eb[t]
                exact_sum = tri_sum + (dot_tdos_eb[t] if SQUARED else 0)
                print(f"  t={t+1:>3d}  ε{sym}={total_error[t]:.3e}  "
                      f"TDOS{sym}={tdos[t]:.3e}  EB{sym}={eb[t]:.3e}  "
                      f"sum={tri_sum:.3e}" +
                      (f"  exact={exact_sum:.3e}  cross/ε²={dot_tdos_eb[t]/(total_error[t]+1e-30):+.3f}" if SQUARED else "") +
                      f"  cos(TDOS,EB)={cos_tdos_eb[t]:+.3f}")
            else:
                tri_sum   = td[t] + os_err[t] + eb[t]
                exact_sum = tri_sum + (dot_eb_os[t] + dot_eb_td[t] + dot_os_td[t] if SQUARED else 0)
                print(f"  t={t+1:>3d}  ε{sym}={total_error[t]:.3e}  "
                      f"Σd{sym}={tri_sum:.3e}" +
                      (f"  exact={exact_sum:.3e}  cross/ε²={(dot_eb_os[t]+dot_eb_td[t]+dot_os_td[t])/(total_error[t]+1e-30):+.3f}" if SQUARED else "") +
                      f"  cos(EB,OS)={cos_eb_os[t]:+.3f}  cos(EB,TD)={cos_eb_td[t]:+.3f}  cos(OS,TD)={cos_os_td[t]:+.3f}")

    if NO_PROXY:
        results[result_key] = {
            "total_error": total_error,
            "total_error_std": total_error_std,
            "num_seeds": NUM_SEEDS,
            "td": td,
            "td_std": td_std,
            "os": oseb,
            "os_std": oseb_std,
            "os_gt": oseb_gt,
            "cross_td_os": cross_td_oseb,
            "cos_td_os": cos_td_oseb,
            # Per-seed raw arrays (NUM_SEEDS, AR_STEPS), kept so downstream
            # fitting scripts (e.g. the epsilon(t+1) = K*epsilon(t) + os(t+1)
            # recurrence fit) can pool over seeds instead of only the
            # seed-averaged curve.
            "total_error_seeds": total_error_seeds,
            "td_seeds": td_seeds,
            "os_seeds": oseb_seeds,
        }
    elif TWO_TERM:
        results[result_key] = {
            "total_error": total_error,
            "total_error_std": total_error_std,
            "num_seeds": NUM_SEEDS,
            "tdos": tdos,
            "eb": eb,
            "dot_tdos_eb": dot_tdos_eb,
            "cos_tdos_eb": cos_tdos_eb,
        }
    else:
        results[result_key] = {
            "total_error": total_error,
            "total_error_std": total_error_std,
            "num_seeds": NUM_SEEDS,
            "td": td,
            "os_err": os_err,
            "eb": eb,
            "dot_eb_os": dot_eb_os,
            "dot_eb_td": dot_eb_td,
            "dot_os_td": dot_os_td,
            "cos_eb_os": cos_eb_os,
            "cos_eb_td": cos_eb_td,
            "cos_os_td": cos_os_td,
        }


# ──────────────────────────────────────────────────────────────────────
# Raw-array dump — one .npz per run, keyed by result_key, holding every
# per-seed array (NUM_SEEDS, AR_STEPS) plus the seed-averaged curves. Lets
# downstream scripts (e.g. fit_error_recurrence.py) refit quantities like
# the epsilon(t+1) = K*epsilon(t) + os(t+1) recurrence without rerunning
# inference.
# ──────────────────────────────────────────────────────────────────────
if NO_PROXY:
    _raw_dump = {"config_names": np.array(list(results.keys()), dtype=object)}
    for _cname, _r in results.items():
        for _k, _v in _r.items():
            _raw_dump[f"{_cname}||{_k}"] = np.asarray(_v)
    np.savez(FIG_DIR / "raw_arrays.npz", **_raw_dump)
    print(f"Saved raw_arrays.npz -> {FIG_DIR / 'raw_arrays.npz'}")

# ──────────────────────────────────────────────────────────────────────
# Ranking benchmark (§11.03.2026)
# ──────────────────────────────────────────────────────────────────────
config_names = list(results.keys())

# Stack arrays: shape (n_models, AR_STEPS)
total_error_mat = np.stack([results[c]["total_error"] for c in config_names])
total_error_std_mat = np.stack([results[c]["total_error_std"] for c in config_names])
# Standard error of the across-seed mean: the relevant scale for deciding
# whether two methods' total errors are actually distinguishable.
_n_seeds_used = max(results[config_names[0]].get("num_seeds", 1), 1)
total_error_sem_mat = total_error_std_mat / np.sqrt(_n_seeds_used)

timesteps = np.arange(1, AR_STEPS + 1)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ── Total error per model (no decomposition) ───────────────────────────
fig_te, ax_te = plt.subplots(figsize=(8, 4))
fig_te.suptitle(f"Total squared L2 norm per model  |  K={K}  |  {SCENARIO_NAME}", fontsize=11)

sym = "²" if SQUARED else ""
for i, name in enumerate(config_names):
    ax_te.plot(timesteps, total_error_mat[i],
               color=colors[i % len(colors)],
               marker=["o", "s", "^", "D", "v"][i % 5],
               ms=4, lw=2, label=f"{name}  ε{sym}")
    # ±1 SEM band across seeds (clipped to stay positive on the log axis).
    lower = np.clip(total_error_mat[i] - total_error_sem_mat[i], 1e-30, None)
    upper = total_error_mat[i] + total_error_sem_mat[i]
    ax_te.fill_between(timesteps, lower, upper,
                       color=colors[i % len(colors)], alpha=0.20, lw=0)

ax_te.set_yscale("log")
ax_te.set_xlabel("AR step t")
ax_te.set_ylabel("Squared L2 norm" if SQUARED else "L2 norm")
ax_te.legend(fontsize=8, ncol=2)
ax_te.set_title(f"ε{sym}(t) per model — no decomposition  "
                f"(band = ±1 SEM over {_n_seeds_used} seeds)", fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / "total_error_per_model.pdf", dpi=150)
plt.savefig(FIG_DIR / "total_error_per_model.png", dpi=150)
print("Saved total_error_per_model.pdf / .png")

# ──────────────────────────────────────────────────────────────────────
# Barplot: percentage of each component in total error, per AR step & training method
# ──────────────────────────────────────────────────────────────────────
n_methods = len(config_names)

# Stacking order: bottom=TD, middle=dot_product, top=OSEB
if NO_PROXY:
    comp_labels = [r"$\|TD\|^2$", r"$2\langle TD, OSEB\rangle$", r"$\|OSEB\|^2$"]
    comp_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    comp_keys   = ["td", "cross_td_os", "os"]
elif TWO_TERM:
    comp_labels = [r"$\|TDOS\|^2$", r"$2\langle TDOS, EB\rangle$", r"$\|EB\|^2$"]
    comp_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    comp_keys   = ["tdos", "dot_tdos_eb", "eb"]
else:
    comp_labels = [r"$\|TD\|^2$", r"$\|OS\|^2$",
                   r"$2\langle OS,TD\rangle$", r"$2\langle EB,TD\rangle$", r"$2\langle EB,OS\rangle$",
                   r"$\|EB\|^2$"]
    comp_colors = ["#2196F3", "#FF9800", "#795548", "#F44336", "#9C27B0", "#4CAF50"]
    comp_keys   = ["td", "os_err", "dot_os_td", "dot_eb_td", "dot_eb_os", "eb"]

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
fig3.suptitle(f"Error decomposition ratios per AR step  |  K={K}  |  {SCENARIO_NAME}", fontsize=11)

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
    parts = cname.split("|")
    net_lbl   = parts[0] if len(parts) >= 2 else ""
    train_lbl = parts[1] if len(parts) >= 2 else cname
    optim_lbl = parts[2] if len(parts) >= 3 else ""
    optim_lbl = optim_lbl.split(";")[1]
    title = f"{net_lbl}\ntrain: {train_lbl}" + (f"\nepochs: {optim_lbl}" if optim_lbl else "")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x_pos)

axes3[0].set_ylabel("% of total error $\\epsilon^2$")
# Place legend inside the last subplot (most space typically)
axes3[-1].legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(FIG_DIR / "component_ratios.pdf", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "component_ratios.png", dpi=150, bbox_inches="tight")
print("Saved component_ratios.pdf / .png")

# ── Unnormalized version: absolute values, shared y-axis ──────────────
abs_arr = np.zeros((n_methods, AR_STEPS, n_comps))
for i, c in enumerate(config_names):
    r = results[c]
    for j, key in enumerate(comp_keys):
        abs_arr[i, :, j] = r[key]

fig4, axes4 = plt.subplots(1, n_methods, figsize=(4 * n_methods, 5), sharey=True)
if n_methods == 1:
    axes4 = [axes4]
fig4.suptitle(f"Error decomposition (absolute)  |  K={K}  |  {SCENARIO_NAME}", fontsize=11)

for i, (ax, cname) in enumerate(zip(axes4, config_names)):
    x_pos = np.arange(1, AR_STEPS + 1)
    bottoms_pos = np.zeros(AR_STEPS)
    bottoms_neg = np.zeros(AR_STEPS)
    for j in range(n_comps):
        vals = abs_arr[i, :, j]
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        ax.bar(x_pos, pos_vals, bar_width, bottom=bottoms_pos,
               label=comp_labels[j], color=comp_colors[j], edgecolor="white", linewidth=0.5)
        ax.bar(x_pos, neg_vals, bar_width, bottom=bottoms_neg,
               color=comp_colors[j], edgecolor="white", linewidth=0.5)
        bottoms_pos += pos_vals
        bottoms_neg += neg_vals

    ax.plot(x_pos, results[cname]["total_error"], "k-", lw=1.5, label=r"$\epsilon^2$ (total)")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("AR step t")
    parts = cname.split("|")
    net_lbl   = parts[0] if len(parts) >= 2 else ""
    train_lbl = parts[1] if len(parts) >= 2 else cname
    optim_lbl = parts[2] if len(parts) >= 3 else ""
    title = f"{net_lbl}\ntrain: {train_lbl}" + (f"\noptim: {optim_lbl}" if optim_lbl else "")
    ax.set_title(title, fontsize=8)
    ax.set_xticks(x_pos)

axes4[0].set_ylabel("Squared L2 norm" if SQUARED else "L2 norm")
axes4[-1].legend(fontsize=7, loc="upper left")

plt.tight_layout()
plt.savefig(FIG_DIR / "component_ratios_abs.pdf", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "component_ratios_abs.png", dpi=150, bbox_inches="tight")
print("Saved component_ratios_abs.pdf / .png")

# ── Temporal evolution of each decomposition term (one line per config) ──
if NO_PROXY:
    term_data = [
        ("td",        r"$\|TD\|^2$",                   "Trajectory Drift"),
        ("cross_td_os", r"$2\langle TD, OS\rangle$",   "Dot-product (cross term)"),
        ("os",      r"$\|OS\|^2$",                 "One-step + Exposure Bias"),
    ]
elif TWO_TERM:
    term_data = [
        ("tdos",      r"$\|TDOS\|^2$",                "Drift + One-step"),
        ("dot_tdos_eb", r"$2\langle TDOS, EB\rangle$",  "Dot-product (cross term)"),
        ("eb",        r"$\|EB\|^2$",                   "Exposure Bias"),
    ]
else:
    term_data = [
        ("td", r"$\|TD\|^2$", "Trajectory Drift"),
        ("os_err", r"$\|OS\|^2$", "One-step Error"),
        ("eb", r"$\|EB\|^2$", "Exposure Bias"),
    ]

_colors6 = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_ylabel6 = "Squared L2 norm" if SQUARED else "L2 norm"

_skip_first = {"td", "cross_td_os"}

for key, math_label, title in term_data:
    ts = timesteps[1:] if key in _skip_first else timesteps
    add_normalized = (key == "os" and NO_PROXY)
    ncols = 2 if add_normalized else 1
    fig6, axes6 = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes6 = [axes6]
    fig6.suptitle(f"{math_label} — {title}  |  K={K}  |  {SCENARIO_NAME}", fontsize=10)
    for i, (cname, res) in enumerate(results.items()):
        parts = cname.split("|")
        train_lbl = parts[1] if len(parts) >= 2 else cname
        vals = res[key][1:] if key in _skip_first else res[key]
        axes6[0].plot(ts, vals, color=_colors6[i % len(_colors6)],
                      marker="o", ms=3, label=train_lbl)
        # ±1 SEM band across seeds when this term has a stored std.
        if f"{key}_std" in res:
            n_s = max(res.get("num_seeds", 1), 1)
            sem = res[f"{key}_std"] / np.sqrt(n_s)
            sem = sem[1:] if key in _skip_first else sem
            axes6[0].fill_between(ts, np.clip(vals - sem, 1e-30, None), vals + sem,
                                  color=_colors6[i % len(_colors6)], alpha=0.18, lw=0)
        if add_normalized:
            ratio = vals / (res["os_gt"] + 1e-30)
            axes6[1].plot(ts, ratio, color=_colors6[i % len(_colors6)],
                          marker="o", ms=3, label=train_lbl)
    axes6[0].set_xlabel("AR step t")
    axes6[0].set_ylabel(_ylabel6)
    axes6[0].set_yscale("log")
    axes6[0].legend(fontsize=7)
    if add_normalized:
        axes6[1].set_xlabel("AR step t")
        axes6[1].set_ylabel(r"$\|OS\|^2_{\rm model} \;/\; \|OS\|^2_{\rm GT}$")
        axes6[1].set_title("Normalized (model OS / GT OS)", fontsize=9)
        axes6[1].legend(fontsize=7)
        axes6[1].axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.5)
    plt.tight_layout()
    fname = f"term_{key}"
    plt.savefig(FIG_DIR / f"{fname}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print(f"Saved {fname}.pdf / .png")

# ── Cosine similarity between the two main error vectors (one line per config) ──
# Unlike the raw dot-product cross term above (which is scale-dependent and
# shrinks if either vector simply grows in magnitude), this isolates whether
# the error *directions* are actually becoming more orthogonal with AR step /
# more unrolling, independent of magnitude.
if NO_PROXY:
    cos_key, cos_label = "cos_td_os", r"$\cos(TD, OSEB)$"
elif TWO_TERM:
    cos_key, cos_label = "cos_tdos_eb", r"$\cos(TDOS, EB)$"
else:
    cos_key, cos_label = None, None

fig_cos, ax_cos = plt.subplots(figsize=(7, 4))
fig_cos.suptitle(f"Cosine similarity of error components  |  K={K}  |  {SCENARIO_NAME}", fontsize=11)
if cos_key is not None:
    for i, (cname, res) in enumerate(results.items()):
        parts = cname.split("|")
        train_lbl = parts[1] if len(parts) >= 2 else cname
        ts = timesteps[1:]
        ax_cos.plot(ts, res[cos_key][1:], color=_colors6[i % len(_colors6)],
                    marker="o", ms=3, label=train_lbl)
    ax_cos.set_title(cos_label, fontsize=10)
else:
    for i, (cname, res) in enumerate(results.items()):
        parts = cname.split("|")
        train_lbl = parts[1] if len(parts) >= 2 else cname
        for key, ls in [("cos_eb_os", "-"), ("cos_eb_td", "--"), ("cos_os_td", ":")]:
            ax_cos.plot(timesteps, res[key], color=_colors6[i % len(_colors6)],
                        marker="o", ms=3, ls=ls, label=f"{train_lbl} {key}")

ax_cos.axhline(0.0, color="k", ls="-", lw=0.5)
ax_cos.set_xlabel("AR step t")
ax_cos.set_ylabel("Cosine similarity")
ax_cos.set_ylim(-1.05, 1.05)
ax_cos.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(FIG_DIR / "cosine_similarity.pdf", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "cosine_similarity.png", dpi=150, bbox_inches="tight")
plt.close(fig_cos)
print("Saved cosine_similarity.pdf / .png")

# ── Scatter: full error at t vs TD at t+1 (one colour per training config) ──
if NO_PROXY:
    td_key, td_label = "td", r"$\|TD\|^2(t+1)$"
elif TWO_TERM:
    td_key, td_label = "tdos", r"$\|TDOS\|^2(t+1)$"
else:
    td_key, td_label = "td", r"$\|TD\|^2(t+1)$"

fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
fig_sc.suptitle(
    f"Full error at t vs TD at t+1  |  K={K}  |  {SCENARIO_NAME}", fontsize=11
)
for i, (cname, res) in enumerate(results.items()):
    parts = cname.split("|")
    train_lbl = parts[1] if len(parts) >= 2 else cname
    # t runs 0..AR_STEPS-2 so that t+1 is valid
    x_vals = res["total_error"][:-1]
    y_vals = res[td_key][1:]
    ax_sc.scatter(
        x_vals, y_vals,
        color=_colors6[i % len(_colors6)],
        label=train_lbl,
        s=40, alpha=0.85, zorder=3,
    )

ax_sc.set_xscale("log")
ax_sc.set_yscale("log")
ax_sc.set_xlabel(r"$\epsilon^2(t)$  (full error)" if SQUARED else r"$\epsilon(t)$  (full error)")
ax_sc.set_ylabel(td_label)
ax_sc.legend(fontsize=8, ncol=1)
plt.tight_layout()
plt.savefig(FIG_DIR / "scatter_error_vs_td_next.pdf", dpi=150, bbox_inches="tight")
plt.savefig(FIG_DIR / "scatter_error_vs_td_next.png", dpi=150, bbox_inches="tight")
plt.close(fig_sc)
print("Saved scatter_error_vs_td_next.pdf / .png")

# ──────────────────────────────────────────────────────────────────────
# Training loss curves — convergence / optimization sanity check.
# Verifies that each unrolling method actually converged (and to what level),
# so ranking differences can be attributed to the method rather than to some
# model being under-trained. One line per train config; shaded band = ±1 std
# of the loss across seeds at each update step.
# (Only covers apebench's regular train configs; chained configs trained via
#  mix_chain_training are not logged here.)
# ──────────────────────────────────────────────────────────────────────
if loss_df is not None and len(loss_df):
    loss_df.to_csv(FIG_DIR / "training_loss.csv", index=False)

    # Group by (net, train, optim_config) so an LR sweep yields one curve per
    # (unrolling method, learning rate). optim_config is present because
    # apebench records it in the scenario_kwargs identity (parsed by read_in_kwargs).
    group_cols = [c for c in ("net", "train", "optim_config") if c in loss_df.columns]
    grouped = list(loss_df.groupby(group_cols))

    def _lr_of(optim_config_str):
        """Peak LR = 5th field of adam;steps;warmup_cosine;init;peak;warmup."""
        parts = str(optim_config_str).split(";")
        return parts[4] if len(parts) > 4 else optim_config_str

    fig_loss, ax_loss = plt.subplots(figsize=(8, 4.5))
    fig_loss.suptitle(f"Training loss per unrolling method  |  {SCENARIO_NAME}", fontsize=11)
    for gi, (gkey, gdf) in enumerate(grouped):
        agg = (gdf.groupby("update_step")["train_loss"]
                  .agg(["mean", "std"]).reset_index())
        steps = agg["update_step"].to_numpy()
        mean  = agg["mean"].to_numpy()
        std   = np.nan_to_num(agg["std"].to_numpy())
        gvals = dict(zip(group_cols, gkey if isinstance(gkey, tuple) else (gkey,)))
        lbl = gvals.get("train", "")
        if "optim_config" in gvals:
            lbl = f"{lbl} | lr={_lr_of(gvals['optim_config'])}"
        color = _colors6[gi % len(_colors6)]
        ax_loss.plot(steps, mean, color=color, lw=1.3, label=lbl)
        ax_loss.fill_between(steps, np.clip(mean - std, 1e-30, None), mean + std,
                             color=color, alpha=0.18, lw=0)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("update step")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_title("band = ±1 std across seeds", fontsize=9)
    ax_loss.legend(fontsize=8, ncol=2, title="train | lr")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_loss_curves.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "training_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_loss)
    print("Saved training_loss_curves.pdf / .png / training_loss.csv")

    # Final-loss summary: mean ± std over seeds at the last logged update step.
    final_step = loss_df["update_step"].max()
    final = loss_df[loss_df["update_step"] == final_step]
    summary = (final.groupby(group_cols)["train_loss"]
                    .agg(["mean", "std", "count"]))
    print(f"\nFinal training loss @ update_step={final_step} (mean ± std over seeds):")
    print(summary.to_string())
else:
    print("No apebench loss_df available — skipping training loss curves.")
