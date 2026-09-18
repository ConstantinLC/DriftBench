## Error Decomposition for Autoregressive Neural PDE Emulators

`exponax_start.py` is the main entry point. It trains one or more neural
emulators on a PDE scenario (via [`exponax`](https://github.com/Ceyron/exponax)
/ [`apebench`](https://github.com/tum-pbs/apebench)), rolls them out
autoregressively, and decomposes their trajectory error against a ground-truth
(GT) rollout of the reference simulator.

### **Core decomposition (default, `NO_PROXY = True`):**

At each autoregressive step $t$, given the model's own state $\hat{x}_t$ and
its next prediction $\hat{x}_{t+1} = NN(\hat{x}_t)$:

```
GT_on_NN(t) = GT(x̂_t)                       # one reference-simulator step from the model's own state
TD(t)   = x_{t+1} - GT(x̂_t)                  # Trajectory Drift
OSEB(t) = GT(x̂_t) - x̂_{t+1}                  # One-Step error + Exposure Bias (combined)
ε(t+1)  = x̂_{t+1} - x_{t+1} = -(TD(t) + OSEB(t))
```

- **Trajectory Drift (TD):** how far the *true* trajectory has already
  diverged from the model's current state, measured by applying one step of
  the real simulator from $\hat{x}_t$. This is error already "baked in" from
  earlier steps — physics can't fix it.
- **One-Step + Exposure Bias (OSEB):** the model's own one-step error, but
  evaluated at its own (possibly out-of-distribution) state $\hat{x}_t$
  rather than at a true state. Without an explicit HR/proxy correction this
  conflates ordinary one-step error and exposure bias into a single term
  (`oseb_gt` is also logged: the same one-step error evaluated at the *true*
  state $x_t$, used to normalize/compare against).
- **Exact Pythagorean check:** since $\varepsilon = TD + OSEB$ (vector
  identity, not an approximation), the script also logs the cross term
  $2\langle TD, OSEB\rangle$ and $\cos(TD, OSEB)$ so you can see how close the
  decomposition is to a true orthogonal (Pythagorean) split at each step.

A legacy three/two-term variant (`NO_PROXY = False`) is still implemented: it
optimizes an initial condition so that $K$ simulator steps reproduce the
model's prediction (the original HR-proxy idea from this repo's earlier
Kolmogorov-flow-specific setup), then splits error into Trajectory Drift +
One-Step-error + Exposure-Bias as three separate terms. This path is off by
default and much more expensive (an `N_OPT_STEPS`-iteration optimization per
timestep, per sample).

### **What the script actually does, end to end:**

1. **Configure a scenario.** Any `apebench` scenario, selected by
   `--scenario_name` (e.g. `phy_kolm_flow`, `norm_ks`, `norm_kdv`,
   `phy_burgers`, ...). `--predict_steps` can wrap the reference stepper so
   the model is trained/evaluated to jump several simulator steps ahead in a
   single forward call.
2. **Train (or load cached) emulators** for every combination of
   `--net_configs` (e.g. `UNet;12;2;relu`, `FNO;12;18;4;gelu`) ×
   `--train_configs` × learning rate in `--lrs`, over `--num_seeds` seeds.
   - Plain configs (`one`, `sup;N`, ...) go through `apebench.run_study_convenience`.
   - Chained configs (`cfg1@epochs1->cfg2@epochs2->...`) and the custom
     unrolling losses defined in `mix_chain_training.py` (`mix;M;K`, `fg;N`,
     `tdcross;N;...`, `tdcrossos;N;...`) are trained stage-by-stage via
     `mix_chain_training.train_chain`.
   - Trained weights are cached under `outputs/<scenario>/weights/`, keyed by
     their own net/train/optim identity, so unrelated config changes (e.g.
     `--ar_steps`) don't force retraining.
3. **Roll out** each trained model for `--ar_steps` steps from a shared,
   warmed-up set of initial conditions, alongside the GT reference rollout.
4. **Compute the decomposition** above at every AR step, averaged (and
   std/SEM'd) over seeds.
5. **Save figures and raw arrays** to `outputs/<scenario>/<config-hash>/`
   (the hash covers every run parameter, so different configs never
   overwrite each other's outputs; the resolved config is dumped to
   `config.json` in the same folder):
   - `total_error_per_model.png` — $\varepsilon^2(t)$ per model/config.
   - `component_ratios.png` / `component_ratios_abs.png` — stacked bar of
     TD / cross-term / OSEB as a % (and absolute value) of total error, per
     AR step and per training config.
   - `term_<td|cross_td_os|os>.png` — each term's evolution over AR steps
     (with an extra normalized OSEB/GT-OSEB panel).
   - `cosine_similarity.png` — $\cos(TD, OSEB)$ over AR steps.
   - `scatter_error_vs_td_next.png` — $\varepsilon^2(t)$ vs. $TD^2(t+1)$.
   - `training_loss_curves.png` / `training_loss.csv` — convergence check
     per training method/LR.
   - `raw_arrays.npz` — every per-seed array, for downstream refitting
     (e.g. `fit_error_recurrence.py`) without rerunning inference.

### **Usage:**

```bash
python exponax_start.py \
  --scenario_name phy_kolm_flow \
  --net_configs "UNet;12;2;relu" \
  --train_configs one "sup;2" "sup;4" "sup;8" \
  --lrs 1e-3 \
  --num_seeds 3 \
  --ar_steps 10
```

Run `python exponax_start.py --help` for the full argument list
(`--gpu_id`, `--n_epochs`, `--predict_steps`, etc.).

### **Resolutions / scenarios:**

Scenario resolution and dynamics are whatever the chosen `apebench` scenario
defines (e.g. `phy_kolm_flow` is 2D Kolmogorov Flow) — there is no longer a
fixed HR(256)/coarse(64) split baked into the main pipeline; that setup only
still applies inside the legacy `NO_PROXY = False` proxy path.
