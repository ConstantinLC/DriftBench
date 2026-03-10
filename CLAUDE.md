# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Goal:** Analyze how neural emulators (ML models) for fluid simulations diverge from physics over long horizons, by decomposing error into two interpretable components: exposure bias and trajectory drift.

**Physical Domain:** Kolmogorov Flow turbulence dynamics

**Core Insight:** A trained ML model can make "good" one-step predictions, yet systematically diverge from physics when applied autoregressively—either because it encounters out-of-distribution states (exposure bias) or because early errors propagate irreversibly (trajectory drift).

## Three-Resolution Architecture

The core architecture involves three coupled simulation levels:

1. **HR Simulation (256×256):** Ground truth high-resolution physics simulator (differentiable JAX)
2. **Coarse Resolution (64×64):** Reduced resolution where the ML model operates (coarsen HR by 4×)
3. **Proxy HR (256×256):** Optimized HR state that, when evolved and coarsened, matches the model's coarse prediction

**Workflow:**
- Generate trajectories at coarse resolution using the trained ML model
- For each timestep, *optimize* an HR initial condition so that simulating K steps and coarsening matches the model's prediction
- Result: physics-consistent HR equivalent of the model's prediction
- Evaluate two metrics using this proxy

## Two Key Evaluation Metrics

### Element 1: Exposure Bias Indicator
**Metric:** `||M(x̂_t) - GT(x̃_t)||`

Measures how well the model generalizes to physics-consistent states:
- `M(x̂_t)`: Model's prediction starting from its own coarse prediction
- `GT(x̃_t)`: Ground truth simulator (HR) applied to the optimized proxy, then coarsened

**Interpretation:** An increasing trend indicates the model becomes unreliable when applied to states outside its training distribution.

### Element 2: Trajectory Drift
**Metric:** `||M(x̂_t) - x_{t+1}|| - ||M(x̂_t) - GT(x̃_t)||`

Decomposes total model error into recoverable and irreducible components:
- First term: Direct model error vs true state
- Second term: Error remaining after perfect physics correction
- The gap: Irreducible drift that accumulated before time t

**Interpretation:** Positive and growing drift means the trajectory has passed a point of no return—future physics correction cannot fix this error.

## Critical Implementation Details

**Key Parameters** (in scripts):
- `K` (simulator calls): Number of HR steps per coarse step (typically K=2 for Kolmogorov)
- `n_optimization_steps`: Iterations to optimize HR proxy per timestep (~1000)
- `n_warmup_calls`: Steps to reach steady state before recording (~64)
- `ar_steps`: Autoregressive evaluation horizon (typically 5-10)
- `COARSEN_FACTOR`: 4 (256÷64)

**JAX Patterns:**
- Use `@jax.jit` for optimization loops (JAX compiled gradients)
- Use `jax.lax.scan` for unrolled simulator steps
- Memory: Set `os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"` to prevent over-allocation

**Simulation Integration:**
- HR simulator: `from sda.mcs import KolmogorovFlow` (external dependency, differentiable)
- Call: `sim._transition(state)` → one HR evolution step
- Gradient flow: Full backprop through K simulator steps + coarsening

## Directory Structure

```
drift-bench/
├── src/                      # Core modules
│   ├── model.py             # Unet architecture
│   ├── model_diffusion.py   # Diffusion-based models
│   ├── trainer.py           # Main training loop
│   ├── data_loader.py       # Dataset utilities
│   ├── dataset.py           # Data pipeline
│   ├── utils.py             # Helpers (run_model, correlation, vorticity, etc.)
│   └── data_transformations.py
├── configs/                  # Training configuration JSON files
├── checkpoints/             # Saved model weights
├── results/                 # Training-time evaluation outputs
├── proxy_gt_results/        # Outputs from proxy GT evaluation
├── proxy_gt_evaluation.py   # Main evaluation script (Element 1 & 2)
├── find_hr_matching_state.py # Inverse problem: optimize HR IC to match model
├── eval_kolmo.py            # Basic trajectory evaluation
└── data_assimilation.py     # Experimental data assimilation script
```

## Main Scripts & How to Use

### 1. Proxy GT Evaluation (Primary Analysis)
```bash
python proxy_gt_evaluation.py \
  --config configs/config_kolmo_unet.json \
  --checkpoints ModelName=/path/to/checkpoint.pth \
  --batch-size 5 \
  --output-dir ./proxy_gt_results
```

**Outputs:**
- `proxy_gt_metrics.png`: Two plots (Element 1 & 2 evolution over time)
- `proxy_gt_metrics.json`: Numerical results

**What it does:**
1. Loads model from checkpoint
2. Generates model predictions (coarse, 10 steps)
3. For each timestep: optimizes HR proxy state (~1000 iterations per step via JAX/optax)
4. Computes Element 1 & 2 metrics
5. Visualizes and saves results

### 2. Find HR Matching State (Inverse Problem)
```bash
python find_hr_matching_state.py \
  --config configs/config_kolmo_unet.json \
  --checkpoint /path/to/checkpoint.pth \
  --n_samples 5
```

**Purpose:** Standalone proxy optimization without full evaluation pipeline. Useful for debugging or analyzing single trajectories.

### 3. Eval Kolmo (Basic Trajectory Metrics)
```bash
python eval_kolmo.py \
  --config configs/config_kolmo_unet.json \
  --checkpoint /path/to/checkpoint.pth
```

**Metrics:** Correlation, MSE, vorticity correlation over trajectory

### 4. Data Assimilation (Experimental)
Investigates how observation assimilation affects model drift.

## Configuration Format

Config JSON (e.g., `config_kolmo_unet.json`):

```json
{
  "train_params": {
    "num_epochs": 10001,
    "learning_rate_start": 0.00001,
    "device": "cuda"
  },
  "data_params": {
    "data_path": "/mnt/SSD2/constantin/sda/data",
    "resolution": 64,
    "sequence_length": [3, 1],
    "batch_size": 64,
    "limit_trajectories_train": 819,
    "limit_trajectories_val": 102
  },
  "loss_params": {
    "name": "mse",
    "eval_traj_metrics": ["mse", "corr"],
    "primary_metric": "corr"
  },
  "model_params": {
    "dim": 64,
    "channels": 2,
    "padding_mode": "circular"
  },
  "checkpoint": "/path/to/checkpoint.pth"
}
```

## Key Module Functions

**src/utils.py:**
- `run_model(model, state_batch)`: Runs single inference step, handles model type
- `evaluate_trajectory(model, traj_loader, device, metrics=[...])`: Computes trajectory metrics
- `parse_checkpoint_args(checkpoint_str)`: Parses `--checkpoints ModelName=path` syntax
- `correlation(a, b)`: Pearson correlation
- `vorticity(state)`: Computes vorticity field
- `fsd_torch_radial(state)`: Frequency spectrum density

**src/model.py:**
- `Unet`: Standard U-Net architecture with ConvNeXt blocks, circular padding

**src/model_diffusion.py:**
- `DiffusionModel`: Diffusion-based emulator (alternative to deterministic)

**src/data_loader.py:**
- `get_data_loaders()`: Returns train/val/trajectory dataloaders

## Typical Workflow

1. **Train a model** using `src/trainer.py` (configured via JSON)
   - Outputs: `checkpoints/KolmogorovFlow/Unet/epoch_XXXX.pth`

2. **Run proxy GT evaluation** on the trained model
   - Computes Element 1 & 2 metrics over 5-10 autoregressive steps
   - Inspect output plots and JSON to identify where exposure bias/drift occur

3. **Interpret results:**
   - **Element 1 increasing early** → Model overtrained to training distribution
   - **Element 2 increasing late** → Irreversible error accumulation (inherent to one-step error)
   - **Peak around t=6-8** → Typical cross-over point where recovery becomes impossible

## Common Development Tasks

### Running a single evaluation with a different checkpoint
```bash
python proxy_gt_evaluation.py \
  --config configs/config_kolmo_unet.json \
  --checkpoints MyModel=/path/to/my/checkpoint.pth \
  --batch-size 10 \
  --output-dir ./my_results
```

### Testing new metrics
- Modify `compute_exposure_bias()` or `compute_trajectory_drift()` in `proxy_gt_evaluation.py`
- Add per-timestep breakdown dicts for analysis
- Metrics already computed: MSE, vorticity correlation, frequency spectrum

### Debugging optimization convergence
- In `find_hr_matching_state.py`, increase `--verbose` flag
- Check `loss_history` output to see if optimizer is stuck
- Reduce `learning_rate` or increase `n_optimization_steps` if proxy doesn't converge

### Batch processing multiple models
```bash
for ckpt in checkpoints/KolmogorovFlow/Unet/*.pth; do
  python proxy_gt_evaluation.py \
    --config configs/config_kolmo_unet.json \
    --checkpoints Model=$ckpt \
    --output-dir ./results/$(basename $ckpt .pth)
done
```

## Dependencies

**Core:**
- PyTorch, JAX, optax
- NumPy, SciPy
- Matplotlib

**External Simulator:**
- `sda.mcs.KolmogorovFlow` (must be importable; assumes `/mnt/SSD2/constantin/sda/` exists)

**Data:**
- Kolmogorov Flow trajectories at `/mnt/SSD2/constantin/sda/data`

## Key Insights from Documentation

- **Proxy initialization matters**: Optimized HR proxy starts from `HR_preds[t-K-1]` (true state) for faster convergence
- **Metrics are interdependent**: Element 2 = Element 1 + Drift; they partition the direct model error
- **Non-linear divergence**: Drift often remains low for first few steps, then explodes around t=6-8
- **Physics consistency**: Proxy states are guaranteed physically realizable (evolved from true HR states), unlike raw model predictions

## References

See project documentation for theoretical background:
- `Projet.md`: Goal statement and three-error decomposition
- `Proxy GT.md`: Inverse problem formulation and methodology
- `EXPERIMENTAL_SETUP.md`: Detailed hyperparameter choices and validation strategy
