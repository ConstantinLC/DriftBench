#!/usr/bin/env python
"""
PROXY GT EVALUATION: Exposure Bias and Trajectory Drift Analysis

ELEMENTS IMPLEMENTED:
====================
1. EXPOSURE BIAS INDICATOR:
   Evolution of ||M(x̂_t) - GT(x̃_t)||
   Where:
   - x̂_t = Model's coarse prediction at time t
   - x̃_t = HR proxy equivalent found via optimization
   - M(·) = Model's next-step prediction
   - GT(·) = Ground truth HR simulator, coarsened

   An INCREASE indicates exposure bias accumulation: the model becomes less reliable
   when applied to optimized HR states (which differ from training distribution).

2. TRAJECTORY DRIFT:
   Evolution of ||M(x̂_t) - x_{t+1}|| - ||M(x̂_t) - GT(x̃_t)||
   Where:
   - ||M(x̂_t) - x_{t+1}|| = Direct model error (one-step model prediction vs true state)
   - ||M(x̂_t) - GT(x̃_t)|| = Error after applying HR proxy (can't be reduced even with simulator)

   The DIFFERENCE is the irreducible trajectory drift: error accumulated up to step t
   that cannot be compensated by using perfect physics simulation.

PURPOSE:
========
- Understand how exposure bias affects long-horizon predictions
- Identify which timesteps exhibit critical divergence
- Evaluate the gap between model errors and what physics can fix
- Guide decisions on proxy HR fidelity requirements
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as rng
import optax
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
import subprocess
import datetime

import hydra
from omegaconf import DictConfig

# Project Imports
from src.data_loader import get_data_loaders
from src.model_loader import load_model
from src.utils import run_model

# Simulator Import
from sda.mcs import KolmogorovFlow

# ============================================================================
# MODULE-LEVEL GLOBALS (set from config before first JIT call)
# ============================================================================
sim = None
optimizer = None
n_simulator_calls_estimator = None
COARSEN_FACTOR = None

# ============================================================================
# JAX HELPER FUNCTIONS
# ============================================================================

def coarsen_jax(x, r=None):
    """JAX coarsening via average pooling."""
    if r is None:
        r = COARSEN_FACTOR
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // r, r, w // r, r)
    return x.mean(axis=(3, 5))

def loss_fn(hr_initial_condition, model_predicted_coarse_trajectory):
    """
    Loss function for finding HR proxy state.
    Optimizes initial condition so that k steps of HR simulation
    matches the coarse model prediction.
    """
    hr_state = hr_initial_condition

    def step_fn(carry, _):
        next_state = sim._transition(carry)
        return next_state, None

    hr_state_final, _ = jax.lax.scan(step_fn, hr_state, None,
                                     length=n_simulator_calls_estimator)
    simulated_coarse = coarsen_jax(hr_state_final, r=COARSEN_FACTOR)

    return jnp.mean((simulated_coarse - model_predicted_coarse_trajectory) ** 2)

@jax.jit
def optimization_step(opt_state, hr_initial_condition, model_predicted_coarse_trajectory):
    """Single optimization step with gradient computation."""
    loss, grads = jax.value_and_grad(loss_fn)(hr_initial_condition,
                                               model_predicted_coarse_trajectory)
    updates, opt_state = optimizer.update(grads, opt_state)
    hr_initial_condition = optax.apply_updates(hr_initial_condition, updates)
    return hr_initial_condition, opt_state, loss

# ============================================================================
# ELEMENT 1: EXPOSURE BIAS INDICATOR
# ============================================================================

def compute_exposure_bias(model_pred_t, proxy_hr_t):
    """
    Compute: ||Model_preds[t] - coarsen(Proxy_HR_preds[t])||

    Args:
        model_pred_t: Model prediction at time t (coarse JAX array, shape: (batch, 2, 64, 64))
        proxy_hr_t: HR proxy state at time t (JAX array, shape: (batch, 2, 256, 256))

    Returns:
        exposure_bias_distance: Mean MSE across batch
        per_sample_mse: Per-sample MSE values
    """
    proxy_coarsened = coarsen_jax(proxy_hr_t, r=COARSEN_FACTOR)
    per_sample_mse = jnp.mean((model_pred_t - proxy_coarsened) ** 2, axis=(1, 2, 3))
    exposure_bias_distance = jnp.mean(per_sample_mse)

    return float(exposure_bias_distance), np.array(per_sample_mse)

# ============================================================================
# ELEMENT 2: TRAJECTORY DRIFT
# ============================================================================

def compute_trajectory_drift(model_pred_t, proxy_hr_t, true_coarse_t):
    """
    Compute trajectory drift:
        ||Model_preds[t] - coarsen(HR_preds[t])|| - ||Model_preds[t] - coarsen(Proxy_HR_preds[t])||

    Args:
        model_pred_t: Model prediction at time t (coarse JAX array, shape: (batch, 2, 64, 64))
        proxy_hr_t: HR proxy state at time t (JAX array, shape: (batch, 2, 256, 256))
        true_coarse_t: True coarse state at time t (JAX array, shape: (batch, 2, 64, 64))

    Returns:
        trajectory_drift: The difference in errors
        drift_per_sample: Per-sample drift values
        breakdown: dict with component values
    """
    proxy_coarsened = coarsen_jax(proxy_hr_t, r=COARSEN_FACTOR)

    # Component 1: ||Model_preds[t] - x_t|| (total model error)
    direct_error_per_sample = jnp.mean(
        (model_pred_t - true_coarse_t) ** 2, axis=(1, 2, 3)
    )
    direct_error = jnp.mean(direct_error_per_sample)

    # Component 2: ||Model_preds[t] - coarsen(Proxy_HR_preds[t])|| (exposure bias)
    proxy_error_per_sample = jnp.mean(
        (model_pred_t - proxy_coarsened) ** 2, axis=(1, 2, 3)
    )
    proxy_error = jnp.mean(proxy_error_per_sample)

    # Trajectory drift: error accumulated before step t that physics can't fix
    trajectory_drift = direct_error - proxy_error
    drift_per_sample = direct_error_per_sample - proxy_error_per_sample

    breakdown = {
        'direct_error': float(direct_error),
        'proxy_error': float(proxy_error),
        'drift': float(trajectory_drift),
    }

    return float(trajectory_drift), np.array(drift_per_sample), breakdown

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def save_reproducibility_info(output_dir):
    """Save git hash, package versions, and timestamp to run_info.json."""
    info = {
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["git_hash"] = result.stdout.strip()
        result_dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, timeout=5
        )
        info["git_dirty"] = result_dirty.returncode != 0
    except Exception:
        info["git_hash"] = None

    # Package versions
    info["versions"] = {
        "jax": jax.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    path = os.path.join(output_dir, "run_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved reproducibility info to: {path}")

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

@hydra.main(version_base=None, config_path="conf", config_name="proxy_gt_eval")
def main(cfg: DictConfig):
    global sim, optimizer, n_simulator_calls_estimator, COARSEN_FACTOR

    # Set globals from config before any JIT call
    n_simulator_calls_estimator = cfg.simulator.n_calls_per_step
    COARSEN_FACTOR = cfg.simulator.coarsen_factor
    ar_steps = cfg.optimization.ar_steps
    n_optimization_steps = cfg.optimization.n_steps
    batch_size = cfg.batch_size

    output_dir = os.getcwd()  # Hydra manages output directory

    # Load JSON config for data params
    with open(cfg.data.config_path, 'r') as f:
        config = json.load(f)

    print(f"=== PROXY GT EVALUATION ===")
    print(f"Model: {cfg.model.type}, Batch Size: {batch_size}, AR Steps: {ar_steps}\n")

    # Load Model
    active_model = load_model(
        model_type=cfg.model.type,
        checkpoint_path=cfg.model.checkpoint,
        config_path=cfg.data.config_path,
        device=cfg.device,
    )
    print(f"Loaded model: {cfg.model.type} from {cfg.model.checkpoint}\n")

    # Load data
    _, val_loader, _ = get_data_loaders(config['data_params'])
    datapoint = next(iter(val_loader))["data"]

    # Setup Simulator
    sim = KolmogorovFlow(size=cfg.simulator.size, dt=cfg.simulator.dt)

    print("Generating trajectories and optimizing HR proxies...")
    print("-" * 70)

    # Generate initial HR state and warmup
    key = rng.PRNGKey(cfg.seed)
    keys = rng.split(key, batch_size)
    hr_prior = sim._prior(keys)

    n_warmup_calls = cfg.simulator.n_warmup_calls

    last_states = []
    hr_state = hr_prior
    for t in range(1, n_warmup_calls + 1):
        hr_state = sim._transition(hr_state)
        if t > n_warmup_calls - n_simulator_calls_estimator:
            last_states.append(hr_state)

    HR_preds = {i - len(last_states) + 1: last_states[i] for i in range(len(last_states))}
    Model_preds = {0: coarsen_jax(HR_preds[0], r=COARSEN_FACTOR)}
    Proxy_HR_preds = {}

    # Generate GT trajectory
    for t in range(1, ar_steps + 1):
        HR_preds[t] = sim._transition(HR_preds[t-1])

    # Generate model predictions
    with torch.no_grad():
        model_curr = torch.tensor(np.array(Model_preds[0])).to(cfg.device)
        for t in range(1, ar_steps + 1):
            model_curr = run_model(active_model, model_curr)
            model_curr_jax = jnp.array(model_curr.detach().cpu().numpy())
            Model_preds[t] = model_curr_jax

    # Optimize HR proxies
    for t in range(2, ar_steps + 1):
        optimizer = optax.adam(cfg.optimization.learning_rate)
        opt_state = optimizer.init(HR_preds[t - n_simulator_calls_estimator - 1])

        hr_refined = HR_preds[t - n_simulator_calls_estimator - 1]
        for iteration in range(n_optimization_steps):
            hr_refined, opt_state, loss = optimization_step(
                opt_state, hr_refined, Model_preds[t-1]
            )
            if iteration % 200 == 0:
                print(f"  Step {t}/{ar_steps}, Iteration {iteration:4d} | Loss: {loss:.8f}")

        # Final HR state: evolved from optimized initial condition
        hr_refined_final = hr_refined
        for _ in range(n_simulator_calls_estimator + 1):
            hr_refined_final = sim._transition(hr_refined_final)

        Proxy_HR_preds[t] = hr_refined_final

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    print("\n" + "="*70)
    print("COMPUTING METRICS: EXPOSURE BIAS & TRAJECTORY DRIFT")
    print("="*70 + "\n")

    # Storage for metrics
    exposure_bias_metrics = {}
    trajectory_drift_metrics = {}
    detailed_breakdown = {}

    for t in range(2, ar_steps + 1):
        print(f"Timestep {t}:")

        # Element 1: Exposure Bias
        exp_bias_avg, exp_bias_per_sample = compute_exposure_bias(
            Model_preds[t], Proxy_HR_preds[t]
        )
        exposure_bias_metrics[t] = {
            'average': exp_bias_avg,
            'per_sample': exp_bias_per_sample
        }
        print(f"  [1] Exposure Bias (||M(x̂_t) - GT(x̃_t)||):")
        print(f"      Average MSE: {exp_bias_avg:.8f}")
        print(f"      Per-sample:  {exp_bias_per_sample}")

        # Element 2: Trajectory Drift
        true_coarse_t = coarsen_jax(HR_preds[t], r=COARSEN_FACTOR)

        drift_avg, drift_per_sample, breakdown = compute_trajectory_drift(
            Model_preds[t], Proxy_HR_preds[t], true_coarse_t
        )
        trajectory_drift_metrics[t] = {
            'average': drift_avg,
            'per_sample': drift_per_sample
        }
        detailed_breakdown[t] = breakdown

        print(f"  [2] Trajectory Drift:")
        print(f"      Direct error (||Model_preds[t] - x_t||):          {breakdown['direct_error']:.8f}")
        print(f"      Proxy error (||Model_preds[t] - Proxy_GT[t]||):   {breakdown['proxy_error']:.8f}")
        print(f"      Trajectory drift (difference):           {drift_avg:.8f}")
        print(f"      Per-sample drift: {drift_per_sample}")
        print()

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Exposure Bias Evolution
    timesteps = sorted(exposure_bias_metrics.keys())
    exposure_bias_values = [exposure_bias_metrics[t]['average'] for t in timesteps]

    axes[0].plot(timesteps, exposure_bias_values, 'o-', linewidth=2, markersize=8, color='tab:blue')
    axes[0].set_xlabel('Autoregressive Step (t)', fontsize=12)
    axes[0].set_ylabel('||M(x̂_t) - GT(x̃_t)|| (MSE)', fontsize=12)
    axes[0].set_title('Element 1: Exposure Bias Indicator', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(timesteps)

    # Plot 2: Trajectory Drift Evolution
    drift_values = [trajectory_drift_metrics[t]['average'] for t in timesteps]
    direct_error_values = [detailed_breakdown[t]['direct_error'] for t in timesteps]
    proxy_error_values = [detailed_breakdown[t]['proxy_error'] for t in timesteps]

    axes[1].plot(timesteps, direct_error_values, 'o-', linewidth=2, markersize=8,
                label='Direct error ($||\\hat{x}_t - x_t||$)', color='tab:red')
    axes[1].plot(timesteps, proxy_error_values, 's-', linewidth=2, markersize=8,
                label='Proxy error ($||\\hat{x}_t - \\tilde{x}_t||$)', color='tab:green')
    axes[1].fill_between(timesteps, proxy_error_values, direct_error_values,
                         alpha=0.2, color='orange', label='Trajectory drift')
    axes[1].set_xlabel('Autoregressive Step (t)', fontsize=12)
    axes[1].set_ylabel('Error (MSE)', fontsize=12)
    axes[1].set_title('Element 2: Trajectory Drift', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(timesteps)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'proxy_gt_metrics.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to: {plot_path}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    results = {
        'element_1_exposure_bias': {
            int(t): {
                'average': float(exposure_bias_metrics[t]['average']),
                'per_sample': exposure_bias_metrics[t]['per_sample'].tolist()
            }
            for t in timesteps
        },
        'element_2_trajectory_drift': {
            int(t): {
                'average': float(trajectory_drift_metrics[t]['average']),
                'per_sample': trajectory_drift_metrics[t]['per_sample'].tolist(),
                'direct_error': float(detailed_breakdown[t]['direct_error']),
                'proxy_error': float(detailed_breakdown[t]['proxy_error']),
            }
            for t in timesteps
        }
    }

    results_path = os.path.join(output_dir, 'proxy_gt_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")

    # Save reproducibility info
    save_reproducibility_info(output_dir)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Element 1 - Exposure Bias:")
    print(f"  Mean over timesteps: {np.mean(exposure_bias_values):.8f}")
    print(f"  Trend: {'Increasing' if exposure_bias_values[-1] > exposure_bias_values[0] else 'Decreasing'}")
    print(f"\nElement 2 - Trajectory Drift:")
    print(f"  Mean over timesteps: {np.mean(drift_values):.8f}")
    print(f"  Max at timestep: {timesteps[np.argmax(drift_values)]}")
    print(f"  Interpretation: Error that CANNOT be fixed even with perfect physics")

if __name__ == "__main__":
    main()
