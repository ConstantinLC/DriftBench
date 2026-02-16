#!/usr/bin/env python
#PROXY GT EVALUATION: Exposure Bias and Trajectory Drift Analysis

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
from hydra.core.hydra_config import HydraConfig

# Project Imports
from src.data_loader import get_data_loaders
from src.model_loader import load_model
from src.utils import run_model

# Simulator Import
from sda.mcs import KolmogorovFlow
from drift_utils import coarsen_jax
from drift_utils import optimize_initial_conditions, get_trajectory_predictions, make_loss_and_grad_fn

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

@hydra.main(version_base=None, config_path="conf", config_name="conf")
def main(cfg: DictConfig):
    # Create simulator and other configuration
    sim = KolmogorovFlow(size=cfg.simulation.size, dt=cfg.simulation.dt)
    coarsen_factor = cfg.simulation.coarsen_factor
    n_trajectories = cfg.simulation.n_trajectories
    n_warmup_calls =  cfg.simulation.n_warmup_calls
    n_ar_steps = cfg.simulation.n_ar_steps
    n_calls_per_step = cfg.optimization.n_calls_per_step
    optimization_steps = cfg.optimization.optimization_steps
    learning_rate = cfg.optimization.learning_rate
    initialization_type = cfg.optimization.initialization_type

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    seed = cfg.seed
    device = cfg.device


    print(f"Model: {cfg.model.type}, Number of trajectories: {cfg.simulation.n_trajectories}",
          f"AR Steps: {cfg.simulation.n_ar_steps}\n")

    # Load Model
    active_model = load_model(
        model_type=cfg.model.type,
        checkpoint_path=cfg.model.checkpoint,
        model_params=cfg.model,
        device=cfg.device,
    )
    active_model.eval()
    print(f"Loaded model: {cfg.model.type} from {cfg.model.checkpoint}\n")

    print("Generating trajectories and optimizing HR proxies...")

    hr_preds, model_preds = get_trajectory_predictions(active_model, sim, n_warmup_calls,
                                                       n_trajectories, seed, n_ar_steps,
                                                       n_calls_per_step, coarsen_factor,
                                                       device)

    # Build JIT-compiled loss+grad function once, reuse across all timesteps
    compute_loss_and_grads = make_loss_and_grad_fn(sim, n_calls_per_step, coarsen_factor)

    proxy_preds = {}
    model_preds_on_proxy = {}
    model_preds_on_gt = {}
    for t in range(1, cfg.simulation.n_ar_steps + 1):
        optimized_initial_condition, _ = optimize_initial_conditions(n_calls_per_step,
                                                                    optimization_steps,
                                                                    hr_preds,
                                                                    model_preds,
                                                                    learning_rate,
                                                                    t,
                                                                    sim,
                                                                    coarsen_factor,
                                                                    initialization_type,
                                                                    compute_loss_and_grads=compute_loss_and_grads
                                                                    )

        # Final HR state: evolved from optimized initial condition
        hr_refined_final = optimized_initial_condition
        for step in range(n_calls_per_step + 1):
            hr_refined_final = sim._transition(hr_refined_final)
            if step == n_calls_per_step - 1:
                # MODEL PRED ON PROXY
                coarse_state = coarsen_jax(hr_refined_final, r=coarsen_factor)
                model_input = torch.from_numpy(np.asarray(coarse_state)).float().to(device)
                model_preds_on_proxy[t] = np.array(run_model(active_model, model_input).cpu().detach())

        # PROXY PRED
        proxy_preds[t] = coarsen_jax(hr_refined_final, r=coarsen_factor)

        # MODEL PRED ON GT TRAJ
        coarse_state = coarsen_jax(hr_preds[t-1], r=coarsen_factor)
        model_input = torch.from_numpy(np.asarray(coarse_state)).float().to(device)
        model_preds_on_gt[t] = np.array(run_model(active_model, model_input).cpu().detach())

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    print("\n" + "="*70)
    print("COMPUTING METRICS: EXPOSURE BIAS & TRAJECTORY DRIFT")
    print("="*70 + "\n")

    # Storage for metrics
    proxy_gt_errors = {}
    proxy_gt_vs_model_on_proxy_errors = {}
    trajectory_errors = {}
    gt_vs_model_on_gt_errors = {}

    for t in range(1, n_ar_steps + 1):
        print(f"Timestep {t}:")

        error_per_sample = np.mean((model_preds[t] - proxy_preds[t])**2, axis=(1,2,3))
        error_avg = np.mean(error_per_sample)
        proxy_gt_errors[t] = {
            'average': error_avg,
            'per_sample': error_per_sample
        }

        error_per_sample = np.mean((model_preds_on_proxy[t] - proxy_preds[t])**2, axis=(1,2,3))
        error_avg = np.mean(error_per_sample)
        proxy_gt_vs_model_on_proxy_errors[t] = {
            'average': error_avg,
            'per_sample': error_per_sample
        }

        coarse_hr_preds = coarsen_jax(hr_preds[t], r=coarsen_factor)
        error_per_sample = np.mean((model_preds[t] - coarse_hr_preds)**2, axis=(1,2,3))
        error_avg = np.mean(error_per_sample)
        trajectory_errors[t] = {
            'average': error_avg,
            'per_sample': error_per_sample
        }

        coarse_hr_preds = coarsen_jax(hr_preds[t], r=coarsen_factor)
        error_per_sample = np.mean((model_preds_on_gt[t] - coarse_hr_preds)**2, axis=(1,2,3))
        error_avg = np.mean(error_per_sample)
        gt_vs_model_on_gt_errors[t] = {
            'average': error_avg,
            'per_sample': error_per_sample
        }

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average Proxy GT Error
    timesteps = sorted(proxy_gt_errors.keys())
    proxy_gt_values = [proxy_gt_errors[t]['average'] for t in timesteps]
    proxy_gt_vs_model_on_proxy_values = [proxy_gt_vs_model_on_proxy_errors[t]['average'] for t in timesteps]
    gt_vs_model_on_gt_values = [gt_vs_model_on_gt_errors[t]['average'] for t in timesteps]

    axes[0].plot(timesteps, proxy_gt_values, 'o-', linewidth=2, 
                 markersize=8, color='tab:blue', label='Model on Own Traj VS Proxy GT')
    axes[0].plot(timesteps, proxy_gt_vs_model_on_proxy_values, 'o-', linewidth=2, 
                 markersize=8, color='tab:green', label='Model on Proxy vs Proxy GT')
    axes[0].plot(timesteps, gt_vs_model_on_gt_values, 'o-', linewidth=2, 
                 markersize=8, color='tab:orange', label='Model on GT vs GT')
    axes[0].set_xlabel('Autoregressive Step (t)', fontsize=12)
    axes[0].set_ylabel('Error (MSE)', fontsize=12)
    axes[0].set_title('One-step errors', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(timesteps)

    # Plot 2: Trajectory Errors and Average Proxy GT Error
    trajectory_values = [trajectory_errors[t]['average'] for t in timesteps]

    axes[1].plot(timesteps, trajectory_values, 'o-', linewidth=2, markersize=8,
                label='Model on Own Traj vs GT Traj', color='tab:red')
    axes[1].plot(timesteps, proxy_gt_values, 's-', linewidth=2, markersize=8,
                label='Model on Own Traj vs Proxy GT', color='tab:blue')
    axes[1].set_xlabel('Autoregressive Step (t)', fontsize=12)
    axes[1].set_ylabel('Error (MSE)', fontsize=12)
    axes[1].set_title('Trajectory Errors', fontsize=13, fontweight='bold')
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
        'proxy_gt_errors': {
            int(t): {
                'average': float(proxy_gt_errors[t]['average']),
                'per_sample': proxy_gt_errors[t]['per_sample'].tolist()
            }
            for t in timesteps
        },
        'trajectory_errors': {
            int(t): {
                'average': float(trajectory_errors[t]['average']),
                'per_sample': trajectory_errors[t]['per_sample'].tolist()
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
    print(f"Average Proxy GT Error:")
    print(f"  Mean over timesteps: {np.mean(proxy_gt_values):.8f}")
    print(f"  Trend: {'Increasing' if proxy_gt_values[-1] > proxy_gt_values[0] else 'Decreasing'}")
    print(f"\nTrajectory Errors:")
    print(f"  Mean over timesteps: {np.mean(trajectory_values):.8f}")
    print(f"  Max at timestep: {timesteps[np.argmax(trajectory_values)]}")

if __name__ == "__main__":
    main()
