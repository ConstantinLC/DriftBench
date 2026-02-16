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

@hydra.main(version_base=None, config_path="conf", config_name="conf_grid_ncallsperstep")
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
    n_calls_per_step_list = cfg.optimization.n_calls_per_step_list
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
                                                        max(n_calls_per_step_list), coarsen_factor,
                                                        device)    

    # Build JIT-compiled loss+grad function once, reuse across all timesteps
    compute_loss_and_grads_dict = {}
    for n_calls in n_calls_per_step_list:
        compute_loss_and_grads_dict[n_calls] = make_loss_and_grad_fn(sim, n_calls, coarsen_factor)

    # 3-level structure: proxy_preds[n_calls][t][optim_step]
    proxy_preds = {}

    for n_calls in n_calls_per_step_list:
        proxy_preds[n_calls] = {}
        compute_loss_and_grads = compute_loss_and_grads_dict[n_calls]

        for t in range(1, cfg.simulation.n_ar_steps + 1):
            proxy_preds[n_calls][t] = {}

            

            optimized_initial_condition, optim_history = optimize_initial_conditions(n_calls,
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

            # Store intermediate optimization steps
            if optim_history is not None:
                for optim_step, hr_state in optim_history.items():
                    # Final HR state: evolved from HR state at this optimization step
                    hr_refined_final = hr_state
                    for _ in range(n_calls + 1):
                        hr_refined_final = sim._transition(hr_refined_final)
                    proxy_preds[n_calls][t][optim_step] = coarsen_jax(hr_refined_final, r=coarsen_factor)
            else:
                # Fallback: just store final result if history not available
                hr_refined_final = optimized_initial_condition
                for _ in range(n_calls + 1):
                    hr_refined_final = sim._transition(hr_refined_final)
                proxy_preds[n_calls][t]['final'] = coarsen_jax(hr_refined_final, r=coarsen_factor)

    # ========================================================================
    # COMPUTE METRICS
    # ========================================================================

    print("\n" + "="*70)
    print("COMPUTING METRICS: EXPOSURE BIAS & TRAJECTORY DRIFT")
    print("="*70 + "\n")

    # Storage for metrics: metrics[n_calls][t][optim_step]
    proxy_gt_errors = {}
    trajectory_errors = {}

    for n_calls in n_calls_per_step_list:
        proxy_gt_errors[n_calls] = {}
        trajectory_errors[n_calls] = {}

        for t in range(1, n_ar_steps + 1):
            print(f"Timestep {t} (n_calls={n_calls}):")
            proxy_gt_errors[n_calls][t] = {}
            trajectory_errors[n_calls][t] = {}

            # Iterate over optimization steps
            for optim_step, proxy_pred in proxy_preds[n_calls][t].items():
                error_per_sample = np.mean((model_preds[t] - proxy_pred)**2, axis=(1,2,3))
                error_avg = np.mean(error_per_sample)
                proxy_gt_errors[n_calls][t][optim_step] = {
                    'average': error_avg,
                    'per_sample': error_per_sample
                }

            # Trajectory error (independent of optimization steps)
            coarse_hr_preds = coarsen_jax(hr_preds[t], r=coarsen_factor)
            error_per_sample = np.mean((model_preds[t] - coarse_hr_preds)**2, axis=(1,2,3))
            error_avg = np.mean(error_per_sample)
            trajectory_errors[n_calls][t] = {
                'average': error_avg,
                'per_sample': error_per_sample
            }

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    # Create subplots for each n_calls value
    n_calls_values = sorted(n_calls_per_step_list)
    fig, axes = plt.subplots(len(n_calls_values), 2, figsize=(14, 5*len(n_calls_values)))

    # Handle single n_calls case
    if len(n_calls_values) == 1:
        axes = np.array([axes])

    for idx, n_calls in enumerate(n_calls_values):
        timesteps = sorted(proxy_gt_errors[n_calls].keys())

        # Get all optim_step values for each timestep
        all_optim_steps = set()
        proxy_gt_by_step = {}  # Store values for each optimization step

        for t in timesteps:
            optim_steps = sorted(proxy_gt_errors[n_calls][t].keys())
            all_optim_steps.update(optim_steps)
            for opt_step in optim_steps:
                if opt_step not in proxy_gt_by_step:
                    proxy_gt_by_step[opt_step] = []
                proxy_gt_by_step[opt_step].append(proxy_gt_errors[n_calls][t][opt_step]['average'])

        all_optim_steps = sorted(all_optim_steps)
        final_optim = all_optim_steps[-1]
        proxy_gt_values = np.array(proxy_gt_by_step[final_optim])

        # Plot 1: Average Proxy GT Error - show all intermediate steps
        # Plot intermediate steps with faded colors
        for opt_step in all_optim_steps[:-1]:
            values = np.array(proxy_gt_by_step[opt_step])
            axes[idx, 0].plot(timesteps, values, 'o-', linewidth=1, markersize=4,
                            color='tab:blue', alpha=0.3)

        # Plot final step in bold
        axes[idx, 0].plot(timesteps, proxy_gt_values, 'o-', linewidth=2, markersize=8,
                         color='tab:blue', label=f'Final (optim_step={final_optim})')
        axes[idx, 0].set_xlabel('Autoregressive Step (t)', fontsize=12)
        axes[idx, 0].set_ylabel('Error (MSE)', fontsize=12)
        axes[idx, 0].set_title(f'Average Proxy GT Error - All Optimization Steps (n_calls={n_calls})',
                              fontsize=13, fontweight='bold')
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].set_xticks(timesteps)
        axes[idx, 0].legend(fontsize=9)

        # Plot 2: Trajectory Errors and Average Proxy GT Error
        trajectory_values = [trajectory_errors[n_calls][t]['average'] for t in timesteps]

        # Plot intermediate proxy GT error steps with faded colors
        for opt_step in all_optim_steps[:-1]:
            values = np.array(proxy_gt_by_step[opt_step])
            axes[idx, 1].plot(timesteps, values, 's-', linewidth=1, markersize=4,
                            color='tab:blue', alpha=0.3)

        axes[idx, 1].plot(timesteps, trajectory_values, 'o-', linewidth=2, markersize=8,
                    label='Trajectory errors', color='tab:red')
        axes[idx, 1].plot(timesteps, proxy_gt_values, 's-', linewidth=2, markersize=8,
                    label=f'Proxy GT error (final)', color='tab:blue')
        axes[idx, 1].set_xlabel('Autoregressive Step (t)', fontsize=12)
        axes[idx, 1].set_ylabel('Error (MSE)', fontsize=12)
        axes[idx, 1].set_title(f'Trajectory Errors vs Proxy GT Error (n_calls={n_calls})', fontsize=13, fontweight='bold')
        axes[idx, 1].legend(fontsize=10)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].set_xticks(timesteps)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'proxy_gt_metrics.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to: {plot_path}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    results = {
        'proxy_gt_errors': {},
        'trajectory_errors': {}
    }

    for n_calls in n_calls_per_step_list:
        results['proxy_gt_errors'][int(n_calls)] = {}
        results['trajectory_errors'][int(n_calls)] = {}

        timesteps = sorted(proxy_gt_errors[n_calls].keys())

        for t in timesteps:
            results['proxy_gt_errors'][int(n_calls)][int(t)] = {}
            for optim_step in proxy_gt_errors[n_calls][t]:
                results['proxy_gt_errors'][int(n_calls)][int(t)][str(optim_step)] = {
                    'average': float(proxy_gt_errors[n_calls][t][optim_step]['average']),
                    'per_sample': proxy_gt_errors[n_calls][t][optim_step]['per_sample'].tolist()
                }

            results['trajectory_errors'][int(n_calls)][int(t)] = {
                'average': float(trajectory_errors[n_calls][t]['average']),
                'per_sample': trajectory_errors[n_calls][t]['per_sample'].tolist()
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

    for n_calls in n_calls_per_step_list:
        timesteps = sorted(proxy_gt_errors[n_calls].keys())
        # Get final optim_step values for summary
        proxy_gt_values = []
        for t in timesteps:
            optim_steps = sorted(proxy_gt_errors[n_calls][t].keys())
            final_optim = optim_steps[-1]
            proxy_gt_values.append(proxy_gt_errors[n_calls][t][final_optim]['average'])
        proxy_gt_values = np.array(proxy_gt_values)

        trajectory_values = [trajectory_errors[n_calls][t]['average'] for t in timesteps]

        print(f"\nn_calls={n_calls}:")
        print(f"  Average Proxy GT Error:")
        print(f"    Mean over timesteps: {np.mean(proxy_gt_values):.8f}")
        print(f"    Trend: {'Increasing' if proxy_gt_values[-1] > proxy_gt_values[0] else 'Decreasing'}")
        print(f"  Trajectory Errors:")
        print(f"    Mean over timesteps: {np.mean(trajectory_values):.8f}")
        print(f"    Max at timestep: {timesteps[np.argmax(trajectory_values)]}")

if __name__ == "__main__":
    main()
