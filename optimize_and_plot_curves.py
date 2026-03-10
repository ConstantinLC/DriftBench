#!/usr/bin/env python
"""
SIMPLIFIED: Track proxy GT optimization progress across different K values

This script:
1. Runs proxy GT evaluation for different n_calls_per_step (K) values
2. Records optimization loss and metrics at intermediate steps
3. Creates a grid plot showing error evolution during optimization

Usage:
    python optimize_and_plot_curves.py \
        --config conf/config.yaml \
        --n-calls 1 2 3 4 \
        --output-dir ./opt_curves \
        --batch-size 3
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
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import yaml

# Project Imports
from src.data_loader import get_data_loaders
from src.model_loader import load_model
from src.utils import run_model
from sda.mcs import KolmogorovFlow

# ============================================================================
# GLOBALS
# ============================================================================
sim = None
optimizer = None
n_simulator_calls = None
coarsen_factor = None

# ============================================================================
# JAX HELPERS
# ============================================================================

def coarsen(x, factor):
    """Coarsen JAX array via average pooling."""
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // factor, factor, w // factor, factor)
    return x.mean(axis=(3, 5))

def loss_fn(hr_ic, model_pred_coarse):
    """Loss: match simulator output to model prediction."""
    hr = hr_ic
    for _ in range(n_simulator_calls):
        hr = sim._transition(hr)
    sim_coarse = coarsen(hr, coarsen_factor)
    return jnp.mean((sim_coarse - model_pred_coarse) ** 2)

@jax.jit
def opt_step(opt_state, hr_ic, model_pred):
    """One optimization step."""
    loss, grads = jax.value_and_grad(loss_fn)(hr_ic, model_pred)
    updates, opt_state = optimizer.update(grads, opt_state)
    hr_ic = optax.apply_updates(hr_ic, updates)
    return hr_ic, opt_state, loss

def compute_metrics(model_pred_t, proxy_hr_t, true_coarse_t):
    """Compute both metrics at once."""
    proxy_coarse = coarsen(proxy_hr_t, coarsen_factor)

    # Exposure bias
    exp_bias = jnp.mean((model_pred_t - proxy_coarse) ** 2)

    # Trajectory drift components
    direct_error = jnp.mean((model_pred_t - true_coarse_t) ** 2)
    proxy_error = jnp.mean((model_pred_t - proxy_coarse) ** 2)
    drift = direct_error - proxy_error

    return float(exp_bias), float(drift), float(direct_error), float(proxy_error)

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_with_n_calls(n_calls, cfg, batch_size=3, track_every=200):
    """
    Run evaluation with specific n_calls value, tracking progress.

    Returns:
        {timestep: {opt_iteration: metrics_dict}}
    """
    global sim, optimizer, n_simulator_calls, coarsen_factor

    n_simulator_calls = n_calls
    coarsen_factor = cfg['simulator']['coarsen_factor']

    ar_steps = cfg['optimization']['ar_steps']
    n_opt_steps = cfg['optimization']['n_steps']
    lr = cfg['optimization']['learning_rate']

    print(f"\n{'='*70}")
    print(f"Evaluating: n_calls = {n_calls} | Batch: {batch_size} | AR steps: {ar_steps}")
    print(f"{'='*70}\n")

    # Load model
    model = load_model(
        model_type=cfg['model']['type'],
        checkpoint_path=cfg['model']['checkpoint'],
        model_params=cfg['model'],
        device=cfg['device'],
    )
    model.eval()

    # Load data
    data_cfg = {
        'dataset_name': cfg['data'].get('dataset_name', 'kolmogorov'),
        'data_path': cfg['data']['data_path'],
        'resolution': cfg['data']['resolution'],
        'sequence_length': cfg['data'].get('sequence_length', [3, 1]),
        'trajectory_sequence_length': cfg['data'].get('trajectory_sequence_length', 100),
        'frames_per_time_step': cfg['data'].get('frames_per_time_step', 1),
        'limit_trajectories_train': cfg['data'].get('limit_trajectories_train', 819),
        'limit_trajectories_val': cfg['data'].get('limit_trajectories_val', 102),
        'batch_size': batch_size,
        'val_batch_size': batch_size,
    }

    with torch.no_grad():
        _, val_loader, _ = get_data_loaders(data_cfg)
        _ = next(iter(val_loader))

    # Init simulator
    sim = KolmogorovFlow(size=cfg['simulator']['size'], dt=cfg['simulator']['dt'])

    # Generate initial HR state and warmup
    key = rng.PRNGKey(cfg.get('seed', 1))
    keys = rng.split(key, batch_size)
    hr = sim._prior(keys)

    for t in range(cfg['simulator']['n_warmup_calls']):
        hr = sim._transition(hr)

    # Build ground truth trajectory
    HR_traj = {0: hr}
    for t in range(1, ar_steps + 1):
        HR_traj[t] = sim._transition(HR_traj[t-1])

    # Build model trajectory
    Model_traj = {0: coarsen(HR_traj[0], coarsen_factor)}
    with torch.no_grad():
        m_curr = torch.tensor(np.array(Model_traj[0])).to(cfg['device'])
        for t in range(1, ar_steps + 1):
            m_curr = run_model(model, m_curr)
            Model_traj[t] = jnp.array(m_curr.detach().cpu().numpy())

    # Optimize proxies and track progress
    results_by_t = {}

    for t in range(1, ar_steps + 1):
        optimizer = optax.adam(lr)

        # Init HR IC from previous model prediction or true state
        if t - n_calls - 1 < 0:
            ic_coarse = coarsen(HR_traj[t - n_calls - 1], 4)
        else:
            ic_coarse = Model_traj[t - n_calls - 1]

        # Upsample to HR
        ic_hr = jax.image.resize(ic_coarse, shape=(batch_size, 2, 256, 256), method='nearest')
        opt_st = optimizer.init(ic_hr)

        metrics_at_step = {}
        hr_refined = ic_hr

        for it in range(n_opt_steps):
            hr_refined, opt_st, loss = opt_step(opt_st, hr_refined, Model_traj[t-1])

            # Record at checkpoints
            if it % track_every == 0 or it == n_opt_steps - 1:
                # Evolve to final state
                hr_final = hr_refined
                for _ in range(n_calls + 1):
                    hr_final = sim._transition(hr_final)

                # Compute metrics
                true_coarse = coarsen(HR_traj[t], coarsen_factor)
                exp_bias, drift, d_err, p_err = compute_metrics(Model_traj[t], hr_final, true_coarse)

                metrics_at_step[it] = {
                    'loss': float(loss),
                    'exp_bias': exp_bias,
                    'drift': drift,
                    'd_error': d_err,
                    'p_error': p_err,
                }

                print(f"  t={t}/{ar_steps}, iter={it:4d}: loss={loss:.6f}, "
                      f"exp_bias={exp_bias:.6f}, drift={drift:.6f}")

        results_by_t[t] = metrics_at_step

    return results_by_t

# ============================================================================
# PLOTTING
# ============================================================================

def plot_optimization_grid(results_by_n_calls, n_calls_list, output_path):
    """
    Create grid: rows=K values, cols=timesteps.
    Each cell shows exposure bias and drift vs optimization step.
    """
    n_k = len(n_calls_list)
    timesteps = sorted(results_by_n_calls[n_calls_list[0]].keys())

    fig, axes = plt.subplots(n_k, len(timesteps), figsize=(18, 4*n_k))
    if n_k == 1:
        axes = axes.reshape(1, -1)

    for i, n_calls in enumerate(n_calls_list):
        for j, t in enumerate(timesteps):
            ax = axes[i, j]
            metrics = results_by_n_calls[n_calls][t]

            iters = sorted(metrics.keys())
            exp_bias_vals = [metrics[it]['exp_bias'] for it in iters]
            drift_vals = [metrics[it]['drift'] for it in iters]

            ax2 = ax.twinx()

            ax.plot(iters, exp_bias_vals, 'o-', linewidth=2.5, markersize=7,
                   color='#2E86AB', label='Exposure Bias', markerfacecolor='#A23B72', alpha=0.7)
            ax2.plot(iters, drift_vals, 's-', linewidth=2.5, markersize=7,
                    color='#F18F01', label='Trajectory Drift', markerfacecolor='#C73E1D', alpha=0.7)

            ax.set_ylabel('Exposure Bias (MSE)', fontsize=11, color='#2E86AB', fontweight='bold')
            ax2.set_ylabel('Trajectory Drift', fontsize=11, color='#F18F01', fontweight='bold')
            ax.set_xlabel('Optimization Iteration', fontsize=11)
            ax.set_title(f'K={n_calls}, t={t}', fontsize=12, fontweight='bold')

            ax.grid(True, alpha=0.2, linestyle='--')
            ax.tick_params(axis='y', labelcolor='#2E86AB')
            ax2.tick_params(axis='y', labelcolor='#F18F01')

            # Set tight x limits
            ax.set_xlim(left=0)

    # Add legend (single combined legend at top)
    fig.text(0.5, 0.98, 'Exposure Bias (blue circles) | Trajectory Drift (orange squares)',
            ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_path}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Track optimization curves')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                       help='Config YAML path')
    parser.add_argument('--n-calls', type=int, nargs='+', default=[1, 2, 3],
                       help='Values of n_calls_per_step to test')
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--track-every', type=int, default=200,
                       help='Record metrics every N iterations')
    parser.add_argument('--output-dir', type=str, default='./opt_curves')
    parser.add_argument('--output-plot', type=str, default='opt_curves.png')
    parser.add_argument('--output-json', type=str, default='opt_curves.json')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        # Load base config
        cfg = yaml.safe_load(f)

    # Load subconfigs (defaults)
    defaults = cfg.get('defaults', [])
    for default in defaults:
        if isinstance(default, dict):
            for key, val in default.items():
                if key != '_self_':
                    subdir = key
                    subfile = f"conf/{subdir}/{val}.yaml"
                    if os.path.exists(subfile):
                        with open(subfile, 'r') as f:
                            subcfg = yaml.safe_load(f)
                            if key not in cfg:
                                cfg[key] = {}
                            cfg[key].update(subcfg)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluations
    all_results = {}
    for n_calls in args.n_calls:
        try:
            results = evaluate_with_n_calls(n_calls, cfg, args.batch_size, args.track_every)
            all_results[n_calls] = results
        except Exception as e:
            print(f"❌ Error with n_calls={n_calls}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_json_path = os.path.join(args.output_dir, args.output_json)
    with open(output_json_path, 'w') as f:
        # Convert to serializable format
        json_data = {}
        for n_calls, ts_results in all_results.items():
            json_data[str(n_calls)] = {}
            for t, opt_results in ts_results.items():
                json_data[str(n_calls)][str(t)] = {}
                for it, metrics in opt_results.items():
                    json_data[str(n_calls)][str(t)][str(it)] = metrics
        json.dump(json_data, f, indent=2)

    print(f"✓ Saved JSON to: {output_json_path}")

    # Create plot
    output_plot_path = os.path.join(args.output_dir, args.output_plot)
    plot_optimization_grid(all_results, args.n_calls, output_plot_path)

    print(f"\n{'='*70}")
    print("✓ OPTIMIZATION CURVES COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
