# Experimental Setup: Proxy GT Evaluation (Elements 1 & 2)

## Metrics

### Element 1: Exposure Bias Indicator
**Metric:** `||M(x̂_t) - GT(x̃_t)||`

Compares one-step-ahead predictions from two sources:
- `M(x̂_t)`: ML model applied to its own prediction
- `GT(x̃_t)`: HR simulator applied to the physics-consistent proxy, then coarsened

An **increasing trend** reveals exposure bias: the model becomes unreliable when applied to states that are physically consistent but differ from its training distribution.

### Element 2: Trajectory Drift
**Metric:** `||M(x̂_t) - x_{t+1}|| - ||M(x̂_t) - GT(x̃_t)||`

The difference between:
- Direct model error (model prediction vs ground truth)
- Proxy-corrected error (model prediction vs physics-evolved proxy)

This gap is the **irreducible accumulated error** — what cannot be fixed even with a perfect downstream simulator. Positive and growing drift means the trajectory has passed its point of no return.

## Pipeline

```
Phase 1: Generate trajectories
  HR_preds[t]    = sim(HR_preds[t-1])           # Ground truth (256×256)
  Model_preds[t] = model(Model_preds[t-1])      # ML predictions (64×64)

Phase 2: Find HR proxies (for t = 2..T)
  Optimize hr_ic so that:
    coarsen(sim^K(hr_ic)) ≈ Model_preds[t-1]
  Then evolve K+1 more steps → Proxy_HR_preds[t]

Phase 3: Evaluate metrics (for t = 2..T)
  Element 1: MSE( model(Model_preds[t]), coarsen(sim(Proxy_HR_preds[t])) )
  Element 2: MSE( model(Model_preds[t]), coarsen(HR_preds[t]) )  -  Element 1
```

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| K (simulator calls) | 2 | HR steps per coarse step |
| Optimization iterations | 1000 | Per-timestep proxy search |
| AR steps | 10 | Evaluation horizon |
| Warmup | 64 | Steps to reach steady state |
| Batch size | 5 | Samples for averaging |

## Usage

```bash
python proxy_gt_evaluation.py \
  --config configs/config.json \
  --checkpoints ModelName=/path/to/ckpt.pth \
  --batch-size 5 \
  --output-dir ./proxy_gt_results
```

Outputs `proxy_gt_metrics.png` (two plots) and `proxy_gt_metrics.json` (numerical results).

## Design Decisions

- **Proxy_HR_preds[t] starts from HR_preds[t-K-1]**: initialization close to the true state helps optimization converge.
- **K=2 HR steps ≈ 1 coarse step**: matches the temporal resolution ratio for Kolmogorov flow with dt=0.2.
- **Both metrics share the `M(x̂_t)` computation**: Element 2 decomposes the direct error into a physics-correctable part (Element 1) and an irreducible remainder (the drift).
- **Metrics start at t=2**: proxy optimization requires at least one prior model prediction to match against.
