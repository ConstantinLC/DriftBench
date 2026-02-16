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

def coarsen_jax(x, r=None):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // r, r, w // r, r)
    return x.mean(axis=(3, 5))


def get_trajectory_predictions(active_model, sim, n_warmup_calls, n_trajectories, 
                               seed, n_ar_steps, n_calls_per_step, coarsen_factor, device):
    # Generate initial HR state and warmup
    key = rng.PRNGKey(seed)
    keys = rng.split(key, n_trajectories)
    hr_prior = sim._prior(keys)

    hr_preds = {}
    model_preds = {}

    hr_state = hr_prior
    for t in range(1, n_warmup_calls + 1):
        hr_state = sim._transition(hr_state)
        if t > n_warmup_calls - n_calls_per_step - 2:
            hr_preds[t-n_warmup_calls] = hr_state
            with torch.no_grad():
                coarse_state = coarsen_jax(hr_state, r=coarsen_factor)
                coarse_state = torch.tensor(np.array(coarse_state)).to(device)
                model_curr = run_model(active_model, coarse_state)
                model_curr_jax = jnp.array(model_curr.detach().cpu().numpy())
                model_preds[t-n_warmup_calls+1] = model_curr_jax

    # Generate GT trajectory
    for t in range(1, n_ar_steps + 1):
        hr_preds[t] = sim._transition(hr_preds[t-1])

    # Generate model predictions
    with torch.no_grad():
        coarse_state = coarsen_jax(hr_preds[0], r=coarsen_factor)
        model_curr = torch.tensor(np.array(coarse_state)).to(device)
        for t in range(1, n_ar_steps + 1):
            model_curr = run_model(active_model, model_curr)
            model_curr_jax = jnp.array(model_curr.detach().cpu().numpy())
            model_preds[t] = model_curr_jax

    return hr_preds, model_preds


def optimize_initial_conditions(n_calls_per_step, n_optimization_steps, HR_preds,
                                 Model_preds, learning_rate, ar_idx, sim, coarsen_factor,
                                 initialization_type="model_traj", compute_loss_and_grads=None):

    t = ar_idx
    optimizer = optax.adam(learning_rate)

    # Build JIT-compiled loss+grad function if not provided
    if compute_loss_and_grads is None:
        compute_loss_and_grads = make_loss_and_grad_fn(sim, n_calls_per_step, coarsen_factor)

    # Instead of taking HR trajectory as initialization to initial condition,
    # Take the upsampled LR trajectory
    # This removes a bias that can lead the proxy to converge better for early AR steps.
    if initialization_type == "model_traj":
        initialization_coarse = Model_preds[t - n_calls_per_step - 1]

        batch_size = HR_preds[0].shape[0]
        initialization = jax.image.resize(
            initialization_coarse,
            shape=(batch_size, 2, 256, 256),
            method='nearest'
        )
    elif initialization_type == "noise":
        batch_size = HR_preds[0].shape[0]
        initialization = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 2, 256, 256))
    else :
        raise Exception("Unknown Initialization Type for Initial Condition")

    opt_state = optimizer.init(initialization)

    interm_refined = {}
    hr_refined = initialization
    for iteration in range(n_optimization_steps):
        hr_refined, opt_state, loss = optimization_step(
            opt_state, hr_refined, optimizer, compute_loss_and_grads, Model_preds[t-1]
        )
        if iteration % 200 == 0 and iteration > 0 :
            print(f"  Step {t}, Iteration {iteration:4d} | Loss: {loss:.8f}")
            interm_refined[iteration] = hr_refined

    return hr_refined, interm_refined

def make_loss_and_grad_fn(sim, n_calls_per_step, coarsen_factor):
    """Factory that returns a JIT-compiled loss+grad function with sim captured in closure."""

    def loss_fn(hr_initial_condition, model_predicted_coarse_trajectory):
        hr_state = hr_initial_condition

        def step_fn(carry, _):
            next_state = sim._transition(carry)
            return next_state, None

        hr_state_final, _ = jax.lax.scan(step_fn, hr_state, None,
                                         length=n_calls_per_step)
        simulated_coarse = coarsen_jax(hr_state_final, r=coarsen_factor)

        return jnp.mean((simulated_coarse - model_predicted_coarse_trajectory) ** 2)

    @jax.jit
    def compute_loss_and_grads(hr_initial_condition, model_predicted_coarse_trajectory):
        loss, grads = jax.value_and_grad(loss_fn)(hr_initial_condition,
                                                   model_predicted_coarse_trajectory)
        return loss, grads

    return compute_loss_and_grads


def optimization_step(opt_state, hr_initial_condition, optimizer,
                      compute_loss_and_grads,
                      model_predicted_coarse_trajectory):
    """Single optimization step (JIT loss computation, Python optimizer update)."""
    loss, grads = compute_loss_and_grads(hr_initial_condition,
                                         model_predicted_coarse_trajectory)
    updates, opt_state = optimizer.update(grads, opt_state)
    hr_initial_condition = optax.apply_updates(hr_initial_condition, updates)
    return hr_initial_condition, opt_state, loss