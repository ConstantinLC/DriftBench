import torch
import torch.optim as optim
import wandb
import os
from src.utils import evaluate_trajectory
from src.diffusion_utils import compute_estimate, betas_from_sqrtOneMinusAlphasCumprod
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts
import json
import collections
from torch.nn import functional as F

import os
import torch

def traj_eval_step(traj_loader, epoch, epoch_sampling_frequency, model, device, all_configs, log_dict, checkpoint_dir):
    """
    Evaluates trajectory metrics using the unified evaluate_trajectory function
    and handles logging and best-model saving.
    """
    # Check if we should evaluate this epoch
    if traj_loader and epoch % epoch_sampling_frequency == 0 and epoch > 0:
        model.eval()
        print("Evaluating on trajectories...")

        # 1. Retrieve configuration
        metrics_list = all_configs["loss_params"]["eval_traj_metrics"]
        primary_metric = all_configs["loss_params"]["primary_metric"]

        # 2. Run the unified evaluation (computes all requested metrics in one pass)
        results = evaluate_trajectory(model, traj_loader, device, metrics=metrics_list)

        # 3. Log detailed per-timestep metrics
        # Map result keys to the log_dict keys you expect
        if 'vort_corr' in metrics_list:
            for t, val in enumerate(results['vort_corr_per_ts']):
                log_dict[f'vorticity_correlation_t_{t+1}'] = val
        
        if 'mse' in metrics_list:
            for t, val in enumerate(results['mse_per_ts']):
                log_dict[f'mse_t_{t+1}'] = val
        
        if 'corr' in metrics_list:
            for t, val in enumerate(results['corr_per_ts']):
                log_dict[f'correlation_t_{t+1}'] = val

        # 4. Handle Best Model Selection
        if primary_metric == "vort_corr" and 'vort_corr' in metrics_list:
            # Metric: Time until correlation drops below threshold
            current_traj_time = results['vort_corr_time_under_threshold']
            log_dict['time_under_0.8'] = current_traj_time
            print(f"Vorticity correlation time under 0.8: {current_traj_time}")

            # Check for improvement (Higher time is better)
            if current_traj_time > log_dict.get('best_traj_time', -1):
                best_traj_time = current_traj_time
                log_dict['best_traj_time'] = best_traj_time
                
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ New best model saved to {best_checkpoint_path} with trajectory time: {best_traj_time}")

        elif primary_metric == "corr" and 'corr' in metrics_list:
            # Metric: Time until correlation drops below threshold
            current_traj_time = results['corr_time_under_threshold']
            log_dict['time_under_0.8'] = current_traj_time
            print(f"Correlation time under 0.8: {current_traj_time}")

            # Check for improvement (Higher time is better)
            if current_traj_time > log_dict.get('best_traj_time', -1):
                best_traj_time = current_traj_time
                log_dict['best_traj_time'] = best_traj_time
                
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ New best model saved to {best_checkpoint_path} with trajectory time: {best_traj_time}")

        elif primary_metric == "mse" and 'mse' in metrics_list:
            # Metric: Mean MSE over trajectory
            current_traj_error = results['mean_mse']
            print(f"📉 Mean Trajectory MSE: {current_traj_error:.2e}")

            # Initialize best_traj_error if missing
            if log_dict.get('best_traj_error') is None:
                log_dict['best_traj_error'] = float('inf')

            # Check for improvement (Lower MSE is better)
            if current_traj_error < log_dict['best_traj_error']:
                best_traj_error = current_traj_error
                log_dict['best_traj_error'] = best_traj_error
                
                best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ New best model saved to {best_checkpoint_path} with trajectory error: {best_traj_error:.2e}")

        # 5. Routine Checkpoint Saving
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved for epoch {epoch+1} at {checkpoint_path}")

    return log_dict


def train_diffusion_model(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir, device, is_master):
    """
    Trains a diffusion model with Cosine Annealing Learning Rate and DDP support.
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    
    # --- Config extraction ---
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    # Note: Model is already moved to device and wrapped in DDP in main.py.
    # We use the passed 'device' for data transfers.

    # --- 1. Optimizer & Scheduler Setup ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    
    # T_max is the number of epochs until the LR reaches the minimum
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_params["T_max"], 
        eta_min=lr_end
    )

    if is_master:
        print(f"Starting Diffusion training on {device} for {num_epochs} epochs...")
        print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")

    best_traj_error = float('inf')
    best_traj_time = 0

    for epoch in range(num_epochs):
        # Enforce LR floor manually if epoch exceeds T_max schedule
        if epoch >= train_params["T_max"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_end

        # --- DDP: Set Epoch for Sampler ---
        # Crucial for shuffling to work correctly across GPUs in DistributedSampler
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]
            
            optimizer.zero_grad()
            
            # Diffusion specific forward pass
            noise, predicted_noise = model(conditioning_frame, target_frame)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # --- Validation Loop ---
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            # If using DistributedSampler for validation, set epoch
            if hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)

            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data_val = sample_val["data"].to(device)
                    if data_val.shape[1] != 2:
                        continue
                    conditioning_frame_val = data_val[:, 0]
                    target_frame_val = data_val[:, 1]
                    
                    # Diffusion specific validation pass
                    noise, predicted_noise = model(conditioning_frame_val, target_frame_val)
                    loss_val = criterion(predicted_noise, noise)
                    running_val_loss += loss_val.item()
            
            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- 2. Step Scheduler & Get Current LR ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Logging (Master Only) ---
        if is_master:
            log_dict = {
                "epoch": epoch + 1,
                "training_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "learning_rate": current_lr,
                "best_traj_time": best_traj_time,
                "best_traj_error": best_traj_error
            }
            
            # --- Trajectory Evaluation ---
            # We access the underlying model (.module) if it's wrapped in DDP
            # This ensures that when traj_eval_step saves the checkpoint, 
            # it doesn't have the "module." prefix in the state_dict keys.
            if isinstance(model, DDP):
                model_to_eval = model.module
            else:
                model_to_eval = model

            # Run trajectory evaluation periodically
            # (Assuming traj_eval_step is imported from your utils)
            log_dict = traj_eval_step(
                traj_loader, 
                epoch, 
                epoch_sampling_frequency, 
                model_to_eval, 
                device, 
                all_configs, 
                log_dict, 
                checkpoint_dir
            )

            # Update local bests
            if 'best_traj_time' in log_dict:
                best_traj_time = log_dict['best_traj_time']
            if 'best_traj_error' in log_dict:
                best_traj_error = log_dict['best_traj_error']

            wandb.log(log_dict)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

    if is_master:
        print("Training complete!")
        
    return model



def train_diffusion_model_multisteps(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir):

    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    device = torch.device(train_params["device"])
    
    # --- Config extraction ---
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    model.to(device)

    # --- 1. Optimizer & Scheduler Setup ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    
    # T_max is the number of epochs until the LR reaches the minimum
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_params["T_max"], 
        eta_min=lr_end
    )

    n_proxy_steps = train_params["n_proxy_steps"]
    backgrad = train_params["backgrad"]

    best_val_loss = float('inf')
    best_traj_time = 0
    best_traj_error = 100000000

    print(f"Starting Multi-steps Diffusion training on {device} for {train_params['num_epochs']} epochs...")

    for epoch in range(num_epochs):
        if epoch >= train_params["T_max"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_end

        model.train()
        running_train_loss = 0.0
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device)
            optimizer.zero_grad()
            loss = torch.Tensor([0]).to(device)

            for t in range(data.shape[1] - 1):
                if t == 0:
                    conditioning_frame = data[:, t]
                else:
                    conditioning_frame = x0_estimate
                target_frame = data[:, t + 1]
            
                if t < data.shape[1] - 2:
                    noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)

                    if backgrad:
                        x0_estimate = compute_estimate(model, n_proxy_steps, conditioning_frame, target_frame)
                    else:
                        with torch.no_grad():
                            x0_estimate = compute_estimate(model, n_proxy_steps, conditioning_frame, target_frame)

                if t > 0:
                    noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)

                loss += criterion(predicted_noise, noise)

            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data = sample["data"].to(device)
                    loss = torch.Tensor([0]).to(device)

                    for t in range(data.shape[1] - 1):
                        conditioning_frame = data[:, t]
                        target_frame = data[:, t + 1]
                    
                        if t < data.shape[1] - 2:
                            noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)
                        else:
                            noise, predicted_noise = model(conditioning_frame, target_frame, return_x0_estimate=False)

                        loss += criterion(predicted_noise, noise)

                    running_val_loss += loss.item()
                
            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        # --- Base Logging Logic ---
        log_dict = {
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "learning_rate": current_lr,  # Log LR to visualize the cosine curve
            "best_traj_time": best_traj_time,
            "best_traj_error": best_traj_time
        }
        
        log_dict = traj_eval_step(traj_loader, epoch, epoch_sampling_frequency, model, device, all_configs, log_dict, checkpoint_dir)

        wandb.log(log_dict)

        print(f"Epoch [{epoch+1}/{train_params['num_epochs']}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}")


    print("Training complete!")
    return model


import torch
import torch.optim as optim
import wandb
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_unet(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir, device, is_master):
    """
    Trains a U-Net model with DDP support.
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    
    # --- Config extraction ---
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    # Note: Model is already moved to device in main.py before DDP wrapping
    # But strictly ensuring input data goes to the right device is handled in the loop.

    # --- 1. Optimizer & Scheduler Setup ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    
    # T_max is the number of epochs until the LR reaches the minimum
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_params["T_max"], 
        eta_min=lr_end
    )

    if is_master:
        print(f"Starting Training on {device} for {num_epochs} epochs...")
        print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")

    best_traj_error = float('inf')
    best_traj_time = 0

    for epoch in range(num_epochs):
        if epoch >= train_params["T_max"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_end

        # --- DDP: Set Epoch for Sampler ---
        # Crucial for shuffling to work correctly across GPUs
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]
            
            optimizer.zero_grad()
            pred = model(conditioning_frame, time=None)
            loss = criterion(pred, target_frame)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # --- Validation Loop ---
        # Note: In DDP, each GPU calculates val loss on its subset. 
        # We typically just log Rank 0's validation loss to save overhead.
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            # If using DistributedSampler for validation, set epoch as well
            if hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)
                
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data_val = sample_val["data"].to(device)
                    if data_val.shape[1] != 2:
                        continue
                    conditioning_frame_val = data_val[:, 0]
                    target_frame_val = data_val[:, 1]
                    
                    pred = model(conditioning_frame_val, time=None)
                    loss_val = criterion(pred, target_frame_val)
                    running_val_loss += loss_val.item()
            
            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- 2. Step Scheduler & Get Current LR ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Logging (Master Only) ---
        if is_master:
            log_dict = {
                "epoch": epoch + 1,
                "training_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "learning_rate": current_lr,
                "best_traj_time": best_traj_time,
                "best_traj_error": best_traj_error
            }
            
            # --- Trajectory Evaluation ---
            # We access the underlying model (.module) if it's wrapped in DDP
            # This ensures that when traj_eval_step saves the checkpoint, 
            # it doesn't have the "module." prefix in the state_dict keys.
            if isinstance(model, DDP):
                model_to_eval = model.module
            else:
                model_to_eval = model

            # Run trajectory evaluation periodically (only on master)
            log_dict = traj_eval_step(
                traj_loader, 
                epoch, 
                epoch_sampling_frequency, 
                model_to_eval, 
                device, 
                all_configs, 
                log_dict, 
                checkpoint_dir
            )

            # Update local bests (traj_eval_step updates the log_dict)
            if 'best_traj_time' in log_dict:
                best_traj_time = log_dict['best_traj_time']
            if 'best_traj_error' in log_dict:
                best_traj_error = log_dict['best_traj_error']

            wandb.log(log_dict)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

    if is_master:
        print("Training complete!")
        
    return model



def train_unet_multisteps(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir, device, is_master):
    """
    Trains a U-Net model using unrolled (autoregressive) training steps.
    The model predicts the next frame, and that prediction is fed back 
    as input for the subsequent step calculation.
    """
    epoch_sampling_frequency = train_params["epoch_sampling_frequency"]
    
    # --- Config extraction ---
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    
    # --- 1. Optimizer & Scheduler Setup ---
    optimizer = optim.Adam(model.parameters(), lr=lr_start)
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=train_params["T_max"], 
        eta_min=lr_end
    )

    if is_master:
        print(f"Starting Multi-Step U-Net Training on {device} for {num_epochs} epochs...")
        print(f"LR Schedule: Cosine Annealing from {lr_start:.1e} to {lr_end:.1e}")

    best_traj_error = float('inf')
    best_traj_time = 0

    for epoch in range(num_epochs):
        # Enforce minimum LR after T_max
        if epoch >= train_params["T_max"]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_end

        # --- DDP: Set Epoch for Sampler ---
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- Training Loop ---
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, sample in enumerate(train_loader):
            data = sample["data"].to(device) # Shape: (B, T, C, H, W)
            optimizer.zero_grad()
            
            # Accumulate loss over the trajectory
            loss = torch.tensor(0.0, device=device)
            
            # Initial input is the first ground truth frame
            current_input = data[:, 0]

            # Iterate through time steps (0 -> 1, 1 -> 2, ...)
            # We predict T-1 transitions
            steps = data.shape[1] - 1
            
            for t in range(steps):
                target_frame = data[:, t + 1]

                # Forward pass: Predict next frame
                # U-Net typically ignores the 'time' argument used in diffusion
                pred = model(current_input, time=None)
                
                # Accumulate MSE loss
                loss += criterion(pred, target_frame)
                
                # Autoregressive step: Use prediction as input for next step
                # (Detach to prevent gradient explosion if sequence is very long, 
                # though usually for short unrolls we keep gradients)
                current_input = pred

            # Backpropagation on the accumulated loss
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # --- Validation Loop ---
        if val_loader is None:
            avg_val_loss = 0.0
        else:
            if hasattr(val_loader.sampler, "set_epoch"):
                val_loader.sampler.set_epoch(epoch)
                
            running_val_loss = 0.0
            with torch.no_grad():
                for batch_idx_val, sample_val in enumerate(val_loader):
                    data_val = sample_val["data"].to(device)
                    if data_val.shape[1] < 2:
                        continue
                        
                    loss_val = torch.tensor(0.0, device=device)
                    current_input_val = data_val[:, 0]
                    steps_val = data_val.shape[1] - 1

                    for t in range(steps_val):
                        target_frame_val = data_val[:, t + 1]
                        pred_val = model(current_input_val, time=None)
                        loss_val += criterion(pred_val, target_frame_val)
                        current_input_val = pred_val

                    running_val_loss += loss_val.item()
            
            avg_val_loss = running_val_loss / (batch_idx_val + 1) if (batch_idx_val + 1) > 0 else 0

        # --- Scheduler Step ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Logging (Master Only) ---
        if is_master:
            log_dict = {
                "epoch": epoch + 1,
                "training_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "learning_rate": current_lr,
                "best_traj_time": best_traj_time,
                "best_traj_error": best_traj_error
            }
            
            # Access underlying model if wrapped in DDP
            if isinstance(model, DDP):
                model_to_eval = model.module
            else:
                model_to_eval = model

            # Run trajectory evaluation
            log_dict = traj_eval_step(
                traj_loader, 
                epoch, 
                epoch_sampling_frequency, 
                model_to_eval, 
                device, 
                all_configs, 
                log_dict, 
                checkpoint_dir
            )

            # Update local bests
            if 'best_traj_time' in log_dict:
                best_traj_time = log_dict['best_traj_time']
            if 'best_traj_error' in log_dict:
                best_traj_error = log_dict['best_traj_error']

            wandb.log(log_dict)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

    if is_master:
        print("Training complete!")
        
    return model