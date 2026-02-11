import torch
import torch.optim as optim
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.diffusion_utils import evaluate_dw_train_inf_gap

def train_diffusion_single_noise_level(model, train_loader, val_loader, traj_loader, train_params, criterion, all_configs, checkpoint_dir, device, is_master):
    """
    Trains a diffusion model on a single noise level with DDP support.
    Breaks training if (own_prediction_error / clean_prediction_error) < tau.
    """
    num_epochs = train_params["num_epochs"]
    lr_start = train_params["learning_rate_start"]
    lr_end = train_params["learning_rate_end"]
    tau = train_params["tau"]  # Threshold for early stopping
    
    # Get world size for averaging metrics
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=lr_start)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_params["T_max"], eta_min=lr_end)

    success = False

    if is_master:
        print(f"Starting Single Level Training on {device}.")
        print(f"Stop Threshold (Tau): {tau}")

    for epoch in range(num_epochs):
        # Enforce LR floor

        # --- DDP: Set Epoch for Sampler ---
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
            
            # Forward pass
            noise, predicted_noise = model(conditioning_frame, target_frame)
            loss = criterion(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        # Calculate local average loss
        avg_train_loss = running_train_loss / (batch_idx + 1)
        
        # --- Validation & Ratio Check ---
        # 1. Prepare for Evaluation
        model.eval()
        
        # Use underlying model for evaluation utility if wrapped
        model_to_eval = model.module if isinstance(model, DDP) else model

        # 2. Run Evaluation (Locally on each GPU)
        # Note: This assumes val_loader is distributed or identical. 
        # If distributed, each GPU computes error on its subset.
        evals = evaluate_dw_train_inf_gap(
            {'model': model_to_eval}, 
            val_loader, 
            device=device, 
            n_batches=10, 
            metric='mse', 
            input_types=['clean', 'own-pred']
        )
        
        # 3. Extract Errors
        # .mean() returns a scalar tensor. 
        # CRITICAL: We must force it to be on the GPU (.to(device)) for NCCL to work.
        local_clean_err = evals['mse_clean']['model'].float().mean().to(device)
        local_own_err = evals['mse_own_pred']['model'].float().mean().to(device)
        
        # 4. Synchronize Metrics across GPUs
        # Now that they are on CUDA, NCCL can handle them
        dist.all_reduce(local_clean_err, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_own_err, op=dist.ReduceOp.SUM)
        
        # Compute global averages
        global_clean_err = local_clean_err / world_size
        global_own_err = local_own_err / world_size
        
        # 5. Compute Ratio
        # Add epsilon to avoid division by zero
        ratio = global_own_err / (global_clean_err + 1e-8)
        
        # --- Scheduler Step ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # --- Logging & Decision ---
        if is_master:
            log_dict = {
                "epoch": epoch + 1,
                "training_loss": avg_train_loss,
                "learning_rate": current_lr,
                "ratio": ratio.item(),
                "clean_err": global_clean_err.item(),
                "own_err": global_own_err.item()
            }
            wandb.log(log_dict)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_train_loss:.6f} | "
                  f"Ratio: {ratio:.4f} (Tau: {tau}) | LR: {current_lr:.2e}")

        # --- Break Condition ---
        # Since 'ratio' is derived from synchronized tensors, this condition 
        # is guaranteed to evaluate identically on all ranks.
        if ratio < tau:
            success = True
            if is_master:
                print(f" >>> SUCCESS: Ratio {ratio:.4f} < {tau}. Converged.")
            break
            
        scheduler.step()

    return model, success