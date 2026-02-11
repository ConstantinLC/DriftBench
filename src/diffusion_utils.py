import torch
import numpy as np
import torch.nn.functional as F

### OBTAIN BETAS FROM SNR

def betas_from_sqrtOneMinusAlphasCumprod(sqrtOneMinusAlphasCumprod: torch.Tensor) -> torch.Tensor:
    """
    Given sqrtOneMinusAlphasCumprod (shape [T] or [T, 1, 1, 1]), reconstructs a stable betas schedule.
    """
    # Flatten
    sqrtOneMinusAlphasCumprod = sqrtOneMinusAlphasCumprod.flatten().float()

    # 1. Compute alphas_cumprod
    alphas_cumprod = 1.0 - sqrtOneMinusAlphasCumprod ** 2  # shape [T]

    # Numerical safety
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1.0)

    # 2. Compute alphas from ratio
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    alphas = alphas_cumprod / alphas_cumprod_prev
    alphas = torch.clamp(alphas, min=1e-8, max=1.0)

    # 3. Compute betas
    betas = 1.0 - alphas

    # 4. Clamp betas for stability
    betas = torch.clamp(betas, min=1e-8, max=0.999999)

    return betas


### BETA SCHEDULES
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    if timesteps < 10:
        raise ValueError("Warning: Less than 10 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (500/timesteps) # adjust reference values determined for 500 steps
    beta_end = 0.02 * (500/timesteps)
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return torch.clip(betas, 0.0001, 0.9999)

def quadratic_beta_schedule(timesteps):
    if timesteps < 20:
        raise ValueError("Warning: Less than 20 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (1000/timesteps) # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000/timesteps)
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    return torch.clip(betas, 0.0001, 0.9999)

def cubic_beta_schedule(timesteps):
    if timesteps < 20:
        raise ValueError("Warning: Less than 20 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (1000/timesteps) # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000/timesteps)
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 3
    return 2*torch.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_schedule(timesteps):
    if timesteps < 20:
        raise Warning("Warning: Less than 20 timesteps require adjustments to this schedule!")

    beta_start = 0.0001 * (1000/timesteps) # adjust reference values determined for 1000 steps
    beta_end = 0.02 * (1000/timesteps)
    betas = torch.linspace(-6, 6, timesteps)
    betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    return torch.clip(betas, 0.0001, 0.9999)

def psd_beta_schedule(timesteps):
    return torch.Tensor([ 4.53887314e-4, 2.85259073e-5, 2.85267210e-5, 2.85275348e-5, 3.43625704e-5, 4.60325957e-5, 4.60347148e-5, 4.60368341e-5, 4.10608473e-5, 3.85733780e-5, 3.85748659e-5, 3.85763540e-5, 9.50367681e-5, 9.50458010e-5, 9.50548356e-5, 9.24530131e-5, 8.72393612e-5, 8.72469726e-5, 8.72545853e-5, 1.11271477e-4, 1.23289835e-4, 1.23305037e-4, 1.23320243e-4, 1.38707811e-4, 1.38727053e-4, 1.38746301e-4, 1.77679095e-4, 2.55551583e-4, 2.55616906e-4, 2.55682263e-4, 2.67478209e-4, 2.73416620e-4, 2.73491397e-4, 2.73566215e-4, 4.05702254e-4, 4.05866915e-4, 4.06031710e-4, 4.71263714e-4, 6.01681417e-4, 6.02043655e-4, 6.02406330e-4, 6.51330849e-4, 6.76051886e-4, 6.76509241e-4, 6.76967216e-4, 7.92570006e-4, 7.93198671e-4, 7.93828335e-4, 9.96352187e-4, 1.40153499e-3, 1.40350204e-3, 1.40547463e-3, 2.23737102e-3, 2.65827770e-3, 2.66536298e-3, 2.67248612e-3, 3.71762129e-3, 3.73149357e-3, 3.74546977e-3, 4.59707837e-3, 6.30109965e-3, 6.34105527e-3, 6.38152085e-3, 8.53294772e-3, 9.67068796e-3, 9.76512342e-3, 9.86142141e-3, 1.84721510e-2, 1.88197930e-2, 1.91807712e-2, 2.46602855e-2, 3.57507445e-2, 3.70762479e-2, 3.85038253e-2, 4.00154242e-2, 4.16676139e-2, 4.34792922e-2, 4.54556727e-2, 7.70881607e-2, 8.35271121e-2, 9.11397525e-2, 1.05886062e-1, 1.30967473e-1, 1.50704915e-1, 1.77447059e-1, 1.88729694e-1, 2.15995741e-1, 2.75503275e-1, 3.80268488e-1, 7.68084062e-2, 8.31987712e-2, 9.07489743e-2, 1.24347840e-1, 1.98059165e-1, 2.46974784e-1, 3.27976779e-1, 2.19751730e-1, 1.09715958e-1, 1.23237027e-1, 1.40559114e-1 ])

def cosine_sigma_schedule(sigma_min, sigma_max, T):
    t = torch.linspace(0, T, T)
    sigmas = sigma_min + (sigma_max - sigma_min) / 2 * (1 - torch.cos(t * np.pi / T))
    return sigmas

def low_nl_max_out_beta_schedule(timesteps, min_log_nl):
    noise_levels = torch.linspace(10**min_log_nl, 10**(-0.0001), timesteps)
    betas = betas_from_sqrtOneMinusAlphasCumprod(noise_levels)
    return betas

def low_and_high_nl_focus(timesteps, min_log_nl):
    start_log = min_log_nl
    
    n_mid = (timesteps-1)//2
    n_high = (timesteps)//2
    
    # Construct parts (handling endpoints to avoid duplicates)
    p1 = torch.tensor([10**start_log])
    p2 = torch.linspace(0.1, 0.95, n_mid + 1)[:-1]
    p3 = torch.linspace(0.95, 10**-0.0001, n_high)

    noise_levels = torch.cat((p1, p2, p3))
    betas = betas_from_sqrtOneMinusAlphasCumprod(noise_levels)
    return betas

def initial_exploration_beta_schedule(min_log_value, timesteps):
    start = 10**min_log_value
    end = 10**(-0.0001)

    power = 0.5
    noise_levels = torch.linspace(start**power, end**power, timesteps)**(1/power)
    betas = betas_from_sqrtOneMinusAlphasCumprod(noise_levels)
    return betas


### ESTIMATE OF DIFFUSION SAMPLE

def predict_start_from_noise(x_t, t, predictedNoise, diff_model):
    return (x_t - diff_model.sqrtOneMinusAlphasCumprod[t] * predictedNoise)/diff_model.sqrtAlphasCumprod[t]        

def compute_estimate(model, k, cond, gt):

    interm_estimates = []
    
    dNoisy = model.sqrtAlphasCumprod[k] * gt + model.sqrtOneMinusAlphasCumprod[k] * torch.randn_like(gt).to('cuda')

    for i in reversed(range(0, k+1, 1)):

        t = i * torch.ones(dNoisy.shape[0]).to('cuda').long()
        dNoiseCond = torch.concat((cond, dNoisy), dim=1)

        predictedNoiseCond = model.unet(dNoiseCond, t)

        # use model (noise predictor) to predict mean
        modelMean = model.sqrtRecipAlphas[t] * (dNoiseCond - model.betas[t] * predictedNoiseCond / model.sqrtOneMinusAlphasCumprod[t])
        dNoisy = modelMean[:, cond.shape[1]:]

        if i != 0:
        
            dNoisy = dNoisy + model.sqrtPosteriorVariance[t] * torch.randn_like(dNoisy)

        estimate = (dNoiseCond[:, cond.shape[1]:]  - model.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:])/model.sqrtAlphasCumprod[t]

        interm_estimates.append(estimate)

    return dNoisy


def evaluate_dw_train_inf_gap0(models, val_loader, device, n_batches, input_types=['ancestor', 'clean', 'own-pred', 'prev-pred']):
    """
    Runs autoregressive rollout for multiple models over the FULL dataset.
    """
    # 1. Set models to eval mode
    for model in list(models.values()):
        model.eval()
    
    total_samples = 0

    print(f"Starting evaluation over full dataset ({len(val_loader)} batches)...")

    with torch.no_grad():

        mse_ancestor_all = {name: [] for name in models}
        mse_clean_all = {name: [] for name in models}
        mse_clean_own_pred_all = {name: [] for name in models}
        mse_clean_prev_pred_all = {name: [] for name in models}

        for batch_idx, sample in enumerate(val_loader):

            if batch_idx == n_batches:
                break
            
            # --- A. Prepare Batch ---
            data = sample["data"].to(device) # (B, T_total, C, H, W)
            batch_size = data.shape[0]
            total_samples += batch_size

            # Initial Condition (t=0)
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]

            for name in models:
                # Store prediction
                model = models[name]

                _, x0_estimates = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="ancestor")
                _, x0_estimates_clean = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="clean")
                _, x0_estimates_clean_own_pred = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="own-pred")
                _, x0_estimates_clean_prev_pred = model(conditioning=conditioning_frame, data=target_frame, return_x0_estimate=True, input_type="prev-pred")

                
                mse_ancestor = [(torch.mean((x0_estimates[t] - target_frame)**2)).item()
                        for t in range(len(x0_estimates))]
                mse_clean = [(torch.mean((x0_estimates_clean[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]
            
                mse_clean_own_pred = [(torch.mean((x0_estimates_clean_own_pred[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]
                
                mse_clean_prev_pred = [(torch.mean((x0_estimates_clean_prev_pred[t] - target_frame)**2)).item()
                         for t in range(len(x0_estimates))]
                
                
                mse_ancestor_all[name].append(mse_ancestor)
                mse_clean_all[name].append(mse_clean)
                mse_clean_own_pred_all[name].append(mse_clean_own_pred)
                mse_clean_prev_pred_all[name].append(mse_clean_prev_pred)


    # 3. Aggregate results
   
    mean_mse_ancestor = {}
    mean_mse_clean = {}
    mean_mse_clean_own_pred = {}
    mean_mse_clean_prev_pred = {}
    for name in models:
        # Concatenate all batches along dimension 0
        mean_mse_ancestor[name] = torch.mean(torch.tensor(mse_ancestor_all[name]), dim=0)
        mean_mse_clean[name] = torch.mean(torch.tensor(mse_clean_all[name]), dim=0)
        mean_mse_clean_own_pred[name] = torch.mean(torch.tensor(mse_clean_own_pred_all[name]), dim=0)
        mean_mse_clean_prev_pred[name] = torch.mean(torch.tensor(mse_clean_prev_pred_all[name]), dim=0)

    return {
        "mse_ancestor": mean_mse_ancestor,   # (N_total, T, C, H, W)
        "mse_clean": mean_mse_clean,   # (N_total, T, C, H, W)
        "mse_clean_own_pred": mean_mse_clean_own_pred,   # (N_total, T, C, H, W)
        "mse_clean_prev_pred": mean_mse_clean_prev_pred,   # (N_total, T, C, H, W)
    }



def evaluate_dw_train_inf_gap(models, val_loader, n_batches, metric, device, input_types=['ancestor', 'clean', 'own-pred', 'prev-pred']):
    """
    Runs autoregressive rollout for multiple models over the FULL dataset.
    Dynamically handles different input types to avoid code duplication.
    """
    # 1. Set models to eval mode
    for model in models.values():
        model.eval()
    
    # 2. Initialize storage dictionary dynamically based on input_types
    # Structure: raw_metrics[input_type][model_name] = [list of batch results]
    raw_metrics = {itype: {name: [] for name in models} for itype in input_types}

    print(f"Starting evaluation over full dataset ({len(val_loader)} batches) for types: {input_types}...")

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            if batch_idx == n_batches:
                break
            
            # --- Prepare Batch ---
            data = sample["data"].to(device)
            
            # Initial Condition (t=0) and Target
            conditioning_frame = data[:, 0]
            target_frame = data[:, 1]

            for name, model in models.items():
                # Loop through the requested input types to avoid duplication
                for itype in input_types:
                    # Run model with specific input type
                    _, x0_estimates = model(
                        conditioning=conditioning_frame, 
                        data=target_frame, 
                        return_x0_estimate=True, 
                        input_type=itype
                    )

                    # Calculate MSE for each time step in the diffusion process
                    # x0_estimates is typically a list of tensors over diffusion timesteps
                    if metric == 'mse':
                        criterion = F.mse_loss
                    elif metric == 'mae':
                        criterion = F.l1_loss
                    mse_list = [
                        criterion(estimate, target_frame).item()
                        for estimate in x0_estimates
                    ]
                    
                    raw_metrics[itype][name].append(mse_list)

    # 3. Aggregate results
    final_results = {}
    
    for itype in input_types:
        # Create a dictionary for this specific input type (e.g., mean_mse_ancestor)
        # We format the key to match the original style (e.g., 'ancestor' -> 'mse_ancestor')
        metric_key = f"mse_{itype.replace('-', '_')}"
        final_results[metric_key] = {}
        
        for name in models:
            # Concatenate all batches along dimension 0 and compute mean
            if raw_metrics[itype][name]:
                # Convert list of lists to tensor: (N_batches, T_steps) -> Mean over batches -> (T_steps)
                final_results[metric_key][name] = torch.mean(
                    torch.tensor(raw_metrics[itype][name]), dim=0
                )
            else:
                final_results[metric_key][name] = torch.tensor([])

    return final_results