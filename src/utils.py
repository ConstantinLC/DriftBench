import torch
import numpy as np
from scipy.stats import binned_statistic
from scipy.stats import pearsonr
import os
from src.model_diffusion import DiffusionModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import torch

def compute_pearson_correlation(x, y, epsilon=1e-8):
    """
    Computes the Pearson correlation coefficient between two tensors x and y 
    along all dimensions except the batch dimension (dim 0).
    
    Args:
        x (torch.Tensor): Input tensor 1. Shape (B, ...).
        y (torch.Tensor): Input tensor 2. Must have the same shape as x.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: A 1D tensor of shape (B,) containing the correlation 
                      coefficient for each sample in the batch.
    """
    assert x.shape == y.shape, "Inputs must have the same shape"
    
    # 1. Flatten all dimensions except the batch dimension
    # Shape changes from (B, C, H, W) or (B, L) -> (B, N_features)
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)

    # 2. Compute means along the feature dimension
    x_mean = torch.mean(x_flat, dim=1, keepdim=True)
    y_mean = torch.mean(y_flat, dim=1, keepdim=True)

    # 3. Center the data (subtract mean)
    x_centered = x_flat - x_mean
    y_centered = y_flat - y_mean

    # 4. Compute numerator: Covariance (unscaled)
    numerator = torch.sum(x_centered * y_centered, dim=1)

    # 5. Compute denominator: Product of standard deviations (unscaled)
    x_var = torch.sum(x_centered ** 2, dim=1)
    y_var = torch.sum(y_centered ** 2, dim=1)
    denominator = torch.sqrt(x_var * y_var)

    # 6. Compute correlation
    correlation = numerator / (denominator + epsilon)

    return correlation

def compute_image_correlation(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    if img1.shape != img2.shape:
        raise ValueError(f"Input image shapes must be identical. Got {img1.shape} and {img2.shape}")
    if img1.dim() != 4:
        raise ValueError(f"Input images must be 4-dimensional (N, C, H, W). Got {img1.dim()} dimensions.")

    if img1.device != img2.device:
        img2 = img2.to(img1.device)

    img1 = img1.float()
    img2 = img2.float()

    mean1 = img1.mean(dim=(-2, -1), keepdim=True)
    mean2 = img2.mean(dim=(-2, -1), keepdim=True)
    centered_img1 = img1 - mean1
    centered_img2 = img2 - mean2

    numerator = (centered_img1 * centered_img2).sum(dim=(-2, -1), keepdim=True)
    sum_sq_centered1 = (centered_img1**2).sum(dim=(-2, -1), keepdim=True)
    sum_sq_centered2 = (centered_img2**2).sum(dim=(-2, -1), keepdim=True)
    denominator = torch.sqrt(sum_sq_centered1 * sum_sq_centered2)

    correlation = torch.where(
        denominator == 0,
        torch.tensor(float('nan'), device=img1.device, dtype=img1.dtype),
        numerator / denominator
    )
    return torch.mean(correlation.squeeze(-1).squeeze(-1))

import torch
import numpy as np
from scipy.stats import binned_statistic

def compute_radial_psd_error(
    predictions: torch.Tensor,
    ground_truths: torch.Tensor,
    num_bands: int = 12,
    statistic: str = 'mean',
    device: str = 'cpu'
) -> dict:
    """
    Computes the error between predictions and ground truths in radially-averaged
    frequency bands.

    For each image, it calculates the mean power in each radial band. It then
    aggregates these results across the entire batch using the specified statistic
    (either 'mean' or 'median').
    """
    if predictions.shape != ground_truths.shape:
        raise ValueError("Predictions and ground truths must have the same shape.")

    if statistic not in ['mean', 'median']:
        raise ValueError("The 'statistic' argument for final aggregation must be either 'mean' or 'median'.")

    predictions = predictions.to(device)
    ground_truths = ground_truths.to(device)

    predictions_np = predictions.cpu().numpy()
    ground_truths_np = ground_truths.cpu().numpy()

    N, C, H, W = predictions_np.shape
    
    # Pre-calculate coordinates and radii for efficiency
    y, x = np.indices((H, W))
    center_y, center_x = H // 2, W // 2
    radii = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_radius = np.max(radii)
    bin_edges = np.linspace(0, max_radius, num_bands + 1)
    
    # Store the radially-averaged PSD for each image individually
    all_radial_psds = []
    
    # Process each image in the batch
    for n in range(N):
        for c in range(C):
            error_img = predictions_np[n, c, :, :] - ground_truths_np[n, c, :, :]
            fft_error = np.fft.fftshift(np.fft.fft2(error_img))
            psd = np.abs(fft_error)**2
            
            # For each image, ALWAYS calculate the mean PSD in the radial bins
            radial_psd_mean, _, _ = binned_statistic(
                radii.ravel(),
                psd.ravel(),
                statistic='mean',
                bins=bin_edges
            )
            all_radial_psds.append(np.nan_to_num(radial_psd_mean))

    # Convert the list of 1D arrays into a 2D array
    # The shape will be (number_of_images, num_bands)
    psd_matrix = np.array(all_radial_psds)

    # Now, aggregate across the batch of images using the chosen statistic
    if statistic == 'mean':
        final_psd_by_band = np.mean(psd_matrix, axis=0)
    elif statistic == 'median':
        final_psd_by_band = np.median(psd_matrix, axis=0)
    
    # Format the output dictionary
    band_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(num_bands)]
    result = {label: value for label, value in zip(band_labels, final_psd_by_band)}

    return result

def correlation(qa, qb):
    return pearsonr(qa.ravel(), qb.ravel())[0]

def vorticity(x: torch.Tensor) -> torch.Tensor:
    """Computes the vorticity of a 2D vector field."""
    *batch, _, h, w = x.shape
    y = x.reshape(-1, 2, h, w)
    # Pad for circular boundary conditions
    y = torch.nn.functional.pad(y, pad=(1, 1, 1, 1), mode='circular')

    du, = torch.gradient(y[:, 0], dim=-1)
    dv, = torch.gradient(y[:, 1], dim=-2)

    # Crop back to original size
    vort = (dv - du)[:, 1:-1, 1:-1]
    # Reshape to original batch dimensions
    vort = vort.reshape(*batch, h, w)
    return vort


def evaluate_trajectory(model, loader, device, metrics=['mse', 'corr', 'vort_corr'], corr_threshold=0.8):
    """
    Performs autoregressive rollout and computes specified metrics efficiently in a single pass.
    
    Args:
        model: The neural network model.
        loader: DataLoader returning (N, T, C, H, W).
        device: 'cuda' or 'cpu'.
        metrics: List of metrics to compute ['mse', 'corr', 'vort_corr'].
        corr_threshold: Threshold for 'time_under_threshold' (specific to 'vort_corr').
    
    Returns:
        Dictionary containing time-series arrays and scalar summaries for each metric.
    """
    model.eval()
    
    # storage for all batches [metric_name -> list of lists]
    history = {m: [] for m in metrics}

    with torch.no_grad():
        for sample in loader:
            data = sample["data"].to(device)
            # Shapes: N (Batch), T (Time), C, H, W
            T = data.shape[1]
            
            current_pred = data[:, 0] # Initial frame (t=0)
            
            # Temporary buffers for this batch
            batch_results = {m: [] for m in metrics}

            for t in range(1, T):
                # 1. Autoregressive Step
                predicted_frame = run_model(model, current_pred)
                ground_truth_frame = data[:, t]
                
                # 2. Compute Metrics
                # --- MSE ---
                if 'mse' in metrics:
                    mse_val = torch.mean((predicted_frame - ground_truth_frame)**2, dim=list(range(1, len(predicted_frame.shape))))
                    batch_results['mse'].append(mse_val.cpu().numpy())

                # --- Pixel Correlation ---
                if 'corr' in metrics:
                    corr_val = compute_pearson_correlation(predicted_frame, ground_truth_frame)
                    batch_results['corr'].append(corr_val.cpu().numpy())

                # --- Vorticity Correlation ---
                if 'vort_corr' in metrics:
                    # Calculate vorticity
                    pred_vort = vorticity(predicted_frame)
                    gt_vort = vorticity(ground_truth_frame)
                    
                    # Compute correlation (unsqueeze to add channel dim for helper function)
                    v_corr_val = compute_pearson_correlation(pred_vort, gt_vort)
                    batch_results['vort_corr'].append(v_corr_val.cpu().numpy())
                
                # 3. Update State
                current_pred = predicted_frame
            
            # Append batch results to history
            for m in metrics:
                # Stack to get shape (Time, Batch) -> Transpose to (Batch, Time) later if needed
                # Here we append list of shape (Time, Batch)
                history[m].append(np.array(batch_results[m]))

    # --- Aggregation ---
    output = {}
    
    for m in metrics:
        # Concatenate all batches along the batch dimension
        # history[m] is a list of arrays, each array is (Time, Batch_Size)
        # We want to average over all batches.
        
        # 1. Combine all batches: shape (Total_Batches, Time_Steps)
        # Transpose internal arrays to be (Batch, Time) before stacking
        full_data = np.concatenate([np.transpose(batch_arr) for batch_arr in history[m]], axis=0)
        print("full_data", full_data.shape)
        # 2. Compute mean over batch dimension -> shape (Time_Steps,)
        metric_curve = np.mean(full_data, axis=0)
        output[f'{m}_per_ts'] = metric_curve
        
        # 3. Compute scalar summary (mean over time)
        output[f'mean_{m}'] = np.mean(metric_curve)

        # 4. Special metric: Time until correlation drops (only for vort_corr)
        if m == 'vort_corr':
            drop_indices = np.where(metric_curve < corr_threshold)[0]
            if len(drop_indices) > 0:
                time_under = drop_indices[0] + 1 # +1 because loop starts at t=1
            else:
                time_under = len(metric_curve) + 1
            output['vort_corr_time_under_threshold'] = time_under

        # 4. Special metric: Time until correlation drops (only for vort_corr)
        if m == 'corr':
            drop_indices = np.where(metric_curve < corr_threshold)[0]
            if len(drop_indices) > 0:
                time_under = drop_indices[0] + 1 # +1 because loop starts at t=1
            else:
                time_under = len(metric_curve) + 1
            output['corr_time_under_threshold'] = time_under

    return output

def evaluate_trajectory_vorticity(model, loader, device, threshold=0.8):
    """
    Performs autoregressive rollout and computes vorticity correlation.
    """
    model.eval()
    all_correlations = []

    with torch.no_grad():
        for sample in loader:
            data = sample["data"].to(device)
            N, T, C, H, W = data.shape
            
            current_pred = data[:, 0] # Initial frame
            correlations_for_batch = []

            for t in range(1, T):
                # Predict the next frame
                predicted_frame = model(current_pred, None)
                ground_truth_frame = data[:, t]

                print(torch.mean((predicted_frame - ground_truth_frame)**2))
                
                # Calculate vorticity for both prediction and ground truth
                pred_vort = vorticity(predicted_frame)
                gt_vort = vorticity(ground_truth_frame)
                
                # Compute correlation of the vorticities
                corr = compute_image_correlation(pred_vort.unsqueeze(1), gt_vort.unsqueeze(1))
                correlations_for_batch.append(corr.item())
                
                # The new prediction becomes the input for the next step
                current_pred = predicted_frame
            
            all_correlations.append(correlations_for_batch)

    # Average the correlations across all batches
    mean_correlations = np.mean(np.array(all_correlations), axis=0)
    
    # Find the time when correlation drops below the threshold
    drop_off_indices = np.where(mean_correlations < threshold)[0]
    time_under_threshold = drop_off_indices[0] + 1 if len(drop_off_indices) > 0 else T

    return {
        'mean_correlations': mean_correlations,
        'time_under_threshold': time_under_threshold
    }


def evaluate_trajectory_mse(model, loader, device, threshold=0.8):
    """
    Performs autoregressive rollout and computes vorticity correlation.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for sample in loader:
            data = sample["data"].to(device)
            T = data.shape[1]

            current_pred = data[:, 0] # Initial frame
            errors_for_batch = []

            for t in range(1, T):
                # Predict the next frame
                predicted_frame = model(current_pred, None)
                ground_truth_frame = data[:, t]
                
                # Compute correlation of the vorticities
                error = torch.mean((predicted_frame - ground_truth_frame)**2)
                errors_for_batch.append(error.item())
                
                # The new prediction becomes the input for the next step
                current_pred = predicted_frame
            
            all_errors.append(errors_for_batch)

    # Average the correlations across all batches
    errors_per_ts = np.mean(np.array(all_errors), axis=0)
    mean_error = np.mean(errors_per_ts)

    return {
        'errors_per_ts': errors_per_ts,
        'mean_error': mean_error
    }


def make_freq_radius(H, W, device):
    fy = torch.fft.fftfreq(H, device=device)
    fx = torch.fft.fftfreq(W, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    return torch.sqrt(FX**2 + FY**2)  # (H, W)


def shuffle_batch_dim(x):
    idx = torch.randperm(x.size(0))
    x_shuffled = x[idx]
    return x_shuffled

def get_next_run_number(base_dir):
    """
    Finds the next available run number by checking directories named 'run_*'.
    """
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('run_')]
    if not existing_runs:
        return 1
    else:
        run_numbers = [int(d.split('_')[1]) for d in existing_runs]
        return max(run_numbers) + 1
    
def parse_checkpoint_args(args_list):
    """
    Parses a list of strings ["name=path", "name2=path2"] into a dictionary.
    """
    ckpt_dict = {}
    for item in args_list:
        if '=' in item:
            name, path = item.split('=', 1)
        else:
            # Fallback for old behavior (auto-name)
            path = item
            name = os.path.basename(os.path.dirname(path)) + "_" + os.path.basename(path)
        
        ckpt_dict[name.strip()] = path.strip()
    return ckpt_dict

def run_model(model, x):
    if isinstance(model, DiffusionModel):
        return model(x)
    else:
        return model(x, time=None)
    

def _fft_magnitude(x, take_log=True):
    """Compute magnitude of 2D FFT for tensor (N,C,H,W)."""
    fft = torch.fft.fft2(x)
    mag = torch.abs(fft)
    if take_log:
        mag = torch.log1p(mag)
    return mag

def _cov_torch(x, eps=1e-6):
    """Covariance along batch dimension."""
    x = x - x.mean(dim=0, keepdim=True)
    N = x.shape[0]
    cov = (x.T @ x) / (N - 1)
    cov += torch.eye(cov.shape[0], device=x.device, dtype=x.dtype) * eps
    return cov

def _sqrtm_symmetric_torch(mat):
    """Matrix square root via eigen-decomposition (symmetric PSD case)."""
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=0)
    sqrt_eigvals = torch.sqrt(eigvals)
    return (eigvecs * sqrt_eigvals.unsqueeze(0)) @ eigvecs.T

def frechet_distance_torch(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute squared Fréchet distance between two Gaussians."""
    diff = mu1 - mu2
    diff_sq = diff.dot(diff)
    cov_prod = sigma1 @ sigma2
    covmean = _sqrtm_symmetric_torch(cov_prod)
    trace_term = torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)
    trace_term = torch.clamp(trace_term, min=0)
    return diff_sq + trace_term

def _radial_average_vectorized(psd2d):
    """Vectorized calculation of radial profiles."""
    N, C, H, W = psd2d.shape
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H, device=psd2d.device),
                          torch.arange(W, device=psd2d.device),
                          indexing='ij')
    r = torch.sqrt((x - cx)**2 + (y - cy)**2).long()
    nbins = int(r.max()) + 1
    
    psd_flat = psd2d.view(N, C, -1)
    r_flat = r.flatten() 
    
    radial_profiles = torch.zeros((N, C, nbins), device=psd2d.device, dtype=psd2d.dtype)
    
    for i in range(nbins):
        mask = (r_flat == i)
        if mask.any():
            val = psd_flat[..., mask].mean(dim=-1)
            radial_profiles[..., i] = val

    return radial_profiles.view(N, -1)

def fsd_torch_radial(real, gen, take_log=True, eps=1e-6):
    """Computes FSD based on 1D Radial Power Spectra (Optimized)."""
    real_mag = _fft_magnitude(real, take_log)**2 
    gen_mag = _fft_magnitude(gen, take_log)**2
    
    Xr = _radial_average_vectorized(real_mag)
    Xg = _radial_average_vectorized(gen_mag)
    
    mu_r = Xr.mean(dim=0)
    mu_g = Xg.mean(dim=0)
    cov_r = _cov_torch(Xr, eps)
    cov_g = _cov_torch(Xg, eps)

    fsd_sq = frechet_distance_torch(mu_r, cov_r, mu_g, cov_g)
    return fsd_sq