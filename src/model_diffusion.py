### Mainly copied from https://github.com/tum-pbs/autoreg-pde-diffusion/blob/main/src/turbpred/model_diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion_utils import linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from src.diffusion_utils import cosine_beta_schedule, cubic_beta_schedule, initial_exploration_beta_schedule
from src.diffusion_utils import psd_beta_schedule, cosine_sigma_schedule
from src.diffusion_utils import betas_from_sqrtOneMinusAlphasCumprod
from src.model import Unet
from src.model_acdm_arch import UnetACDM
from src.model_1d import Unet1D

### DIFFUSION MODEL WITH CONDITIONING
class DiffusionModel(nn.Module):

    def __init__(self, dimension, dataSize, condChannels, dataChannels, diffSchedule, diffSteps, 
                 inferenceSamplingMode, inferenceConditioningIntegration, diffCondIntegration,
                 inferenceInitialSampling = "random", padding_mode='circular',
                architecture="ours", checkpoint="", load_betas=False):
        super(DiffusionModel, self).__init__()

        self.dimension = dimension

        self.timesteps = diffSteps
        # sampling settings
        self.inferencePosteriorSampling = "random"      # "random" or "same", ddpm only
        self.inferenceInitialSampling = inferenceInitialSampling       # "random" or "same"
        self.inferenceConditioningIntegration = inferenceConditioningIntegration # "noisy" or "clean"
        self.inferenceSamplingMode = inferenceSamplingMode # "ddpm" or "ddim"
        self.diffCondIntegration = diffCondIntegration # "noisy" or "clean"
        self.architecture = architecture
        
        if diffSchedule == "linear":
            betas = linear_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "cosine":
            betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "cubic":
            betas = cubic_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "psd":
            betas = psd_beta_schedule(timesteps=self.timesteps)
        elif diffSchedule == "transonicIteration7":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.cat((torch.logspace(-1.2, -1, 14), torch.logspace(-0.16, -0.00005, 6))))
        elif diffSchedule == "Log-2-Min":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.8, 85),
                            torch.logspace(-1.8, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
            self.timesteps=100
        elif diffSchedule == "Log-2-Min+":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.99, 85),
                            torch.logspace(-1.99, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
            self.timesteps=100
        elif diffSchedule == "Log-2-Min+Closeto1":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-2, -1.99, 85),
                            torch.logspace(-1.99, -0.1, 10),
                            torch.logspace(-0.1, -0.000001, 5))))
            self.timesteps=100
        elif diffSchedule == "Log-1.5-Min+":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1.5, -1.49, 85),
                            torch.logspace(-1.49, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
            self.timesteps=100
        elif diffSchedule == "Log-1-Min+":
            betas = betas_from_sqrtOneMinusAlphasCumprod(torch.concatenate((torch.logspace(-1, -0.99, 85),
                            torch.logspace(-0.99, -0.1, 10),
                            torch.logspace(-0.1, -0.0001, 5))))
            self.timesteps=100
        elif diffSchedule == "inverseCosLog-1.25":
            sigmas = cosine_sigma_schedule(10**-1.25, 10**-0.0001, 20).to(device)
            training_sigmas = torch.concatenate((torch.ones(80).to(device)*sigmas[0], sigmas))
            betas = betas_from_sqrtOneMinusAlphasCumprod(training_sigmas)
            self.timesteps=100
        elif diffSchedule == "inverseCosLog-1.15":
            sigmas = cosine_sigma_schedule(10**-1.15, 10**-0.0001, 20).to(device)
            training_sigmas = torch.concatenate((torch.ones(80).to(device)*sigmas[0], sigmas))
            betas = betas_from_sqrtOneMinusAlphasCumprod(training_sigmas)
            self.timesteps=100
        elif diffSchedule == "inverseCosLog-0.8":
            sigmas = cosine_sigma_schedule(10**-0.8, 10**-0.0001, 20).to(device)
            training_sigmas = torch.concatenate((torch.ones(80).to(device)*sigmas[0], sigmas))
            betas = betas_from_sqrtOneMinusAlphasCumprod(training_sigmas)
            self.timesteps=100
        elif diffSchedule == "initial_exploration":
            betas = initial_exploration_beta_schedule(min_log_value=-2.5, timesteps=self.timesteps)
        else:
            raise ValueError("Unknown variance schedule")
        
        self.condChannels = condChannels
        self.dataChannels = dataChannels
        if self.architecture == "ACDM":
            self.unet = UnetACDM(dim=128,
                channels= condChannels+dataChannels,
                sigmas=torch.zeros(1),
                dim_mults=(1,1,1),
                use_convnext=True,
                convnext_mult=1)
        
        elif self.architecture == "Unet2D":
            self.unet = Unet(
            dim=dataSize[0],
            sigmas=torch.zeros(1),
            channels=condChannels+dataChannels,
            dim_mults=(1,1,1),
            use_convnext=True,
            convnext_mult=1,
            padding_mode=padding_mode,
            )

        elif self.architecture == "Unet1D":
            self.unet = Unet1D(
            dim=dataSize[0],
            sigmas=torch.zeros(1),
            channels=condChannels+dataChannels,
            dim_mults=(1,1,1),
            convnext_mult=1,
            padding_mode=padding_mode
            )
        
        else : raise Exception
        """if checkpoint != "":
            print(f"Loading Checkpoint from {checkpoint}")
            ckpt = torch.load(checkpoint)
            if 'stateDictDecoder' in ckpt.keys():
                ckpt = ckpt['stateDictDecoder']
            checkpoint_unet = {key[5:]:ckpt[key] for key in ckpt if 'unet' in key and not 'sigmas' in key}
            self.unet.load_state_dict(checkpoint_unet)
            if load_betas and 'betas' in ckpt: 
                betas = ckpt['betas']"""
        
        if checkpoint != "":
            print(f"Loading Checkpoint from {checkpoint}")
            ckpt = torch.load(checkpoint)
            if 'stateDictDecoder' in ckpt.keys():
                ckpt = ckpt['stateDictDecoder']
            if 'model_state_dict' in ckpt.keys():
                ckpt = ckpt['model_state_dict']
            if not 'sigmas' in ckpt.keys():
                ckpt['sigmas'] = torch.tensor([1])
            #checkpoint_unet = {key[5:]:ckpt[key] for key in ckpt if 'unet' in key and not 'sigmas' in key}
            self.unet.load_state_dict(ckpt)
            if load_betas and 'betas' in ckpt: 
                betas = ckpt['betas']

        self.compute_schedule_variables(betas)

    def compute_schedule_variables(self, betas=None, sigmas=None):
        if betas is None and sigmas is None:
            raise Exception
        elif sigmas is not None:
            betas = betas_from_sqrtOneMinusAlphasCumprod(sigmas)
        
        # 1. Flatten to 1D for calculation
        betas = betas.ravel()
        alphas = 1.0 - betas
        alphasCumprod = torch.cumprod(alphas, axis=0)
        
        # Pad with 1.0 at the beginning for t=0 anterior value
        alphasCumprodPrev = F.pad(alphasCumprod[:-1], (1, 0), value=1.0)
        
        sqrtRecipAlphas = torch.sqrt(1.0 / alphas)
        sqrtAlphasCumprod = torch.sqrt(alphasCumprod)
        sqrtOneMinusAlphasCumprod = torch.sqrt(1. - alphasCumprod)
        
        posteriorVariance = betas * (1. - alphasCumprodPrev) / (1. - alphasCumprod)
        sqrtPosteriorVariance = torch.sqrt(posteriorVariance)

        # 2. Determine Broadcasting Shape
        # If 1D data (Batch, Channel, Length) -> params need (T, 1, 1)
        # If 2D data (Batch, Channel, Height, Width) -> params need (T, 1, 1, 1)
        if self.dimension == 1:
            reshape_shape = (-1, 1, 1)
        else:
            reshape_shape = (-1, 1, 1, 1)

        def prep(tensor):
            return tensor.view(*reshape_shape)

        # 3. Register buffers with correct broadcasting shape
        self.register_buffer("betas", prep(betas)) 
        self.register_buffer("sqrtRecipAlphas", prep(sqrtRecipAlphas))
        self.register_buffer("sqrtAlphasCumprod", prep(sqrtAlphasCumprod))
        self.register_buffer("sqrtOneMinusAlphasCumprod", prep(sqrtOneMinusAlphasCumprod))
        self.register_buffer("sqrtPosteriorVariance", prep(sqrtPosteriorVariance))

        sigmas = (sqrtAlphasCumprod / sqrtOneMinusAlphasCumprod)
        self.unet.sigmas = sigmas
        self.timesteps = len(betas)

    def forward(self, conditioning:torch.Tensor, data:torch.Tensor = None, return_x0_estimate:bool = False, 
                input_type:str = "ancestor", lower_timestep_limit:int = 0) -> torch.Tensor:

        device = conditioning.device
        
        # Initialize data if None (Training target or Inference starting point)
        if data is None:
            if self.dimension == 1:
                N, C_cond, L = conditioning.shape
                shape_data = (N, self.dataChannels, L)
            else:
                N, C_cond, H, W = conditioning.shape
                shape_data = (N, self.dataChannels, H, W)
            data = torch.zeros(shape_data, device=device)
            
        d = data
        cond = conditioning

        # ==========================
        # TRAINING
        # ==========================
        if self.training:
            t = torch.randint(0, self.timesteps, (d.shape[0],), device=device).long()

            dNoise = torch.randn_like(d)
            dNoisy = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * dNoise
            noise = torch.cat((cond, dNoise), dim=1)
            dNoisy = torch.cat((cond, dNoisy), dim=1)

            # Predict noise
            predictedNoise = self.unet(dNoisy, t)[:, cond.shape[1]:]
            noise = noise[:, cond.shape[1]:]

            if return_x0_estimate:
                # Direct access via [t]
                return (dNoisy[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t] * predictedNoise) / self.sqrtAlphasCumprod[t]
                        
            return noise, predictedNoise

        # ==========================
        # INFERENCE
        # ==========================
        
        else:
            # Setup initial noise
            if self.inferenceInitialSampling == "random":
                dNoise = torch.randn_like(d)
                cNoise = torch.randn_like(cond) # Used if needed for specialized integration
            else:
                # Expand dims for scalar broadcasting if using fixed noise
                expand_dims = [-1] * (d.ndim - 1)
                dNoise = torch.randn((1,) + d.shape[1:], device=device).expand(d.shape[0], *expand_dims)
                cNoise = torch.randn((1,) + cond.shape[1:], device=device).expand(cond.shape[0], *expand_dims)

            sampleStride = 1
            if return_x0_estimate:
                all_x0_estimates = []

            for i in reversed(range(0, self.timesteps, sampleStride)):
                t = torch.full((cond.shape[0],), i, device=device, dtype=torch.long)
                condNoisy = cond

                # Optional: Pre-calculation/Manipulation of dNoise input based on input_type
                if input_type == "clean":
                    noise_temp = torch.randn_like(d)
                    dNoise = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * noise_temp

                elif input_type == "prev-pred":
                    # Look ahead to the "previous" step in diffusion terms (higher noise)
                    t_prev = torch.clamp(t + 1, max=self.timesteps - 1)
                    
                    noise_temp = torch.randn_like(d)                    
                    # Create noisy input at t_prev
                    dNoise_prev = self.sqrtAlphasCumprod[t_prev] * d + self.sqrtOneMinusAlphasCumprod[t_prev] * noise_temp
                    if i == self.timesteps - 2:
                        dNoise_prev = torch.randn_like(dNoise_prev)
                    
                    dNoiseCond = torch.cat((condNoisy, dNoise_prev), dim=1)
                    predictedNoiseCond = self.unet(dNoiseCond, t_prev)
                    
                    # Estimate x0 from t_prev
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t_prev] * predictedNoiseCond[:, cond.shape[1]:]) / self.sqrtAlphasCumprod[t_prev]
                    
                    # Re-noise to current t
                    dNoise = self.sqrtAlphasCumprod[t] * x0_estimate + self.sqrtOneMinusAlphasCumprod[t] * torch.randn_like(d)
                
                elif input_type == "own-pred":                
                    # 1. Forward diffusion to current t
                    dNoise = self.sqrtAlphasCumprod[t] * d + self.sqrtOneMinusAlphasCumprod[t] * torch.randn_like(d)
                
                    # 2. Predict x0 using current t
                    dNoiseCond = torch.cat((condNoisy, dNoise), dim=1)
                    predictedNoiseCond = self.unet(dNoiseCond, t)
                    x0_estimate = (dNoiseCond[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:]) / self.sqrtAlphasCumprod[t]
                    
                    # 3. Re-noise to current t (refresh noise)
                    dNoise = self.sqrtAlphasCumprod[t] * x0_estimate + self.sqrtOneMinusAlphasCumprod[t] * torch.randn_like(d)

                # --- Standard Reverse Step ---
                dNoiseCond = torch.cat((condNoisy, dNoise), dim=1)
                predictedNoiseCond = self.unet(dNoiseCond, t)

                x0_current = (dNoiseCond[:, cond.shape[1]:] - self.sqrtOneMinusAlphasCumprod[t] * predictedNoiseCond[:, cond.shape[1]:]) / self.sqrtAlphasCumprod[t]

                if i > 0:
                    dNoise = self.sqrtAlphasCumprod[t-1] * x0_current + self.sqrtOneMinusAlphasCumprod[t-1] * torch.randn_like(x0_current)
                else:
                    dNoise = x0_current

                if return_x0_estimate:
                    # Calculate x0 estimate at current step t
                    
                    all_x0_estimates.append(x0_current)

            if return_x0_estimate:
                return dNoise, torch.stack(all_x0_estimates)
            return dNoise