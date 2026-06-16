
import math
import os
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import subprocess
from datetime import datetime
import swanlab
import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, resample
from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 32
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 50
BETA = 2.5 # weight for KL term
RANDOM_SEED = 42
CONDITION_COMPRESS_DIM_PRIOR = 32
CONDITION_COMPRESS_DIM_DECODER = 32
IMG_COMPRESS_DIM = 32
VERSION = "CVAE_large_noisy_data"
IRF_MAT_PATH = "F:/OneDrive/foam_imaging_project/experiment_setup/matlab_all_code/IRF/IRF_noLens_10avg_20260612_2210.mat"
IRF_MAT_KEY = "hist"
IRF_CENTER_INDEX = (15, 15)  # MATLAB pixel (16, 16), converted to zero-based Python indices.
IRF_TIME_BINS = 227
MODEL_TIME_BINS = 121
SAVE_TRAIN_FIGURES = False
TRAIN_FIGURE_INTERVAL = 10


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std) 
    return mu + eps * std

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)

def freq_filter(input_y_all, filter = True, cutoff = 0.1, order = 3):
    if not filter:
        return input_y_all
    b, a = butter(order, cutoff / 0.5, btype='low')
    input_y_all = filtfilt(b, a, input_y_all, axis=2)
    np.maximum(input_y_all, 0, out=input_y_all)
    return input_y_all

# load IRF matrix from .mat file, return as float32 numpy array
def load_irf_mat(mat_path, key=IRF_MAT_KEY):
    mat_data = loadmat(mat_path)
    if key not in mat_data:
        available_keys = [name for name in mat_data.keys() if not name.startswith("__")]
        raise KeyError(f"IRF key '{key}' not found in {mat_path}. Available keys: {available_keys}")
    return np.asarray(mat_data[key], dtype=np.float32)

def extract_center_irf(irf_cube, center_index=IRF_CENTER_INDEX):
    if irf_cube.ndim != 3:
        raise ValueError(f"Expected a 3D IRF cube, got shape {irf_cube.shape}")
    row, col = center_index
    return np.asarray(irf_cube[row, col, :], dtype=np.float32).squeeze()

def circular_convolve_time_axis(data, irf, axis=-1, normalize_irf=True):
    data = np.asarray(data, dtype=np.float32)
    irf = np.asarray(irf, dtype=np.float32).squeeze()
    time_bins = data.shape[axis]
    if irf.ndim != 1:
        raise ValueError(f"Expected 1D IRF, got shape {irf.shape}")
    if irf.shape[0] != time_bins:
        raise ValueError(f"IRF length {irf.shape[0]} does not match data time bins {time_bins}")

    if normalize_irf:
        irf_sum = float(np.sum(irf))
        if irf_sum > 0:
            irf = irf / irf_sum

    data_fft = np.fft.rfft(data, n=time_bins, axis=axis)
    irf_fft = np.fft.rfft(irf, n=time_bins)
    kernel_shape = [1] * data.ndim
    kernel_shape[axis % data.ndim] = irf_fft.shape[0]
    convolved = np.fft.irfft(data_fft * irf_fft.reshape(kernel_shape), n=time_bins, axis=axis)
    convolved = convolved.astype(np.float32)
    np.maximum(convolved, 0, out=convolved)
    return convolved

def resample_time_axis(data, output_bins=MODEL_TIME_BINS, axis=-1, preserve_sum=True):
    input_bins = data.shape[axis]
    resampled = resample(data, output_bins, axis=axis).astype(np.float32)
    np.maximum(resampled, 0, out=resampled)
    if preserve_sum and output_bins > 0:
        resampled *= input_bins / output_bins
    return resampled

def apply_irf_and_resample_tpsf(input_y_all, irf, output_bins=MODEL_TIME_BINS):
    input_y_all = np.asarray(input_y_all, dtype=np.float32)
    if input_y_all.ndim != 4:
        raise ValueError(f"Expected input_y_all shape [B, 3, 3, T], got {input_y_all.shape}")
    convolved = circular_convolve_time_axis(input_y_all, irf, axis=-1, normalize_irf=True)
    return resample_time_axis(convolved, output_bins=output_bins, axis=-1, preserve_sum=True)

def sample_wise_max_normalize_tpsf(input_y_all):
    input_y_all = np.asarray(input_y_all, dtype=np.float32)
    if input_y_all.ndim != 4:
        raise ValueError(f"Expected input_y_all shape [B, 3, 3, T], got {input_y_all.shape}")
    max_vals = input_y_all.max(axis=(1, 2, 3), keepdims=True)
    return np.divide(input_y_all, max_vals, out=np.zeros_like(input_y_all), where=max_vals > 0)

class RobustAugmentationLayer(nn.Module):
    def __init__(self, scale_range=(0.95, 1.20), photon_count=10000, target_traces="all", shift_range=(0, 0),
                 apply_filter=True, cutoff=0.1, order=3):
        """
        Args:
            scale_range (tuple): Multiplicative scale range [min, max] for system noise.
            photon_count (int): Photon count used for shot-noise normalization; <= 0 disables Poisson noise.
            target_traces (str, int, tuple, list): Traces where scale noise and time shift are applied.
            shift_range (tuple): Random time-shift range [min_shift, max_shift].
            apply_filter (bool): Whether to apply the temporal low-pass filter.
            cutoff (float): Low-pass filter cutoff frequency.
            order (int): Low-pass filter order.
        """
        self.min_scale = scale_range[0]
        self.max_scale = scale_range[1]
        self.photon_count = photon_count
        self.target_traces = target_traces
        
        # Parse the requested time-shift range.
        self.min_shift = int(shift_range[0])
        self.max_shift = int(shift_range[1])
        self.do_shift = (self.min_shift != 0 or self.max_shift != 0)
        
        # Store low-pass filter parameters.
        self.apply_filter = apply_filter
        self.cutoff = cutoff
        self.order = order

    def __call__(self, x):
        """
        x shape: [B, 1, 121, 3, 3] (B, C, D, H, W)
        Axis 2 is the time/depth dimension; axes 3 and 4 are spatial.
        """
        # Clone as float so augmentation does not modify the original tensor.
        x_aug = x.clone().float()
        device = x_aug.device
        B, C, D, H, W = x_aug.shape
        
        # Initialize per-sample scale and shift tensors.
        scales = torch.ones((B, 1, 1, H, W), dtype=torch.float32, device=device)
        shifts = torch.zeros((B, H, W), dtype=torch.long, device=device)
        
        def get_rand_scale(size):
            return torch.empty(size, device=device).uniform_(self.min_scale, self.max_scale)
            
        def get_rand_shift(size):
            return torch.randint(self.min_shift, self.max_shift + 1, size, device=device)

        # ================================================================
        # 0. Resolve target traces and sample random scale/shift parameters.
        # ================================================================
        if self.target_traces == "all":
            scales[...] = get_rand_scale((B, 1, 1, H, W))
            if self.do_shift:
                shifts[...] = get_rand_shift((B, H, W))
                
        elif self.target_traces == "center":
            scales[:, 0, 0, H // 2, W // 2] = get_rand_scale((B,))
            if self.do_shift:
                shifts[:, H // 2, W // 2] = get_rand_shift((B,))
                
        elif isinstance(self.target_traces, (int, np.integer)):
            h_idx, w_idx = self.target_traces // W, self.target_traces % W
            scales[:, 0, 0, h_idx, w_idx] = get_rand_scale((B,))
            if self.do_shift:
                shifts[:, h_idx, w_idx] = get_rand_shift((B,))
                
        elif isinstance(self.target_traces, (list, tuple)):
            if len(self.target_traces) == 2 and isinstance(self.target_traces[0], (int, np.integer)):
                coords = [self.target_traces] 
            else:
                coords = self.target_traces   
                
            for coord in coords:
                if isinstance(coord, (int, np.integer)):
                    h_idx, w_idx = coord // W, coord % W
                else:
                    h_idx, w_idx = coord
                    
                scales[:, 0, 0, h_idx, w_idx] = get_rand_scale((B,))
                if self.do_shift:
                    shifts[:, h_idx, w_idx] = get_rand_shift((B,))
        else:
            raise ValueError("Unsupported target_traces format.")

        # ================================================================
        # Step 1: Poisson photon shot noise.
        # ================================================================
        if self.photon_count > 0:
            x_aug = F.relu(x_aug)  # keep the signal non-negative
            
            # Normalize each sample independently using its maximum value.
            max_val = x_aug.amax(dim=(1, 2, 3, 4), keepdim=True)
            
            # Avoid division by zero when forming the photon scale factor.
            scale_factor = torch.where(max_val > 0, self.photon_count / max_val, torch.zeros_like(max_val))
            
            x_photons = torch.round(x_aug * scale_factor)
            x_noisy = torch.poisson(x_photons)
            
            new_max = x_noisy.amax(dim=(1, 2, 3, 4), keepdim=True)
            x_aug = torch.where(new_max > 0, x_noisy / new_max, x_noisy)

        # ================================================================
        # Step 2: frequency-domain low-pass filtering.
        # ================================================================
        if self.apply_filter:
                b_coeff, a_coeff = butter(self.order, self.cutoff / 0.5, btype='low')
                x_np = x_aug.cpu().numpy()
                x_np = filtfilt(b_coeff, a_coeff, x_np, axis=2)
                # copy() removes negative strides from scipy output before converting back to torch.
                x_aug = torch.from_numpy(x_np.copy()).to(device)
                x_aug = F.relu(x_aug)

        # ================================================================
        # Step 3: multiplicative system noise and optional time jitter.
        # ================================================================
        x_aug *= scales
        
        if self.do_shift:
            for b in range(B):
                for h in range(H):
                    for w in range(W):
                        s = shifts[b, h, w].item()
                        if s > 0:
                            # Positive shift pads zeros at the beginning of the time axis.
                            x_aug[b, 0, :, h, w] = F.pad(x_aug[b, 0, :-s, h, w], (s, 0), value=0)
                        elif s < 0:
                            # Negative shift pads zeros at the end of the time axis.
                            x_aug[b, 0, :, h, w] = F.pad(x_aug[b, 0, -s:, h, w], (0, -s), value=0)
                            
        return x_aug

import torch.nn as nn

class ConditionCompressLayerPrior(nn.Module):
    def __init__(self, compress_dim=CONDITION_COMPRESS_DIM_PRIOR):
        """
        Condition encoder for PriorNet with a large temporal receptive field.
        Stride-1 convolutions plus pooling improve robustness to temporal shifts.
        """
        super(ConditionCompressLayerPrior, self).__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1: extract low-level features and downsample.
            # Use stride-1 convolutions plus pooling for local shift robustness.
            nn.Conv3d(1, 32, kernel_size=(15, 3, 3), stride=(1, 1, 1), padding=(7, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1), ceil_mode=True),
            
            # Layer 2: extract intermediate features.
            nn.Conv3d(32, 64, kernel_size=(9, 1, 1), stride=(1, 1, 1), padding=(4, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            
            # Layer 3: extract deeper features.
            nn.Conv3d(64, 128, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            
            # Layer 4: final feature refinement or upsampling.
            nn.Conv3d(128, 256, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ELU(inplace=True),
            
            # Global average pooling removes the final position dependence.
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        self.flatten_dim = 256
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(self.flatten_dim, compress_dim),
            nn.BatchNorm1d(compress_dim),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        features = self.encoder(x)
        latent = self.head(features)
        return latent

import torch.nn as nn

class ConditionCompressLayerDecoder(nn.Module):
    def __init__(self, compress_dim=CONDITION_COMPRESS_DIM_DECODER):
        """
        Condition encoder for the Decoder branch.
        Uses stride-1 convolutions, pooling, and global average pooling.
        This reduces sensitivity to absolute timing along the temporal axis.
        """
        super(ConditionCompressLayerDecoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1: extract low-level features and downsample.
            # Use stride-1 convolutions plus pooling for local shift robustness.
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            
            # Layer 2: extract intermediate features.
            nn.Conv3d(32, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            
            # Layer 3: extract deeper features.
            nn.Conv3d(64, 128, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            
            # Layer 4: final feature refinement or upsampling.
            nn.Conv3d(128, 256, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ELU(inplace=True),
            
            # Global average pooling keeps the feature vector shift tolerant.
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Flattened feature dimension is fixed at 256 channels.
        self.flatten_dim = 256
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2), 
            nn.Linear(self.flatten_dim, compress_dim),
            nn.BatchNorm1d(compress_dim),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        features = self.encoder(x)
        latent = self.head(features)
        return latent

class ImgCompressLayer(nn.Module):
    def __init__(self, compress_dim = IMG_COMPRESS_DIM):
        """
        Image encoder: compress [B, 1, 50, 50] images into a compact feature vector.
        """
        super(ImgCompressLayer, self).__init__()
        
        self.compress = nn.Sequential(
            # Layer 1: extract low-level features and downsample.
            # Input: [B, 1, 50, 50]
            # Conv: (50 + 2*1 - 3) / 2 + 1 = 25
            # Output: [B, 32, 25, 25]
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            
            # Layer 2: extract intermediate features.
            # Input: [B, 32, 25, 25]
            # Conv: (25 + 2*1 - 3) / 2 + 1 = 12 + 1 = 13
            # Output: [B, 64, 13, 13]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            
            # Layer 3: extract deeper features.
            # Input: [B, 64, 13, 13]
            # Conv: (13 + 2*1 - 3) / 2 + 1 = 6 + 1 = 7
            # Output: [B, 128, 7, 7]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )
        
        # Flattened feature dimension: 128 * 7 * 7 = 6272.
        self.flatten_dim = 128 * 7 * 7
        
        # Projection head.
        # Dropout helps prevent overfitting and encourages both branches to contribute.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.flatten_dim, compress_dim),
            nn.ELU(inplace=True)
        )
        
    def forward(self, x):
        # x: [B, 1, 50, 50]
        features = self.compress(x) # [B, 128, 7, 7]
        img_feature = self.fc(features) # [B, compress_dim]
        return img_feature   

def smooth_tensor_121(data, method='savgol', **kwargs):
    """
    Smooth along the 121-bin time dimension.
    Accepts either NumPy arrays or PyTorch tensors.
    """
    is_tensor = isinstance(data, torch.Tensor)
    device = None
    
    if is_tensor:
        device = data.device
        # Detach from autograd, move to CPU, and convert to NumPy.
        data_np = data.detach().cpu().numpy().astype(np.float32)
    else:
        # Ensure NumPy float32 input.
        data_np = np.asarray(data).astype(np.float32)
    
    # In [B, 1, 121, 3, 3], axis 2 is the 121-bin time axis.
    # axis=2 corresponds to the 121-bin time dimension.
    target_axis = 2
    
    if method == 'savgol':
        window = kwargs.get('window_length', 11)
        poly = kwargs.get('polyorder', 3)
        smoothed = savgol_filter(data_np, window_length=window, polyorder=poly, axis=target_axis)
        
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 2)
        smoothed = gaussian_filter1d(data_np, sigma=sigma, axis=target_axis)
        
    elif method == 'moving_avg':
        size = kwargs.get('size', 5)
        smoothed = uniform_filter1d(data_np, size=size, axis=target_axis)
        
    else:
        raise ValueError("Unknown method")
        
    if is_tensor:
        # Convert back to a tensor on the original device.
        return torch.from_numpy(smoothed).to(device)
    
    return smoothed

def count_parameters():
    from torchinfo import summary
    cond_layer = ConditionCompressLayerPrior().to(device)
    ICL = ImgCompressLayer().to(device)
    x_img = torch.randn(1, 1, 50, 50).to(device)
    x_cond = torch.randn(1, 1, 121, 3, 3).to(device)
    latent = torch.randn(1, LATENT_DIM).to(device)  

    print(summary(cond_layer, input_data=[x_cond]))
    print(summary(ICL, input_data=[x_img]))

    print(32*16*3*3)

    print(16*11*11)

count_parameters()


class PriorNet(nn.Module):
    def __init__(self, condition_dim = CONDITION_COMPRESS_DIM_PRIOR, latent_dim = LATENT_DIM, hidden_dim = 256):
        """
        PriorNet: predict latent distribution parameters mu and logvar from the condition input.
        
        Args:
            condition_dim (int): Output dimension of the condition feature branch.
            latent_dim (int): Latent variable dimension.
            hidden_dim (int): Number of hidden units in the fusion layer.
        """
        super(PriorNet, self).__init__()
        
        # 1. Condition feature backbone.
        # Reuse the temporal-spatial condition compressor.
        self.feature_extractor = ConditionCompressLayerPrior()

        # 2. MLP feature processor.
        # MLP increases nonlinearity before the distribution heads.
        # Dropout helps prevent overfitting and encourages both branches to contribute.
        self.middle_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(condition_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
        )

        # 3. Latent distribution heads.
        # Initialize BatchNorm layers.
        # Output heads do not use BatchNorm or activation.
        
        # Mu head predicts the latent mean.
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)

        # LogVar head predicts the latent log variance.
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
      
        # Initialize network weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            # Kaiming initialization works well with ELU activations.
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Initialize BatchNorm layers.
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # VAE-specific output initialization.
        # Keep mu and logvar heads close to zero at startup.
        # This starts the posterior near N(0, 1) and helps avoid KL spikes.
        nn.init.normal_(self.fc_mu.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc_mu.bias, 0)
        
        nn.init.normal_(self.fc_logvar.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc_logvar.bias, 0)

    def forward(self, condition):
        # 1. Extract condition features.
        features = self.feature_extractor(condition)
        
        # 2. Process features through the MLP.
        # Run the intermediate MLP before the output heads.
        x = self.middle_layer(features)
        
        # 3. Predict latent distribution parameters.
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Encoder(nn.Module):
    def __init__(self, condition_dim = CONDITION_COMPRESS_DIM_PRIOR, img_dim = IMG_COMPRESS_DIM, latent_dim = LATENT_DIM, hidden_dim = 512):
        """
        Encoder: fuse image features and photon-condition features to predict q(z|x, c).
        
        Args:
            condition_dim (int): Output dimension of the condition feature branch.
            img_dim (int): Output dimension of the image feature branch.
            latent_dim (int): Latent variable dimension.
            hidden_dim (int): Number of hidden units in the fusion layer.
        """
        super(Encoder, self).__init__()
        
        # 1. Feature extractor branches.
        # Feature extractor branches.
        self.CCL = ConditionCompressLayerPrior()
        self.ICL = ImgCompressLayer()

        # 2. Fusion layer for concatenated features.
        # Fuse concatenated image and condition features.
        combined_dim = condition_dim + img_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Dropout(0.1), # prevent overfitting and encourage both branches to contribute
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
        )

        # 3. Latent distribution heads.
        # Distribution heads are plain linear layers without activation or BatchNorm.
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 4. Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            # Standard layer initialization.
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # VAE output heads start near zero, close to a standard normal posterior.
        # This improves early training stability.
        nn.init.normal_(self.fc_mu.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc_mu.bias, 0)
        
        nn.init.normal_(self.fc_logvar.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc_logvar.bias, 0)

    def forward(self, x, condition):
        # x: Target Image [B, 1, 50, 50]
        # condition: Photon Hist [B, 1, 121, 3, 3]
    
        # 1. Extract condition and image features separately.
        feat_c = self.CCL(condition) # [B, condition_dim]
        feat_i = self.ICL(x)         # [B, img_dim]

        # 2. Concatenate features.
        combined = torch.cat([feat_c, feat_i], dim=1) # [B, condition_dim + img_dim]
        
        # 3. Fuse features.
        hidden = self.fusion_layer(combined) # [B, hidden_dim]

        # 4. Predict latent distribution parameters.
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim = LATENT_DIM, condition_dim = CONDITION_COMPRESS_DIM_DECODER):
        
        """
        Args:
            latent_dim (int): Latent variable dimension.
            condition_dim (int): Output dimension of the condition feature branch.
        """
        super(Decoder, self).__init__()
        
        # 1. Condition Processing
        self.CCL = ConditionCompressLayerDecoder()
        
        # 2. Projection parameters.
        self.input_dim = latent_dim + condition_dim
        self.reshape_channels = 256
        self.reshape_size = 4
        self.flattened_dim = self.reshape_channels * self.reshape_size * self.reshape_size # 256 * 4 * 4 = 4096
        
        # 3. Project latent-condition vector into a feature map.
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.input_dim, self.flattened_dim),
            nn.BatchNorm1d(self.flattened_dim),
            nn.ELU(inplace=True),
        )
        
        # 4. Upsampling decoder with transposed convolutions.
        self.decoder = nn.Sequential(
            # Layer 1: 4x4 -> 7x7
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            
            # Layer 2: 7x7 -> 13x13
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            
            # Layer 3: 13x13 -> 25x25
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            
            # Layer 4: final feature refinement or upsampling.
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Final layer uses Sigmoid only, without BatchNorm.
            nn.Sigmoid() 
        )
    
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            # Kaiming initialization works well with ELU activations.
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                # Check bias before initializing it.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, w, condition):
        # 1. Extract condition features.
        c_feat = self.CCL(condition)        
        
        # 2. Concatenate latent vector and condition features.
        combined = torch.cat([w, c_feat], dim=1)
        
        # 3. Project to flattened feature map.
        x = self.fc(combined)           
        
        # 4. Reshape into decoder feature-map layout.
        x = x.reshape(-1, self.reshape_channels, self.reshape_size, self.reshape_size)             
        
        # 5. Decode to [B, 1, 50, 50] image output.
        output = self.decoder(x)         
        
        return output

# -------------------- CVAE model --------------------
class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim).to(device)
        self.prior = PriorNet(latent_dim).to(device)
        self.decoder = Decoder(latent_dim).to(device)
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, max=10) # prevent overflow
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x, y):
        # q(z|x,y)
        q_mu, q_logvar = self.encoder(x, y)
        z = self.reparameterize(q_mu, q_logvar)
        # prior p(z|y)
        p_mu, p_logvar = self.prior(y)
        # decode
        x_rec = self.decoder(z, y)
        return x_rec, q_mu, q_logvar, p_mu, p_logvar

#---------------------view model summary---------------------
from torchinfo import summary
encoder = Encoder().to(device)
prior = PriorNet().to(device)
decoder = Decoder().to(device)

x_img = torch.randn(1, 1, 50, 50).to(device)
x_cond = torch.randn(1, 1, 121, 3, 3).to(device)
latent = torch.randn(1, LATENT_DIM).to(device)  

print(summary(encoder, input_data=[x_img, x_cond]))
print('\n')
print(summary(prior, input_data =[x_cond]))
print('\n')
print(summary(decoder, input_data=[latent, x_cond]))  


def get_beta(epoch, total_epochs, n_cycle = 2, ratio = 0.5, max_beta = BETA, stop_cycle_ratio=0.9):
    """
    stop_cycle_ratio: 
    """
    stop_epoch = total_epochs * stop_cycle_ratio
    
    if epoch >= stop_epoch:
        return max_beta
    
    period = stop_epoch / n_cycle
    step = (epoch % period) / period
    
    if step < ratio:
        return (step / ratio) * max_beta
    else:
        return max_beta

def reconstruction_loss(x_rec, x):
    # MSE loss per image summed over pixels
    return F.mse_loss(x_rec, x, reduction='none').view(x.size(0), -1).sum(dim=1)

# ================ KL divergence ===================
def kl_gaussian(q_mu, q_logvar, p_mu, p_logvar):
    """KL divergence between two diagonal Gaussians KL(q||p)
    q_logvar is log variance; returns per-batch sum.
    """
    # KL(q||p) = 0.5 * sum( log|Sigma_p|/|Sigma_q| - k + tr(Sigma_p^{-1} Sigma_q)
    #                       + (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q) )
    
    q_logvar = torch.clamp(q_logvar, min=-8, max=8)
    p_logvar = torch.clamp(p_logvar, min=-8, max=8)
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar)
    term1 = (p_logvar - q_logvar)
    term2 = (q_var + (q_mu - p_mu).pow(2)) / (p_var)
    kld = 0.5 * (term1 + term2 - 1).sum(dim=1)  # sum over latent dims, keep batch
    return kld

def train_epoch(model: CVAE, dataloader: DataLoader, optim, epoch):
    model.train()
    total_loss = 0.0
    rec_loss_sum = 0.0
    kl_loss_sum = 0.0
    current_beta = 0.0
    # noise_layer = RobustAugmentationLayer()
    
    for x, y in tqdm(dataloader, desc="Train batches"):
        x = x.to(device)
        y = y.to(device)

        optim.zero_grad()
        x_rec, q_mu, q_logvar, p_mu, p_logvar = model(x, y)
        rec = reconstruction_loss(x_rec, x)  # per-sample
        kl = kl_gaussian(q_mu, q_logvar, p_mu, p_logvar)  # per-sample

        current_beta = get_beta(epoch, NUM_EPOCHS) # circle beta

        # current_beta = beta     # fixed beta
        # loss = (0.5*rec + current_beta*kl + NOISE_CONTROL_BETA*kl_noisy_control_encoder + NOISE_CONTROL_BETA*kl_noisy_control_prior).mean() # modified to weight KL more
        loss = (0.5*rec + current_beta*kl).mean() # modified to weight KL more

        loss.backward() # calculate gradients
        
        optim.step()    # update weights
        total_loss += loss.item() * x.size(0)
        rec_loss_sum += rec.sum().item()
        kl_loss_sum += kl.sum().item()
  
    n = len(dataloader.dataset)
    
    return total_loss / n, rec_loss_sum / n, kl_loss_sum / n, current_beta

def validate_epoch(model: CVAE, dataloader: DataLoader, epoch):
    model.eval()
    total_loss = 0.0
    rec_loss_sum = 0.0
    kl_loss_sum = 0.0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Val batches"):
            x = x.to(device)
            y = y.to(device)
            x_rec, q_mu, q_logvar, p_mu, p_logvar = model(x, y)
            rec = reconstruction_loss(x_rec, x)
            kl = kl_gaussian(q_mu, q_logvar, p_mu, p_logvar)

            current_beta = get_beta(epoch, NUM_EPOCHS) # circle beta
            # current_beta = beta # fixed beta
            loss = (0.5*rec + current_beta*kl).mean() # modified to weight KL more

            total_loss += loss.item() * x.size(0)
            rec_loss_sum += rec.sum().item()
            kl_loss_sum += kl.sum().item()

    n = len(dataloader.dataset)
    return total_loss / n, rec_loss_sum / n, kl_loss_sum / n

def save_checkpoints(model: CVAE, epoch:int, beta:float, save_dir, current_time):
    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
        'prior': model.prior.state_dict(),
        'epoch': epoch
    }, os.path.join(save_dir, f'cvae_epoch_{epoch}_{beta}_{current_time}.pth'))

#=================the function to pick 6 images in val dataset to test=============
def visualize_reconstruction(model: CVAE, dataloader: DataLoader, n=6):
    model.eval()
    x, y = next(iter(dataloader))
    x = x[:n].to(device)
    y = y[:n].to(device)
    
    with torch.no_grad():
        x_rec, *_ = model(x, y)
    
    x_all = torch.cat([x.cpu(), x_rec.cpu()], dim=0)
    grid = vutils.make_grid(x_all, nrow=n, normalize=False, pad_value=1.0)
    
    # 1. Create the figure.
    fig = plt.figure(figsize=(12, 6))
    
    # 2. Draw the reconstruction grid.
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Top: ground truth, Bottom: reconstruction via encoder')
    
    return fig

def visualize_reconstruction_prior(prior: PriorNet, decoder: Decoder, dataloader: DataLoader, n=6):
    prior.eval()
    decoder.eval()
    x, y = next(iter(dataloader))
    x = x[:n].to(device)
    y = y[:n].to(device)
    
    with torch.no_grad():
        p_mu, p_logvar = prior(y)
        w = reparameterize(p_mu, p_logvar)
        x_rec = decoder(w, y)
    
    x_all = torch.cat([x.cpu(), x_rec.cpu()], dim=0)
    grid = vutils.make_grid(x_all, nrow=n, normalize=False, pad_value=1.0)
    
    # 1. Create the figure.
    fig = plt.figure(figsize=(12, 6))
    
    # 2. Draw the reconstruction grid.
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.title('Top: ground truth, Bottom: reconstruction via prior')
    
    # 3. Return the figure for display or logging.
    return fig

# -------------------- main training loop --------------------
class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data  # shape: (30000, 1, 50, 50)
        self.y_data = y_data  # shape: (30000, 1, 121, 3, 3)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x.float(), y.float()



def train(input_x,input_y, save_dir, current_time):

    # ============ generate noisy data ==================
    noise_layer = RobustAugmentationLayer(
                 scale_range=(0.95,1.13), photon_count=300, target_traces="all", shift_range=(0, 0),
                 apply_filter=True, cutoff=0.1, order=3)
    
    start_idx, end_idx = 0,7000
    aug_times = 5  # number of augmentations per sample
    sub_y = input_y[start_idx:end_idx]
    sub_x = input_x[start_idx:end_idx]
    augmented_y_list = [noise_layer(sub_y) for _ in range(aug_times)]
    augmented_x_list = [sub_x] * aug_times
    # input_y = torch.cat([input_y] + augmented_y_list, dim=0)
    # input_x = torch.cat([input_x] + augmented_x_list, dim=0)
    input_y = torch.cat(augmented_y_list, dim=0)
    input_x = torch.cat(augmented_x_list, dim=0)

    # input_y[:,:,0:40,:,:] = 0  # zero out the first 40 time bins to prevent model from overfitting to early peaks

    total_len = len(input_x)
    train_len = int(total_len * 0.95)
    val_len = total_len - train_len
    
    # random split train and val
    dataset = MyDataset(input_x, input_y)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    dataset, valset = random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CVAE(LATENT_DIM).to(device)
    # optim = torch.optim.Adam(model.parameters(), lr=lr)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    figure_dir = os.path.join(save_dir, "train_figures")
    if SAVE_TRAIN_FIGURES:
        os.makedirs(figure_dir, exist_ok=True)

    print("Training Log:")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        train_loss, train_rec, train_kl, current_beta = train_epoch(model, train_loader, optim, epoch)
        val_loss, val_rec, val_kl = validate_epoch(model, val_loader, epoch)

        # swanlab.log({"epoch": epoch, "train_loss": train_loss,"train_rec":train_rec,"train_kl": train_kl, "val_loss": val_rec,'beta':current_beta})

        print(f"Epoch {epoch}/{NUM_EPOCHS} --------------------------------")
        print(f"  Train loss: {train_loss:.6f}  Rec: {train_rec:.6f}  KL: {train_kl:.6f}")
        print(f"  Val loss:   {val_rec:.6f}")
        
        if SAVE_TRAIN_FIGURES and epoch % TRAIN_FIGURE_INTERVAL == 0:
            fig1 = visualize_reconstruction(model, val_loader, n=8)
            fig2 = visualize_reconstruction_prior(model.prior, model.decoder, val_loader, n=8)
            fig1.savefig(os.path.join(figure_dir, f"epoch_{epoch:04d}_encoder.png"), dpi=150, bbox_inches="tight")
            fig2.savefig(os.path.join(figure_dir, f"epoch_{epoch:04d}_prior.png"), dpi=150, bbox_inches="tight")
            plt.close(fig1)
            plt.close(fig2)
            print(f"  Saved reconstruction figures to: {figure_dir}")

        print("\n" + "="*50 + "\n")

        # if epoch % 10 == 0:
            # swanlab.log({"Reconstruction via encoder": [swanlab.Image(fig1, caption=f"Epoch {epoch}")]})
            # swanlab.log({"Reconstruction via prior": [swanlab.Image(fig2, caption=f"Epoch {epoch}")]})

    #========save the best model=========
    save_checkpoints(model, NUM_EPOCHS, BETA, save_dir, current_time)
    print("saved model")

    return model,train_loader,val_loader


# -------------------- If run as script --------------------
current_date = datetime.now().strftime("%d%m%y")
save_dir = f"F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/emnist_batch_results/train_results/checkpoints/cvae_recon_checkpoints_{current_date}"
current_time = datetime.now().strftime("%H%M") # to uniquely identify checkpoint files
os.makedirs(save_dir, exist_ok=True)
print(f"Checkpoints will be saved to: {save_dir}")

if __name__ == '__main__':

    input_x_all = []
    input_y_all = []

    input_x_all = np.load(f'F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/emnist_batch_results/20260615_220457_emnist_pmcx_3x3_multisource_batch/templates_50x50_uint8.npy') # [7000,50,50]
    input_y_all = np.load(f'F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/emnist_batch_results/20260615_220457_emnist_pmcx_3x3_multisource_batch/raw_tpsf_3x3x228_float32.npy') # [7000,3,3,228]
    input_y_all = input_y_all[:, :, :, :IRF_TIME_BINS] # trim to 227 bins before IRF convolution

    IRF = extract_center_irf(load_irf_mat(IRF_MAT_PATH)) # (227,)
    print(f"Loaded IRF from {IRF_MAT_PATH}, key='{IRF_MAT_KEY}', shape={IRF.shape}")
    input_y_all = apply_irf_and_resample_tpsf(input_y_all, IRF, output_bins=MODEL_TIME_BINS) # [7000,3,3,121]
    print(f"Applied circular IRF convolution and resampled TPSF to shape={input_y_all.shape}")
    input_y_all = sample_wise_max_normalize_tpsf(input_y_all)
    print("Applied sample-wise max normalization to each 3x3x121 TPSF.")

    if input_x_all.ndim == 3:
        input_x_all = input_x_all[:, None, :, :]

    if input_y_all.ndim == 4:
        input_y_all = input_y_all.transpose(0, 3, 1, 2)[:, None, :, :, :]

    input_x_all = torch.from_numpy(input_x_all.copy())
    input_y_all = torch.from_numpy(input_y_all.copy())

    # swanlab.init(
    #     project="foam_imaging_CVAE_recon",
    #     workspace="qiqiwang",
    #     name = f"convNet_{VERSION}_{current_date}{current_time}",
    #     config={
    #         "learning_rate": LR,
    #         "architecture": "CVAE with conv nets",
    #         "epochs": NUM_EPOCHS,
    #         "beta": BETA,
    #         "latent_dim": LATENT_DIM,
    #         "irf_mat_path": IRF_MAT_PATH,
    #         "irf_mat_key": IRF_MAT_KEY,
    #         "irf_center_index": IRF_CENTER_INDEX,
    #         "irf_time_bins": IRF_TIME_BINS,
    #         "model_time_bins": MODEL_TIME_BINS,
    #     }
    # ) # ini for viewing in swanlab

    # quick test run
    print('Device:', device)
    
    model, train_loader, val_loader = train(input_x_all, input_y_all, save_dir, current_time)
    
    # reconstruct from last saved best
    latest = max([os.path.join(save_dir, f) for f in os.listdir(save_dir)], key=os.path.getmtime)





