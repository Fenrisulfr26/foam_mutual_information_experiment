import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.signal import resample


LATENT_DIM = 32
CONDITION_COMPRESS_DIM_PRIOR = 32
CONDITION_COMPRESS_DIM_DECODER = 32
MODEL_TIME_BINS = 121
RAW_TIME_BINS = 227
ROI_SLICE = (slice(10, 22), slice(10, 22))  # MATLAB [11:22, 11:22]

DEFAULT_DATA_DIR = (
    "F:/OneDrive/foam_imaging_project/experiment_setup/matlab_all_code/data/"
    "3x3_grid_scan_20260615_155728_15mm_deg_0_exp_2us_frames_100000_avg_10_A_DC"
)

# 15 mm spacing weight
DEFAULT_CHECKPOINT = (
    "F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/emnist_batch_results/"
    "train_results/checkpoints/cvae_recon_checkpoints_160626/cvae_epoch_50_2.5_1136.pth"
)

# # 30 mm spacing weight
# DEFAULT_CHECKPOINT = (
#     "F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/emnist_batch_results/"
#     "train_results/checkpoints/cvae_recon_checkpoints_170626/cvae_epoch_50_2.5_1240.pth"
# )

DEFAULT_OUTPUT_ROOT = (
    "F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/emnist_batch_results/"
    "exp_infer_results"
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ConditionCompressLayerPrior(nn.Module):
    def __init__(self, compress_dim=CONDITION_COMPRESS_DIM_PRIOR):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(15, 3, 3), stride=(1, 1, 1), padding=(7, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1), ceil_mode=True),
            nn.Conv3d(32, 64, kernel_size=(9, 1, 1), stride=(1, 1, 1), padding=(4, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            nn.Conv3d(64, 128, kernel_size=(5, 1, 1), stride=(1, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            nn.Conv3d(128, 256, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, compress_dim),
            nn.BatchNorm1d(compress_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class ConditionCompressLayerDecoder(nn.Module):
    def __init__(self, compress_dim=CONDITION_COMPRESS_DIM_DECODER):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            nn.Conv3d(32, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            nn.Conv3d(64, 128, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ELU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), ceil_mode=True),
            nn.Conv3d(128, 256, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, compress_dim),
            nn.BatchNorm1d(compress_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class PriorNet(nn.Module):
    def __init__(self, condition_dim=CONDITION_COMPRESS_DIM_PRIOR, latent_dim=LATENT_DIM, hidden_dim=256):
        super().__init__()
        self.feature_extractor = ConditionCompressLayerPrior()
        self.middle_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(condition_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, condition):
        features = self.feature_extractor(condition)
        x = self.middle_layer(features)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, condition_dim=CONDITION_COMPRESS_DIM_DECODER):
        super().__init__()
        self.CCL = ConditionCompressLayerDecoder()
        self.input_dim = latent_dim + condition_dim
        self.reshape_channels = 256
        self.reshape_size = 4
        self.flattened_dim = self.reshape_channels * self.reshape_size * self.reshape_size
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.input_dim, self.flattened_dim),
            nn.BatchNorm1d(self.flattened_dim),
            nn.ELU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, w, condition):
        c_feat = self.CCL(condition)
        x = self.fc(torch.cat([w, c_feat], dim=1))
        x = x.reshape(-1, self.reshape_channels, self.reshape_size, self.reshape_size)
        return self.decoder(x)


def sorted_point_files(data_dir):
    files = []
    for path in Path(data_dir).glob("hist_*point*.mat"):
        match = re.search(r"point(\d+)", path.name)
        if match:
            files.append((int(match.group(1)), path))
    files = [path for _, path in sorted(files)]
    if len(files) != 9:
        raise RuntimeError(f"Expected 9 point files, found {len(files)} in {data_dir}")
    return files


def load_experiment_grid(data_dir):
    grid = np.zeros((3, 3, RAW_TIME_BINS), dtype=np.float32)
    point_files = sorted_point_files(data_dir)
    for i, path in enumerate(point_files):
        mat = loadmat(path)
        if "hist" not in mat:
            raise KeyError(f"Missing 'hist' in {path}")
        hist = np.asarray(mat["hist"], dtype=np.float32)
        if hist.shape != (32, 32, RAW_TIME_BINS):
            raise ValueError(f"Expected hist shape (32, 32, {RAW_TIME_BINS}), got {hist.shape} in {path}")
        row, col = divmod(i, 3)
        grid[row, col, :] = hist[ROI_SLICE[0], ROI_SLICE[1], :].sum(axis=(0, 1))
        print(f"Loaded point {i + 1}: {path.name}, ROI sum={grid[row, col, :].sum():.3f}")
    return grid, point_files


def preprocess_condition(grid_3x3x227, normalize=True):
    grid_3x3x121 = resample(grid_3x3x227, MODEL_TIME_BINS, axis=-1).astype(np.float32)
    grid_3x3x121 *= RAW_TIME_BINS / MODEL_TIME_BINS
    np.maximum(grid_3x3x121, 0, out=grid_3x3x121)
    if normalize:
        # Match training preprocessing: each 3x3x121 sample is normalized by its own maximum.
        max_val = float(grid_3x3x121.max())
        if max_val > 0:
            grid_3x3x121 /= max_val
    condition = grid_3x3x121.transpose(2, 0, 1)[None, None, :, :, :]
    return grid_3x3x121, torch.from_numpy(condition.copy()).float().to(device)


def load_models(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    prior = PriorNet().to(device)
    decoder = Decoder().to(device)
    prior.load_state_dict(checkpoint["prior"])
    decoder.load_state_dict(checkpoint["decoder"])
    prior.eval()
    decoder.eval()
    return prior, decoder, checkpoint


def infer_reconstructions(condition, prior, decoder, n_samples=100, n_random=8):
    with torch.no_grad():
        mu, logvar = prior(condition)
        most_likely = decoder(mu, condition)
        samples = []
        for _ in range(n_samples):
            z = reparameterize(mu, logvar)
            samples.append(decoder(z, condition).cpu().numpy().squeeze())

    samples = np.stack(samples, axis=0).astype(np.float32)
    mean_img = samples.mean(axis=0)
    var_img = samples.var(axis=0)
    random_imgs = samples[: min(n_random, n_samples)]
    return {
        "most_likely": most_likely.cpu().numpy().squeeze().astype(np.float32),
        "mean": mean_img,
        "variance": var_img,
        "random_samples": random_imgs,
        "all_samples": samples,
    }


def save_overview(results, condition_grid, output_dir):
    random_imgs = results["random_samples"]
    n_random = len(random_imgs)
    n_cols = max(4, n_random)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 5.6))
    axes = np.asarray(axes)

    panels = [
        ("Most likely", results["most_likely"], "gray"),
        ("Mean", results["mean"], "gray"),
        ("Variance", results["variance"], "magma"),
        ("Condition sum", condition_grid.sum(axis=-1), "viridis"),
    ]
    for ax, (title, image, cmap) in zip(axes[0, :4], panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[0, 4:]:
        ax.axis("off")

    for idx, ax in enumerate(axes[1]):
        ax.axis("off")
        if idx < n_random:
            ax.imshow(random_imgs[idx], cmap="gray")
            ax.set_title(f"Sample {idx + 1}")

    fig.tight_layout()
    out_path = Path(output_dir) / "exp_cvae_reconstruction_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Infer CVAE reconstructions from 3x3 experimental TPSF data.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--random-show", type=int, default=8)
    parser.add_argument("--no-normalize", action="store_true", help="Debug only: disable sample-wise max normalization after downsampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"exp_grid_infer_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Experiment data: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {output_dir}")

    grid_3x3x227, point_files = load_experiment_grid(args.data_dir)
    grid_3x3x121, condition = preprocess_condition(grid_3x3x227, normalize=not args.no_normalize)
    print(f"Condition grid shape: {grid_3x3x121.shape}; model tensor shape: {tuple(condition.shape)}")

    prior, decoder, checkpoint = load_models(args.checkpoint)
    print(f"Loaded checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")

    results = infer_reconstructions(condition, prior, decoder, n_samples=args.samples, n_random=args.random_show)
    overview_path = save_overview(results, grid_3x3x121, output_dir)

    np.savez_compressed(
        output_dir / "exp_cvae_reconstruction_results.npz",
        condition_3x3x227=grid_3x3x227,
        condition_3x3x121=grid_3x3x121,
        most_likely=results["most_likely"],
        mean=results["mean"],
        variance=results["variance"],
        random_samples=results["random_samples"],
        all_samples=results["all_samples"],
        point_files=np.array([str(p) for p in point_files]),
        checkpoint=str(args.checkpoint),
    )
    print(f"Saved overview: {overview_path}")
    print(f"Saved arrays: {output_dir / 'exp_cvae_reconstruction_results.npz'}")


if __name__ == "__main__":
    main()
