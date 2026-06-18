import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from scipy.io import loadmat
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader, random_split


LATENT_DIM = 32
BATCH_SIZE = 64
RANDOM_SEED = 42
CONDITION_COMPRESS_DIM_PRIOR = 32
CONDITION_COMPRESS_DIM_DECODER = 32
IMG_COMPRESS_DIM = 32
IRF_MAT_PATH = "F:/OneDrive/foam_imaging_project/experiment_setup/matlab_all_code/IRF/IRF_noLens_10avg_20260612_2210.mat"
IRF_MAT_KEY = "hist"
IRF_CENTER_INDEX = (15, 15)
IRF_TIME_BINS = 227
MODEL_TIME_BINS = 121
BATCH_DIR = "F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/emnist_batch_results/20260615_220457_emnist_pmcx_3x3_multisource_batch"
CHECKPOINT_ROOT = "F:/OneDrive/foam_imaging_project/experiment_setup/MCX_simulation/emnist_batch_results/train_results/checkpoints"
EXP_DATA_DIR = (
    "F:/OneDrive/foam_imaging_project/experiment_setup/matlab_all_code/data/"
    "3x3_grid_scan_20260615_153557_15mm_deg_0_exp_2us_frames_100000_avg_10_X_DC"
)
EXP_ROI_SLICE = (slice(10, 22), slice(10, 22))  # MATLAB [11:22, 11:22]

# -------------------- Run settings --------------------
# Edit these values directly before running this script.
RUN_BATCH_DIR = BATCH_DIR
RUN_EXP_DATA_DIR = EXP_DATA_DIR
RUN_CHECKPOINT = None  # None means use the newest .pth under CHECKPOINT_ROOT.
RUN_LETTER = "X"
RUN_LETTER_OCCURRENCE = 0  # 0 means the first X sample in batch_manifest.csv.
RUN_N_RECON = 8
RUN_BATCH_SIZE = BATCH_SIZE
RUN_SKIP_RECON = True  # True only shows the TPSF curve comparison.


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


class ImgCompressLayer(nn.Module):
    def __init__(self, compress_dim=IMG_COMPRESS_DIM):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 7 * 7, compress_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.fc(self.compress(x))


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
        x = self.middle_layer(self.feature_extractor(condition))
        return self.fc_mu(x), self.fc_logvar(x)


class Encoder(nn.Module):
    def __init__(self, condition_dim=CONDITION_COMPRESS_DIM_PRIOR, img_dim=IMG_COMPRESS_DIM, latent_dim=LATENT_DIM, hidden_dim=512):
        super().__init__()
        self.CCL = ConditionCompressLayerPrior()
        self.ICL = ImgCompressLayer()
        self.fusion_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(condition_dim + img_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, condition):
        feat_c = self.CCL(condition)
        feat_i = self.ICL(x)
        hidden = self.fusion_layer(torch.cat([feat_c, feat_i], dim=1))
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, condition_dim=CONDITION_COMPRESS_DIM_DECODER):
        super().__init__()
        self.CCL = ConditionCompressLayerDecoder()
        self.reshape_channels = 256
        self.reshape_size = 4
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(latent_dim + condition_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
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


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.prior = PriorNet()
        self.decoder = Decoder()

    def forward(self, x, y):
        q_mu, q_logvar = self.encoder(x, y)
        z = reparameterize(q_mu, q_logvar)
        p_mu, p_logvar = self.prior(y)
        return self.decoder(z, y), q_mu, q_logvar, p_mu, p_logvar


class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx].float(), self.y_data[idx].float()


def load_latest_checkpoint(root):
    checkpoints = sorted(Path(root).glob("**/*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    return str(checkpoints[0])


def load_irf():
    irf_cube = loadmat(IRF_MAT_PATH)[IRF_MAT_KEY].astype(np.float32)
    return irf_cube[IRF_CENTER_INDEX[0], IRF_CENTER_INDEX[1], :].squeeze()


def circular_convolve_time_axis(data, irf):
    data = np.asarray(data, dtype=np.float32)
    irf = np.asarray(irf, dtype=np.float32).squeeze()
    irf_sum = float(irf.sum())
    if irf_sum > 0:
        irf = irf / irf_sum
    data_fft = np.fft.rfft(data, n=data.shape[-1], axis=-1)
    irf_fft = np.fft.rfft(irf, n=data.shape[-1]).reshape(1, 1, 1, -1)
    convolved = np.fft.irfft(data_fft * irf_fft, n=data.shape[-1], axis=-1).astype(np.float32)
    return np.maximum(convolved, 0)


def sample_wise_max_normalize_tpsf(data):
    max_vals = data.max(axis=(1, 2, 3), keepdims=True)
    return np.divide(data, max_vals, out=np.zeros_like(data), where=max_vals > 0)


def load_preprocessed_dataset(batch_dir):
    batch_dir = Path(batch_dir)
    x = np.load(batch_dir / "templates_50x50_uint8.npy").astype(np.float32)
    y = np.load(batch_dir / "raw_tpsf_3x3x228_float32.npy").astype(np.float32)
    y = y[:, :, :, :IRF_TIME_BINS]
    y = circular_convolve_time_axis(y, load_irf())
    y = resample(y, MODEL_TIME_BINS, axis=-1).astype(np.float32)
    y *= IRF_TIME_BINS / MODEL_TIME_BINS
    y = np.maximum(y, 0)
    y = sample_wise_max_normalize_tpsf(y)
    y = y.transpose(0, 3, 1, 2)[:, None, :, :, :]
    x = x[:, None, :, :]
    return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def find_letter_sample_index(batch_dir, letter="X", occurrence=0):
    manifest_path = Path(batch_dir) / "batch_manifest.csv"
    matches = []
    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("letter") == letter and row.get("status") == "ok":
                matches.append(int(row["matrix_index"]))
    if not matches:
        raise ValueError(f"No samples with letter '{letter}' found in {manifest_path}")
    if occurrence < 0 or occurrence >= len(matches):
        raise IndexError(f"Requested occurrence {occurrence}, but only found {len(matches)} samples for letter '{letter}'")
    return matches[occurrence], len(matches)


def sorted_exp_point_files(exp_data_dir):
    files = []
    for path in Path(exp_data_dir).glob("hist_*point*.mat"):
        name = path.name
        marker = "point"
        if marker in name:
            point_text = name.split(marker, 1)[1][:2]
            if point_text.isdigit():
                files.append((int(point_text), path))
    files = [path for _, path in sorted(files)]
    if len(files) != 9:
        raise RuntimeError(f"Expected 9 experimental point files, found {len(files)} in {exp_data_dir}")
    return files


def load_experiment_condition(exp_data_dir):
    exp_grid = np.zeros((3, 3, IRF_TIME_BINS), dtype=np.float32)
    for idx, path in enumerate(sorted_exp_point_files(exp_data_dir)):
        mat = loadmat(path)
        if "hist" not in mat:
            raise KeyError(f"Missing 'hist' in {path}")
        hist = np.asarray(mat["hist"], dtype=np.float32)
        row, col = divmod(idx, 3)
        exp_grid[row, col, :] = hist[EXP_ROI_SLICE[0], EXP_ROI_SLICE[1], :].sum(axis=(0, 1))
    exp_grid = resample(exp_grid, MODEL_TIME_BINS, axis=-1).astype(np.float32)
    exp_grid *= IRF_TIME_BINS / MODEL_TIME_BINS
    exp_grid = np.maximum(exp_grid, 0)
    max_val = float(exp_grid.max())
    if max_val > 0:
        exp_grid /= max_val
    return exp_grid


def load_model(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CVAE().to(device)
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.prior.load_state_dict(checkpoint["prior"])
    model.eval()
    return model, checkpoint


def plot_reconstructions(model, dataloader, n=8):
    x, y = next(iter(dataloader))
    x = x[:n].to(device)
    y = y[:n].to(device)
    with torch.no_grad():
        q_mu, _ = model.encoder(x, y)
        p_mu, _ = model.prior(y)
        x_encoder = model.decoder(q_mu, y)
        x_prior = model.decoder(p_mu, y)

    rows = [
        ("Ground truth", x.cpu()),
        ("Encoder recon", x_encoder.cpu()),
        ("Prior recon", x_prior.cpu()),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(2 * n, 6))
    if len(rows) == 1:
        axes = [axes]
    for ax, (title, imgs) in zip(axes, rows):
        grid = vutils.make_grid(imgs, nrow=n, normalize=False, pad_value=1.0)
        ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def plot_tpsf_comparison(sim_y_tensor, exp_grid, sample_index, letter="X"):
    sim_grid = sim_y_tensor.squeeze(0).numpy().transpose(1, 2, 0)
    time_axis = np.arange(MODEL_TIME_BINS)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.plot(time_axis, sim_grid[row, col, :], label=f"sim {letter}", linewidth=1.8)
            ax.plot(time_axis, exp_grid[row, col, :], label="experiment", linewidth=1.8)
            ax.set_title(f"Point ({row + 1}, {col + 1})")
            ax.grid(True, alpha=0.3)
            if row == 2:
                ax.set_xlabel("Time bin")
            if col == 0:
                ax.set_ylabel("Normalized TPSF")
    axes[0, 0].legend()
    fig.suptitle(f"Processed 3x3x121 TPSF comparison: dataset {letter} sample index {sample_index} vs experiment", fontsize=13)
    fig.tight_layout()
    plt.show()


def main():
    checkpoint_path = RUN_CHECKPOINT or load_latest_checkpoint(CHECKPOINT_ROOT)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Loading and preprocessing dataset...")
    x, y = load_preprocessed_dataset(RUN_BATCH_DIR)
    sample_index, total_letter_matches = find_letter_sample_index(RUN_BATCH_DIR, RUN_LETTER, RUN_LETTER_OCCURRENCE)
    print(f"Using letter '{RUN_LETTER}' sample matrix_index={sample_index} ({RUN_LETTER_OCCURRENCE + 1}/{total_letter_matches}).")
    exp_grid = load_experiment_condition(RUN_EXP_DATA_DIR)
    plot_tpsf_comparison(y[sample_index], exp_grid, sample_index, letter=RUN_LETTER)

    if RUN_SKIP_RECON:
        return

    dataset = MyDataset(x, y)
    train_len = int(len(dataset) * 0.95)
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, valset = random_split(dataset, [train_len, val_len], generator=generator)
    val_loader = DataLoader(valset, batch_size=max(RUN_N_RECON, RUN_BATCH_SIZE), shuffle=False, num_workers=0)
    model, checkpoint = load_model(checkpoint_path)
    print(f"Loaded checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Validation samples: {len(valset)}")
    plot_reconstructions(model, val_loader, n=RUN_N_RECON)


if __name__ == "__main__":
    main()
