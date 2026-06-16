import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Display CVAE validation/test-set reconstruction results.")
    parser.add_argument("--batch-dir", default=BATCH_DIR)
    parser.add_argument("--checkpoint", default=None, help="Defaults to latest .pth under train_results/checkpoints.")
    parser.add_argument("--n", type=int, default=8, help="Number of validation samples to display.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or load_latest_checkpoint(CHECKPOINT_ROOT)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Loading and preprocessing dataset...")
    x, y = load_preprocessed_dataset(args.batch_dir)
    dataset = MyDataset(x, y)
    train_len = int(len(dataset) * 0.95)
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, valset = random_split(dataset, [train_len, val_len], generator=generator)
    val_loader = DataLoader(valset, batch_size=max(args.n, args.batch_size), shuffle=False, num_workers=0)
    model, checkpoint = load_model(checkpoint_path)
    print(f"Loaded checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Validation samples: {len(valset)}")
    plot_reconstructions(model, val_loader, n=args.n)


if __name__ == "__main__":
    main()
