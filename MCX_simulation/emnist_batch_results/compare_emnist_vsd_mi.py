from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(r"F:\OneDrive\foam_imaging_project")
DEFAULT_DATASETS = [
    (
        "30 mm",
        PROJECT_ROOT
        / "experiment_setup"
        / "MCX_simulation"
        / "emnist_batch_results"
        / "20260616_211244_emnist_pmcx_3x3_multisource_batch_30mm",
    ),
    (
        "15 mm",
        PROJECT_ROOT
        / "experiment_setup"
        / "MCX_simulation"
        / "emnist_batch_results"
        / "20260615_220457_emnist_pmcx_3x3_multisource_batch_15m",
    ),
]

BASE_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two EMNIST pMCX datasets with the same two-stage "
            "variational distillation relative-information score used by "
            "run_variational_distillation_scan.py. Results are printed and "
            "shown as a plot; no files are written."
        )
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--img-ae-epochs", type=int, default=40)
    parser.add_argument("--meas-ae-epochs", type=int, default=50)
    parser.add_argument("--distill-epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--ae-batch-size", type=int, default=512)
    parser.add_argument("--distill-batch-size", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=1024)
    parser.add_argument("--ae-lr", type=float, default=2e-3)
    parser.add_argument("--distill-lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--aligned-seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--shuffled-seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--no-plot", action="store_true", help="Only print numeric results.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    np.random.default_rng(BASE_SEED).shuffle(idx)
    split = int(0.8 * n)
    return idx[:split], idx[split:]


def load_templates(dataset_dir: Path) -> np.ndarray:
    imgs = np.load(dataset_dir / "templates_50x50_uint8.npy").astype(np.float32)
    if imgs.ndim != 3 or imgs.shape[1:] != (50, 50):
        raise ValueError(f"Unexpected template shape in {dataset_dir}: {imgs.shape}")
    if imgs.max() > 1.0:
        imgs /= 255.0
    return imgs[:, None, :, :]


def normalize_measurement_layout(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected measurement rank: {arr.shape}")
    if arr.shape[1:3] == (3, 3):
        return arr.astype(np.float32, copy=False)
    if arr.shape[2:4] == (3, 3):
        return np.transpose(arr, (0, 2, 3, 1)).astype(np.float32, copy=False)
    raise ValueError(f"Unexpected measurement shape: {arr.shape}")


def preprocess_measurements(arr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    arr = arr.astype(np.float32, copy=True)
    sample_max = np.max(arr.reshape(arr.shape[0], -1), axis=1)
    sample_max = np.maximum(sample_max, 1e-12).astype(np.float32)
    arr = arr / sample_max[:, None, None, None]
    offset = 0.0
    arr_min = float(arr.min())
    if arr_min <= 0.0:
        offset = -arr_min + 1e-6
        arr = arr + offset
    arr = np.log1p(arr)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-6
    arr = (arr - mean) / std
    return arr.astype(np.float32), {
        "offset": offset,
        "sample_max_mean_before_norm": float(sample_max.mean()),
        "sample_max_std_before_norm": float(sample_max.std()),
        "sample_max_min_before_norm": float(sample_max.min()),
        "sample_max_max_before_norm": float(sample_max.max()),
    }


def load_measurements(dataset_dir: Path) -> tuple[np.ndarray, dict[str, float]]:
    meas = normalize_measurement_layout(np.load(dataset_dir / "raw_tpsf_3x3x228_float32.npy"))
    meas_norm, info = preprocess_measurements(meas)
    return meas_norm.reshape(len(meas_norm), -1), info


def load_spacing(dataset_dir: Path) -> float | None:
    settings_path = dataset_dir / "batch_settings.json"
    if not settings_path.exists():
        return None
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    return float(settings["user_config"]["scan_spacing"])


class ImageAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        y = self.decoder_fc(z).view(-1, 64, 7, 7)
        y = self.decoder(y)[:, :, :50, :50]
        return y, z


class MeasurementAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class VariationalDistiller(nn.Module):
    def __init__(self, latent_dim: int, beta: float):
        super().__init__()
        self.beta = beta
        self.teacher_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.student_mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.student_logvar = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def aligned_terms(self, img_lat: torch.Tensor, meas_lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.teacher_proj(img_lat).detach()
        mu = self.student_mu(meas_lat)
        logvar = self.student_logvar(meas_lat).clamp(min=-6.0, max=4.0)
        inv_var = torch.exp(-logvar)
        nll = 0.5 * (((target - mu) ** 2) * inv_var + logvar + math.log(2.0 * math.pi))
        nll = nll.sum(dim=1).mean()
        kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar).sum(dim=1).mean()
        elbo = -(nll + self.beta * kl)
        return elbo, nll, kl


def train_autoencoder(
    model: nn.Module,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    epochs: int,
    binary: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(TensorDataset(train_x), batch_size=args.ae_batch_size, shuffle=True, drop_last=False)
    best_val = 1e18
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(recon, xb) if binary else nn.functional.mse_loss(recon, xb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            recon_val, _ = model(val_x.to(device))
            val_loss = (
                nn.functional.binary_cross_entropy_with_logits(recon_val, val_x.to(device)).item()
                if binary
                else nn.functional.mse_loss(recon_val, val_x.to(device)).item()
            )
        print(f"ae epoch={epoch:03d}/{epochs} val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_loss": float(best_val)}


def encode_latents(model: nn.Module, x: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device)
            _, z = model(xb)
            outputs.append(z.cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def evaluate_distiller(model: VariationalDistiller, img_lat: torch.Tensor, meas_lat: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        elbo, nll, kl = model.aligned_terms(img_lat.to(device), meas_lat.to(device))
    return {
        "elbo_nats": float(elbo.detach().cpu()),
        "nll_nats": float(nll.detach().cpu()),
        "kl_nats": float(kl.detach().cpu()),
    }


def run_vsd(
    img_lat_np: np.ndarray,
    meas_lat_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    shuffle_meas: bool,
    args: argparse.Namespace,
) -> dict[str, float | int | bool]:
    set_seed(seed)
    img_lat = torch.from_numpy(img_lat_np.astype(np.float32))
    meas_lat = torch.from_numpy(meas_lat_np.astype(np.float32))
    if shuffle_meas:
        meas_lat = meas_lat[np.random.default_rng(seed + 1000).permutation(len(meas_lat))]

    train_loader = DataLoader(
        TensorDataset(img_lat[train_idx], meas_lat[train_idx]),
        batch_size=args.distill_batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = VariationalDistiller(args.latent_dim, args.beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.distill_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.distill_epochs, eta_min=args.distill_lr * 0.2
    )

    best_val = -1e18
    best_epoch = -1
    best_state = None
    for epoch in range(1, args.distill_epochs + 1):
        model.train()
        train_elbos = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            elbo, _, _ = model.aligned_terms(xb, yb)
            (-elbo).backward()
            optimizer.step()
            train_elbos.append(float(elbo.detach().cpu()))
        scheduler.step()
        if epoch % args.eval_every == 0 or epoch == args.distill_epochs:
            val_metrics = evaluate_distiller(model, img_lat[val_idx], meas_lat[val_idx])
            print(
                f"vsd seed={seed} shuffle={shuffle_meas} epoch={epoch:03d}/{args.distill_epochs} "
                f"train_elbo={float(np.mean(train_elbos)):.4f} val_elbo={val_metrics['elbo_nats']:.4f}"
            )
            if val_metrics["elbo_nats"] > best_val:
                best_val = val_metrics["elbo_nats"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    full_metrics = evaluate_distiller(model, img_lat, meas_lat)
    return {
        "seed": seed,
        "shuffle_measurements": shuffle_meas,
        "best_epoch": best_epoch,
        "best_val_elbo_nats": float(best_val),
        "full_dataset_elbo_nats": full_metrics["elbo_nats"],
        "full_dataset_nll_nats": full_metrics["nll_nats"],
        "full_dataset_kl_nats": full_metrics["kl_nats"],
    }


def summarize_runs(runs: list[dict[str, float | int | bool]]) -> dict[str, float | list[float]]:
    elbo_arr = np.array([r["full_dataset_elbo_nats"] for r in runs], dtype=np.float32)
    nll_arr = np.array([r["full_dataset_nll_nats"] for r in runs], dtype=np.float32)
    kl_arr = np.array([r["full_dataset_kl_nats"] for r in runs], dtype=np.float32)
    return {
        "mean_elbo_nats": float(elbo_arr.mean()),
        "std_elbo_nats": float(elbo_arr.std()),
        "mean_nll_nats": float(nll_arr.mean()),
        "mean_kl_nats": float(kl_arr.mean()),
        "full_values": elbo_arr.tolist(),
    }


def print_table(results: list[dict[str, object]]) -> None:
    print("\nRelative-information comparison")
    print(
        "label     spacing_mm  aligned_elbo      shuffled_elbo     "
        "gap_nats          gap_bits"
    )
    for result in results:
        rel = result["relative_information"]
        gap_nats = float(rel["gap_vs_shuffle_nats"])
        print(
            f"{str(result['label']):<9} "
            f"{float(result['scan_spacing_mm']):>10.3f} "
            f"{float(rel['aligned_mean_elbo_nats']):>16.6f} "
            f"{float(rel['shuffled_mean_elbo_nats']):>16.6f} "
            f"{gap_nats:>16.6f} "
            f"{gap_nats / math.log(2.0):>16.6f}"
        )
    print("\nRaw data:")
    print(json.dumps(results, indent=2))


def build_plot(results: list[dict[str, object]]) -> None:
    labels = [str(r["label"]) for r in results]
    aligned = np.array([r["relative_information"]["aligned_mean_elbo_nats"] for r in results], dtype=np.float32)
    shuffled = np.array([r["relative_information"]["shuffled_mean_elbo_nats"] for r in results], dtype=np.float32)
    aligned_std = np.array([r["relative_information"]["aligned_std_elbo_nats"] for r in results], dtype=np.float32)
    gap = np.array([r["relative_information"]["gap_vs_shuffle_nats"] for r in results], dtype=np.float32)

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    axes[0].bar(x - width / 2, aligned, width, yerr=aligned_std, capsize=5, label="Aligned", color="#238b45")
    axes[0].bar(x + width / 2, shuffled, width, label="Shuffled", color="#b2182b")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("ELBO-like score (nats)")
    axes[0].set_title("VSD Score")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(labels, gap, color="#2166ac")
    for i, value in enumerate(gap):
        axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Aligned - shuffled (nats)")
    axes[1].set_title("Relative Information Gap")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("EMNIST pMCX 3x3 Multisource Dataset Comparison", fontsize=13)
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print("device =", device)
    for _, dataset_dir in DEFAULT_DATASETS:
        if not dataset_dir.exists():
            raise FileNotFoundError(dataset_dir)

    images_by_label = {label: load_templates(dataset_dir) for label, dataset_dir in DEFAULT_DATASETS}
    sample_counts = {label: imgs.shape[0] for label, imgs in images_by_label.items()}
    if len(set(sample_counts.values())) != 1:
        raise ValueError(f"Dataset sample counts differ: {sample_counts}")

    first_label = DEFAULT_DATASETS[0][0]
    first_images = images_by_label[first_label]
    same_templates = all(np.array_equal(first_images, imgs) for imgs in images_by_label.values())
    if not same_templates:
        print("warning: template arrays are not identical; image AE is trained on concatenated templates.")

    n_samples = first_images.shape[0]
    train_idx, val_idx = split_indices(n_samples)
    if same_templates:
        img_train_np = first_images[train_idx]
        img_val_np = first_images[val_idx]
    else:
        img_train_np = np.concatenate([imgs[train_idx] for imgs in images_by_label.values()], axis=0)
        img_val_np = np.concatenate([imgs[val_idx] for imgs in images_by_label.values()], axis=0)
    img_train = torch.from_numpy(img_train_np.astype(np.float32))
    img_val = torch.from_numpy(img_val_np.astype(np.float32))

    set_seed(BASE_SEED)
    image_ae = ImageAutoencoder(args.latent_dim).to(device)
    print("\ntraining shared image autoencoder")
    train_autoencoder(image_ae, img_train, img_val, args.img_ae_epochs, binary=True, args=args)

    img_latents_by_label = {
        label: encode_latents(image_ae, torch.from_numpy(imgs.astype(np.float32)), args.encode_batch_size)
        for label, imgs in images_by_label.items()
    }

    results = []
    for label, dataset_dir in DEFAULT_DATASETS:
        print("=" * 80)
        print(f"dataset {label}: {dataset_dir}")
        meas_np, preprocess_info = load_measurements(dataset_dir)
        if meas_np.shape[0] != n_samples:
            raise ValueError(f"{label} measurement count differs from template count: {meas_np.shape[0]} vs {n_samples}")
        meas_train = torch.from_numpy(meas_np[train_idx].astype(np.float32))
        meas_val = torch.from_numpy(meas_np[val_idx].astype(np.float32))

        set_seed(BASE_SEED)
        meas_ae = MeasurementAutoencoder(meas_np.shape[1], args.latent_dim).to(device)
        print("training measurement autoencoder")
        train_autoencoder(meas_ae, meas_train, meas_val, args.meas_ae_epochs, binary=False, args=args)
        meas_latents = encode_latents(meas_ae, torch.from_numpy(meas_np.astype(np.float32)), args.encode_batch_size)
        img_latents = img_latents_by_label[label]

        aligned_runs = [run_vsd(img_latents, meas_latents, train_idx, val_idx, seed, False, args) for seed in args.aligned_seeds]
        shuffled_runs = [run_vsd(img_latents, meas_latents, train_idx, val_idx, seed, True, args) for seed in args.shuffled_seeds]
        aligned_summary = summarize_runs(aligned_runs)
        shuffled_summary = summarize_runs(shuffled_runs)

        results.append(
            {
                "label": label,
                "dataset_dir": str(dataset_dir),
                "scan_spacing_mm": load_spacing(dataset_dir),
                "preprocess_info": preprocess_info,
                "aligned_summary": aligned_summary,
                "shuffled_summary": shuffled_summary,
                "relative_information": {
                    "aligned_mean_elbo_nats": aligned_summary["mean_elbo_nats"],
                    "aligned_std_elbo_nats": aligned_summary["std_elbo_nats"],
                    "shuffled_mean_elbo_nats": shuffled_summary["mean_elbo_nats"],
                    "shuffled_std_elbo_nats": shuffled_summary["std_elbo_nats"],
                    "gap_vs_shuffle_nats": float(aligned_summary["mean_elbo_nats"])
                    - float(shuffled_summary["mean_elbo_nats"]),
                },
            }
        )

    print_table(results)
    if not args.no_plot:
        build_plot(results)


if __name__ == "__main__":
    main()
