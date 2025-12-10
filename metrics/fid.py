"""
Fréchet Inception Distance (FID) – pełna, samodzielna implementacja.

Wersja z:
- limitem liczby obrazów (np. 2000 na zbiór),
- losowym próbkowaniem,
- logami czasu (REAL / GENERATED / FID / łącznie),
- prostymi statystykami cech,
- wykresem rozkładu norm cech (real vs generated).
"""

from __future__ import annotations


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import random
...

import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Sequence, Optional, List

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    from scipy.linalg import sqrtm  # type: ignore
except ImportError:
    sqrtm = None


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # pozwala wczytywać lekko ucięte obrazy

import matplotlib.pyplot as plt  # do wykresów


# ==========================
# 1. Dataset + DataLoader
# ==========================

class ImageFolderDataset(Dataset):
    """
    Prosty dataset: wczytuje wszystkie JPG/PNG z podanego folderu (rekurencyjnie).
    Obrazy są konwertowane do RGB i przekształcane za pomocą transformacji.
    Można opcjonalnie ograniczyć liczbę obrazów (max_images).
    """

    def __init__(
            self,
            root: str | Path,
            transform: Optional[nn.Module] = None,
            max_images: Optional[int] = None,
            seed: int = 42,
    ) -> None:
        self.root = Path(root)  # ⬅⬅⬅ TEGO BRAKOWAŁO
        self.transform = transform

        # zbierz wszystkie obrazy
        self.paths: List[Path] = sorted(
            list(self.root.rglob("*.png"))
            + list(self.root.rglob("*.jpg"))
            + list(self.root.rglob("*.jpeg"))
        )
        if not self.paths:
            raise ValueError(f"No images found in {self.root}")

        original_count = len(self.paths)

        # losowe ograniczenie liczby obrazów
        if max_images is not None and original_count > max_images:
            random.seed(seed)
            random.shuffle(self.paths)
            self.paths = self.paths[:max_images]
            self.paths.sort()
            print(
                f"[INFO] Using random subset of {len(self.paths)} images "
                f"out of {original_count} candidates in {self.root}"
            )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except OSError as e:
            print(f"[WARN] Problem z plikiem {path}: {e}, zastępuję czarnym obrazem 299x299")
            img = Image.new("RGB", (299, 299), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)
        return img


def make_dataloader(
    root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 299,
    max_images: Optional[int] = None,
    seed: int = 42,
) -> DataLoader:
    """
    Tworzy DataLoader z odpowiednimi transformacjami dla Inception v3,
    z opcjonalnym limitem liczby obrazów.
    """

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = ImageFolderDataset(root=root, transform=transform,
                                 max_images=max_images, seed=seed)
    print(f"[INFO] Final dataset size for {root}: {len(dataset)} images")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# ==========================
# 2. Ekstrakcja cech – Inception v3
# ==========================

class InceptionEmbedding(nn.Module):
    """
    Sieć wyciągająca cechy z Inception v3 (warstwa "pool3" – 2048-D).
    """

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()

        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        )
        inception.aux_logits = False

        inception.eval()

        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),  # "pool3"
        )

        self.device = torch.device(device)
        self.to(self.device)
        self.eval()

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca cechy o kształcie [N, 2048].
        """
        x = x.to(self.device, non_blocking=True)
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return feats


@torch.no_grad()
def get_activations(
    loader: DataLoader,
    model: Optional[InceptionEmbedding] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Wylicza wektory cech dla wszystkich obrazów z DataLoadera.

    Zwraca:
        numpy array o kształcie [N, 2048]
    """
    if model is None:
        model = InceptionEmbedding(device=device)
    model.eval()

    activations: List[torch.Tensor] = []

    for i, batch in enumerate(loader):
        feats = model(batch)
        activations.append(feats.cpu())
        if (i + 1) % 20 == 0:
            print(f"[INFO] Processed {(i + 1) * loader.batch_size} samples...")

    acts = torch.cat(activations, dim=0)
    return acts.numpy()


# ==========================
# 3. Statystyki i FID
# ==========================

def _compute_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Liczy średnią i macierz kowariancji po wymiarze próbek (N).
    """
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {features.shape}")

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _sqrtm_product(sigma_r: np.ndarray, sigma_g: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Liczy macierzowy pierwiastek z iloczynu sigma_r * sigma_g.
    """
    cov_prod = sigma_r @ sigma_g

    if sqrtm is not None:
        covmean = sqrtm(cov_prod)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return covmean

    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    eigvals = np.clip(eigvals, a_min=eps, a_max=None)
    covmean = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    return covmean


@dataclass
class FID:
    """
    Główna klasa metryki FID.
    """

    device: str = "cpu"
    batch_size: int = 32
    num_workers: int = 0
    image_size: int = 299
    eps: float = 1e-6

    def from_stats(
        self,
        mu_r: np.ndarray,
        sigma_r: np.ndarray,
        mu_g: np.ndarray,
        sigma_g: np.ndarray,
    ) -> float:
        diff = mu_r - mu_g
        diff_sq = diff.dot(diff)

        covmean = _sqrtm_product(sigma_r, sigma_g, eps=self.eps)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_r.shape[0]) * self.eps
            covmean = _sqrtm_product(sigma_r + offset, sigma_g + offset, eps=self.eps)

        fid_value = diff_sq + np.trace(sigma_r + sigma_g - 2.0 * covmean)

        # małe zabezpieczenie: jeśli z powodu numeryki wyjdzie minimalnie poniżej zera
        if fid_value < 0 and abs(fid_value) < 1e-3:
            print(f"[WARN] FID slightly negative due to numeric issues ({fid_value:.6f}), clamping to 0.")
            fid_value = 0.0

        return float(fid_value)


# ==========================
# 4. Główna funkcja
# ==========================

def main():
    REAL_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\real"
    GEN_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\generated"
    DEVICE = "cuda"

    MAX_IMAGES = 2000      # ⬅ LIMIT obrazów na zbiór
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    SEED = 42

    fid = FID(
        device=DEVICE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=299,
    )

    print("====================================")
    print("[FID] CONFIG:")
    print(f"  REAL_DIR    = {REAL_DIR}")
    print(f"  GEN_DIR     = {GEN_DIR}")
    print(f"  DEVICE      = {DEVICE}")
    print(f"  BATCH_SIZE  = {fid.batch_size}")
    print(f"  NUM_WORKERS = {fid.num_workers}")
    print(f"  MAX_IMAGES  = {MAX_IMAGES}")
    print("====================================\n")

    # ---------- REAL ----------
    print("[INFO] Extracting REAL features...")
    t0 = time.perf_counter()

    real_loader = make_dataloader(
        REAL_DIR,
        batch_size=fid.batch_size,
        num_workers=fid.num_workers,
        image_size=fid.image_size,
        max_images=MAX_IMAGES,
        seed=SEED,
    )

    model = InceptionEmbedding(device=DEVICE)
    real_feats = get_activations(real_loader, model=model)
    mu_r, sigma_r = _compute_stats(real_feats)

    n_real = real_feats.shape[0]
    t1 = time.perf_counter()
    real_time = t1 - t0
    print(f"[TIME] REAL features extracted in {real_time:.2f} s")
    print(f"[STATS] REAL: feats shape = {real_feats.shape}")
    real_norms = np.linalg.norm(real_feats, axis=1)
    print(f"[STATS] REAL norms: mean={real_norms.mean():.3f}, std={real_norms.std():.3f}, "
          f"min={real_norms.min():.3f}, max={real_norms.max():.3f}")
    print(f"[TIME] REAL: {real_time / n_real:.4f} s / image "
          f"({real_time * 1000 / n_real:.1f} ms / image)")
    print(f"[TIME] REAL: {real_time / n_real * 1000:.2f} s / 1000 images\n")

    # ---------- GENERATED ----------
    print("[INFO] Extracting GENERATED features...")
    t2 = time.perf_counter()

    gen_loader = make_dataloader(
        GEN_DIR,
        batch_size=fid.batch_size,
        num_workers=fid.num_workers,
        image_size=fid.image_size,
        max_images=MAX_IMAGES,
        seed=SEED + 1,
    )
    gen_feats = get_activations(gen_loader, model=model)
    mu_g, sigma_g = _compute_stats(gen_feats)

    n_gen = gen_feats.shape[0]
    t3 = time.perf_counter()
    gen_time = t3 - t2
    print(f"[TIME] GENERATED features extracted in {gen_time:.2f} s")
    print(f"[STATS] GENERATED: feats shape = {gen_feats.shape}")
    gen_norms = np.linalg.norm(gen_feats, axis=1)
    print(f"[STATS] GENERATED norms: mean={gen_norms.mean():.3f}, std={gen_norms.std():.3f}, "
          f"min={gen_norms.min():.3f}, max={gen_norms.max():.3f}")
    print(f"[TIME] GENERATED: {gen_time / n_gen:.4f} s / image "
          f"({gen_time * 1000 / n_gen:.1f} ms / image)")
    print(f"[TIME] GENERATED: {gen_time / n_gen * 1000:.2f} s / 1000 images\n")

    # ---------- FID ----------
    print("[INFO] Computing FID...")
    t4 = time.perf_counter()
    score = fid.from_stats(mu_r, sigma_r, mu_g, sigma_g)
    t5 = time.perf_counter()
    fid_time = t5 - t4

    total_time = t5 - t0

    print("\n\n====================================")
    print(f"FID RESULT = {score:.4f}")
    print("====================================")
    print(f"[TIME] FID computation time = {fid_time:.2f} s")
    print(f"[TIME] TOTAL time (features + FID) = {total_time:.2f} s\n")

    # ---------- WYKRES NORM CECH ----------
    print("[INFO] Plotting feature norms histogram...")
    plt.figure(figsize=(8, 5))
    plt.hist(real_norms, bins=50, alpha=0.5, label="REAL", density=True)
    plt.hist(gen_norms, bins=50, alpha=0.5, label="GENERATED", density=True)
    plt.xlabel("Norma wektora cech Inception (L2)")
    plt.ylabel("Gęstość")
    plt.title("Rozkład norm cech Inception: REAL vs GENERATED")
    plt.legend()
    plt.tight_layout()
    out_plot = "fid_feature_norms.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"[SAVED] Histogram norm cech zapisany do: {out_plot}")


def sanity_real_real():
    """
    Test sanity-check: FID(real, real) powinien być bliski 0.
    Tu też korzystamy z limitu MAX_IMAGES.
    """
    REAL_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\g1"
    DEVICE = "cuda"
    MAX_IMAGES = 2000
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    SEED = 42

    fid = FID(device=DEVICE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, image_size=299)

    loader = make_dataloader(
        REAL_DIR,
        batch_size=fid.batch_size,
        num_workers=fid.num_workers,
        image_size=fid.image_size,
        max_images=MAX_IMAGES,
        seed=SEED,
    )

    model = InceptionEmbedding(device=DEVICE)
    feats = get_activations(loader, model=model)
    mu, sigma = _compute_stats(feats)

    score = fid.from_stats(mu, sigma, mu, sigma)
    print("FID(real, real) =", score)


if __name__ == "__main__":
    # tu wybierasz, co odpalić:
   main()
   # sanity_real_real()
