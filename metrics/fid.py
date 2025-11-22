"""
Fréchet Inception Distance (FID) – pełna implementacja.

- Ekstrakcja cech z sieci Inception v3 (warstwa "pool3", 2048-D).
- Liczenie średniej i kowariancji cech dla obu zbiorów.
- Wyznaczenie FID wg klasycznego wzoru.

Ten plik NIE korzysta z gotowych bibliotek do FID – wszystko jest policzone ręcznie.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Sequence, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

try:
    # używamy sqrtm ze scipy, ale mamy też fallback gdyby ktoś nie miał scipy
    from scipy.linalg import sqrtm  # type: ignore
except ImportError:
    sqrtm = None


# ==========================
# 1. Dataset + DataLoader
# ==========================

class ImageFolderDataset(Dataset):
    """
    Bardzo prosty dataset: wczytuje wszystkie JPG/PNG z podanego folderu.
    Obrazy są konwertowane do RGB i przekształcane za pomocą transformacji.
    """

    def __init__(self, root: str | Path, transform: Optional[nn.Module] = None) -> None:
        self.root = Path(root)
        self.paths: Sequence[Path] = sorted(
            list(self.root.glob("*.png"))
            + list(self.root.glob("*.jpg"))
            + list(self.root.glob("*.jpeg"))
        )
        if not self.paths:
            raise ValueError(f"No images found in {self.root}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def make_dataloader(
    root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 299,
) -> DataLoader:
    """
    Tworzy DataLoader z odpowiednimi transformacjami dla Inception v3.
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

    dataset = ImageFolderDataset(root=root, transform=transform)
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

        # ładowanie wstępnie wytrenowanej sieci Inception v3
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=False,
        )
        inception.eval()

        # bierzemy wszystko oprócz końcowych warstw klasyfikacyjnych
        # (do AdaptiveAvgPool2d włącznie)
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

    activations: list[torch.Tensor] = []

    for batch in loader:
        feats = model(batch)
        activations.append(feats.cpu())

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
    Liczy macierzową pierwiastek z iloczynu sigma_r * sigma_g.
    Korzysta z scipy.linalg.sqrtm jeśli dostępne, w przeciwnym razie z własnej
    implementacji poprzez rozkład własny.
    """
    cov_prod = sigma_r @ sigma_g

    if sqrtm is not None:
        covmean = sqrtm(cov_prod)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return covmean

    # fallback: rozkład własny
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    eigvals = np.clip(eigvals, a_min=eps, a_max=None)
    covmean = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    return covmean


@dataclass
class FID:
    """
    Główna klasa metryki FID.

    Użycie – wersja na dwóch folderach z obrazami:

        from metrics.fid import FID

        fid = FID(device="cuda")
        score = fid.from_directories("data/real", "data/generated")

    Użycie – jeśli masz już cechy (np. z Inception):

        score = fid.from_features(real_feats, gen_feats)
    """

    device: str = "cpu"
    batch_size: int = 32
    num_workers: int = 0
    image_size: int = 299
    eps: float = 1e-6

    # ---------- API wysokiego poziomu ----------

    def from_directories(self, real_dir: str | Path, gen_dir: str | Path) -> float:
        """
        Liczy FID między obrazami w dwóch folderach.
        """
        # DataLoadery
        real_loader = make_dataloader(
            real_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
        )
        gen_loader = make_dataloader(
            gen_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
        )

        # model Inception
        model = InceptionEmbedding(device=self.device)

        # cechy
        real_feats = get_activations(real_loader, model=model)
        gen_feats = get_activations(gen_loader, model=model)

        return self.from_features(real_feats, gen_feats)

    def from_features(self, real_features: np.ndarray, gen_features: np.ndarray) -> float:
        """
        Liczy FID z gotowych wektorów cech.
        """
        mu_r, sigma_r = _compute_stats(real_features)
        mu_g, sigma_g = _compute_stats(gen_features)

        diff = mu_r - mu_g
        diff_sq = diff.dot(diff)

        covmean = _sqrtm_product(sigma_r, sigma_g, eps=self.eps)

        # dodaj małe eps na diagonali, jeśli macierz jest zdegenerowana
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_r.shape[0]) * self.eps
            covmean = _sqrtm_product(sigma_r + offset, sigma_g + offset, eps=self.eps)

        fid_value = diff_sq + np.trace(sigma_r + sigma_g - 2.0 * covmean)
        return float(fid_value)
