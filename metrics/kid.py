from __future__ import annotations
from scipy.linalg import sqrtm

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, List

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# =====================================
# 1. Dataset + DataLoader
# =====================================

class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, transform: Optional[nn.Module] = None):
        self.root = Path(root)
        self.paths: Sequence[Path] = sorted(
            list(self.root.rglob("*.png")) +
            list(self.root.rglob("*.jpg")) +
            list(self.root.rglob("*.jpeg"))
        )
        if not self.paths:
            raise ValueError(f"No images found in {self.root}")

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def make_dataloader(root: str | Path,
                    batch_size=32,
                    num_workers=0,
                    image_size=299) -> DataLoader:

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = ImageFolderDataset(root, transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader


# =====================================
# 2. Inception v3 embedding
# =====================================

class InceptionEmbedding(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        inception.aux_logits = False
        inception.eval()

        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.device = torch.device(device)
        self.to(self.device)

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        feats = self.features(x)
        return feats.view(feats.size(0), -1)


@torch.no_grad()
def get_activations(loader: DataLoader, model=None, device="cpu") -> np.ndarray:
    if model is None:
        model = InceptionEmbedding(device=device)
    model.eval()

    outs = []
    for batch in loader:
        feats = model(batch)
        outs.append(feats.cpu())

    return torch.cat(outs, dim=0).numpy()


# =====================================
# 3. KID (MMD^2 z jądrem wielomianowym)
# =====================================
def calculate_fid_from_features(real_features: np.ndarray,
                                gen_features: np.ndarray,
                                eps: float = 1e-6) -> float:
    """
    FID liczony na cechach (np. Inception 2048-D), zgodnie ze wzorem Heusel et al. (2017).

    real_features: [N_r, D]
    gen_features : [N_g, D]
    """
    # 1. Średnie wektory
    mu_r = real_features.mean(axis=0)
    mu_g = gen_features.mean(axis=0)

    # 2. Macierze kowariancji (rowvar=False => każda kolumna to zmienna)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)

    # 3. Pierwiastek macierzy sigma_r * sigma_g
    cov_prod = sigma_r @ sigma_g
    # zabezpieczenie przed numeryczną osobliwością
    cov_prod += np.eye(cov_prod.shape[0]) * eps

    covmean = sqrtm(cov_prod)

    # sqrtm może zwrócić liczby zespolone z powodu błędów numerycznych -> bierzemy część rzeczywistą
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 4. FID = ||mu_r - mu_g||^2 + Tr(sigma_r + sigma_g - 2*sqrt(sigma_r sigma_g))
    diff = mu_r - mu_g
    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))

    return fid

def polynomial_mmd_unbiased(X: np.ndarray, Y: np.ndarray,
                            degree=3, gamma=None, coef0=1.0) -> float:
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K_XX = (gamma * X @ X.T + coef0) ** degree
    K_YY = (gamma * Y @ Y.T + coef0) ** degree
    K_XY = (gamma * X @ Y.T + coef0) ** degree

    m = X.shape[0]
    n = Y.shape[0]

    sum_K_XX = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1))
    sum_K_YY = (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1))
    mean_K_XY = K_XY.mean()

    return float(sum_K_XX + sum_K_YY - 2.0 * mean_K_XY)



def one_nn_test(real_features: np.ndarray,
                gen_features: np.ndarray):
    """
    Test 1-NN w przestrzeni cech (np. Inception).

    Zwraca:
        overall_acc  - dokładność 1-NN dla wszystkich próbek
        real_acc     - dokładność dla próbek real
        gen_acc      - dokładność dla próbek generated
    """
    X_real = real_features
    X_gen = gen_features

    n_real = X_real.shape[0]
    n_gen = X_gen.shape[0]

    # 1. Sklejamy cechy i etykiety
    X = np.concatenate([X_real, X_gen], axis=0)   # [N, D]
    y = np.concatenate([
        np.zeros(n_real, dtype=int),     # 0 = real
        np.ones(n_gen, dtype=int)        # 1 = gen
    ], axis=0)                           # [N]

    # (opcjonalnie) normalizacja L2 – często pomaga
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    X_norm = X / norms

    # 2. Macierz odległości L2^2 (N x N)
    # dist^2(x_i, x_j) = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    sq_norms = np.sum(X_norm**2, axis=1, keepdims=True)  # [N, 1]
    dist2 = sq_norms + sq_norms.T - 2.0 * (X_norm @ X_norm.T)

    # ignorujemy odległość do siebie samego
    np.fill_diagonal(dist2, np.inf)

    # 3. Najbliższy sąsiad
    nn_idx = np.argmin(dist2, axis=1)   # [N]
    y_pred = y[nn_idx]

    # 4. Accuracy
    correct = (y_pred == y)
    overall_acc = float(correct.mean())

    real_acc = float(correct[:n_real].mean())        # tylko próbki real
    gen_acc = float(correct[n_real:].mean())         # tylko próbki gen

    return overall_acc, real_acc, gen_acc

@dataclass
class KID:
    device: str = "cpu"   # <-- zamiast "gpu"
    batch_size: int = 32
    num_workers: int = 0
    image_size: int = 299
    subset_size: int = 1000
    num_subsets: int = 100
    scale: float = 1000.0  # standardowo raportuje się KID * 1000

    def from_directories(self, real_dir, gen_dir):
        real_loader = make_dataloader(real_dir, self.batch_size,
                                      self.num_workers, self.image_size)
        gen_loader = make_dataloader(gen_dir, self.batch_size,
                                     self.num_workers, self.image_size)

        model = InceptionEmbedding(device=self.device)

        real_feats = get_activations(real_loader, model)
        gen_feats = get_activations(gen_loader, model)

        return self.from_features(real_feats, gen_feats)

    def from_features(self, real_features, gen_features):
        m = real_features.shape[0]
        n = gen_features.shape[0]
        subset = min(self.subset_size, m, n)

        rng = np.random.default_rng()
        vals = []

        for _ in range(self.num_subsets):
            idx_r = rng.choice(m, subset, replace=False)
            idx_g = rng.choice(n, subset, replace=False)

            X = real_features[idx_r]
            Y = gen_features[idx_g]

            mmd2 = polynomial_mmd_unbiased(X, Y)
            vals.append(mmd2)

        vals = np.array(vals)
        return float(vals.mean() * self.scale), float(vals.std(ddof=1) * self.scale)


# =====================================
# 4. MAIN bez argparse
# =====================================
if __name__ == "__main__":

    REAL_DIR = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\real"
    GEN_DIR  = r"C:\Users\zprzy\Desktop\inzynierka\image-generation-metrics\data\generated"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    kid = KID(
        device=DEVICE,
        batch_size=32,
        num_workers=0,
        image_size=299,
        subset_size=1000,
        num_subsets=100,
        scale=1000.0,
    )

    print("[INFO] Extracting REAL & GENERATED features (Inception)...")

    # --- 1) Ładujemy obrazki i liczymy cechy raz ---
    real_loader = make_dataloader(REAL_DIR, kid.batch_size,
                                  kid.num_workers, kid.image_size)
    gen_loader  = make_dataloader(GEN_DIR,  kid.batch_size,
                                  kid.num_workers, kid.image_size)

    model = InceptionEmbedding(device=DEVICE)

    real_feats = get_activations(real_loader, model)
    gen_feats  = get_activations(gen_loader,  model)

    # --- 2) KID z cech ---
    mean_kid, std_kid = kid.from_features(real_feats, gen_feats)

    print("\n====================================")
    print(f"KID RESULT = {mean_kid:.4f} ± {std_kid:.4f}   (×10^-3)")
    print("====================================\n")
    # --- 2a) FID z tych samych cech ---
    fid_value = calculate_fid_from_features(real_feats, gen_feats)

    print("FID RESULT = {:.4f}".format(fid_value))

    # --- 3) Test 1-NN z tych samych cech ---
    overall_acc, real_acc, gen_acc = one_nn_test(real_feats, gen_feats)

    print("1-NN TEST (Inception feature space)")
    print("-----------------------------------")
    print(f"Overall accuracy : {overall_acc*100:.2f}%")
    print(f"Real accuracy    : {real_acc*100:.2f}%")
    print(f"Gen  accuracy    : {gen_acc*100:.2f}%")
    print()
