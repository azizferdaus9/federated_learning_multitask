# model_utils.py
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ------------------------------
# Model (single-target: total_UPDRS)
# ------------------------------
class ParkinsonsNetSingle(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns shape (batch,), not (batch,1)
        return self.net(x).squeeze(1)


# ------------------------------
# Data helpers
# ------------------------------
def load_site_data(
    site_path: Path,
    feature_list: List[str],
    target_list: List[str],
):
    """
    Load per-site CSVs and return tensors:
        - X: float32 [N, F]
        - y: float32 [N]  (single target: total_UPDRS)
    Expect files: site_path/{train.csv,val.csv,test.csv}
    """
    assert len(target_list) == 1, "Expected a single target, e.g. ['total_UPDRS']"

    def to_tensors(df: pd.DataFrame):
        X = torch.tensor(df[feature_list].values, dtype=torch.float32)
        y = torch.tensor(df[target_list[0]].values, dtype=torch.float32)  # 1-D target
        return X, y

    df_train = pd.read_csv(site_path / "train.csv")
    df_val = pd.read_csv(site_path / "val.csv")
    df_test = pd.read_csv(site_path / "test.csv")

    return to_tensors(df_train), to_tensors(df_val), to_tensors(df_test)


def make_loaders(
    train_tensors: Tuple[torch.Tensor, torch.Tensor],
    val_tensors: Tuple[torch.Tensor, torch.Tensor],
    test_tensors: Tuple[torch.Tensor, torch.Tensor],
    batch_size: int = 32,
    shuffle_train: bool = True,
):
    Xtr, ytr = train_tensors
    Xva, yva = val_tensors
    Xte, yte = test_tensors

    tr_ds = TensorDataset(Xtr, ytr)
    va_ds = TensorDataset(Xva, yva)
    te_ds = TensorDataset(Xte, yte)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle_train)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader


# ------------------------------
# Training loop (single-target)
# ------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
        n += len(X)
    return total_loss / max(n, 1)
