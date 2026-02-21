# client.py
from pathlib import Path
import sys

# Ensure local imports work when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from model_utils import (
    ParkinsonsNetSingle as ParkinsonsNet,  # single-target net
    load_site_data,
    make_loaders,
    train_one_epoch,
)

# -------------------------------
# Config
# -------------------------------
SERVER_ADDRESS = "127.0.0.1:8080"
FEATURES_FILE = Path("clients/_features.txt")
TARGET_LIST = ["total_UPDRS"]   # single target

LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 3e-4
DEVICE = torch.device("cpu")  # change to "cuda" if you have GPU


# -------------------------------
# Metrics (single target)
# -------------------------------
def _eval_metrics_common(model, loader, loss_fn):
    """Compute MSE/MAE/R² for total_UPDRS on the given loader."""
    model.eval()
    losses = []
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(X)                 # shape: (batch,)
            loss = loss_fn(pred, y)         # y shape: (batch,)
            losses.append(loss.item() * len(X))
            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())
    n = sum(len(a) for a in y_true_all)
    mse = float(np.sum(losses) / n)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return mse, mae, r2


def eval_val_metrics(model, loader, loss_fn):
    return _eval_metrics_common(model, loader, loss_fn)


def eval_test_metrics(model, loader, loss_fn):
    return _eval_metrics_common(model, loader, loss_fn)


# -------------------------------
# Personalization helper
# -------------------------------
def personalize_and_eval(model, train_loader, test_loader, loss_fn, epochs=2, lr=1e-4):
    """Fine-tune ONLY the last layer locally, then evaluate on test."""
    import copy
    m = copy.deepcopy(model)  # don't change the original global model
    m.to(DEVICE)
    m.train()

    # Freeze all but last Linear layer
    for p in m.parameters():
        p.requires_grad = False

    last_linear = None
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            last_linear = mod
    assert last_linear is not None, "Could not find final Linear layer"

    for p in last_linear.parameters():
        p.requires_grad = True

    opt = optim.AdamW(last_linear.parameters(), lr=lr, weight_decay=0.0)

    for _ in range(epochs):
        for X, y in train_loader:
            X = X.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad()
            pred = m(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

    mse, mae, r2 = eval_test_metrics(m, test_loader, loss_fn)
    return mse, mae, r2


# -------------------------------
# Flower Client
# -------------------------------
class ParkinsonClient(fl.client.NumPyClient):
    def __init__(self, site_path: Path):
        self.site_path = site_path

        # Load feature list
        self.feature_list = [
            ln.strip() for ln in FEATURES_FILE.read_text().splitlines() if ln.strip()
        ]

        # Load data tensors
        train_tensors, val_tensors, test_tensors = load_site_data(
            site_path, self.feature_list, TARGET_LIST
        )
        self.train_loader, self.val_loader, self.test_loader = make_loaders(
            train_tensors, val_tensors, test_tensors, batch_size=BATCH_SIZE
        )

        # Model / loss / optimizer
        self.model = ParkinsonsNet(in_features=len(self.feature_list)).to(DEVICE)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber-like loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, weight_decay=1e-4)

    # ----- Flower API -----
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for _, p in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state = self.model.state_dict()
        for (k, _), v in zip(state.items(), parameters):
            state[k] = torch.tensor(v)
        self.model.load_state_dict(state)

    def fit(self, parameters, config):
        # Receive global weights
        self.set_parameters(parameters)

        # Local training
        for _ in range(LOCAL_EPOCHS):
            train_one_epoch(self.model, self.train_loader, self.optimizer, self.loss_fn)

        # Validation metrics after local training (aggregated on server)
        val_mse, val_mae, val_r2 = eval_val_metrics(self.model, self.val_loader, self.loss_fn)

        # Return updated weights, num_examples, and validation metrics
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "val_mse": val_mse,
            "val_mae": val_mae,
            "val_r2": val_r2,
        }

    def evaluate(self, parameters, config):
        # Receive latest global weights
        self.set_parameters(parameters)

        # Global test metrics (no personalization)
        mse, mae, r2 = eval_test_metrics(self.model, self.test_loader, self.loss_fn)
        print(f"[{self.site_path.name}] GLOBAL TEST -> MSE={mse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")

        # Personalized test metrics (last-layer fine-tune on local train)
        p_mse, p_mae, p_r2 = personalize_and_eval(
            self.model, self.train_loader, self.test_loader, self.loss_fn, epochs=2, lr=1e-4
        )
        print(f"[{self.site_path.name}] PERSONALIZED TEST -> MSE={p_mse:.3f} | MAE={p_mae:.3f} | R2={p_r2:.3f}")

        # Return both sets so the server aggregates them
        return mse, len(self.test_loader.dataset), {
            "mse": mse,
            "mae_total": mae,   # global metrics (kept for server consistency)
            "r2_total": r2,
            "p_mse": p_mse,           # personalized metrics
            "p_mae_total": p_mae,
            "p_r2_total": p_r2,
        }


def start_client(site_id: int):
    site_path = Path(f"clients_prepared/site{site_id}")
    print(f"🚀 Starting client for {site_path}")
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=ParkinsonClient(site_path),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=int, required=True, help="Site id (1..5)")
    args = parser.parse_args()
    start_client(args.site)
