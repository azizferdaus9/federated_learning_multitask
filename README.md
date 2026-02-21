# Federated Multi-Task Learning for Privacy-Preserving Parkinson’s Voice Analysis under Non-IID Labels

This repository provides an implementation of a **Federated Multi-Task Learning (FMTL)** framework for Parkinson’s voice analysis under **non-IID data distributions** and **heterogeneous label availability** across clients.

The framework trains a **shared encoder** with **task-specific heads** (classification + regression), enabling multiple institutions to collaboratively learn a robust representation **without sharing raw patient data**.

---

## Overview

- Supports **non-IID client distributions**
- Supports **heterogeneous labels** (classification-only, regression-only, or both)
- Implements **Flower-based federated learning**
- Includes **centralized/local baseline comparison**
- Designed for reproducible experiments

---

## Repository Structure

```
main_preprocess.py   # Dataset download/cleaning + preprocessing + feature preparation
fe_engineer.py       # Feature engineering utilities
make_clients.py      # Create non-IID client splits / partitions
prepare_clients.py   # Prepare client folders/files for federated runs
client.py            # Flower (FL) client implementation
server.py            # Flower (FL) server implementation
local_baseline.py    # Centralized/local baseline training
model_utils.py       # Model architecture (encoder + heads) and helper functions
```

---

# 1. Setup

## Requirements

- Python **3.9+** (recommended)
- Windows / Linux / macOS
- (Optional) GPU for faster training

---

## Create Virtual Environment

### Option A — Using `venv`

```bash
python -m venv .venv
```

### Activate Environment

#### Windows (PowerShell)

```bash
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet:

```bash
pip freeze > requirements.txt
```

---

# 2. Data Preparation

## Step 1 — Run Preprocessing

This step prepares the dataset(s) into the format expected by the federated learning pipeline.

```bash
python main_preprocess.py
```

Typical outputs:

- Processed feature files (`.csv`, `.npz`, `.pkl`)
- Clean dataset directory for client partitioning

To view available arguments:

```bash
python main_preprocess.py --help
```

---

# 3. Create Non-IID Federated Clients

## Step 2 — Create Client Partitions

This step creates non-IID splits and assigns heterogeneous label availability.

Examples:
- Some clients → classification labels only
- Some clients → regression labels only
- Some clients → both tasks

```bash
python make_clients.py
```

---

## Step 3 — Prepare Client Folders

Finalizes local datasets and configuration files for each client.

```bash
python prepare_clients.py
```

After this step, you should see:

```
clients/
├── client_0/
├── client_1/
├── client_2/
└── client_3/
```

(Exact structure depends on your implementation.)

---

# 4. Run Federated Training

## Step 4A — Start FL Server (Terminal 1)

```bash
python server.py
```

Keep this terminal running.

---

## Step 4B — Start Clients (Terminal 2, 3, 4…)

Open multiple terminals (one per client):

```bash
python client.py --cid 0
python client.py --cid 1
python client.py --cid 2
python client.py --cid 3
```

If `--cid` is not supported, run according to your implementation.

To check supported arguments:

```bash
python client.py --help
```

---

# 5. Baselines (Optional)

To run centralized or local-only baseline training:

```bash
python local_baseline.py
```

This is useful for comparison against federated performance.

---

# 6. Experimental Configuration

For reproducibility, document the following in your experiments:

- Number of clients
- Number of federated rounds
- Local epochs
- Learning rate
- Non-IID split method (e.g., Dirichlet α)
- Label distribution per client
- Random seed

---

# 7. Outputs

Depending on your configuration, you may obtain:

- Saved model checkpoints
- Per-round logs
- Accuracy / F1-score (classification)
- MAE / RMSE (regression)
- Final global model

---

# 8. Reproducibility

To ensure consistent results, set a fixed seed:

```bash
python main_preprocess.py --seed 42
python make_clients.py --seed 42
```

---

# 9. Troubleshooting

### 1) ModuleNotFoundError

Ensure environment is activated:

```bash
pip install -r requirements.txt
```

---

### 2) Clients Cannot Connect to Server

- Ensure server and client ports match.
- Use `127.0.0.1` consistently for local execution.

---

### 3) Results Change Every Run

- Fix random seeds
- Ensure deterministic data partitioning

---

# Citation

If you use this repository, please cite:

> Federated Multi-Task Learning for Privacy-Preserving Parkinson’s Voice Analysis under Non-IID Labels  
> ACM Conference Submission, 2026

---

# License

This project is intended for academic and research purposes.
