# Federated Multi-Task Learning for Privacy-Preserving Parkinson’s Voice Analysis under Non-IID Labels

This repository provides an implementation of a **Federated Multi-Task Learning (FMTL)** framework for Parkinson’s voice analysis under **non-IID** and **heterogeneous label availability** across clients.  
The system trains a **shared encoder** with **task-specific heads** (classification + regression), enabling collaboration between institutions without sharing raw data.

## Repository Structure

- `main_preprocess.py` — dataset download/cleaning + preprocessing + feature preparation  
- `fe_engineer.py` — feature engineering utilities (if used by preprocessing)
- `make_clients.py` — create non-IID client splits / partitions
- `prepare_clients.py` — prepares client folders/files for federated runs
- `client.py` — Flower (FL) client implementation
- `server.py` — Flower (FL) server implementation
- `local_baseline.py` — centralized or local baseline training/evaluation
- `model_utils.py` — model architectures + helpers (encoder, heads, metrics, etc.)

---

## 1) Setup

### Requirements
- Python **3.9+** recommended
- OS: Windows / Linux / macOS (tested conceptually on Windows + Linux)
- (Optional) GPU for faster training

### Create environment

#### Option A: venv
```bash
python -m venv .venv

##Activate:

##Windows (PowerShell)

.venv\Scripts\Activate.ps1

##macOS/Linux

source .venv/bin/activate

##Install packages:

pip install -U pip
pip install -r requirements.txt

If you don’t have requirements.txt yet, create one (example):

pip freeze > requirements.txt
2) Data Preparation
Step 1 — Run preprocessing

This step prepares the dataset(s) into the format expected by the FL pipeline.

python main_preprocess.py

Typical outputs (example):

processed feature files (e.g., .csv, .npz, .pkl)

a clean dataset directory for later partitioning

If your preprocessing script supports arguments, you can document them here (recommended):

python main_preprocess.py --help
3) Create Non-IID Federated Clients
Step 2 — Create client partitions

This creates client splits (non-IID) and assigns different label availability (e.g., some clients only have classification labels, others only regression labels).

python make_clients.py
Step 3 — Prepare client folders/files

This step finalizes each client’s local dataset and config.

python prepare_clients.py

After this, you should have a folder like:

clients/

client_0/

client_1/

...

(Your actual folder names may differ depending on your scripts.)

4) Run Federated Training
Step 4A — Start the FL server (Terminal 1)
python server.py

Keep this running.

Step 4B — Start clients (Terminal 2, 3, 4…)

Open multiple terminals (one per client), then run:

python client.py --cid 0
python client.py --cid 1
python client.py --cid 2
python client.py --cid 3

If your client.py does not accept --cid, then run it as your code expects (for example it may read from a config file automatically).
To check available flags:

python client.py --help
5) Baselines (Optional)
Local / Centralized baseline

To compare with a centralized or local-only training baseline:

python local_baseline.py
6) Results & Reproducibility

Recommended items to include in your runs:

number of clients

number of rounds

local epochs

learning rate

non-IID split method (Dirichlet alpha / label-skew etc.)

which clients have which labels (classification/regression/both)

If your scripts save outputs, you may see:

saved model checkpoints

logs per round

metrics such as Accuracy / F1 / MAE / RMSE

Troubleshooting
1) “Module not found”

Make sure your environment is activated and dependencies installed:

pip install -r requirements.txt
2) Server starts but clients don’t connect

Confirm server address/port in server.py and client.py match.

If running locally, use 127.0.0.1 consistently.

3) Different results each run

Set a fixed seed (recommended). If you already support it:

python main_preprocess.py --seed 42
python make_clients.py --seed 42
Citation

If you use this code, please cite the corresponding paper:

Federated Multi-Task Learning for Privacy-Preserving Parkinson’s Voice Analysis under Non-IID Labels
(ACM conference submission, 2026)
