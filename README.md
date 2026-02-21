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

