# fe_engineer.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import List

# Columns we engineer deltas/rollings for (same as your FEATURES_FILE)
ACOUSTIC_COLS = [
    'age','sex','Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP',
    'Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11',
    'Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE'
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # sort by subject and time to preserve causality
    df = df.sort_values(['subject#', 'test_time']).reset_index(drop=True)

    # time normalization per subject (bounded in [0,1])
    max_time = df.groupby('subject#', observed=True)['test_time'].transform('max').clip(lower=1e-8)
    df['time_norm'] = df['test_time'] / max_time

    # First-order deltas per subject (causal)
    for col in ACOUSTIC_COLS:
        dname = f'delta_{col}'
        df[dname] = df.groupby('subject#', observed=True)[col].diff().fillna(0.0)

    # Rolling mean (window=3, min_periods=1) per subject (causal)
    for col in ACOUSTIC_COLS:
        rname = f'roll3_{col}'
        df[rname] = (
            df.groupby('subject#', observed=True)[col]
              .rolling(window=3, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )

    return df

def process_site_folder(site_path: Path, target: str = 'total_UPDRS'):
    for split in ['train','val','test']:
        csv_path = site_path / f'{split}.csv'
        df = pd.read_csv(csv_path)
        df = add_features(df)
        df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    base = Path("clients_prepared")
    for k in range(1, 6):
        process_site_folder(base / f"site{k}")
    print("✅ Feature engineering added to all sites.")
