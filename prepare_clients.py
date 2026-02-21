import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np

ID_COL = "subject#"
TIME_COL = "test_time"
TARGETS = ["motor_UPDRS", "total_UPDRS"]

# Load feature list saved earlier
features = pd.read_csv("clients/_features.txt", header=None).iloc[:,0].tolist()

in_dir = Path("clients")
out_dir = Path("clients_prepared")
out_dir.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

def temporal_split(df_subj, train_frac=0.6, val_frac=0.2):
    """Split one subject's rows by time into train/val/test."""
    df_subj = df_subj.sort_values(TIME_COL)
    n = len(df_subj)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = n - n_train - n_val
    if n_test < 1:
        # push one from val to test if needed
        n_val = max(1, n_val - 1)
        n_test = n - n_train - n_val
    idx = np.arange(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return df_subj.iloc[train_idx], df_subj.iloc[val_idx], df_subj.iloc[test_idx]

def scale_site(train_df, val_df, test_df, feature_cols):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val   = scaler.transform(val_df[feature_cols])
    X_test  = scaler.transform(test_df[feature_cols])

    train_df_scaled = train_df.copy()
    val_df_scaled   = val_df.copy()
    test_df_scaled  = test_df.copy()

    train_df_scaled[feature_cols] = X_train
    val_df_scaled[feature_cols]   = X_val
    test_df_scaled[feature_cols]  = X_test
    return train_df_scaled, val_df_scaled, test_df_scaled, scaler

# Process each siteN.csv
for site_csv in sorted(in_dir.glob("site*.csv")):
    site_name = site_csv.stem  # e.g., "site1"
    df = pd.read_csv(site_csv)

    # Subject-wise temporal split, then concatenate
    trains, vals, tests = [], [], []
    for sid, g in df.groupby(ID_COL):
        tr, va, te = temporal_split(g, train_frac=0.6, val_frac=0.2)
        trains.append(tr)
        vals.append(va)
        tests.append(te)
    train_df = pd.concat(trains).reset_index(drop=True)
    val_df   = pd.concat(vals).reset_index(drop=True)
    test_df  = pd.concat(tests).reset_index(drop=True)

    # Scale per site using train only
    train_scaled, val_scaled, test_scaled, scaler = scale_site(
        train_df, val_df, test_df, features
    )

    # Save
    site_out = out_dir / site_name
    site_out.mkdir(exist_ok=True)
    train_scaled.to_csv(site_out / "train.csv", index=False)
    val_scaled.to_csv(site_out / "val.csv", index=False)
    test_scaled.to_csv(site_out / "test.csv", index=False)

    # Save a small meta file
    meta = {
        "n_train": len(train_scaled),
        "n_val": len(val_scaled),
        "n_test": len(test_scaled),
        "features": features,
        "targets": TARGETS,
        "id_col": ID_COL,
        "time_col": TIME_COL,
    }
    pd.Series(meta, dtype="object").to_json(site_out / "meta.json")
    print(f"{site_name}: train={len(train_scaled)}, val={len(val_scaled)}, test={len(test_scaled)}")

print("\n✅ Done. Per-site temporal splits created and standardized (fit on train only).")
