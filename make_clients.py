import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Load
df = pd.read_csv("Dataset/parkinsons_updrs.csv")

# ---- Define features & targets (used later) ----
ID_COL = "subject#"
TARGETS = ["motor_UPDRS", "total_UPDRS"]
IGNORE_COLS = [ID_COL, "test_time"] + TARGETS
FEATURES = [c for c in df.columns if c not in IGNORE_COLS]

# Save a small schema file for later checks
schema = {
    "features": FEATURES,
    "targets": TARGETS,
    "id_col": ID_COL,
}
Path("clients").mkdir(exist_ok=True)
pd.Series(schema["features"]).to_csv("clients/_features.txt", index=False, header=False)

# ---- Make subject-level stratification on baseline severity ----
# Use each subject's FIRST record to estimate severity bin for stratification
first_rows = df.sort_values("test_time").groupby(ID_COL, as_index=False).first()
severity_bin = pd.qcut(first_rows["total_UPDRS"], q=5, labels=False)  # 5 bins

N_CLIENTS = 5
skf = StratifiedKFold(n_splits=N_CLIENTS, shuffle=True, random_state=42)
fold_ids = {}
for fold_idx, (_, test_idx) in enumerate(skf.split(first_rows, severity_bin), start=1):
    subj_ids = first_rows.iloc[test_idx][ID_COL].tolist()
    fold_ids[fold_idx] = subj_ids

# ---- Write per-client CSVs (ALL rows of the subjects in that fold) ----
for k, subj_list in fold_ids.items():
    client_df = df[df[ID_COL].isin(subj_list)].copy()
    out_path = Path(f"clients/site{k}.csv")
    client_df.to_csv(out_path, index=False)
    print(f"site{k}: subjects={len(subj_list)}, rows={len(client_df)}")

print("\nFeatures used:")
print(FEATURES)
