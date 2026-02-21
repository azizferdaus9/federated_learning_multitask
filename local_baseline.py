import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

BASE = Path("clients_prepared")

targets = ["motor_UPDRS", "total_UPDRS"]
features = pd.read_csv("clients/_features.txt", header=None).iloc[:,0].tolist()

for site in sorted(BASE.glob("site*")):
    print(f"\n===== {site.name} =====")
    df_train = pd.read_csv(site / "train.csv")
    df_val   = pd.read_csv(site / "val.csv")
    df_test  = pd.read_csv(site / "test.csv")

    X_train, y_train = df_train[features], df_train[targets]
    X_val,   y_val   = df_val[features],   df_val[targets]
    X_test,  y_test  = df_test[features],  df_test[targets]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train["total_UPDRS"])

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test["total_UPDRS"], preds)
    r2  = r2_score(y_test["total_UPDRS"], preds)
    print(f"Test MAE={mae:.3f} | R²={r2:.3f}")
