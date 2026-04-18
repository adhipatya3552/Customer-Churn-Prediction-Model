import pandas as pd

def encode_features(df):

    # -----------------------------
    # 1. Fix target column
    # -----------------------------
    if "Churn" in df.columns:
        target_col = "Churn"
    elif "Churn Label" in df.columns:
        target_col = "Churn Label"
    else:
        raise Exception("Target column not found")

    df["Churn"] = df[target_col].map({"Yes":1, "No":0})

    # -----------------------------
    # 2. Remove original target column
    # -----------------------------
    if target_col != "Churn":
        df.drop(target_col, axis=1, inplace=True)

        # -----------------------------
    # REMOVE DATA LEAKAGE (FINAL FIX)
    # -----------------------------
    cols_to_drop = [
        "Churn Value",
        "Churn Score",
        "Churn Reason",
        "CLTV",          # also remove this (future value leakage)
        "Count"          # useless column
    ]

    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # -----------------------------
    # 4. One-hot encoding
    # -----------------------------
    df = pd.get_dummies(df, drop_first=True)

    return df