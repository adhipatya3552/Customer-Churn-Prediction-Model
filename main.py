from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import encode_features
from src.train_model import train

# -----------------------------
# 1. Load data
# -----------------------------
df = load_and_clean_data("data/raw/churn.csv")

# -----------------------------
# 2. Feature engineering
# -----------------------------
df = encode_features(df)

# -----------------------------
# 3. SAVE processed dataset (NEW)
# -----------------------------
df.to_csv("data/processed/clean_data.csv", index=False)

# -----------------------------
# 4. Train model
# -----------------------------
train(df)