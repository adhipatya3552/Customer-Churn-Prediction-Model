from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

def train(df):

    # -----------------------------
    # 1. Split features & target
    # -----------------------------
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -----------------------------
    # 2. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 3. SCALE DATA (IMPORTANT FIX)
    # -----------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # 4. Models
    # -----------------------------
    models = {
        "logistic": LogisticRegression(max_iter=3000),
        "rf": RandomForestClassifier(),
        "xgb": xgb.XGBClassifier(eval_metric="logloss")
    }

    best_model = None
    best_score = 0

    # -----------------------------
    # 5. Train & Evaluate
    # -----------------------------
    for name, model in models.items():

        if name == "logistic":
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

        print(name, score)

        if score > best_score:
            best_model = model
            best_score = score

    # -----------------------------
    # 6. Save model
    # -----------------------------
    pickle.dump(best_model, open("models/model.pkl", "wb"))

    # ADD THIS ↓
    import json
    with open("models/columns.json", "w") as f:
        json.dump(list(X.columns), f)