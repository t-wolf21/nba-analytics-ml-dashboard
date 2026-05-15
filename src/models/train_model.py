from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

FEATURE_COLUMNS = [
    "last_10_win_pct_diff",
    "avg_points_last_10_diff",
    "avg_plus_minus_last_10_diff",
]

TARGET_COLUMN = "home_team_won"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_model(training_data_path: str = "data/processed/training_games.csv"):
    training_data_path = PROJECT_ROOT / training_data_path
    data = pd.read_csv(training_data_path)
    data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    data["game_date"] = pd.to_datetime(data["game_date"])
    data = data.sort_values("game_date")

    split_index = int(len(data) * 0.8)

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Log loss: {loss:.3f}")

    model_path = PROJECT_ROOT / "data/models/baseline_logistic_regression.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Saved model to {model_path}")

    print(f"Train games: {len(train_data)}")
    print(f"Test games: {len(test_data)}")
    print(f"Train date range: {train_data['game_date'].min()} to {train_data['game_date'].max()}")
    print(f"Test date range: {test_data['game_date'].min()} to {test_data['game_date'].max()}")

    return model

if __name__ == "__main__":
    train_model()
