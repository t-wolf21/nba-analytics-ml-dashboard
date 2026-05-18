from pathlib import Path

import joblib
import pandas as pd

from src.data.nba_data_client import NBADataClient
from src.features.team_features import calculate_team_features
from src.features.matchup_features import build_matchup_features
from src.models.train_model import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data/models/baseline_logistic_regression.joblib"


def predict_game(
    home_team_id: int,
    away_team_id: int,
    game_date,
    season: str = "2025-26",
    client: NBADataClient | None = None,
    model=None,
    team_games_cache: dict[int, pd.DataFrame] | None = None,
) -> dict:
    if client is None:
        client = NBADataClient(season=season)

    if model is None:
        model = joblib.load(MODEL_PATH)

    if team_games_cache is None:
        team_games_cache = {}

    if home_team_id not in team_games_cache:
        team_games_cache[home_team_id] = client.fetch_team_games(home_team_id)

    if away_team_id not in team_games_cache:
        team_games_cache[away_team_id] = client.fetch_team_games(away_team_id)

    home_games = team_games_cache[home_team_id]
    away_games = team_games_cache[away_team_id]

    home_features = calculate_team_features(home_games, game_date)
    away_features = calculate_team_features(away_games, game_date)

    matchup_features = build_matchup_features(home_features, away_features)

    features_df = pd.DataFrame([matchup_features])
    X = features_df[FEATURE_COLUMNS]

    home_win_probability = float(model.predict_proba(X)[0][1])

    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_win_probability": home_win_probability,
        "away_win_probability": 1 - home_win_probability,
        "features": matchup_features,
    }

def predict_games_for_date(game_date, season: str = "2025-26") -> pd.DataFrame:
    client = NBADataClient(season=season)
    model = joblib.load(MODEL_PATH)
    team_games_cache = {}
    games = client.fetch_schedule_for_date(game_date)

    rows = []

    for _, game in games.iterrows():
        prediction = predict_game(
            home_team_id=int(game["home_team_id"]),
            away_team_id=int(game["away_team_id"]),
            game_date=game["game_date"],
            season=season,
            client=client,
            model=model,
            team_games_cache=team_games_cache,
        )

        features = prediction["features"]

        rows.append(
            {
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "status": game["status"],
                "away_team": game["away_team_code"],
                "home_team": game["home_team_code"],
                "away_team_name": game["away_team_name"],
                "home_team_name": game["home_team_name"],
                "home_win_probability": prediction["home_win_probability"],
                "away_win_probability": prediction["away_win_probability"],
                "last_10_win_pct_diff": features["last_10_win_pct_diff"],
                "avg_points_last_10_diff": features["avg_points_last_10_diff"],
                "avg_plus_minus_last_10_diff": features["avg_plus_minus_last_10_diff"],
                "home_season_win_pct": features["home_season_win_pct"],
                "away_season_win_pct": features["away_season_win_pct"],
                "season_win_pct_diff": features["season_win_pct_diff"],
                "season_avg_points_diff": features["season_avg_points_diff"],
                "season_avg_plus_minus_diff": features["season_avg_plus_minus_diff"],
            }
        )

    return pd.DataFrame(rows)
