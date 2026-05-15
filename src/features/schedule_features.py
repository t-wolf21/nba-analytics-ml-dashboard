import pandas as pd

def add_record_features(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()

    games["home_games_played"] = games["home_wins"] + games["home_losses"]
    games["away_games_played"] = games["away_wins"] + games["away_losses"]

    games["home_win_pct"] = games["home_wins"] / games["home_games_played"].replace(0, pd.NA)
    games["away_win_pct"] = games["away_wins"] / games["away_games_played"].replace(0, pd.NA)

    games["win_pct_diff"] = games["home_win_pct"] - games["away_win_pct"]

    return games