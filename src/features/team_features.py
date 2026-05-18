import pandas as pd

def get_games_before_date(team_games: pd.DataFrame, game_date) -> pd.DataFrame:
    games = team_games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date
    return games[games["GAME_DATE"] < game_date].copy()

def get_last_n_games_before_date(team_games: pd.DataFrame, game_date, n: int = 10) -> pd.DataFrame:
    games = team_games.copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date

    previous_games = games[games["GAME_DATE"] < game_date]
    previous_games = previous_games.sort_values("GAME_DATE", ascending=False)

    return previous_games.head(n).reset_index(drop=True)

def calculate_team_features(team_games: pd.DataFrame, game_date) -> dict:
    last_10 = get_last_n_games_before_date(team_games, game_date, n=10)
    previous_games = get_games_before_date(team_games, game_date)

    if previous_games.empty:
        rest_days = pd.NA
        is_back_to_back = pd.NA
    else:
        previous_games = previous_games.sort_values("GAME_DATE", ascending=False)
        last_game_date = previous_games.iloc[0]["GAME_DATE"]
        rest_days = (game_date - last_game_date).days
        is_back_to_back = int(rest_days == 1)


    return {
        "last_10_games_played": len(last_10),
        "last_10_win_pct": (last_10["WL"] == "W").mean(),
        "last_10_avg_points": last_10["PTS"].mean(),
        "last_10_avg_plus_minus": last_10["PLUS_MINUS"].mean(),

        "season_games_played": len(previous_games),
        "season_win_pct": (previous_games["WL"] == "W").mean(),
        "season_avg_points": previous_games["PTS"].mean(),
        "season_avg_plus_minus": previous_games["PLUS_MINUS"].mean(),

        "rest_days": rest_days,
        "is_back_to_back": is_back_to_back,
    }