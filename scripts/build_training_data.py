from pathlib import Path

import pandas as pd

from src.data.nba_data_client import NBADataClient
from src.features.team_features import calculate_team_features
from src.features.matchup_features import build_matchup_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _season_cache_key(season: str) -> str:
    return season.replace("-", "_")


def fetch_schedule_cached(client: NBADataClient, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"schedule_{_season_cache_key(client.season)}.csv"

    if cache_path.exists():
        print(f"Loading cached schedule from {cache_path}...")
        return pd.read_csv(cache_path)

    print(f"Fetching schedule for {client.season}...")
    schedule = client.fetch_schedule()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(cache_path, index=False)

    return schedule


def fetch_team_games_cached(
    client: NBADataClient,
    team_id: int,
    memory_cache: dict[int, pd.DataFrame],
    cache_dir: Path,
) -> pd.DataFrame:
    if team_id in memory_cache:
        return memory_cache[team_id]

    cache_path = cache_dir / f"team_games_{_season_cache_key(client.season)}_{team_id}.csv"

    if cache_path.exists():
        print(f"Loading cached team games for {team_id}...")
        team_games = pd.read_csv(cache_path)
    else:
        print(f"Fetching team games for {team_id}...")
        team_games = client.fetch_team_games(team_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        team_games.to_csv(cache_path, index=False)

    memory_cache[team_id] = team_games
    return team_games


def build_training_data(season: str = "2025-26", max_games: int | None = None) -> pd.DataFrame:
    client = NBADataClient(season=season, timeout=20, max_retries=2)
    raw_cache_dir = PROJECT_ROOT / "data/raw"

    schedule = fetch_schedule_cached(client, raw_cache_dir)
    schedule["gameDate"] = pd.to_datetime(schedule["gameDate"]).dt.date

    finished_games = schedule[
        (schedule["gameStatusText"] == "Final")
        & schedule["homeTeam_score"].notna()
        & schedule["awayTeam_score"].notna()
    ].copy()

    finished_games = finished_games[
        (finished_games["homeTeam_teamId"] >= 1610612737)
        & (finished_games["homeTeam_teamId"] <= 1610612766)
        & (finished_games["awayTeam_teamId"] >= 1610612737)
        & (finished_games["awayTeam_teamId"] <= 1610612766)
        ].copy()

    if max_games is not None:
        finished_games = finished_games.head(max_games)

    print(f"Building training rows for {len(finished_games)} finished games...")

    rows = []
    team_games_cache = {}
    skipped_not_enough_history = 0

    for index, game in finished_games.iterrows():
        game_date = game["gameDate"]
        home_team_id = int(game["homeTeam_teamId"])
        away_team_id = int(game["awayTeam_teamId"])

        home_games = fetch_team_games_cached(
            client=client,
            team_id=home_team_id,
            memory_cache=team_games_cache,
            cache_dir=raw_cache_dir,
        )
        away_games = fetch_team_games_cached(
            client=client,
            team_id=away_team_id,
            memory_cache=team_games_cache,
            cache_dir=raw_cache_dir,
        )

        home_features = calculate_team_features(home_games, game_date)
        away_features = calculate_team_features(away_games, game_date)

        if (
            home_features["last_10_games_played"] < 10
            or away_features["last_10_games_played"] < 10
        ):
            skipped_not_enough_history += 1
            continue

        matchup_features = build_matchup_features(home_features, away_features)

        row = {
            "game_id": game["gameId"],
            "game_date": game_date,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_team_code": game["homeTeam_teamTricode"],
            "away_team_code": game["awayTeam_teamTricode"],
            "home_team_won": int(game["homeTeam_score"] > game["awayTeam_score"]),
            **matchup_features,
        }

        rows.append(row)

        if len(rows) % 25 == 0:
            print(f"Built {len(rows)} rows...")

    print(f"Skipped {skipped_not_enough_history} games with fewer than 10 previous games.")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    training_data = build_training_data(max_games=None)

    print(training_data.head())
    print(training_data.shape)

    output_path = PROJECT_ROOT / "data/processed/training_games.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(output_path, index=False)
