from datetime import date

from src.data.nba_data_client import NBADataClient
from src.features.team_features import calculate_last_10_team_features

client = NBADataClient()
lakers_games = client.fetch_team_games(1610612747)
for column in lakers_games.columns:
    print(column)
features = calculate_last_10_team_features(
    lakers_games,
    date(2026, 3, 3),
)
print(features)

