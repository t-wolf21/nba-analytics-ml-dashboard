from datetime import date

from src.models.predict import predict_game

prediction = predict_game(
    home_team_id=1610612747,  # Lakers
    away_team_id=1610612744,  # Warriors
    game_date=date(2026, 3, 3),
)

print(prediction)