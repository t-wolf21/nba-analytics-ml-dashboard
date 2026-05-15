from nba_api.stats.endpoints import (
    leaguedashteamstats,
    leaguegamefinder,
    scheduleleaguev2
)
from datetime import date
import pandas as pd
import time

class NBADataClient:
    """
    Client for NBA data from the nba_api library.
    """

    def __init__(self, season ="2025-26", timeout: int = 60, max_retries: int = 3):
        self.season = season
        self.timeout = timeout
        self.max_retries = max_retries

    def _run_with_retries(self, request_func):
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return request_func()
            except Exception as error:
                last_error = error
                if attempt == self.max_retries:
                    break

                time.sleep(attempt * 2)

        raise last_error

    def fetch_schedule(self) -> pd.DataFrame:
        return self._run_with_retries(
            lambda: scheduleleaguev2.ScheduleLeagueV2(
                season=self.season,
                timeout=self.timeout,
            ).get_data_frames()[0]
        )

    def fetch_schedule_for_date(self, game_date: date) -> pd.DataFrame:
        schedule = self.fetch_schedule().copy()
        schedule["gameDate"] = pd.to_datetime(schedule["gameDate"]).dt.date
        games = schedule[schedule["gameDate"] == game_date]
        return self.clean_schedule(games).reset_index(drop=True)

    def fetch_team_games(self, team_id: int) -> pd.DataFrame:
        return self._run_with_retries(
            lambda: leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T",
                team_id_nullable=team_id,
                season_nullable=self.season,
                timeout=self.timeout,
            ).get_data_frames()[0]
        )

    def fetch_team_gamestats(self, team_id: str) -> pd.DataFrame:
        return self._run_with_retries(
            lambda: leaguedashteamstats.LeagueDashTeamStats(
                team_id_nullable=team_id,
                season=self.season,
                timeout=self.timeout,
            ).get_data_frames()[0]
        )

    def clean_schedule(self, schedule: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "gameDate",
            "gameId",
            "gameStatusText",
            "arenaName",
            "arenaCity",
            "homeTeam_teamId",
            "homeTeam_teamCity",
            "homeTeam_teamName",
            "homeTeam_teamTricode",
            "homeTeam_wins",
            "homeTeam_losses",
            "awayTeam_teamId",
            "awayTeam_teamCity",
            "awayTeam_teamName",
            "awayTeam_teamTricode",
            "awayTeam_wins",
            "awayTeam_losses",
        ]

        return schedule[columns].rename(
            columns={
                "gameDate": "game_date",
                "gameId": "game_id",
                "gameStatusText": "status",
                "homeTeam_teamId": "home_team_id",
                "homeTeam_teamCity": "home_team_city",
                "homeTeam_teamName": "home_team_name",
                "homeTeam_teamTricode": "home_team_code",
                "homeTeam_wins": "home_wins",
                "homeTeam_losses": "home_losses",
                "awayTeam_teamId": "away_team_id",
                "awayTeam_teamCity": "away_team_city",
                "awayTeam_teamName": "away_team_name",
                "awayTeam_teamTricode": "away_team_code",
                "awayTeam_wins": "away_wins",
                "awayTeam_losses": "away_losses",
            }
        )
