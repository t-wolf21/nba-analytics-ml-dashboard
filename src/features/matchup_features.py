def build_matchup_features(home_features: dict, away_features: dict) -> dict:
    return {
        "home_last_10_win_pct": home_features["last_10_win_pct"],
        "away_last_10_win_pct": away_features["last_10_win_pct"],
        "last_10_win_pct_diff": home_features["last_10_win_pct"] - away_features["last_10_win_pct"],
        "home_avg_points_last_10": home_features["last_10_avg_points"],
        "away_avg_points_last_10": away_features["last_10_avg_points"],
        "avg_points_last_10_diff": home_features["last_10_avg_points"] - away_features["last_10_avg_points"],
        "home_avg_plus_minus_last_10": home_features["last_10_avg_plus_minus"],
        "away_avg_plus_minus_last_10": away_features["last_10_avg_plus_minus"],
        "avg_plus_minus_last_10_diff": (home_features["last_10_avg_plus_minus"] - away_features["last_10_avg_plus_minus"]),
    }
