from datetime import date

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="NBA Matchup Dashboard",
    page_icon="🏀",
    layout="wide",
)


MATCHUPS = [
    {
        "date": date(2026, 5, 12),
        "away_team": "Boston Celtics",
        "home_team": "New York Knicks",
        "home_win_probability": 0.47,
    },
    {
        "date": date(2026, 5, 12),
        "away_team": "Denver Nuggets",
        "home_team": "Los Angeles Lakers",
        "home_win_probability": 0.54,
    },
    {
        "date": date(2026, 5, 13),
        "away_team": "Minnesota Timberwolves",
        "home_team": "Oklahoma City Thunder",
        "home_win_probability": 0.58,
    },
]


def format_matchup(matchup: dict) -> str:
    return (
        f"{matchup['date'].strftime('%Y-%m-%d')} | "
        f"{matchup['away_team']} @ {matchup['home_team']}"
    )


st.title("NBA Matchup Dashboard")
st.caption("Exploring NBA team performance and building a baseline game outcome prediction model.")

matchup = st.selectbox(
    "Matchup",
    MATCHUPS,
    format_func=format_matchup,
)

away_team = matchup["away_team"]
home_team = matchup["home_team"]
home_probability = matchup["home_win_probability"]
away_probability = 1 - home_probability

st.subheader(f"{away_team} @ {home_team}")

col1, col2, col3 = st.columns(3)
col1.metric("Date", matchup["date"].strftime("%Y-%m-%d"))
col2.metric(f"{home_team} win probability", f"{home_probability:.0%}")
col3.metric(f"{away_team} win probability", f"{away_probability:.0%}")

st.progress(home_probability, text=f"{home_team}: {home_probability:.0%}")

st.divider()

st.subheader("Available Matchups")
matchups_df = pd.DataFrame(MATCHUPS)
matchups_df["matchup"] = (
    matchups_df["away_team"] + " @ " + matchups_df["home_team"]
)
matchups_df["home_win_probability"] = matchups_df["home_win_probability"].map(
    lambda value: f"{value:.0%}"
)

st.dataframe(
    matchups_df[
        ["date", "matchup", "home_team", "home_win_probability"]
    ],
    use_container_width=True,
    hide_index=True,
)
