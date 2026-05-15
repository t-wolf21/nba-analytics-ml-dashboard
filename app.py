from datetime import date

import pandas as pd
import streamlit as st

from src.models.predict import MODEL_PATH, predict_games_for_date


st.set_page_config(
    page_title="NBA Matchup Dashboard",
    page_icon=":basketball:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_predictions(game_date: date, season: str) -> pd.DataFrame:
    return predict_games_for_date(game_date=game_date, season=season)


def format_probability(value: float) -> str:
    return f"{value:.1%}"


def format_matchup(row: pd.Series) -> str:
    return f"{row['away_team']} @ {row['home_team']}"


st.title("NBA Matchup Dashboard")
st.caption("Daily NBA game predictions from a baseline logistic regression model.")

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run `python -m src.models.train_model` first."
    )
    st.stop()

control_col1, control_col2 = st.columns([1, 1])

with control_col1:
    selected_date = st.date_input("Game date", value=date.today())

with control_col2:
    selected_season = st.selectbox(
        "Season",
        options=["2025-26"],
        index=0,
    )

with st.spinner("Loading games and predictions..."):
    try:
        predictions = load_predictions(selected_date, selected_season)
    except Exception as error:
        st.error(f"Could not load predictions: {error}")
        st.stop()

if predictions.empty:
    st.info("No games found for this date.")
    st.stop()

predictions = predictions.copy()
predictions["matchup"] = predictions.apply(format_matchup, axis=1)
predictions["home_win_pct"] = predictions["home_win_probability"].map(format_probability)
predictions["away_win_pct"] = predictions["away_win_probability"].map(format_probability)

st.subheader("Games")

display_columns = [
    "game_date",
    "matchup",
    "status",
    "home_win_pct",
    "away_win_pct",
    "last_10_win_pct_diff",
    "avg_points_last_10_diff",
    "avg_plus_minus_last_10_diff",
]

st.dataframe(
    predictions[display_columns],
    column_config={
        "game_date": "Date",
        "matchup": "Matchup",
        "status": "Status",
        "home_win_pct": "Home win",
        "away_win_pct": "Away win",
        "last_10_win_pct_diff": st.column_config.NumberColumn(
            "Last 10 win pct diff",
            format="%.1f",
        ),
        "avg_points_last_10_diff": st.column_config.NumberColumn(
            "Avg points diff",
            format="%.1f",
        ),
        "avg_plus_minus_last_10_diff": st.column_config.NumberColumn(
            "Avg plus-minus diff",
            format="%.1f",
        ),
    },
    use_container_width=True,
    hide_index=True,
)

selected_matchup = st.selectbox(
    "Matchup details",
    predictions["matchup"].tolist(),
)

selected_game = predictions[predictions["matchup"] == selected_matchup].iloc[0]

st.subheader(selected_matchup)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Status", selected_game["status"])
metric_col2.metric(
    f"{selected_game['home_team']} win probability",
    format_probability(selected_game["home_win_probability"]),
)
metric_col3.metric(
    f"{selected_game['away_team']} win probability",
    format_probability(selected_game["away_win_probability"]),
)

st.progress(
    float(selected_game["home_win_probability"]),
    text=f"{selected_game['home_team']}: {format_probability(selected_game['home_win_probability'])}",
)

feature_details = pd.DataFrame(
    [
        {
            "feature": "Last 10 win pct diff",
            "value": selected_game["last_10_win_pct_diff"],
        },
        {
            "feature": "Avg points last 10 diff",
            "value": selected_game["avg_points_last_10_diff"],
        },
        {
            "feature": "Avg plus-minus last 10 diff",
            "value": selected_game["avg_plus_minus_last_10_diff"],
        },
    ]
)

st.subheader("Model Features")
st.dataframe(
    feature_details,
    column_config={
        "feature": "Feature",
        "value": st.column_config.NumberColumn("Value", format="%.1f"),
    },
    use_container_width=True,
    hide_index=True,
)
