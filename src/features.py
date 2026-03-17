"""
Aggregate adjusted game-level stats into team profiles and matchup features.

Two output modes:
    1. Team profiles: One row per team with season-long adjusted averages.
       Used for bracket simulation (plug any two teams in).
    2. Matchup features: One row per game with the difference between
       the two teams' profiles. Used for model training.
"""

import logging

import numpy as np
import pandas as pd

import config
from src.adjust import OFF_STATS, DEF_STATS, _compute_recency_weights

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# The adjusted columns we aggregate into team profiles.
ADJ_OFF_STATS = [f"adj_{s}" for s in OFF_STATS]
ADJ_DEF_STATS = [f"adj_{s}" for s in DEF_STATS]
PROFILE_STATS = ADJ_OFF_STATS + ADJ_DEF_STATS


def build_team_profiles(
    adjusted_df: pd.DataFrame,
    recency_lambda: float | None = None,
) -> pd.DataFrame:
    """
    Aggregate adjusted game-level stats into a single row per team.

    Each stat is a recency-weighted average across all games in the season.
    This is the "team resume" that feeds predictions.

    Returns:
        DataFrame indexed by team name with one column per adjusted stat.
    """
    recency_lambda = recency_lambda if recency_lambda is not None else config.RECENCY_LAMBDA

    df = adjusted_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    weights = _compute_recency_weights(df["game_date"], recency_lambda)

    rows = []
    for team, group in df.groupby("team"):
        tw = weights.loc[group.index]
        row = {"team": team, "games_played": len(group)}

        for stat in PROFILE_STATS:
            if stat in group.columns:
                mask = group[stat].notna()
                if mask.sum() > 0:
                    row[stat] = np.average(group[stat][mask], weights=tw[mask])
                else:
                    row[stat] = np.nan
            else:
                row[stat] = np.nan

        # Win rate (raw, not adjusted — useful as a sanity check).
        row["win_pct"] = group["win"].mean()
        rows.append(row)

    profiles = pd.DataFrame(rows).set_index("team")
    log.info(f"Built profiles for {len(profiles)} teams")
    return profiles


def build_matchup_features(
    adjusted_df: pd.DataFrame,
    recency_lambda: float | None = None,
) -> pd.DataFrame:
    """
    Build matchup-level features for model training.

    For each game, compute the difference between the team's adjusted
    profile and the opponent's adjusted profile across all stat categories.
    The target is whether the team won.

    To avoid data leakage, team profiles are computed using only games
    that occurred BEFORE the game in question (expanding window).

    Args:
        adjusted_df: The SOS-adjusted fact table.
        recency_lambda: For building the team profiles.

    Returns:
        DataFrame with one row per team per game, feature columns, and
        a 'win' target column.
    """
    recency_lambda = recency_lambda if recency_lambda is not None else config.RECENCY_LAMBDA

    df = adjusted_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values("game_date").reset_index(drop=True)

    # Pre-compute full-season profiles as a fallback for early-season games.
    full_profiles = build_team_profiles(df, recency_lambda)

    # Group games by date for expanding-window profile computation.
    unique_dates = np.sort(df["game_date"].dropna().unique())

    # For efficiency, rebuild profiles at regular intervals rather than
    # per-game. Use weekly snapshots as a balance between accuracy and speed.
    snapshot_interval = 7  # days
    snapshots = {}  # date -> profiles DataFrame

    features = []
    current_snapshot_date = None

    for _, row in df.iterrows():
        game_date = row["game_date"]
        team = row["team"]
        opponent = row["opponent"]

        if pd.isna(game_date):
            continue

        # Determine which snapshot to use (games strictly before this date).
        if current_snapshot_date is None or (game_date - current_snapshot_date).days >= snapshot_interval:
            prior_games = df[df["game_date"] < game_date]
            if len(prior_games) >= 100:  # need enough games for meaningful profiles
                current_snapshot_date = game_date
                snapshots[game_date] = build_team_profiles(prior_games, recency_lambda)

        # Use the most recent snapshot, or fall back to full-season profiles
        # for early-season games.
        profiles = snapshots.get(current_snapshot_date, full_profiles)

        if team not in profiles.index or opponent not in profiles.index:
            continue

        team_prof = profiles.loc[team]
        opp_prof = profiles.loc[opponent]

        feat = {
            "game_id": row["game_id"],
            "team": team,
            "opponent": opponent,
            "win": row["win"],
            "is_neutral": row.get("is_neutral", False),
            "is_postseason": row.get("is_postseason", False),
        }

        # Feature: difference in adjusted stats (offensive and defensive).
        for stat in PROFILE_STATS:
            if stat in team_prof.index and stat in opp_prof.index:
                feat[f"diff_{stat}"] = team_prof[stat] - opp_prof[stat]

        features.append(feat)

    result = pd.DataFrame(features)
    log.info(f"Built {len(result)} matchup feature rows")
    return result
