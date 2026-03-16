"""
Transform raw CBBpy data into the core fact table.

The fact table has one row per team per game. Each row contains the team's
offensive stats and the opponent's offensive stats (i.e. what the team allowed
defensively). This gives 2 rows per game — one from each team's perspective.

CBBpy boxscore columns (after its internal parsing):
    game_id, team, player, player_id, position, starter, min,
    fgm, fga, 2pm, 2pa, 3pm, 3pa, ftm, fta,
    oreb, dreb, reb, ast, stl, blk, to, pf, pts

CBBpy game info columns (subset we use):
    game_id, home_team, away_team, home_id, away_id,
    home_win, is_conference, is_neutral, is_postseason,
    tournament, game_day
"""

import os
import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Stats we aggregate from the player-level boxscore to the team level.
# These are the columns we sum across all players on a team for a game.
AGG_STATS = [
    "fgm", "fga", "2pm", "2pa", "3pm", "3pa",
    "ftm", "fta", "oreb", "dreb", "ast", "to", "pts",
]

# The stat categories that get SOS-adjusted (rates and volume).
# Keys are column names in the fact table; values are (made, attempted)
# tuples for computing percentages, or None for counting stats.
STAT_CATEGORIES = {
    "fg_pct_2": ("off_2pm", "off_2pa"),
    "fg_pct_3": ("off_3pm", "off_3pa"),
    "ft_pct":   ("off_ftm", "off_fta"),
    "off_2pa":  None,
    "off_3pa":  None,
    "off_fta":  None,
    "off_oreb": None,
    "off_dreb": None,
    "off_ast":  None,
    "off_to":   None,
}


def _aggregate_boxscore(box_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level boxscore rows to team-level per game.

    Returns one row per team per game with summed counting stats.
    """
    # Filter to real player rows (exclude team totals if present).
    players = box_df[box_df["player"].notna() & (box_df["player"] != "")].copy()

    team_stats = (
        players
        .groupby(["game_id", "team"])[AGG_STATS]
        .sum()
        .reset_index()
    )
    return team_stats


def _build_game_rows(team_stats: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join each team's stats with their opponent's stats for the same game,
    and merge in game metadata. Produces the 2-rows-per-game fact table.
    """
    # Self-join: for each (game_id, team), find the other team in the same game.
    merged = team_stats.merge(
        team_stats,
        on="game_id",
        suffixes=("", "_opp"),
    )
    # Drop self-joins (team matched with itself).
    merged = merged[merged["team"] != merged["team_opp"]].copy()

    # Prefix offensive and defensive columns for clarity.
    rename_map = {}
    for stat in AGG_STATS:
        rename_map[stat] = f"off_{stat}"
        rename_map[f"{stat}_opp"] = f"def_{stat}"
    merged.rename(columns=rename_map, inplace=True)
    merged.rename(columns={"team_opp": "opponent"}, inplace=True)

    # Compute shooting percentages.
    for prefix in ("off", "def"):
        merged[f"{prefix}_fg_pct_2"] = np.where(
            merged[f"{prefix}_2pa"] > 0,
            merged[f"{prefix}_2pm"] / merged[f"{prefix}_2pa"],
            np.nan,
        )
        merged[f"{prefix}_fg_pct_3"] = np.where(
            merged[f"{prefix}_3pa"] > 0,
            merged[f"{prefix}_3pm"] / merged[f"{prefix}_3pa"],
            np.nan,
        )
        merged[f"{prefix}_ft_pct"] = np.where(
            merged[f"{prefix}_fta"] > 0,
            merged[f"{prefix}_ftm"] / merged[f"{prefix}_fta"],
            np.nan,
        )

    # Estimate possessions (standard four-factor approximation).
    for prefix in ("off", "def"):
        merged[f"{prefix}_poss"] = (
            merged[f"{prefix}_fga"]
            - merged[f"{prefix}_oreb"]
            + merged[f"{prefix}_to"]
            + 0.475 * merged[f"{prefix}_fta"]
        )

    # Merge game metadata.
    info_cols = [
        "game_id", "home_team", "away_team",
        "home_win", "is_conference", "is_neutral", "is_postseason",
        "tournament", "game_day",
    ]
    available_cols = [c for c in info_cols if c in info_df.columns]
    info_subset = info_df[available_cols].drop_duplicates(subset=["game_id"])
    merged = merged.merge(info_subset, on="game_id", how="left")

    # Derive per-row fields from the game metadata.
    merged["is_home"] = merged["team"] == merged["home_team"]
    merged["win"] = np.where(
        merged["is_home"],
        merged["home_win"],
        ~merged["home_win"].astype(bool),
    )
    merged["win"] = merged["win"].astype(bool)

    # Parse game date.
    merged["game_date"] = pd.to_datetime(merged["game_day"], errors="coerce")

    # Drop intermediate columns.
    merged.drop(columns=["home_team", "away_team", "home_win", "game_day"], inplace=True, errors="ignore")

    return merged


def build_fact_table(
    info_df: pd.DataFrame,
    box_df: pd.DataFrame,
    season: int | None = None,
) -> pd.DataFrame:
    """
    Build the core fact table from raw CBBpy data.

    Args:
        info_df: Raw game info DataFrame from extract.
        box_df: Raw boxscore DataFrame from extract.
        season: Optional season label to attach.

    Returns:
        DataFrame with one row per team per game, containing offensive stats,
        defensive stats (what the opponent did), and game metadata.
    """
    log.info("Aggregating boxscores to team level...")
    team_stats = _aggregate_boxscore(box_df)

    log.info("Building fact table...")
    fact = _build_game_rows(team_stats, info_df)

    if season is not None:
        fact["season"] = season

    # Filter to completed games with valid stats.
    fact = fact.dropna(subset=["off_pts", "def_pts"])

    log.info(f"Fact table: {len(fact)} rows ({fact['game_id'].nunique()} games)")
    return fact


def save_fact_table(fact_df: pd.DataFrame, season: int) -> str:
    """Save the fact table to the processed data directory."""
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    path = os.path.join(config.PROCESSED_DIR, f"fact_table_{season}.csv")
    fact_df.to_csv(path, index=False)
    log.info(f"Saved fact table to {path}")
    return path
