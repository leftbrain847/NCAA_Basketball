"""
Transform raw extracted data into the core fact table.

The fact table has one row per team per game. Each row contains the team's
offensive stats and the opponent's offensive stats (i.e. what the team allowed
defensively). This gives 2 rows per game — one from each team's perspective.

Handles both data sources transparently:

    sportsdataverse:
        - Boxscores are already team-level (no player aggregation needed).
        - Team names are already location-format ("Duke", not "Duke Blue Devils").

    cbbpy:
        - Boxscores are player-level and need aggregation.
        - Team names use displayName and need normalization via the team map.
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
    "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "to", "pts",
]


def _build_name_map(season: int) -> dict[str, str]:
    """
    Build a mapping from CBBpy's displayName (e.g. "Duke Blue Devils")
    to location name (e.g. "Duke") using CBBpy's internal team map.

    Only used when DATA_SOURCE is "cbbpy". Returns an empty dict if
    the map is unavailable (sportsdataverse names are already normalized).
    """
    try:
        map_path = os.path.join(
            os.path.dirname(__import__("cbbpy").__file__),
            "utils", "mens_team_map.csv",
        )
        team_map = pd.read_csv(map_path)
    except Exception:
        log.warning("Could not load cbbpy team map — team names will not be normalized")
        return {}

    season_map = team_map[team_map["season"] == season]
    if season_map.empty:
        latest = team_map["season"].max()
        log.warning(f"No team map for season {season}, using {latest}")
        season_map = team_map[team_map["season"] == latest]

    return dict(zip(season_map["team"], season_map["location"]))


# The stat categories that get SOS-adjusted (rates and volume).
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


def _is_player_level(box_df: pd.DataFrame) -> bool:
    """Detect whether the boxscore is player-level (cbbpy) or team-level (sportsdataverse)."""
    return "player" in box_df.columns


def _aggregate_boxscore(box_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level boxscore rows to team-level per game.

    Returns one row per team per game with summed counting stats.
    Only needed for cbbpy data — sportsdataverse data is already team-level.
    """
    players = box_df[
        box_df["player"].notna()
        & (box_df["player"] != "")
        & (box_df["player"].str.upper() != "TEAM")
        & (box_df["starter"].notna())
    ].copy()

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
    _true_vals = {True, "True", "true", "TRUE", 1, 1.0}
    merged["home_win"] = merged["home_win"].apply(lambda x: x in _true_vals)
    merged["is_home"] = merged["team"] == merged["home_team"]
    merged["win"] = np.where(merged["is_home"], merged["home_win"], ~merged["home_win"])
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
    Build the core fact table from raw extracted data.

    Works with both sportsdataverse and cbbpy data. Automatically detects
    the format and adapts accordingly.

    Args:
        info_df: Raw game info DataFrame from extract.
        box_df: Raw boxscore DataFrame from extract.
        season: Optional season label to attach.

    Returns:
        DataFrame with one row per team per game, containing offensive stats,
        defensive stats (what the opponent did), and game metadata.
    """
    # Ensure game_id is string in both DataFrames for consistent joining.
    info_df = info_df.copy()
    box_df = box_df.copy()
    info_df["game_id"] = info_df["game_id"].astype(str)
    box_df["game_id"] = box_df["game_id"].astype(str)

    if _is_player_level(box_df):
        # cbbpy path: aggregate player rows to team level, then normalize names.
        log.info("Detected player-level boxscore (cbbpy) — aggregating...")
        team_stats = _aggregate_boxscore(box_df)

        use_season = season if season is not None else config.CURRENT_SEASON
        name_map = _build_name_map(use_season)
        if name_map:
            team_stats["team"] = team_stats["team"].map(name_map).fillna(team_stats["team"])
            for col in ["home_team", "away_team"]:
                if col in info_df.columns:
                    info_df[col] = info_df[col].map(name_map).fillna(info_df[col])
            mapped = sum(1 for v in team_stats["team"].unique() if v in name_map.values())
            log.info(f"Mapped {mapped}/{team_stats['team'].nunique()} team names to location format")
    else:
        # sportsdataverse path: already team-level with location names.
        log.info("Detected team-level boxscore (sportsdataverse) — no aggregation needed")
        team_stats = box_df[["game_id", "team"] + [c for c in AGG_STATS if c in box_df.columns]].copy()

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
