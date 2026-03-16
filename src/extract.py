"""
Extract raw game data from CBBpy and cache to disk.

CBBpy scrapes ESPN, which is slow and rate-sensitive. This module pulls
game info and boxscores for a season, caches the raw DataFrames as CSVs,
and skips re-downloading if cached files already exist.
"""

import os
import logging

import pandas as pd
import cbbpy.mens_scraper as cbb

import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _cache_path(season: int, kind: str) -> str:
    """Return the file path for a cached CSV."""
    return os.path.join(config.RAW_DIR, f"{kind}_{season}.csv")


def _is_cached(season: int, kind: str) -> bool:
    path = _cache_path(season, kind)
    return os.path.exists(path) and os.path.getsize(path) > 0


def extract_season(season: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull game info and boxscores for a single season.

    Args:
        season: The four-digit season year (e.g. 2026 for the 2025-26 season).
        force: If True, re-download even if cached files exist.

    Returns:
        (game_info_df, boxscore_df) — raw DataFrames as returned by CBBpy.
    """
    os.makedirs(config.RAW_DIR, exist_ok=True)

    info_cached = _is_cached(season, "game_info")
    box_cached = _is_cached(season, "boxscore")

    if info_cached and box_cached and not force:
        log.info(f"Loading cached data for {season}")
        info_df = pd.read_csv(_cache_path(season, "game_info"))
        box_df = pd.read_csv(_cache_path(season, "boxscore"))
        return info_df, box_df

    log.info(f"Downloading season {season} from CBBpy (this will take a while)...")
    info_df, box_df, _ = cbb.get_games_season(season, info=True, box=True, pbp=False)

    info_df.to_csv(_cache_path(season, "game_info"), index=False)
    box_df.to_csv(_cache_path(season, "boxscore"), index=False)
    log.info(f"Cached {len(info_df)} games and {len(box_df)} boxscore rows for {season}")

    return info_df, box_df


def extract_all_seasons(
    seasons: list[int] | None = None, force: bool = False
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Extract data for multiple seasons.

    Args:
        seasons: List of season years. Defaults to current + historical from config.
        force: If True, re-download everything.

    Returns:
        Dict mapping season year to (game_info_df, boxscore_df).
    """
    if seasons is None:
        seasons = sorted(set([config.CURRENT_SEASON] + config.HISTORICAL_SEASONS))

    results = {}
    for season in seasons:
        results[season] = extract_season(season, force=force)

    return results


if __name__ == "__main__":
    extract_season(config.CURRENT_SEASON)
