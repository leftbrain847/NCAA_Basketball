"""
Extract raw game data from CBBpy and cache to disk.

CBBpy scrapes ESPN, which is slow and rate-sensitive. This module pulls
game info and boxscores for a season incrementally (one day at a time),
appending to CSVs after each day so that a crash only loses one day of
progress instead of an entire season.
"""

import os
import logging
from datetime import datetime, timedelta

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


def _progress_path(season: int) -> str:
    """Return the path for the incremental progress marker."""
    return os.path.join(config.RAW_DIR, f".progress_{season}.txt")


def _load_last_completed_date(season: int) -> str | None:
    """Read the last fully-scraped date from the progress file."""
    path = _progress_path(season)
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def _save_last_completed_date(season: int, date_str: str) -> None:
    """Write the last fully-scraped date to the progress file."""
    with open(_progress_path(season), "w") as f:
        f.write(date_str)


def _append_to_csv(path: str, df: pd.DataFrame) -> None:
    """Append rows to a CSV, writing the header only if the file is new."""
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def extract_season(season: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull game info and boxscores for a single season, saving incrementally.

    Scrapes one day at a time via ``cbb.get_games_range`` and appends each
    day's results to the cache CSVs.  On restart, resumes from the last
    fully-completed date so that only one day of work is lost on failure.

    Args:
        season: The four-digit season year (e.g. 2026 for the 2025-26 season).
        force: If True, wipe cached files and re-download from scratch.

    Returns:
        (game_info_df, boxscore_df) — raw DataFrames for the full season.
    """
    os.makedirs(config.RAW_DIR, exist_ok=True)

    info_path = _cache_path(season, "game_info")
    box_path = _cache_path(season, "boxscore")
    progress_file = _progress_path(season)

    # If fully cached and not forcing, just load.
    if (
        not force
        and _is_cached(season, "game_info")
        and _is_cached(season, "boxscore")
        and os.path.exists(progress_file)
        and _load_last_completed_date(season) == "DONE"
    ):
        log.info(f"Loading cached data for {season}")
        info_df = pd.read_csv(info_path)
        box_df = pd.read_csv(box_path)
        return info_df, box_df

    # Wipe on force.
    if force:
        for p in (info_path, box_path, progress_file):
            if os.path.exists(p):
                os.remove(p)

    # Determine season date window.
    season_start = datetime(season - 1, 11, 1)
    season_end = min(datetime(season, 5, 1), datetime.today())

    # Resume from last completed date if available.
    last_done = _load_last_completed_date(season)
    if last_done and last_done != "DONE":
        resume_from = datetime.strptime(last_done, "%Y-%m-%d") + timedelta(days=1)
        log.info(f"Resuming season {season} from {resume_from.date()} (after {last_done})")
    else:
        resume_from = season_start

    if resume_from <= season_end:
        log.info(
            f"Downloading season {season}: {resume_from.date()} → {season_end.date()}"
        )

        current = resume_from
        while current <= season_end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                result = cbb.get_games_range(date_str, date_str, info=True, box=True, pbp=False)
            except Exception:
                log.exception(f"Failed on {date_str}, will resume here next run")
                break

            if result and len(result) >= 2:
                day_info, day_box, _ = result
                if len(day_info) > 0:
                    _append_to_csv(info_path, day_info)
                if len(day_box) > 0:
                    _append_to_csv(box_path, day_box)

            _save_last_completed_date(season, date_str)
            current += timedelta(days=1)
        else:
            # Loop completed without break — season fully scraped.
            _save_last_completed_date(season, "DONE")
            log.info(f"Finished downloading season {season}")

    # Load the full accumulated CSVs.
    info_df = pd.read_csv(info_path) if _is_cached(season, "game_info") else pd.DataFrame()
    box_df = pd.read_csv(box_path) if _is_cached(season, "boxscore") else pd.DataFrame()
    log.info(f"Season {season}: {len(info_df)} games, {len(box_df)} boxscore rows")

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
