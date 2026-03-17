"""
Extract raw game data and cache to disk.

Supports two backends controlled by config.DATA_SOURCE:

    "sportsdataverse" (default)
        Downloads pre-built parquet files from the sportsdataverse GitHub
        releases. Each season loads in under a second — all 11 seasons in
        about 3-4 seconds total.

    "cbbpy"
        Scrapes ESPN day-by-day via CBBpy. Much slower (hours per season)
        but works as a fallback.

Both backends produce the same two-DataFrame interface consumed by
transform.build_fact_table():
    - game_info: game metadata (home/away, date, neutral site, etc.)
    - boxscore:  team-level stats per game (2 rows per game)
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────

def _cache_path(season: int, kind: str) -> str:
    """Return the file path for a cached CSV."""
    return os.path.join(config.RAW_DIR, f"{kind}_{season}.csv")


def _is_cached(season: int, kind: str) -> bool:
    path = _cache_path(season, kind)
    return os.path.exists(path) and os.path.getsize(path) > 0


# ── sportsdataverse backend ─────────────────────────────────────────────

def _sdv_extract_season(season: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download team box scores and schedule from sportsdataverse parquet
    files, then reshape into the (game_info, boxscore) pair expected by
    build_fact_table().
    """
    os.makedirs(config.RAW_DIR, exist_ok=True)

    info_path = _cache_path(season, "game_info")
    box_path = _cache_path(season, "boxscore")

    # Return cached CSVs if available.
    if not force and _is_cached(season, "game_info") and _is_cached(season, "boxscore"):
        log.info(f"Loading cached sportsdataverse data for {season}")
        return pd.read_csv(info_path), pd.read_csv(box_path)

    log.info(f"Downloading sportsdataverse parquet files for {season}...")

    base = config.SDV_BASE_URL
    team_box_url = f"{base}/{config.SDV_TEAM_BOX_TAG}/team_box_{season}.parquet"
    schedule_url = f"{base}/{config.SDV_SCHEDULE_TAG}/mbb_schedule_{season}.parquet"

    team_box = pd.read_parquet(team_box_url)
    schedule = pd.read_parquet(schedule_url)

    log.info(f"  team_box: {len(team_box)} rows, schedule: {len(schedule)} rows")

    # ── Build boxscore DataFrame ────────────────────────────────────
    # sportsdataverse team_box already has team-level aggregated stats.
    # We derive 2pm/2pa from fg - 3pt since the source doesn't split them.
    box = pd.DataFrame({
        "game_id":  team_box["game_id"].astype(str),
        "team":     team_box["team_location"],
        "fgm":      team_box["field_goals_made"],
        "fga":      team_box["field_goals_attempted"],
        "3pm":      team_box["three_point_field_goals_made"],
        "3pa":      team_box["three_point_field_goals_attempted"],
        "ftm":      team_box["free_throws_made"],
        "fta":      team_box["free_throws_attempted"],
        "oreb":     team_box["offensive_rebounds"],
        "dreb":     team_box["defensive_rebounds"],
        "ast":      team_box["assists"],
        "stl":      team_box["steals"],
        "blk":      team_box["blocks"],
        "to":       team_box["total_turnovers"],
        "pts":      team_box["team_score"],
    })
    box["2pm"] = box["fgm"] - box["3pm"]
    box["2pa"] = box["fga"] - box["3pa"]

    # ── Build game_info DataFrame ───────────────────────────────────
    # The schedule table has one row per game with home/away details.
    # We need: game_id, home_team, away_team, home_win, is_conference,
    #          is_neutral, is_postseason, tournament, game_day.

    # The schedule uses 'id' as its primary game key in some seasons and
    # 'game_id' in others. We also need home/away display names.
    sched = schedule.copy()

    # Resolve game ID column — prefer 'game_id', fall back to 'id'.
    if "game_id" in sched.columns:
        sched["_gid"] = sched["game_id"].astype(str)
    else:
        sched["_gid"] = sched["id"].astype(str)

    info = pd.DataFrame({
        "game_id":        sched["_gid"],
        "home_team":      sched.get("home_location", sched.get("home_display_name", pd.Series())),
        "away_team":      sched.get("away_location", sched.get("away_display_name", pd.Series())),
        "home_win":       sched.get("home_winner", pd.Series(dtype=bool)),
        "is_neutral":     sched.get("neutral_site", pd.Series(False, index=sched.index)),
        "game_day":       sched.get("game_date", sched.get("start_date", pd.Series())),
    })

    # Conference game flag.
    if "conference_competition" in sched.columns:
        info["is_conference"] = sched["conference_competition"].fillna(False).astype(bool)
    elif "groups_is_conference" in sched.columns:
        info["is_conference"] = sched["groups_is_conference"].fillna(False).astype(bool)
    else:
        info["is_conference"] = False

    # Postseason flag from season_type (3 = postseason in ESPN data).
    if "season_type" in sched.columns:
        info["is_postseason"] = sched["season_type"] == 3
    else:
        info["is_postseason"] = False

    # Tournament name (e.g. "NCAA Tournament - First Round").
    if "notes_headline" in sched.columns:
        info["tournament"] = sched["notes_headline"].fillna("")
    else:
        info["tournament"] = ""

    # Cache to disk.
    info.to_csv(info_path, index=False)
    box.to_csv(box_path, index=False)
    log.info(f"Cached sportsdataverse data for {season} to {config.RAW_DIR}")

    return info, box


# ── cbbpy backend ────────────────────────────────────────────────────────

def _progress_path(season: int) -> str:
    return os.path.join(config.RAW_DIR, f".progress_{season}.txt")


def _load_last_completed_date(season: int) -> str | None:
    path = _progress_path(season)
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return None


def _save_last_completed_date(season: int, date_str: str) -> None:
    with open(_progress_path(season), "w") as f:
        f.write(date_str)


def _append_to_csv(path: str, df: pd.DataFrame) -> None:
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def _cbbpy_extract_season(season: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull game info and boxscores for a single season via CBBpy,
    saving incrementally day-by-day.
    """
    import cbbpy.mens_scraper as cbb

    os.makedirs(config.RAW_DIR, exist_ok=True)

    info_path = _cache_path(season, "game_info")
    box_path = _cache_path(season, "boxscore")
    progress_file = _progress_path(season)

    if (
        not force
        and _is_cached(season, "game_info")
        and _is_cached(season, "boxscore")
        and os.path.exists(progress_file)
        and _load_last_completed_date(season) == "DONE"
    ):
        log.info(f"Loading cached data for {season}")
        return pd.read_csv(info_path), pd.read_csv(box_path)

    if force:
        for p in (info_path, box_path, progress_file):
            if os.path.exists(p):
                os.remove(p)

    season_start = datetime(season - 1, 11, 1)
    season_end = min(datetime(season, 5, 1), datetime.today())

    last_done = _load_last_completed_date(season)
    if last_done and last_done != "DONE":
        resume_from = datetime.strptime(last_done, "%Y-%m-%d") + timedelta(days=1)
        log.info(f"Resuming season {season} from {resume_from.date()} (after {last_done})")
    else:
        resume_from = season_start

    if resume_from <= season_end:
        log.info(f"Downloading season {season}: {resume_from.date()} → {season_end.date()}")
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
            _save_last_completed_date(season, "DONE")
            log.info(f"Finished downloading season {season}")

    info_df = pd.read_csv(info_path) if _is_cached(season, "game_info") else pd.DataFrame()
    box_df = pd.read_csv(box_path) if _is_cached(season, "boxscore") else pd.DataFrame()
    log.info(f"Season {season}: {len(info_df)} games, {len(box_df)} boxscore rows")

    return info_df, box_df


# ── public API ───────────────────────────────────────────────────────────

def extract_season(season: int, force: bool = False, source: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull game info and boxscores for a single season.

    Args:
        season: The four-digit season year (e.g. 2026 for 2025-26).
        force: If True, wipe cached files and re-download.
        source: Override config.DATA_SOURCE for this call.

    Returns:
        (game_info_df, boxscore_df)
    """
    source = source or config.DATA_SOURCE

    if source == "sportsdataverse":
        return _sdv_extract_season(season, force=force)
    elif source == "cbbpy":
        return _cbbpy_extract_season(season, force=force)
    else:
        raise ValueError(f"Unknown DATA_SOURCE: {source!r}. Use 'sportsdataverse' or 'cbbpy'.")


def extract_all_seasons(
    seasons: list[int] | None = None, force: bool = False, source: str | None = None,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Extract data for multiple seasons.

    Args:
        seasons: List of season years. Defaults to current + historical.
        force: If True, re-download everything.
        source: Override config.DATA_SOURCE for this call.

    Returns:
        Dict mapping season year to (game_info_df, boxscore_df).
    """
    if seasons is None:
        seasons = sorted(set([config.CURRENT_SEASON] + config.HISTORICAL_SEASONS))

    results = {}
    for season in seasons:
        results[season] = extract_season(season, force=force, source=source)

    return results


if __name__ == "__main__":
    extract_season(config.CURRENT_SEASON)
