"""
Iterative Strength-of-Schedule adjustment.

For each stat category, every team's raw per-game numbers are adjusted by
the quality of their opponent in that specific category. Because opponent
quality is itself adjusted, this creates a circular dependency resolved
through iterative convergence (fixed-point iteration).

Both offensive and defensive stats are adjusted symmetrically:
    - Offensive stats are adjusted by opponent's defensive averages.
    - Defensive stats are adjusted by opponent's offensive averages.

The process:
    1. Compute each team's weighted average offensive and defensive stats
       (weighted by recency).
    2. For each game row, adjust the team's raw stats using the opponent's
       current adjusted averages.
    3. Recompute team averages from adjusted game stats.
    4. Repeat 2-3 until the max change between iterations is below the
       convergence threshold.

Supports both additive and multiplicative adjustment (set in config).
"""

import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Offensive stat columns to adjust. The corresponding def_ columns are
# derived automatically and adjusted symmetrically.
OFF_STATS = [
    "off_2pa", "off_2pm", "off_fg_pct_2",
    "off_3pa", "off_3pm", "off_fg_pct_3",
    "off_fta", "off_ftm", "off_ft_pct",
    "off_oreb", "off_dreb",
    "off_ast", "off_to",
]

# Defensive counterparts (what the team allowed).
DEF_STATS = [s.replace("off_", "def_") for s in OFF_STATS]

# All stats that go through the convergence loop.
ALL_STATS = OFF_STATS + DEF_STATS

# Kept as public API for features.py to reference.
ADJUSTABLE_STATS = OFF_STATS


def _compute_recency_weights(
    game_dates: pd.Series, recency_lambda: float
) -> pd.Series:
    """
    Compute exponential decay weights based on how many days before the
    most recent game each game occurred.

    Args:
        game_dates: Series of game dates.
        recency_lambda: Decay rate. Higher = more recency bias.

    Returns:
        Series of weights in [0, 1], with the most recent game weighted 1.0.
    """
    if recency_lambda == 0:
        return pd.Series(1.0, index=game_dates.index)

    max_date = game_dates.max()
    days_ago = (max_date - game_dates).dt.days.fillna(0).astype(float)
    weights = np.exp(-recency_lambda * days_ago)
    return weights


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted average, handling NaN values."""
    mask = values.notna()
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def _compute_team_averages(
    fact_df: pd.DataFrame,
    stats: list[str],
    weights: pd.Series,
) -> pd.DataFrame:
    """
    Compute recency-weighted team averages for each stat.

    Args:
        fact_df: Fact table with the columns listed in stats.
        stats: Column names to average.
        weights: Per-row recency weights (same index as fact_df).

    Returns:
        DataFrame indexed by team with one column per stat.
    """
    rows = []
    for team, group in fact_df.groupby("team"):
        team_weights = weights.loc[group.index]
        row = {"team": team}
        for stat in stats:
            if stat in group.columns:
                row[stat] = _weighted_mean(group[stat], team_weights)
            else:
                row[stat] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("team")


def _compute_league_averages(
    fact_df: pd.DataFrame, stats: list[str], weights: pd.Series
) -> dict[str, float]:
    """Compute league-wide weighted averages for each stat."""
    avgs = {}
    for stat in stats:
        if stat in fact_df.columns:
            avgs[stat] = _weighted_mean(fact_df[stat], weights)
    return avgs


def _adjust_game_stats(
    fact_df: pd.DataFrame,
    team_avgs: pd.DataFrame,
    league_avgs: dict[str, float],
    mode: str,
) -> pd.DataFrame:
    """
    Adjust each game row's stats based on opponent quality.

    Offensive stats are adjusted by the opponent's defensive averages:
        additive:       adj_off = raw_off + (league_avg_off - opp_def_avg)
        multiplicative: adj_off = raw_off * (league_avg_off / opp_def_avg)

    Defensive stats are adjusted by the opponent's offensive averages:
        additive:       adj_def = raw_def + (league_avg_def - opp_off_avg)
        multiplicative: adj_def = raw_def * (league_avg_def / opp_off_avg)
    """
    adjusted = fact_df.copy()

    for off_stat, def_stat in zip(OFF_STATS, DEF_STATS):
        # --- Adjust offensive stat by opponent's defensive quality ---
        opp_def_avg = adjusted["opponent"].map(team_avgs[def_stat])
        raw_off = adjusted[off_stat]
        league_avg_off = league_avgs[off_stat]

        if mode == "additive":
            adjusted[f"adj_{off_stat}"] = raw_off + (league_avg_off - opp_def_avg)
        elif mode == "multiplicative":
            safe_opp = opp_def_avg.replace(0, np.nan)
            adjusted[f"adj_{off_stat}"] = raw_off * (league_avg_off / safe_opp)

        # --- Adjust defensive stat by opponent's offensive quality ---
        opp_off_avg = adjusted["opponent"].map(team_avgs[off_stat])
        raw_def = adjusted[def_stat]
        league_avg_def = league_avgs[def_stat]

        if mode == "additive":
            adjusted[f"adj_{def_stat}"] = raw_def + (league_avg_def - opp_off_avg)
        elif mode == "multiplicative":
            safe_opp = opp_off_avg.replace(0, np.nan)
            adjusted[f"adj_{def_stat}"] = raw_def * (league_avg_def / safe_opp)

    return adjusted


def run_sos_adjustment(
    fact_df: pd.DataFrame,
    recency_lambda: float | None = None,
    mode: str | None = None,
    convergence_threshold: float | None = None,
    max_iterations: int | None = None,
) -> pd.DataFrame:
    """
    Run the full iterative SOS adjustment on a fact table.

    Args:
        fact_df: The raw fact table (1 row per team per game).
        recency_lambda: Decay rate for recency weighting. Defaults to config.
        mode: "additive" or "multiplicative". Defaults to config.
        convergence_threshold: Stop when max change < this. Defaults to config.
        max_iterations: Safety cap on iterations. Defaults to config.

    Returns:
        The fact table with adj_ columns added for each stat (off and def).
    """
    recency_lambda = recency_lambda if recency_lambda is not None else config.RECENCY_LAMBDA
    mode = mode if mode is not None else config.ADJUSTMENT_MODE
    convergence_threshold = (
        convergence_threshold if convergence_threshold is not None
        else config.CONVERGENCE_THRESHOLD
    )
    max_iterations = (
        max_iterations if max_iterations is not None
        else config.MAX_ITERATIONS
    )

    df = fact_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Compute recency weights once (they don't change between iterations).
    weights = _compute_recency_weights(df["game_date"], recency_lambda)

    # League averages from raw stats (stable reference point across iterations).
    league_avgs = _compute_league_averages(df, ALL_STATS, weights)

    # Initialize: team averages start from raw stats.
    team_avgs = _compute_team_averages(df, ALL_STATS, weights)

    log.info(f"Starting SOS adjustment (mode={mode}, lambda={recency_lambda})")

    for iteration in range(1, max_iterations + 1):
        # Adjust game-level stats using current team averages.
        adjusted_df = _adjust_game_stats(df, team_avgs, league_avgs, mode)

        # Recompute team averages from the adjusted columns.
        # Build a temporary DataFrame with adj_ values in place of raw columns
        # so _compute_team_averages can operate on them.
        adj_col_map = {f"adj_{s}": s for s in ALL_STATS}
        # Select only the adjusted columns + team + opponent, renaming adj_ to raw names.
        temp_cols = ["team", "opponent"] + [f"adj_{s}" for s in ALL_STATS if f"adj_{s}" in adjusted_df.columns]
        temp_df = adjusted_df[temp_cols].rename(columns=adj_col_map)

        new_team_avgs = _compute_team_averages(temp_df, ALL_STATS, weights)

        # Check convergence: max absolute change in any team's average.
        # Align columns in case of ordering differences.
        common_cols = team_avgs.columns.intersection(new_team_avgs.columns)
        diff = (new_team_avgs[common_cols] - team_avgs[common_cols]).abs()
        max_change = diff.max().max()

        log.info(f"  Iteration {iteration}: max_change = {max_change:.8f}")

        team_avgs = new_team_avgs

        if max_change < convergence_threshold:
            log.info(f"Converged after {iteration} iterations")
            break
    else:
        log.warning(f"Did not converge after {max_iterations} iterations (max_change={max_change:.8f})")

    # Final pass: produce the adjusted fact table with converged team averages.
    result = _adjust_game_stats(df, team_avgs, league_avgs, mode)
    n_adj_cols = len([c for c in result.columns if c.startswith("adj_")])
    log.info(f"SOS adjustment complete. Added {n_adj_cols} adjusted columns.")
    return result
