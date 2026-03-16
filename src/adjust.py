"""
Iterative Strength-of-Schedule adjustment.

For each stat category, every team's raw per-game numbers are adjusted by
the quality of their opponent in that specific category. Because opponent
quality is itself adjusted, this creates a circular dependency resolved
through iterative convergence (fixed-point iteration).

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

# Stat columns to adjust. Each entry is a column name in the fact table.
# "off_" prefixed columns are offensive; the corresponding "def_" column
# (what the opponent did) is derived automatically.
ADJUSTABLE_STATS = [
    "off_2pa", "off_2pm", "off_fg_pct_2",
    "off_3pa", "off_3pm", "off_fg_pct_3",
    "off_fta", "off_ftm", "off_ft_pct",
    "off_oreb", "off_dreb",
    "off_ast", "off_to",
]


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
    Compute recency-weighted team averages for each stat, split into
    offensive (what the team does) and defensive (what the team allows).

    Returns DataFrame indexed by team with columns for each stat in both
    offensive and defensive form.
    """
    rows = []
    for team, group in fact_df.groupby("team"):
        team_weights = weights.loc[group.index]
        row = {"team": team}
        for stat in stats:
            row[stat] = _weighted_mean(group[stat], team_weights)
            # The defensive counterpart: what opponents did against this team.
            def_stat = stat.replace("off_", "def_")
            if def_stat in group.columns:
                row[def_stat] = _weighted_mean(group[def_stat], team_weights)
        rows.append(row)

    return pd.DataFrame(rows).set_index("team")


def _compute_league_averages(
    fact_df: pd.DataFrame, stats: list[str], weights: pd.Series
) -> dict[str, float]:
    """Compute league-wide weighted averages for each stat."""
    avgs = {}
    for stat in stats:
        avgs[stat] = _weighted_mean(fact_df[stat], weights)
    return avgs


def _adjust_game_stats(
    fact_df: pd.DataFrame,
    team_avgs: pd.DataFrame,
    league_avgs: dict[str, float],
    stats: list[str],
    mode: str,
) -> pd.DataFrame:
    """
    Adjust each game row's offensive stats based on the opponent's
    adjusted defensive averages (and vice versa).

    For offensive stat adjustment:
        additive:       adj = raw + (league_avg - opponent_def_avg)
        multiplicative: adj = raw * (league_avg / opponent_def_avg)

    This adjusts a team's performance by how much better/worse than
    average their opponent is at defending that category.
    """
    adjusted = fact_df.copy()

    for stat in stats:
        def_stat = stat.replace("off_", "def_")

        # Look up each row's opponent's adjusted defensive average.
        opp_def_avg = adjusted["opponent"].map(team_avgs[def_stat])

        raw = adjusted[stat]
        league_avg = league_avgs[stat]

        if mode == "additive":
            adjusted[f"adj_{stat}"] = raw + (league_avg - opp_def_avg)
        elif mode == "multiplicative":
            # Guard against division by zero.
            safe_opp = opp_def_avg.replace(0, np.nan)
            adjusted[f"adj_{stat}"] = raw * (league_avg / safe_opp)
        else:
            raise ValueError(f"Unknown adjustment mode: {mode}")

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
        The fact table with adj_ columns added for each adjusted stat.
    """
    recency_lambda = recency_lambda if recency_lambda is not None else config.RECENCY_LAMBDA
    mode = mode or config.ADJUSTMENT_MODE
    convergence_threshold = convergence_threshold or config.CONVERGENCE_THRESHOLD
    max_iterations = max_iterations or config.MAX_ITERATIONS

    df = fact_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Compute recency weights once (they don't change between iterations).
    weights = _compute_recency_weights(df["game_date"], recency_lambda)

    # League averages (computed from raw stats, stable across iterations).
    league_avgs = _compute_league_averages(df, ADJUSTABLE_STATS, weights)

    # Initialize: team averages start from raw stats.
    team_avgs = _compute_team_averages(df, ADJUSTABLE_STATS, weights)

    log.info(f"Starting SOS adjustment (mode={mode}, lambda={recency_lambda})")

    for iteration in range(1, max_iterations + 1):
        # Adjust game-level stats using current team averages.
        adjusted_df = _adjust_game_stats(df, team_avgs, league_avgs, ADJUSTABLE_STATS, mode)

        # Recompute team averages from adjusted stats.
        adj_stats = [f"adj_{s}" for s in ADJUSTABLE_STATS]
        # Temporarily rename adj_ columns so _compute_team_averages can use them.
        rename_to_raw = {f"adj_{s}": s for s in ADJUSTABLE_STATS}
        rename_to_adj = {s: f"adj_{s}" for s in ADJUSTABLE_STATS}

        temp_df = adjusted_df.rename(columns=rename_to_raw)
        new_team_avgs = _compute_team_averages(temp_df, ADJUSTABLE_STATS, weights)

        # Check convergence: max absolute change in any team's average.
        diff = (new_team_avgs - team_avgs).abs()
        max_change = diff.max().max()

        log.info(f"  Iteration {iteration}: max_change = {max_change:.8f}")

        team_avgs = new_team_avgs

        if max_change < convergence_threshold:
            log.info(f"Converged after {iteration} iterations")
            break
    else:
        log.warning(f"Did not converge after {max_iterations} iterations (max_change={max_change:.8f})")

    # Final pass: produce the adjusted fact table.
    result = _adjust_game_stats(df, team_avgs, league_avgs, ADJUSTABLE_STATS, mode)
    log.info(f"SOS adjustment complete. Added {len(ADJUSTABLE_STATS)} adjusted columns.")
    return result
