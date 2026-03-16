"""
Bracket simulation.

Reads the bracket structure from bracket_teams.csv, resolves First Four
games, then simulates each round by predicting matchups using a trained
model and team profiles. Outputs a completed bracket.
"""

import logging

import pandas as pd

import config
from src.model import TrainedModel, predict_matchup

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Bracket positions within each region. Seeds are paired:
# 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
# This ordering reflects the bracket structure (winners play each other).
SEED_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

ROUND_NAMES = [
    "First Round",
    "Second Round",
    "Sweet 16",
    "Elite Eight",
    "Final Four",
    "Championship",
]

# Final Four bracket: which regions play each other in the semis.
# East vs West, South vs Midwest (standard NCAA bracket pairing).
FINAL_FOUR_MATCHUPS = [("East", "West"), ("South", "Midwest")]


def load_bracket() -> pd.DataFrame:
    """Load and return the bracket teams CSV."""
    return pd.read_csv(config.BRACKET_FILE)


def _resolve_first_four(
    bracket_df: pd.DataFrame,
    model: TrainedModel,
    team_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Simulate First Four games. For each pair of FF teams sharing a seed
    and region, predict a winner and keep only that team in the bracket.

    Returns:
        (updated bracket DataFrame, list of First Four game results)
    """
    ff_teams = bracket_df[bracket_df["first_four"] == "FF"].copy()
    non_ff = bracket_df[bracket_df["first_four"] != "FF"].copy()

    results = []
    winners = []

    # Group FF teams by region and seed to find matchups.
    for (region, seed), group in ff_teams.groupby(["region", "seed"]):
        if len(group) != 2:
            log.warning(f"Expected 2 First Four teams for {region} seed {seed}, got {len(group)}")
            continue

        team_a = group.iloc[0]["cbbpy_name"]
        team_b = group.iloc[1]["cbbpy_name"]

        prediction = predict_matchup(model, team_profiles, team_a, team_b)
        winner_name = prediction["predicted_winner"]
        winner_row = group[group["cbbpy_name"] == winner_name].iloc[0].to_dict()
        winners.append(winner_row)

        result = {
            "round": "First Four",
            "region": region,
            "seed": seed,
            **prediction,
        }
        results.append(result)
        log.info(
            f"  First Four: ({seed}) {team_a} vs {team_b} -> "
            f"{winner_name} ({prediction['team_a_win_prob']:.1%})"
        )

    # Rebuild bracket with FF winners replacing FF pairs.
    winners_df = pd.DataFrame(winners)
    bracket_64 = pd.concat([non_ff, winners_df], ignore_index=True)

    return bracket_64, results


def _simulate_region(
    region_teams: pd.DataFrame,
    model: TrainedModel,
    team_profiles: pd.DataFrame,
) -> tuple[str, list[dict]]:
    """
    Simulate all rounds within a single region (First Round through Elite Eight).

    Args:
        region_teams: DataFrame of 16 teams in this region with seed column.
        model: Trained model for predictions.
        team_profiles: Adjusted team profiles.

    Returns:
        (regional champion cbbpy_name, list of game results)
    """
    region = region_teams.iloc[0]["region"]
    results = []

    # Build the initial matchup order from seed pairings.
    seed_to_team = {}
    for _, row in region_teams.iterrows():
        seed_to_team[row["seed"]] = row["cbbpy_name"]

    # Current round's teams, ordered by bracket position.
    current_teams = []
    for high_seed, low_seed in SEED_MATCHUPS:
        current_teams.append((seed_to_team.get(high_seed), seed_to_team.get(low_seed)))

    round_idx = 0
    while len(current_teams) > 0:
        round_name = ROUND_NAMES[round_idx] if round_idx < len(ROUND_NAMES) else f"Round {round_idx + 1}"
        next_round = []

        for team_a, team_b in current_teams:
            if team_a is None or team_b is None:
                winner = team_a or team_b
                next_round.append(winner)
                continue

            prediction = predict_matchup(model, team_profiles, team_a, team_b)
            winner = prediction["predicted_winner"]
            next_round.append(winner)

            result = {
                "round": round_name,
                "region": region,
                **prediction,
            }
            results.append(result)
            log.info(
                f"  {region} {round_name}: {team_a} vs {team_b} -> "
                f"{winner} ({max(prediction['team_a_win_prob'], prediction['team_b_win_prob']):.1%})"
            )

        if len(next_round) == 1:
            return next_round[0], results

        # Pair up winners for the next round (adjacent pairs).
        current_teams = [
            (next_round[i], next_round[i + 1])
            for i in range(0, len(next_round), 2)
        ]
        round_idx += 1

    return next_round[0], results


def simulate_bracket(
    model: TrainedModel,
    team_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate the entire tournament bracket.

    Args:
        model: A trained model.
        team_profiles: Adjusted team profiles (from features.build_team_profiles).

    Returns:
        DataFrame of all game results with columns:
        round, region, team_a, team_b, team_a_win_prob, team_b_win_prob,
        predicted_winner.
    """
    bracket = load_bracket()
    all_results = []

    log.info("=== Resolving First Four ===")
    bracket_64, ff_results = _resolve_first_four(bracket, model, team_profiles)
    all_results.extend(ff_results)

    # Simulate each region through the Elite Eight.
    regional_champs = {}
    for region in ["East", "West", "South", "Midwest"]:
        log.info(f"\n=== {region} Region ===")
        region_teams = bracket_64[bracket_64["region"] == region]
        champ, region_results = _simulate_region(region_teams, model, team_profiles)
        regional_champs[region] = champ
        all_results.extend(region_results)
        log.info(f"  {region} Champion: {champ}")

    # Final Four.
    log.info("\n=== Final Four ===")
    final_four_winners = []
    for region_a, region_b in FINAL_FOUR_MATCHUPS:
        team_a = regional_champs[region_a]
        team_b = regional_champs[region_b]

        prediction = predict_matchup(model, team_profiles, team_a, team_b)
        winner = prediction["predicted_winner"]
        final_four_winners.append(winner)

        result = {
            "round": "Final Four",
            "region": f"{region_a} vs {region_b}",
            **prediction,
        }
        all_results.append(result)
        log.info(
            f"  Final Four: {team_a} ({region_a}) vs {team_b} ({region_b}) -> "
            f"{winner} ({max(prediction['team_a_win_prob'], prediction['team_b_win_prob']):.1%})"
        )

    # Championship.
    log.info("\n=== Championship ===")
    team_a, team_b = final_four_winners
    prediction = predict_matchup(model, team_profiles, team_a, team_b)
    champion = prediction["predicted_winner"]

    result = {
        "round": "Championship",
        "region": "Final",
        **prediction,
    }
    all_results.append(result)
    log.info(f"  Championship: {team_a} vs {team_b} -> {champion}")
    log.info(f"\n{'='*40}")
    log.info(f"  PREDICTED CHAMPION: {champion}")
    log.info(f"{'='*40}")

    return pd.DataFrame(all_results)
