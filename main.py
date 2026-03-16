"""
NCAA March Madness Bracket Predictor — main entry point.

Run the full pipeline:
    python main.py

Or run individual steps:
    python main.py extract
    python main.py transform
    python main.py adjust
    python main.py train
    python main.py simulate
"""

import sys
import os
import logging

import pandas as pd

import config
from src.extract import extract_season
from src.transform import build_fact_table, save_fact_table
from src.adjust import run_sos_adjustment
from src.features import build_team_profiles, build_matchup_features
from src.model import train_model, compare_models
from src.simulate import simulate_bracket

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def step_extract(season: int = config.CURRENT_SEASON) -> tuple:
    """Step 1: Pull raw data from CBBpy."""
    log.info(f"=== EXTRACT (season {season}) ===")
    return extract_season(season)


def step_transform(info_df, box_df, season: int = config.CURRENT_SEASON) -> pd.DataFrame:
    """Step 2: Build the fact table."""
    log.info("=== TRANSFORM ===")
    fact_df = build_fact_table(info_df, box_df, season=season)
    save_fact_table(fact_df, season)
    return fact_df


def step_adjust(fact_df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Run iterative SOS adjustment."""
    log.info("=== SOS ADJUSTMENT ===")
    adjusted_df = run_sos_adjustment(fact_df)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    path = os.path.join(config.PROCESSED_DIR, "adjusted_fact_table.csv")
    adjusted_df.to_csv(path, index=False)
    log.info(f"Saved adjusted fact table to {path}")
    return adjusted_df


def step_train(adjusted_df: pd.DataFrame):
    """Step 4: Build features and train models."""
    log.info("=== TRAIN ===")
    matchup_df = build_matchup_features(adjusted_df)
    team_profiles = build_team_profiles(adjusted_df)

    # Save team profiles for inspection.
    profiles_path = os.path.join(config.PROCESSED_DIR, "team_profiles.csv")
    team_profiles.to_csv(profiles_path)
    log.info(f"Saved team profiles to {profiles_path}")

    # Compare all models.
    models = compare_models(matchup_df)
    best_model = models[0]
    log.info(f"Best model: {best_model.name} (log_loss={best_model.log_loss:.4f})")

    return best_model, team_profiles


def step_simulate(model, team_profiles) -> pd.DataFrame:
    """Step 5: Simulate the bracket."""
    log.info("=== SIMULATE ===")
    results = simulate_bracket(model, team_profiles)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    results_path = os.path.join(config.PROCESSED_DIR, "bracket_results.csv")
    results.to_csv(results_path, index=False)
    log.info(f"Saved bracket results to {results_path}")

    return results


def run_full_pipeline():
    """Run the complete pipeline end-to-end."""
    info_df, box_df = step_extract()
    fact_df = step_transform(info_df, box_df)
    adjusted_df = step_adjust(fact_df)
    model, team_profiles = step_train(adjusted_df)
    results = step_simulate(model, team_profiles)
    return results


def main():
    args = sys.argv[1:]

    if not args:
        run_full_pipeline()
        return

    step = args[0].lower()

    if step == "extract":
        step_extract()
    elif step == "transform":
        info_df = pd.read_csv(os.path.join(config.RAW_DIR, f"game_info_{config.CURRENT_SEASON}.csv"))
        box_df = pd.read_csv(os.path.join(config.RAW_DIR, f"boxscore_{config.CURRENT_SEASON}.csv"))
        step_transform(info_df, box_df)
    elif step == "adjust":
        fact_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, f"fact_table_{config.CURRENT_SEASON}.csv"))
        step_adjust(fact_df)
    elif step == "train":
        adjusted_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "adjusted_fact_table.csv"))
        step_train(adjusted_df)
    elif step == "simulate":
        adjusted_df = pd.read_csv(os.path.join(config.PROCESSED_DIR, "adjusted_fact_table.csv"))
        model, team_profiles = step_train(adjusted_df)
        step_simulate(model, team_profiles)
    else:
        print(f"Unknown step: {step}")
        print("Usage: python main.py [extract|transform|adjust|train|simulate]")
        sys.exit(1)


if __name__ == "__main__":
    main()
