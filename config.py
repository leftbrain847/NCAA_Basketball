"""
Configuration for the NCAA March Madness bracket predictor.

All tunable parameters live here. The pipeline reads from this module
so that experiments are reproducible and changes are centralized.
"""

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
# The primary season to predict. CBBpy uses the later year of the season
# (e.g. the 2025-26 season is 2026).
CURRENT_SEASON = 2026

# Historical seasons used to train on tournament outcomes (Option 2).
# Set to an empty list to skip historical training.
HISTORICAL_SEASONS = list(range(2016, 2026))

# ---------------------------------------------------------------------------
# SOS Adjustment
# ---------------------------------------------------------------------------
# "additive" or "multiplicative"
#   additive:       adjusted = raw + (league_avg - opponent_adjusted)
#   multiplicative: adjusted = raw * (league_avg / opponent_adjusted)
ADJUSTMENT_MODE = "additive"

# Iterative convergence threshold. Iteration stops when the max absolute
# change in any team's adjusted average (across all stat categories)
# falls below this value between passes.
CONVERGENCE_THRESHOLD = 0.0001

# Safety cap on iterations (should never be needed).
MAX_ITERATIONS = 100

# ---------------------------------------------------------------------------
# Recency Weighting
# ---------------------------------------------------------------------------
# Exponential decay applied to game weights based on days before the
# season's final regular-season game.
#   weight = exp(-RECENCY_LAMBDA * days_ago)
#
# Higher lambda = heavier recency bias.
# This is a hyperparameter optimized during model training.
RECENCY_LAMBDA = 0.01  # default starting point

# ---------------------------------------------------------------------------
# Data Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PROCESSED_DIR = f"{DATA_DIR}/processed"
BRACKET_FILE = f"{DATA_DIR}/bracket_teams.csv"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Fraction of data held out for evaluation.
TEST_SIZE = 0.2
RANDOM_STATE = 42

# When optimizing hyperparameters (recency lambda, regularization, etc.),
# use this many cross-validation folds.
CV_FOLDS = 5
