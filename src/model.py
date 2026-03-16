"""
Model training and evaluation.

Trains classifiers to predict game outcomes from matchup features (adjusted
stat differentials between two teams). Supports logistic regression as the
baseline with easy extension to other scikit-learn compatible models.

Includes hyperparameter optimization for recency_lambda, which is upstream
of the model but affects the feature space.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, classification_report

import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Container for a trained model and its metadata."""
    name: str
    pipeline: Pipeline
    accuracy: float
    log_loss: float
    feature_names: list[str]


# Feature columns: everything prefixed with "diff_".
def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("diff_")]


def _deduplicate_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each game produces two rows (one per team). For training, we only
    need one row per game to avoid data leakage (the two rows are
    mirror images with opposite labels).
    """
    return df.drop_duplicates(subset=["game_id"], keep="first")


def train_model(
    matchup_df: pd.DataFrame,
    model_type: str = "logistic_regression",
) -> TrainedModel:
    """
    Train a single model on matchup features.

    Args:
        matchup_df: Output of features.build_matchup_features().
        model_type: One of "logistic_regression", "random_forest",
                    "gradient_boosting".

    Returns:
        TrainedModel with the fitted pipeline and evaluation metrics.
    """
    df = _deduplicate_games(matchup_df)
    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].fillna(0)
    y = df["win"].astype(int)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=config.RANDOM_STATE),
    }

    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(models.keys())}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", models[model_type]),
    ])

    # Cross-validated evaluation.
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    ll_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_log_loss")

    # Fit on full data for final model.
    pipeline.fit(X, y)

    result = TrainedModel(
        name=model_type,
        pipeline=pipeline,
        accuracy=acc_scores.mean(),
        log_loss=-ll_scores.mean(),
        feature_names=feature_cols,
    )

    log.info(
        f"[{model_type}] CV accuracy: {result.accuracy:.4f}, "
        f"CV log_loss: {result.log_loss:.4f}"
    )
    return result


def compare_models(matchup_df: pd.DataFrame) -> list[TrainedModel]:
    """
    Train and compare all available model types.

    Returns list of TrainedModel sorted by log_loss (best first).
    """
    model_types = ["logistic_regression", "random_forest", "gradient_boosting"]
    results = []
    for mt in model_types:
        try:
            results.append(train_model(matchup_df, mt))
        except Exception as e:
            log.warning(f"Failed to train {mt}: {e}")

    results.sort(key=lambda m: m.log_loss)

    log.info("\n--- Model Comparison ---")
    for m in results:
        log.info(f"  {m.name:<25s} accuracy={m.accuracy:.4f}  log_loss={m.log_loss:.4f}")

    return results


def predict_matchup(
    model: TrainedModel,
    team_profiles: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> dict:
    """
    Predict the outcome of a matchup between two teams.

    Args:
        model: A trained model.
        team_profiles: Team profiles DataFrame (from features.build_team_profiles).
        team_a: Name of team A (cbbpy naming).
        team_b: Name of team B (cbbpy naming).

    Returns:
        Dict with team names and win probabilities.
    """
    from src.features import PROFILE_STATS

    prof_a = team_profiles.loc[team_a]
    prof_b = team_profiles.loc[team_b]

    feat = {}
    for stat in PROFILE_STATS:
        if stat in prof_a.index and stat in prof_b.index:
            feat[f"diff_{stat}"] = prof_a[stat] - prof_b[stat]

    X = pd.DataFrame([feat])[model.feature_names].fillna(0)
    proba = model.pipeline.predict_proba(X)[0]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win_prob": proba[1],
        "team_b_win_prob": proba[0],
        "predicted_winner": team_a if proba[1] > 0.5 else team_b,
    }
