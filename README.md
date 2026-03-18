# NCAA March Madness Bracket Predictor (2026)

> **For AI assistants**: This README is the single source of truth for this project.
> Read this file first — it contains everything you need to work on the codebase
> without exploring files. **Update this file every time you make changes.**

A data-driven approach to filling out the 2026 NCAA Men's Basketball Tournament
bracket. The model is built entirely from regular season performance data — no
tournament seedings are used as features.

---

## Architecture Overview

```
EXTRACT (Step 1) ─── Raw game data from ESPN (sportsdataverse parquet or CBBpy scraper)
    │
TRANSFORM (Step 2) ─ Build fact table: 1 row per team per game, off + def stats
    │
ADJUST (Step 3) ──── Iterative Strength-of-Schedule convergence (fixed-point iteration)
    │
TRAIN (Step 4) ───── Team profiles + matchup differentials → 4 classifier models
    │
SIMULATE (Step 5) ── Walk bracket structure, predict each matchup → final bracket
```

---

## File Structure

```
NCAA_Basketball/
├── main.py                 # 132 lines — Entry point, pipeline orchestration
├── config.py               # 75 lines  — All tunable parameters (centralized)
├── requirements.txt        # Dependencies: cbbpy, pandas, numpy, scikit-learn, pyarrow
├── .gitignore              # Excludes data/raw/, data/processed/, .env, __pycache__/
│
├── data/
│   └── bracket_teams.csv   # 70 rows — Seed/region/name mapping for tournament teams
│
└── src/
    ├── __init__.py         # Empty
    ├── extract.py          # 291 lines — Dual-backend data ingestion (SDV + CBBpy)
    ├── transform.py        # 247 lines — Raw → fact table conversion
    ├── adjust.py           # 255 lines — Iterative SOS convergence
    ├── features.py         # 156 lines — Team profiles + matchup feature vectors
    ├── model.py            # 192 lines — 4 classifier models + training/comparison
    └── simulate.py         # 237 lines — Bracket simulation + game prediction
```

**Total**: ~1,585 lines of Python across 8 files.

---

## Module Reference

### `config.py` — Configuration Hub

All tunable parameters live here. No magic numbers elsewhere in the codebase.

| Parameter | Default | Description |
|---|---|---|
| `CURRENT_SEASON` | `2026` | Season to predict |
| `HISTORICAL_SEASONS` | `[2016..2025]` | For historical tournament training |
| `DATA_SOURCE` | `"sportsdataverse"` | `"sportsdataverse"` (fast, parquet) or `"cbbpy"` (slow, scraper) |
| `ADJUSTMENT_MODE` | `"additive"` | `"additive"` or `"multiplicative"` SOS adjustment |
| `CONVERGENCE_THRESHOLD` | `0.0001` | SOS iteration convergence stop criterion |
| `MAX_ITERATIONS` | `100` | Safety cap on convergence iterations |
| `RECENCY_LAMBDA` | `0.01` | Exponential decay rate (higher = more recent-game bias) |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `DATA_DIR` | `"data"` | Base data directory |
| `RAW_DIR` | `"data/raw"` | Cached raw extracts |
| `PROCESSED_DIR` | `"data/processed"` | Pipeline outputs |
| `BRACKET_FILE` | `"data/bracket_teams.csv"` | Tournament team mapping |

Data source URLs (sportsdataverse):
- `SDV_BASE_URL` — GitHub Releases URL for parquet files
- `SDV_TEAM_BOX_TAG` — `"espn_mens_college_basketball_team_boxscores"`
- `SDV_SCHEDULE_TAG` — `"espn_mens_college_basketball_schedules"`

### `main.py` — Pipeline Orchestration

**Entry point**. Runs the full pipeline or individual steps.

```bash
python main.py               # Full pipeline (all 5 steps)
python main.py extract        # Step 1 only
python main.py transform      # Step 2 only
python main.py adjust         # Step 3 only
python main.py train          # Step 4 only
python main.py simulate       # Step 5 only
```

**Functions**:
- `step_extract(season)` → `(game_info_df, boxscore_df)`
- `step_transform(info_df, box_df)` → `fact_df`
- `step_adjust(fact_df)` → `adjusted_df` (saves `adjusted_fact_table.csv`)
- `step_train(adjusted_df)` → `(best_model, team_profiles)` (saves `team_profiles.csv`)
- `step_simulate(model, team_profiles)` → `results_df` (saves `bracket_results.csv`)

### `src/extract.py` — Data Ingestion

**Purpose**: Pull raw game and boxscore data from ESPN. Cache locally to avoid re-fetching.

**Two backends**:

| Backend | Speed | Data Level | Team Name Format |
|---|---|---|---|
| **sportsdataverse** (default) | ~1-2 sec/season | Team-level | `"location"` (e.g., "Duke") |
| **cbbpy** (fallback) | ~hours/season | Player-level | `"displayName"` (e.g., "Duke Blue Devils") |

**Key functions**:
- `extract_season(season, force=False, source=None)` → `(game_info_df, boxscore_df)`
- `extract_all_seasons(seasons=None, force=False, source=None)` → `(game_info_df, boxscore_df)`
- `_sdv_extract_season(season, force=False)` — sportsdataverse backend
- `_cbbpy_extract_season(season, force=False)` — CBBpy backend (supports incremental resume via `.progress_{season}.txt`)

**Caching**: Files saved to `data/raw/game_info_{season}.csv` and `data/raw/boxscore_{season}.csv`. Pass `force=True` to re-download.

**Output schemas**:

`game_info_df`:
```
game_id, home_team, away_team, home_win, is_neutral, is_conference,
is_postseason, tournament, game_day
```

`boxscore_df`:
```
game_id, team, fgm, fga, 2pm, 2pa, 3pm, 3pa, ftm, fta,
oreb, dreb, ast, stl, blk, to, pts
```

### `src/transform.py` — Fact Table Construction

**Purpose**: Convert raw data into the core **1 row per team per game** fact table.

**Key steps**:
1. **Player-to-team aggregation** (CBBpy only): `_aggregate_boxscore(box_df)` — sums player stats
2. **Name normalization** (CBBpy only): `_build_name_map(season)` — maps displayName → location
3. **Game row construction** (both): `_build_game_rows(team_stats, info_df)` — self-join to pair team + opponent stats
4. **Derived stats**: shooting percentages (`off_fg_pct_2`, `off_fg_pct_3`, `off_ft_pct`), estimated possessions (`poss = fga - oreb + to + 0.475*fta`)

**Public API**:
- `transform_season(info_df, box_df, season, source=None)` → `fact_df`

**Design**: A team's `def_` columns are literally the opponent's `off_` columns from the same game. "Defense" = what the opponent did offensively against you.

**Output**: Saved to `data/processed/fact_table_{season}.csv` (~22,000-24,000 rows per season).

**Fact table schema**:
```
game_id, team, opponent, is_home, win,
off_fgm, off_fga, off_2pm, off_2pa, off_3pm, off_3pa, off_ftm, off_fta,
off_oreb, off_dreb, off_ast, off_stl, off_blk, off_to, off_pts,
off_fg_pct_2, off_fg_pct_3, off_ft_pct, off_poss,
def_fgm, def_fga, def_2pm, def_2pa, def_3pm, def_3pa, def_ftm, def_fta,
def_oreb, def_dreb, def_ast, def_stl, def_blk, def_to, def_pts,
def_fg_pct_2, def_fg_pct_3, def_ft_pct, def_poss,
is_conference, is_neutral, is_postseason, tournament, game_date, season
```

### `src/adjust.py` — Strength-of-Schedule Adjustment

**Purpose**: The heart of the system. Iteratively adjust stats for opponent quality via fixed-point iteration (not ML — no overfitting risk).

**Algorithm**:
1. Compute recency-weighted team averages (raw stats)
2. Compute league-wide weighted averages (stable reference)
3. **Loop** (up to `MAX_ITERATIONS`):
   - Adjust game-level stats using current opponent averages
   - Recompute team averages from adjusted stats
   - Check convergence: max change < `CONVERGENCE_THRESHOLD`
4. Converges in ~10-20 iterations for ~360 D-I teams

**Recency weighting**: `weight = exp(-λ × days_ago)` where λ = `RECENCY_LAMBDA`. Most recent game = weight 1.0, exponential decay into past.

**Adjustment formulas**:
```
Additive (default):
  adj_off = raw_off + (league_avg_off − opponent_adj_def_avg)
  adj_def = raw_def + (league_avg_def − opponent_adj_off_avg)

Multiplicative (alternative):
  adj_off = raw_off × (league_avg_off / opponent_adj_def_avg)
  adj_def = raw_def × (league_avg_def / opponent_adj_off_avg)
```

**Adjusted stats** (12 offensive + 12 defensive = 24 total):
```
OFF_STATS: off_2pa, off_fg_pct_2, off_3pa, off_fg_pct_3,
           off_fta, off_ft_pct, off_oreb, off_dreb,
           off_ast, off_stl, off_blk, off_to

DEF_STATS: def_2pa, def_fg_pct_2, def_3pa, def_fg_pct_3,
           def_fta, def_ft_pct, def_oreb, def_dreb,
           def_ast, def_stl, def_blk, def_to
```

**Design decision**: Only adjust attempts + percentages, NOT made counts. Made = pct × attempts, so adjusting all three independently creates redundant/inconsistent features.

**Why it catches inflated stats**: Miami (OH) goes 31-1 vs weak MAC opponents → impressive raw stats. But opponents' adjusted defense is weak → Miami's adjusted offense deflates to reflect true quality. Conversely, losses to strong opponents get partially forgiven.

**Key functions**:
- `adjust_season(fact_df, recency_lambda=None)` → `adjusted_df`
- `_compute_recency_weights(game_dates, lambda_val)` → weights array
- `_compute_team_averages(df, weights, stats)` → per-team weighted averages
- `_adjust_games(df, team_avgs, league_avgs, stats, mode)` → adjusted game stats

**Output**: Original fact table + 24 new `adj_off_*` / `adj_def_*` columns. Saved to `data/processed/adjusted_fact_table.csv`.

### `src/features.py` — Feature Engineering

**Purpose**: Aggregate adjusted game stats into team profiles and matchup feature vectors.

**Two outputs**:

1. **Team Profiles** (season-long resume):
   - `build_team_profiles(adjusted_df, recency_lambda=None)` → one row per team
   - Columns: 24 adjusted stats + `games_played` + `win_pct`
   - Values: recency-weighted averages across all games
   - Used at simulation time for predictions

2. **Matchup Features** (ML training data):
   - `build_matchup_features(adjusted_df, recency_lambda=None)` → one row per team per game
   - Features: `diff_{stat}` = team_profile[stat] − opponent_profile[stat]
   - **Avoids data leakage**: profiles built from games BEFORE each game (expanding window)
   - Efficiency: weekly snapshot intervals instead of per-game profile rebuilds
   - Columns: `game_id, team, opponent, win, is_neutral, is_postseason, diff_adj_off_2pa, diff_adj_off_fg_pct_2, ...`

**Key design**: Features are **differentials** (team − opponent), not raw stats. This reduces multicollinearity, captures competitive balance, and normalizes across teams.

### `src/model.py` — ML Model Training

**Purpose**: Train classifiers to predict game outcomes from matchup differentials.

**Four models**:

| Model | Implementation | Notes |
|---|---|---|
| Logistic Regression | `LogisticRegression(max_iter=1000)` | Baseline, fastest, most interpretable |
| Elastic Net | `LogisticRegressionCV(penalty="elasticnet", l1_ratios=[0.1..0.9], cv=5)` | Auto-tunes C and L1 ratio |
| Random Forest | `RandomForestClassifier(n_estimators=200)` | Non-linear, handles interactions |
| Gradient Boosting | `GradientBoostingClassifier(n_estimators=200)` | Powerful but overfitting-prone |

**TrainedModel dataclass**:
```python
@dataclass
class TrainedModel:
    name: str                   # "logistic_regression", "elastic_net", etc.
    pipeline: Pipeline          # Fitted sklearn Pipeline (StandardScaler + classifier)
    accuracy: float             # CV accuracy
    log_loss: float             # CV log loss
    feature_names: list[str]    # Feature column names
```

**Key functions**:
- `train_model(matchup_df, model_type="logistic_regression")` → `TrainedModel`
- `compare_models(matchup_df)` → `list[TrainedModel]` (sorted by log_loss, best first)
- `predict_matchup(model, team_profiles, team_a, team_b)` → `dict` with `team_a_win_prob`, `team_b_win_prob`, `predicted_winner`
- `_deduplicate_games(matchup_df)` — keeps 1 row per game to avoid leakage

**Training pipeline**: NaN fill → StandardScaler → Classifier → StratifiedKFold(5) CV → final fit on full data.

### `src/simulate.py` — Bracket Simulation

**Purpose**: Walk the tournament bracket structure, predict each matchup, output final bracket.

**Bracket structure**:
- 68 teams → 4 First Four games → 64-team main bracket
- 4 regions (East, West, South, Midwest), seeds 1-16 per region
- Matchup order: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
- Rounds: First Round → Second Round → Sweet 16 → Elite Eight → Final Four → Championship
- Final Four pairings: East vs South, West vs Midwest

**Key functions**:
- `simulate_bracket(model, team_profiles)` → `results_df` (67 rows)
- `load_bracket()` → bracket_df from `bracket_teams.csv`
- `_resolve_first_four(bracket_df, model, team_profiles)` → bracket_64
- `_simulate_region(region_teams, model, team_profiles)` → `(champion, results)`

**Output**: `bracket_results.csv` with columns: `round, region, team_a, team_b, team_a_win_prob, team_b_win_prob, predicted_winner` — 67 rows (4 First Four + 63 main bracket).

---

## Data Files

### Input: `data/bracket_teams.csv` (70 rows)

Maps tournament teams across naming conventions. **Must be manually updated each year.**

| Column | Type | Example | Purpose |
|---|---|---|---|
| `seed` | int | 1-16 | Tournament seed |
| `region` | str | East/West/South/Midwest | Tournament region |
| `bracket_name` | str | "Ohio St." | Official bracket name |
| `cbbpy_name` | str | "Ohio State" | Sportsdataverse/CBBpy name |
| `espn_id` | int | 150 | ESPN team ID (join key) |
| `first_four` | str | "FF" or blank | First Four participant |

**Name mismatches handled**: `St.` → `State`, Cal Baptist → California Baptist, Penn → Pennsylvania, Long Island → Long Island University, PVAMU → Prairie View A&M, Miami (FL) vs Miami (OH), Queens (N.C.) → Queens University, Hawaii → Hawai'i.

### First Four Games (2026)

| Region | Seed | Matchup |
|---|---|---|
| South | 16 | Lehigh vs Prairie View A&M |
| West | 11 | NC State vs Texas |
| Midwest | 16 | Howard vs UMBC |
| Midwest | 11 | SMU vs Miami (OH) |

### Generated Files (gitignored)

| File | Location | Description |
|---|---|---|
| `game_info_{season}.csv` | `data/raw/` | Raw game metadata per season |
| `boxscore_{season}.csv` | `data/raw/` | Raw team boxscores per season |
| `fact_table_{season}.csv` | `data/processed/` | Transformed fact table per season |
| `adjusted_fact_table.csv` | `data/processed/` | SOS-adjusted fact table (all seasons) |
| `team_profiles.csv` | `data/processed/` | Final team profiles (1 row per team) |
| `bracket_results.csv` | `data/processed/` | **Final bracket predictions** (67 games) |

---

## Dependencies

```
cbbpy>=2.1.0          # NCAA basketball scraper (ESPN) — fallback data source
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical computing
scikit-learn>=1.3.0   # ML models, pipelines, evaluation
pyarrow>=14.0.0       # Parquet file support (sportsdataverse)
```

No API keys or `.env` file required. Both data backends (sportsdataverse and CBBpy) are unauthenticated.

---

## Usage

```bash
pip install -r requirements.txt

# Full pipeline (~5-10 minutes first run, faster with cached data):
python main.py

# Individual steps:
python main.py extract      # Pull data from ESPN
python main.py transform    # Build fact table
python main.py adjust       # Run SOS adjustment
python main.py train        # Train and compare models
python main.py simulate     # Fill out the bracket
```

**Typical workflow**: Run full pipeline once. On subsequent runs, skip the expensive extract step by running `transform` through `simulate` individually.

---

## Design Philosophy

1. **No seeds as features** — Seeds encode committee opinion, not team quality. The model forms its own view from data.
2. **SOS adjustment is feature engineering, not prediction** — Fixed-point iteration solves interdependent equations; the ML model is the separate prediction layer.
3. **Differential features** — Matchup features are team−opponent differences, not raw stats. Captures relative strength.
4. **Modular pipeline** — Any classifier can be swapped in without touching the data pipeline. Each step is independent and cacheable.
5. **Centralized config** — All hyperparameters in `config.py`. No magic numbers in module code.

---

## Commit History

| Hash | Description |
|---|---|
| `8f58bca` | Add Elastic Net with CV-tuned regularization to model comparison |
| `89050f4` | Fix DatetimeArray.sort() AttributeError in build_matchup_features |
| `4494592` | Add sportsdataverse as fast alternative data source to CBBpy |
| `fcc5091` | Add incremental day-by-day scraping to extract.py |
| `bc9d1bd` | Fix critical issues from second code review pass |
| `4f8c42a` | Fix critical bugs found in code review |
| `1953d81` | Build full prediction pipeline: extract, transform, adjust, train, simulate |
| `7334150` | Add 2026 tournament bracket team mapping and project README |

---

## Maintenance Notes

- **Updating for a new season**: Change `CURRENT_SEASON` in `config.py`, update `data/bracket_teams.csv` with new tournament field, clear `data/raw/` and `data/processed/` caches.
- **Adding a new model**: Add a branch in `model.py`'s `train_model()` function and include it in `compare_models()`.
- **Changing adjusted stats**: Modify `OFF_STATS` list in `adjust.py`. The rest of the pipeline adapts automatically since feature names are derived from column names.
- **Switching data source**: Set `DATA_SOURCE` in `config.py` to `"cbbpy"` for the scraper fallback. CBBpy is much slower but can be more current.
