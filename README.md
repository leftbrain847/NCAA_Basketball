# NCAA March Madness Bracket Predictor (2026)

A data-driven approach to filling out the 2026 NCAA Men's Basketball Tournament bracket. The model is built entirely from regular season performance data — no tournament seedings are used as features.

## Data Source

All game and team data comes from [CBBpy](https://github.com/dcralph/cbbpy), a Python-based web scraper for NCAA basketball that pulls from ESPN. CBBpy provides:

- Game metadata (scores, records, rankings, venue, TV network, etc.)
- Player and team boxscores
- Play-by-play data

CBBpy does **not** provide tournament seeds or bracket structure. The bracket is sourced separately (see below).

## Bracket Data & Team Name Mapping

### The Problem

The tournament bracket is needed to simulate matchups, but it comes from a different source than the game data. Team names differ between the two:

| Bracket Name | CBBpy Name |
|---|---|
| Ohio St. | Ohio State |
| Michigan St. | Michigan State |
| North Dakota St. | North Dakota State |
| Utah St. | Utah State |
| Kennesaw St. | Kennesaw State |
| Wright St. | Wright State |
| Iowa St. | Iowa State |
| Tennessee St. | Tennessee State |
| Cal Baptist | California Baptist |
| Penn | Pennsylvania |
| Long Island | Long Island University |
| PVAMU / Prairie View | Prairie View A&M |
| Miami (FL) | Miami |
| Miami Ohio | Miami (OH) |
| Queens (N.C.) | Queens University |
| Hawaii | Hawai'i |

Most mismatches follow a `St.` → `State` abbreviation pattern. The remaining are one-off naming differences.

### The Solution

`data/bracket_teams.csv` contains all 68 tournament teams with both naming conventions pre-mapped, along with ESPN team IDs that serve as a reliable join key to CBBpy's internal team data.

### CSV Schema

| Column | Description |
|---|---|
| `seed` | Tournament seed (1–16) |
| `region` | Tournament region (East, West, South, Midwest) |
| `bracket_name` | Team name as shown on the official bracket |
| `cbbpy_name` | Team name as used by CBBpy (`location` field) |
| `espn_id` | ESPN numeric team ID (shared by CBBpy internally) |
| `first_four` | `FF` if the team plays in a First Four game, blank otherwise |

### First Four Games

Eight teams play for four spots in the main 64-team bracket:

| Region | Seed | Matchup |
|---|---|---|
| South | 16 | Lehigh vs Prairie View A&M |
| West | 11 | NC State vs Texas |
| Midwest | 16 | Howard vs UMBC |
| Midwest | 11 | SMU vs Miami (OH) |

---

## Project Structure

```
NCAA_Basketball/
├── main.py                    # Entry point — run full pipeline or individual steps
├── config.py                  # All tunable parameters in one place
├── requirements.txt
├── data/
│   └── bracket_teams.csv      # 68 tournament teams with name mapping
└── src/
    ├── extract.py             # Pull raw data from CBBpy, cache to disk
    ├── transform.py           # Build the core fact table (1 row/team/game)
    ├── adjust.py              # Iterative SOS convergence
    ├── features.py            # Aggregate to team profiles & matchup features
    ├── model.py               # Train/compare classifiers
    └── simulate.py            # Run model through the bracket
```

---

## Pipeline

### Step 1: Extract (`src/extract.py`)

Pulls game info and boxscores from CBBpy for a given season. Results are cached as CSVs in `data/raw/` to avoid re-scraping ESPN on subsequent runs.

### Step 2: Transform (`src/transform.py`)

Converts raw CBBpy data into the **core fact table** — one row per team per game (two rows per game total). Each row contains:

| Category | Columns |
|---|---|
| **Shooting** | 2PA, 2PM, 2PT%, 3PA, 3PM, 3PT%, FTA, FTM, FT% |
| **Rebounds** | Offensive rebounds, Defensive rebounds |
| **Ball control** | Assists, Turnovers |
| **Derived** | Estimated possessions, Points |
| **Metadata** | Game ID, Team, Opponent, Date, Home/Away, W/L, Conference, Neutral site |

All stats appear in both offensive (`off_`) and defensive (`def_`) form. A team's `def_` columns are simply the opponent's `off_` columns from the other row of the same game.

### Step 3: SOS Adjustment (`src/adjust.py`)

The heart of the system. Each team's stats are adjusted for the quality of their opponents **in each specific category** through iterative convergence.

#### How It Works

1. Compute recency-weighted team averages for every stat (offensive and defensive).
2. For each game, adjust the team's raw stats based on the opponent's current adjusted averages.
3. Recompute team averages from the adjusted game stats.
4. Repeat steps 2–3 until the maximum change between iterations falls below the convergence threshold (default: 0.0001).

This is a **fixed-point iteration** — not a learning algorithm. There is no overfitting risk in this step. The system is solving a set of interdependent equations: each team's adjusted stats depend on their opponents' adjusted stats, which depend on *their* opponents' adjusted stats, etc. Convergence typically occurs within 10–20 iterations for ~360 D-I teams.

#### Why This Catches Inflated Stats

A team like Miami (OH) that goes 31-1 against weak MAC opponents will have impressive raw numbers. But:
- Their opponents' adjusted defensive stats are weak (because those teams also play weak offenses).
- When Miami's offense is adjusted against those weak defenses, the impressive raw numbers get deflated to reflect the true quality of competition.
- Conversely, a bad game against a strong out-of-conference opponent gets partially forgiven.

The adjustment is **per-category**: a team that can't defend the 3 but plays in a league that can't shoot 3s gets exposed in that specific stat, even if their overall numbers look fine.

#### Adjustment Formula

Configurable between additive (default, per KenPom's research) and multiplicative:

```
Additive:       adjusted = raw + (league_avg - opponent_adjusted_avg)
Multiplicative: adjusted = raw * (league_avg / opponent_adjusted_avg)
```

KenPom [notes](https://kenpom.com/blog/ratings-methodology-update/) that additive better reflects basketball reality, particularly at the extremes. The multiplicative option is retained as a config flag for experimentation.

#### Recency Weighting

Games are weighted by recency using exponential decay:

```
weight = exp(-λ * days_ago)
```

Where `λ` (lambda) controls how aggressively recent games are favored over early-season games. This is a **tunable hyperparameter** optimized during model training. Higher lambda = more recency bias. A team's February form matters more than their November form.

### Step 4: Features & Training (`src/features.py`, `src/model.py`)

Two modes of operation:

**Team Profiles** — Aggregate each team's adjusted game-level stats into a single row (recency-weighted averages). This is the "team resume" that represents their season-long quality.

**Matchup Features** — For each game, compute the difference between the two teams' profiles across all adjusted stat categories. These differentials become the feature vector for the classifier. The target is win/loss.

#### Training Data Options

- **Option 1 (default):** Train on current-season regular season games (~5,000 games). Learns which stat differentials predict winning.
- **Option 2:** Train on historical tournament games across multiple seasons (configurable in `config.py`). Learns which stats predict *tournament* success specifically. Same pipeline, different filter and label source.

#### Models

Logistic regression is the baseline. Random forest and gradient boosting are available for comparison via `model.compare_models()`. All use the same feature set and are evaluated with stratified k-fold cross-validation on both accuracy and log loss.

### Step 5: Simulate (`src/simulate.py`)

Walks through the bracket structure:
1. Resolves First Four games.
2. Simulates each region (First Round → Elite Eight).
3. Runs the Final Four and Championship.

For each matchup, plugs the two teams' profiles into the trained model to get win probabilities. Outputs a full bracket with predicted winners and confidence levels.

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Or run individual steps
python main.py extract      # Pull data from CBBpy
python main.py transform    # Build fact table
python main.py adjust       # Run SOS adjustment
python main.py train        # Train and compare models
python main.py simulate     # Fill out the bracket
```

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CURRENT_SEASON` | 2026 | Season to predict |
| `HISTORICAL_SEASONS` | 2016–2025 | Seasons for historical tournament training |
| `ADJUSTMENT_MODE` | `"additive"` | `"additive"` or `"multiplicative"` |
| `CONVERGENCE_THRESHOLD` | 0.0001 | SOS iteration stops below this delta |
| `MAX_ITERATIONS` | 100 | Safety cap on convergence iterations |
| `RECENCY_LAMBDA` | 0.01 | Exponential decay rate (higher = more recency bias) |
| `TEST_SIZE` | 0.2 | Train/test split ratio |
| `CV_FOLDS` | 5 | Cross-validation folds |

---

## Philosophy

Tournament seed is deliberately excluded as a feature. Seeds encode the selection committee's opinion — we want the model to form its own view from the data. The bracket structure is only used for *matchups* (who plays whom), not as signal.

The SOS adjustment is **feature engineering**, not prediction. The ML model is the prediction layer. This separation keeps the system modular: any classifier can be swapped in without touching the data pipeline.
