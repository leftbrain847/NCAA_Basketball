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

## Project Structure

```
NCAA_Basketball/
├── README.md
└── data/
    └── bracket_teams.csv     # 68 tournament teams with name mapping
```

## Approach

### Philosophy

The model predicts game outcomes using only regular season performance metrics — tournament seed is deliberately excluded as a feature. Seeds encode selection committee opinion; we want the model to form its own view from the data.

### Pipeline (Planned)

1. **Data Collection** — Pull regular season game data via CBBpy
2. **Feature Engineering** — Transform raw game data into team-level performance metrics
3. **Model Training** — Train on historical regular season → tournament outcome data
4. **Bracket Simulation** — Run the model through the 2026 bracket to predict each game

Model selection is TBD. A logistic regression was used successfully in a prior year.
