# Presentation Script — Flight Delays ML Pipeline

---

## Slide 1 — Title

**Flight Delays and Cancellations: An End-to-End Machine Learning Pipeline**

> Good [morning/afternoon]. Today I'll walk through a complete machine learning pipeline built on US domestic flight delay data from 2015.
> The project has two goals: predict whether a flight will arrive delayed, and identify the main delay profiles by cause.
> I'll cover the data, the modeling decisions, and the results we obtained.

---

## Slide 2 — The Problem

**Why does this matter?**

- In 2015, over **860,000 US domestic flights** arrived at least 15 minutes late.
- Delays cascade: a late aircraft at 8am causes delays throughout the day at every airport it serves.
- Airlines, airports, and passengers all benefit from knowing the probability of a delay before the flight departs.

**Our question:** *Given information available at scheduled departure time, can we predict whether a flight will be delayed?*

---

## Slide 3 — The Dataset

**Source:** US Bureau of Transportation Statistics via Kaggle — 2015 Flight Delays

| File | Records | Content |
|---|---|---|
| `flights.csv` | 5,874,020 | Schedules, delays, routes, delay causes |
| `airlines.csv` | 16 | IATA code → airline name |
| `airports.csv` | ~300 | Airports, cities, coordinates |

**Target variable:** `ARRIVAL_DELAY` binarized
- `LABEL = 1` — flight arrived **≥ 15 minutes late** (FAA official threshold)
- `LABEL = 0` — on time

**Scope:** completed flights only — 5,714,008 records after removing cancelled and diverted flights.

---

## Slide 4 — Exploratory Data Analysis: Key Findings

> We ran a thorough EDA before any modeling. Three findings shaped every decision downstream.

**1. Class imbalance**
- 81% on-time / 19% delayed
- A naive model that always predicts "on time" already hits 81% accuracy — so accuracy alone is a misleading metric.

**2. Informative nulls in delay cause columns**
- The DOT only records the cause of a delay when `ARRIVAL_DELAY ≥ 15 min`.
- Null values in those columns don't mean "missing data" — they mean "no delay occurred."
- Confirmed empirically: zero on-time flights have any cause column filled in.
- Correction: fill nulls with 0 for on-time flights.

**3. Temporal patterns**
- Delays increase sharply through the day — the snowball effect.
- Peak months: June, July, December (summer travel + holidays).
- Friday and Sunday have the highest average delays.

---

## Slide 5 — Exploratory Data Analysis: Leakage Risk

> The most important finding in the EDA was a leakage risk that, if missed, would have produced a model that looks great in testing but fails completely in production.

**`DEPARTURE_DELAY` is highly correlated with `ARRIVAL_DELAY`.**

This makes intuitive sense — a flight that leaves late tends to arrive late. But departure delay is only known *after the aircraft pushes back*. Our model is meant to predict *before departure*, so including it would be cheating.

**Decision:** `DEPARTURE_DELAY` and all operational columns filled post take-off are excluded from the feature set.

The model predicts purely from pre-flight information: schedule, route, airline, and historical patterns.

---

## Slide 6 — Feature Engineering

> We built 16 features across three groups, all computed with strict leakage controls.

**Temporal features**
| Feature | Description |
|---|---|
| `HORA_PARTIDA` | Scheduled departure hour |
| `TURNO` | Time-of-day shift: dawn / morning / afternoon / night |
| `IS_WEEKEND` | Binary flag for Saturday/Sunday |
| `IS_PEAK_MONTH` | Binary flag for June, July, December |

**Route features**
| Feature | Description |
|---|---|
| `ROTA` | Origin–destination pair |
| `FREQ_ROTA` | How often this route operates (training months only) |
| `LOG_DISTANCE` | Log-transformed flight distance |

**Historical delay features** *(computed exclusively on training data)*
- Mean historical delay per airline, origin airport, destination airport, route, airline×weekday, and airline×shift.

**Split strategy:** months 1–10 for training, 11–12 for testing. All aggregates are computed on training months only and joined to test — no future information leaks into any feature.

---

## Slide 7 — Supervised Models: Setup

**Three classifiers evaluated:**

| Model | Role | Imbalance strategy |
|---|---|---|
| Logistic Regression | Baseline — fast, interpretable | `weightCol` |
| Random Forest | Ensemble — captures non-linearity | `weightCol` |
| Gradient Boosted Trees | Boosted ensemble — sequential refinement | Oversampling (~4.35×) |

> GBT does not support the `weightCol` parameter in Spark MLlib, so we replicated the minority class until reaching a 1:1 ratio.

**Naive baseline for reference:** always predict on-time → accuracy = 81.4%, recall on delayed class = 0%.
Any model that cannot beat this on both accuracy and recall for the delayed class provides no value.

---

## Slide 8 — Supervised Models: Results

> *Re-run `classifier.ipynb` and fill in the values below.*

| Model | AUC-ROC | PR-AUC | F1-macro | F1 (delayed) | Recall (delayed) | Accuracy |
|---|---|---|---|---|---|---|
| Naive baseline | — | — | — | 0.0000 | 0.0000 | 0.8139 |
| Logistic Regression | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Random Forest | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Gradient Boosted Trees | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

**Primary metric:** AUC-ROC — robust to class imbalance, measures discriminative power across all thresholds.

**Secondary metric:** F1 and Recall for the delayed class (LABEL=1) — captures how well the model actually catches delays, which is the operationally relevant outcome.

---

## Slide 9 — Supervised Models: Threshold Optimization

> The default classification threshold of 0.5 is arbitrary and suboptimal for imbalanced data.

By lowering the threshold, we can trade precision for recall on the delayed class — catching more true delays at the cost of more false alarms.

The notebook sweeps thresholds from 0.30 to 0.50 on the best-performing model:

| Threshold | F1 (delayed) | Precision | Recall |
|---|---|---|---|
| 0.30 | `[ ]` | `[ ]` | `[ ]` |
| 0.35 | `[ ]` | `[ ]` | `[ ]` |
| 0.40 | `[ ]` | `[ ]` | `[ ]` |
| 0.45 | `[ ]` | `[ ]` | `[ ]` |
| 0.50 | `[ ]` | `[ ]` | `[ ]` |

> The threshold that maximizes F1 for the delayed class is the recommended operating point.

---

## Slide 10 — Feature Importance

> *Add the top-5 features from Random Forest and GBT after re-running the notebook.*

Expected dominant features based on domain knowledge:
- **Historical delay features** (`HIST_DELAY_ROTA`, `HIST_DELAY_AIRLINE`) — past behavior predicts future behavior.
- **`HORA_PARTIDA`** — later departure hours carry accumulated network delays.
- **`MONTH`** — seasonal demand spikes in summer and December.
- **`FREQ_ROTA`** — high-frequency routes tend to have more schedule buffer.

---

## Slide 11 — Unsupervised Clustering: Goal

> Beyond predicting *whether* a flight will be delayed, we want to understand *why* delays happen — which causes dominate across different groups of flights.

**Approach:** K-Means clustering on delayed flights only (1,063,439 flights).

**Features:** 4 delay-cause columns (minutes each cause contributed):

| Feature | Meaning |
|---|---|
| `AIR_SYSTEM_DELAY` | Airspace congestion, runway closure |
| `AIRLINE_DELAY` | Maintenance, crew, boarding |
| `LATE_AIRCRAFT_DELAY` | Aircraft arrived late at origin |
| `WEATHER_DELAY` | Adverse weather conditions |

> `SECURITY_DELAY` was excluded: its mean across all delayed flights is < 0.1 minutes — essentially zero variance, which adds noise without information.

All features are standardized with `StandardScaler` before clustering, since K-Means uses Euclidean distance and scale differences would distort results.

---

## Slide 12 — Unsupervised Clustering: K Selection

**Two metrics evaluated on a 10% stratified sample (≈ 106,000 flights):**

| k | WSSSE | Silhouette |
|---|---|---|
| 2 | 462,072 | 0.913 — only 2 coarse groups |
| 3 | 395,021 | 0.551 |
| 4 | 335,268 | 0.598 |
| **5** | **318,455** | **0.799** ← selected |
| 6 | 258,797 | 0.780 |
| 7 | 194,039 | 0.420 |

**K=5 selected:** meaningful WSSSE reduction over k=4, silhouette recovers to 0.799, and 5 clusters match the 4 distinct delay causes plus a mixed/severe profile expected from domain knowledge.

---

## Slide 13 — Unsupervised Clustering: Results

> *Re-run `clustering.ipynb` with K=5 and 4 features to populate this table.*

| Cluster | Size | % of delayed | Avg delay (min) | Dominant cause |
|---|---|---|---|---|
| 0 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 1 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 2 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 3 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 4 | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

**Reference from prior run (k=4, 5 features):**
- Cluster 3 dominated with 80% of flights — this motivated the change to k=5.
- Cascade (late aircraft) and operational (airline) profiles were clearly separated in smaller clusters.

---

## Slide 14 — Limitations

> Honest about what the model cannot do is as important as what it can.

| Limitation | Impact |
|---|---|
| No departure delay in features | AUC-ROC ceiling ~0.63; informative but not high-precision at inference time |
| 2015 data only | Must retrain for structural changes: new airlines, post-COVID demand shifts, infrastructure changes |
| Historical features are static | `HIST_DELAY_*` become stale as airline punctuality changes; need periodic recomputation |
| Single temporal split | No cross-validation across multiple time windows — confidence intervals not computed |
| K-Means assumes convex clusters | Delay profiles with overlapping causes may not be well-separated in Euclidean space |

---

## Slide 15 — Architecture Summary

```
flights.csv  ──┐
airlines.csv   ├──▶  EDA  ──▶  Feature Engineering  ──▶  flights_features.parquet
airports.csv  ─┘                     │
                                     ├──▶  Classifier  ──▶  LR / RF / GBT models
                                     │                       + threshold sweep
                                     └──▶  Clustering  ──▶  K-Means (k=5)
                                                             delay profiles
```

**Tech stack:** Python 3.13 · PySpark 4.x · Spark MLlib · Pandas · Matplotlib · Seaborn · Parquet

---

## Slide 16 — Conclusions and Next Steps

**What we built:**
- A leakage-safe, temporally consistent ML pipeline on 5.7 million flights.
- Three supervised classifiers with class-imbalance handling and threshold optimization.
- An unsupervised delay profiler with data-driven K selection.

**What comes next (Phase 5):**
- REST API wrapping the best classifier for real-time prediction at departure.
- Model registry and versioning.
- Scheduled recomputation of historical features as new flight data arrives.
- Drift monitoring: alert when the distribution of delay causes shifts significantly.

---

*End of presentation — Q&A*
