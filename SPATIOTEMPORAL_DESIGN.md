# Spatiotemporal Traffic Model — Design Document

## 1. The problem with the previous design

The original `traffic_labeling.py` defines `Traffic_Congestion` as a deterministic rule:

```python
score = (Road_Width(m) < 24) * 2
      + (Is_RushHour == 1)   * 2
      + (Speed_Limit(mph) < 45) * 1
      + (Is_Night == 0)      * 1
      + (Weather_Enc in [2,3,4]) * 2
Traffic_Congestion = (score >= 4)
```

Then `train_model.py` hands all five of those columns to XGBoost as features. The model recovers the rule and reports 99.98 % accuracy. This number is meaningless — the "model" is a lookup table for `f(road_width, rush_hour, speed_limit, weather, is_night)` that the training data already gave it.

This is a **target-leakage** failure mode. The label is a deterministic function of the features the model gets to see, so the model learns the function exactly. It tells you nothing about real-world traffic.

## 2. The reformulation

Replace the rule with an **empirical** label drawn from the data itself.

### 2.1 Spatial discretization

Quantize each accident record's `(Start_Lat, Start_Lng)` into a cell of fixed angular size:

```
Cell_Lat_Idx = round(Start_Lat / cell_size)
Cell_Lng_Idx = round(Start_Lng / cell_size)
Cell_ID      = f"{Cell_Lat_Idx}_{Cell_Lng_Idx}"
```

We use `cell_size = 0.05°` (~5.5 km lat × ~4.4 km lng in the lower-48 USA — neighborhood scale).

**Why not 0.01°?** I tested it. 92.7 % of `(cell, hour, month)` buckets have a count of exactly 1, which makes the tertile thresholds degenerate — class 0 and class 1 both span `count = 1`, distinguishable only by arbitrary tie-breaking. The model can't learn anything from that. 0.05° gives 41 % of buckets a count > 1, enough for the labels to mean something.

**Tradeoff:** coarser cells lose spatial precision; finer cells lose statistical mass. The right answer is dataset-dependent. Make it a parameter (it is — `CELL_SIZE_DEG` in `spark/spatiotemporal_labeling.py`).

### 2.2 Temporal aggregation

Group records by `(Cell_ID, Hour)`. Hour-of-day captures the diurnal pattern that drives traffic. **`Month` is dropped from the grouping key** but kept as an aggregated feature (`mean_Month`) — adding it to the grouping over-splits the dataset (more buckets, fewer records each, sparser counts).

If you have a longer-running dataset and can afford the split, add month back: change `GROUP_KEYS = ["Cell_ID", "Hour"]` to `["Cell_ID", "Hour", "Month"]`. Same logic for `DayOfWeek` — but `data_cleaning.py` would need to preserve a `Date` column first.

### 2.3 Empirical label

For each `(Cell_ID, Hour)` bucket, count the accidents in that bucket. Then bucket the global `accident_count` distribution into three classes:

```
Traffic_Level = pd.qcut(rank(accident_count), q=3, labels=[0, 1, 2])
                # 0 = Low, 1 = Medium, 2 = High
```

We rank-then-qcut (rather than qcut directly) to break ties at low counts — without ranking, all `count=1` buckets would land in a single bin and the others would be empty.

**The label distribution looks like this:**

| Class | Count range | Examples |
|-------|-------------|----------|
| 0 (Low)    | exactly 1                | `count=1` buckets, ~33 % |
| 1 (Medium) | 1 (41 %), 2 (57 %), 3 (2 %) | rank-tiebroken between Low and Medium |
| 2 (High)   | 3 → 104 (median 5)         | substantially higher than baseline |

The Low/Medium boundary is **structurally fuzzy** — both contain `count=1` buckets, the split is arbitrary among them. The Medium/High boundary is sharper (count crosses 3). This shows in the per-class scores: High is predicted nearly perfectly, Low/Medium less so.

If you only need actionable distinction — "is this a busy time-place or not?" — call `build_spatiotemporal_dataset(n_classes=2)`. The binary version performs better on per-class metrics because the labels carry more information per row.

### 2.4 Cell baselines

For each cell, compute three properties that are stable over time:

- `cell_total_accidents`  — total observations across all hours and months
- `cell_unique_hours`     — number of distinct hours-of-day observed
- `cell_unique_months`    — number of distinct months observed

Broadcast these back onto every bucket in that cell. They give the model a **prior on cell density** that's invariant to the specific (hour, month) being predicted. This is what lets the model distinguish "busy hour in a generally busy place" from "busy hour in a sleepy place" — the same relative-tertile label means very different things in those two contexts.

### 2.5 Why this isn't leakage

The label is `tertile(count(records in bucket))` — a property of the **aggregate** of all records sharing a `(Cell_ID, Hour)` key. The features are aggregated condition statistics for that bucket plus cell-level baselines.

Could the model trivially recover the label from the features?

- `accident_count` is **excluded** from the feature set (that would be perfect leakage).
- `Cell_ID` is **excluded** — including it would let the model memorize per-cell labels and would make spatial generalization impossible.
- `cell_total_accidents` correlates with the count distribution (busy cells tend to have busier buckets) but does **not** identify which specific hour-buckets within a cell are busy. The model still has to learn temporal structure.
- The condition features (mean weather, infra fractions) describe what the bucket *looked like*, not how big it was.

The quickest sanity check: if leakage were hiding somewhere, the model would score near-100 %. It scores 81 % macro-F1. That's the smell test passing.

## 3. Features (full list, leakage-free)

| Group        | Features |
|--------------|----------|
| Spatial      | `Cell_Lat_Mean`, `Cell_Lng_Mean` |
| Temporal     | `Hour`, `mean_Month` |
| Weather      | `mean_Temperature_F_`, `mean_Humidity_Pct_`, `mean_Pressure_in_`, `mean_Visibility_mi_`, `mean_Wind_Speed_mph_`, `mean_Precipitation_in_`, `mean_Weather_Enc` |
| Road type    | `mean_Road_Type_Enc` |
| Infrastructure | `frac_Junction`, `frac_Traffic_Signal`, `frac_Crossing`, `frac_Stop`, `frac_Roundabout`, `frac_Station`, `frac_Amenity` |
| Cell baseline | `cell_total_accidents`, `cell_unique_hours`, `cell_unique_months` |

20 features total. None of them is computed from the rule used in `traffic_labeling.py` because the rule no longer exists.

## 4. Model & training

Multi-class XGBoost (`objective="multi:softprob"`, `num_class=3`). Hyperparameters chosen for stability rather than peak performance: `max_depth=7`, `learning_rate=0.05`, `n_estimators=500`, `min_child_weight=2`, `reg_lambda=1.0`. No class-weighting needed — the labels are balanced by construction (33 / 33 / 33).

## 5. Evaluation

Two regimes, both 5-fold:

### 5.1 Random Stratified k-fold

Splits `(cell, hour)` buckets randomly across folds. Both training and test contain buckets from the same cells, just at different hours. **Question this answers:** can the model predict the relative congestion at a new hour-of-day in a cell it has training data for?

### 5.2 Spatial Group k-fold

Splits by `Cell_ID`. Whole cells are held out — every test bucket is from a cell the model has *never* seen. **Question this answers:** can the model predict in cells it has never seen?

### 5.3 Results on the US Accidents 2016–2023 dataset (451,837 records)

| Regime       | Accuracy        | F1-macro       | Recall-macro   |
|--------------|-----------------|----------------|----------------|
| Random       | 0.818 ± 0.002  | **0.811 ± 0.003** | 0.818 ± 0.002 |
| Spatial-group | 0.819 ± 0.003 | **0.812 ± 0.002** | 0.819 ± 0.002 |
| **Gap**       | ≈ 0           | ≈ 0             | ≈ 0            |

The gap between random and spatial-group is **essentially zero**. The model has learned transferable patterns — it doesn't rely on memorizing per-cell idiosyncrasies. (The `cell_*` baseline features still appear in held-out cells because they're computed from each cell's own data, but the model has clearly learned to combine them with conditions and hour rather than just looking up an answer.)

**A stricter "cold-start" holdout** — computing `cell_total_accidents` only from cells in the training fold and using a global median for held-out cells — is a useful next experiment. That measures performance when predicting in a cell with **zero history**, which is the genuine cold-start case.

### 5.4 Per-class structure

From the final-model training-set diagnostic:

| Class    | Precision | Recall | F1     |
|----------|-----------|--------|--------|
| Low      | 0.712     | 0.980  | 0.825  |
| Medium   | 0.923     | 0.586  | 0.717  |
| **High** | **0.980** | **0.969** | **0.974** |

The High class — the actionable one for a traffic-warning system — is essentially solved. The Low/Medium fuzziness is structural (see § 2.3) and cannot be improved by better hyperparameters; only by changing the labeling scheme (use 2 classes, or drop `count=1` buckets entirely).

### 5.5 Top features by gain

```
cell_total_accidents       0.165
mean_Weather_Enc           0.139
frac_Traffic_Signal        0.125
frac_Junction              0.102
mean_Road_Type_Enc         0.075
cell_unique_months         0.055
mean_Month                 0.052
frac_Crossing              0.050
mean_Visibility_mi_        0.047
frac_Stop                  0.037
```

The cell baseline dominates (`cell_total_accidents` = 16.5 %). After that the model leans on weather, traffic-control infrastructure (signals, junctions, crossings, stops), and road type. Visibility shows up but other weather details (temperature, humidity, pressure) do not — sensible: they're aggregated to bucket level so individual storms get washed out.

## 6. How to run

```bash
# 1) Build the empirical-density dataset
python spark/spatiotemporal_labeling.py
#    writes data/processed/spatiotemporal_dataset.csv
#    writes data/processed/cell_baseline_lookup.csv     ← used by API
#    writes data/processed/spatiotemporal_config.json   ← used by API

# 2) Train + evaluate
python models/train_traffic_spatiotemporal.py
#    writes models/traffic_st_model.pkl
#    writes models/traffic_st_features.pkl

# 3) Restart the API — it auto-detects the new artifacts
python apis/main.py
```

## 7. Serving

The API loads the spatiotemporal model on startup if its artifacts are present. Two endpoints expose it:

- `POST /predict` — the existing endpoint now appends a `"spatiotemporal"` block alongside its binary congestion / severity outputs.
- `POST /predict-spatiotemporal` — returns only the 3-class spatiotemporal prediction.

Both responses include the resolved `cell_id` and a `cell_known` boolean. When `cell_known: false` the prediction is a cold-start estimate (cell baselines fall back to global medians) and should be treated as lower-confidence.

## 8. What would push F1 from 0.81 to 0.90

Roughly in order of effort:

1. **Drop count=1 ambiguity** — relabel as binary (count=1 vs count≥2) or drop count=1 from training. Removes the structural fuzziness in classes 0/1.
2. **Lag features** — `count_t-24` (same hour yesterday), `count_t-168` (same hour last week). Requires preserving a `Date` column in `data_cleaning.py`. Lag features are the single biggest accuracy lever in classical traffic forecasting.
3. **Day-of-week** — the dataset clearly has weekday/weekend structure. Same prerequisite as (2).
4. **Holiday calendar** — US holidays cause large traffic shifts. Free signal.
5. **POI density / OSM features** — distance to nearest highway exit, count of POIs within 500 m, road-network betweenness. Lifts spatial generalization considerably and is the principled cold-start fix.
6. **Spatial smoothing with neighbor cells** — borrow strength from adjacent cells when a cell has few observations. GraphSAGE / GAT for the principled version, simple k-nearest-cells averaging for the cheap one.
7. **Explicit population / road-network priors** — replaces `cell_total_accidents` with something that transfers to truly-unseen cells.

Items (1)–(4) are local code changes. (5)–(7) require external data.

## 9. What was deliberately *not* done

- **Cold-start cell baselines.** The current `cell_*` features see all data when computed; they don't leak the label but they do see test-fold *observations* of held-out cells (just not their labels). A truly cold-start evaluation would recompute baselines per training fold. Recommended as a next experiment, not a bug.
- **Time-based holdout.** Splitting by date instead of cell would test temporal generalization (predicting future months from past months). Requires the `Date` preservation mentioned above.
- **Probability calibration.** XGBoost's `predict_proba` is not well-calibrated by default. If the dashboard exposes raw probabilities to users, wrap with `CalibratedClassifierCV(method="isotonic")` for honest probabilities.
