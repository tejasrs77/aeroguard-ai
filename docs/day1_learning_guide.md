# Day 1 learning guide

## 1. The business idea

Predictive maintenance means using evidence about a machine's present condition to estimate when maintenance may be needed. Traditional scheduled maintenance services equipment at fixed intervals. Predictive maintenance attempts to detect degradation and intervene before failure while avoiding unnecessary work.

AeroGuard currently predicts two targets:

- **Remaining Useful Life (RUL):** cycles estimated to remain before failure.
- **Near-term risk:** whether failure is within 30 cycles.

These are decision-support outputs. They do not certify that an aircraft is safe to fly.

## 2. Understanding one row

Suppose a row contains:

```text
engine_id = 7
cycle = 83
operational_setting_1 ... 3
sensor_1 ... 21
```

It means that the row is the sensor snapshot taken from simulated engine 7 during its 83rd operating cycle. The next row for that engine should be cycle 84. It is not a new engine and it is not necessarily one hour or one day later.

## 3. Features and targets

The operating settings and sensor measurements are input **features**. RUL is the value the regression model will learn to predict, so it is the **target**. `failure_within_30_cycles` is a second, binary target for classification.

The target must never be included among the model's input features. Doing so would reveal the answer and produce misleadingly excellent results; this is target leakage.

## 4. Why RUL is calculated differently

Training engines run until simulated failure. If an engine fails at cycle 200, its row at cycle 170 has `200 - 170 = 30` cycles of remaining life.

Test engines stop earlier. If the last observed cycle is 150 and NASA says 40 cycles remain, the last row has RUL 40. Its row at cycle 130 has `40 + 150 - 130 = 60` cycles remaining.

## 5. Validation rules

Before trusting analysis, AeroGuard checks:

- all 26 source columns exist;
- every value is numeric and finite;
- engine IDs and cycles are positive;
- `(engine_id, cycle)` is unique;
- each engine contains consecutive cycles beginning at 1;
- the test truth file contains one value per test engine;
- no supplied RUL value is negative.

Validation protects the model from learning from structurally corrupt data. It does not guarantee that the measurements are scientifically correct.

## 6. Statistical outputs

The Day 1 report calculates lifecycle distribution, sensor ranges and variability, missingness, correlation and a bootstrap confidence interval for mean engine lifetime.

- **Mean:** arithmetic average.
- **Median:** middle value after sorting; less sensitive to extreme engines.
- **Standard deviation:** how widely a sensor varies.
- **Correlation:** strength of a linear relationship between two variables; not proof of causation.
- **Bootstrap confidence interval:** repeatedly resample engines and calculate their mean lifetime to quantify sampling uncertainty without assuming a particular distribution.

Statistics are calculated at the engine level when discussing engine lifetime. Treating every sensor row as an independent engine would give long-lived engines too much weight.

`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19` are constant in FD001. A constant sensor cannot help distinguish healthy from degrading rows. Constant status is determined by distinct-value count, avoiding microscopic floating-point noise in the calculated standard deviation.

## 7. Data leakage

One engine contributes many highly related rows. A random row split could put early cycles from engine 10 in training and later cycles from engine 10 in validation. The model would indirectly see the validation engine during training. Day 2 will split by `engine_id`, so complete engines remain isolated.

## 8. Interview questions and short answers

**What is RUL?**  
The estimated number of operating cycles a machine has left before failure.

**Why is `(engine_id, cycle)` the natural key?**  
It uniquely identifies one sensor snapshot for one engine at one point in its lifecycle.

**Why validate before analysis?**  
Corrupt schemas, duplicates or broken cycle sequences can invalidate statistics and models.

**Why use a 30-cycle label?**  
It converts continuous RUL into an actionable near-term-risk classification target. The threshold is a project assumption and should be configurable in a real system.

**Why are train and test RUL formulas different?**  
Training histories reach failure, while test histories stop early and require NASA's supplied remaining-life value.

**Why use a bootstrap confidence interval?**  
It estimates uncertainty in mean lifecycle using resampling and makes fewer distributional assumptions.

**Why not randomly split rows?**  
Rows from the same engine are dependent. Random row splitting leaks engine-specific patterns across train and validation sets.

**Does correlation identify the failing component?**  
No. Correlation describes linear association and does not establish causation or physical fault identity.

**Why save the archive checksum?**  
It proves which exact bytes were used and helps detect an incomplete or changed source file.

**Is this production aviation software?**  
No. It is an educational decision-support system using simulated data, not a certified safety system.

## 9. Your two-minute Day 1 explanation

> AeroGuard uses NASA's C-MAPSS multivariate time-series dataset. Each row is one engine-cycle snapshot with three operating settings and 21 anonymized sensor values. I first download and fingerprint the official archive, then validate its schema, numeric values, unique engine-cycle keys and sequence continuity. Training engines run to failure, so I calculate RUL as each engine's maximum cycle minus the current cycle. Test engines stop early, so I add NASA's supplied tail RUL. Finally, I produce engine-level lifecycle statistics, sensor summaries and reproducible visualizations. I keep this logic in tested Python modules so the same transformations can later support model training, APIs and monitoring.
