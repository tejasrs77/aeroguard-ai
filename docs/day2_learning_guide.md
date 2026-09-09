# Day 2 learning guide: machine-learning baseline and experiment tracking

## 1. What Day 2 accomplishes

Day 1 turned NASA's raw files into validated sensor histories with a correct Remaining Useful Life (RUL) target. Day 2 teaches models to estimate that target. It does not yet build the LSTM, SHAP explanations, API, dashboard, RAG, or cloud deployment.

The Day 2 question is:

> Given the current operating cycle, settings, and sensor readings for an engine, approximately how many useful cycles remain?

This is **supervised regression** because historical examples contain input features and a numeric answer.

## 2. The three dataset roles

The 100 FD001 training engines are divided by engine identity:

- **Training partition: 80 engines.** Models learn patterns from these engines.
- **Validation partition: 20 different engines.** Candidate models are compared here.
- **Official test set: 100 additional engines.** It is opened only after the champion is selected.

Rows are never randomly mixed. If Engine 7 appeared in both training and validation, the model could recognize its very similar sensor history. That is **data leakage** and would produce an unrealistically optimistic score.

The fixed random seed `42` makes the same engines enter the same partitions on every run. The number 42 is not scientifically special; consistency is the reason for fixing it.

## 3. Inputs and target

The inputs begin with `cycle`, three operating settings, and 21 sensor measurements. `engine_id` is used for grouping but is excluded from training because the number itself has no physical meaning. `rul` and `failure_within_30_cycles` are answers, so presenting them as inputs would be direct target leakage.

Feature removal is learned from the training partition only. Seven FD001 inputs are constant there:

- `operational_setting_3`
- `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19`

A constant feature gives every row the same value, so it cannot help distinguish engine health. Eighteen useful features remain. Missing values would be filled with each training feature's median inside the model pipeline, although the validated FD001 source currently has none.

## 4. Why RUL is capped at 125

The raw training target falls linearly from an engine's first observation to failure. During very early life, however, sensor readings often do not reveal whether 160 or 220 cycles remain. Asking the model to learn that exact difference creates noise without improving maintenance decisions.

Day 2 therefore uses:

```text
model_target = minimum(actual_rul, 125)
```

Examples:

- Raw RUL 200 becomes 125.
- Raw RUL 125 stays 125.
- Raw RUL 30 stays 30.

This is called a piecewise-linear or capped RUL target. It makes the model focus on the degradation period. The report includes both capped and uncapped test metrics so the choice remains transparent.

## 5. The four candidate models

### Mean baseline

It predicts the average capped RUL for every row and ignores sensors. It should be easy to beat. Without a baseline, saying that a complex model is "good" has no reference point.

### Ridge regression

It learns a weighted linear combination of the inputs. Standardization puts features on comparable scales, and Ridge regularization discourages excessively large coefficients. It is fast and interpretable but cannot represent every nonlinear degradation pattern.

### Random Forest

It trains many decision trees on varied samples and averages their predictions. Trees capture nonlinear rules and interactions such as "sensor 11 is high while sensor 4 is also high." Averaging reduces the instability of one tree.

### XGBoost

It builds trees sequentially. Each new tree concentrates on errors made by the existing ensemble. Learning rate, depth, row sampling, and feature sampling control how aggressively it learns and help reduce overfitting.

Random Forest and XGBoost are different: forest trees are largely independent and averaged; boosted trees are created in sequence to correct earlier mistakes.

## 6. How models are evaluated

- **MAE:** average absolute error in cycles. An MAE of 12.3 means predictions miss by about 12.3 cycles on average.
- **RMSE:** square root of average squared error. Large mistakes receive extra weight, so RMSE is normally higher than MAE.
- **R²:** improvement over predicting the mean. `1` is perfect, `0` is about as useful as the mean baseline, and a negative value is worse than that baseline.
- **NASA score:** asymmetric operational penalty. Predicting failure later than reality is penalized more heavily because delayed maintenance is riskier than an early warning. Lower is better.

NASA score is a sum, so it grows when more rows are evaluated. Compare NASA scores only between models evaluated on the same records; do not directly compare the 4,070-row validation score with the 100-engine test score.

The champion is selected only by the lowest validation RMSE. Looking at test results to choose a model would turn the test set into another validation set and make the final result biased.

## 7. Actual results from FD001

| Model | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---:|
| Mean baseline | 36.93 | 41.72 | -0.0002 |
| Ridge regression | 13.14 | 17.36 | 0.827 |
| XGBoost | 10.63 | 15.74 | 0.858 |
| **Random Forest** | **10.60** | **15.25** | **0.866** |

Random Forest won because its validation RMSE was the lowest. It was then trained again using all 100 training engines and evaluated on the final row of each of the 100 official test engines.

Final capped test result:

- MAE: **12.30 cycles**
- RMSE: **16.77 cycles**
- R²: **0.825**
- NASA score: **488.997**

The test result is slightly worse than validation, which is normal for unseen data. It is still far better than the mean baseline.

## 8. Why official testing uses one row per engine

NASA supplies one true RUL number for the last observed point of each test engine. Those 100 endpoints are the official evaluation problem. AeroGuard can calculate labels for earlier test rows for analysis, but final performance is reported only where NASA directly supplies the endpoint truth.

## 9. Reading the figures

- **Model comparison:** shorter MAE/RMSE bars mean fewer cycles of error.
- **Predicted versus actual:** points close to the diagonal red line are accurate.
- **Residual histogram:** zero means perfect; positive values mean the model predicted too much remaining life, and negative values mean an early warning.
- **Feature importance:** larger bars mean a feature contributed more to the forest's decisions. Importance is not proof that the sensor causes degradation.

The champion's strongest inputs are `cycle`, `sensor_11`, `sensor_4`, `sensor_7`, and `sensor_12`. Day 3 will use SHAP for clearer per-prediction explanations.

## 10. What MLflow records

Every candidate receives an MLflow run containing:

- model name and hyperparameters;
- training and validation engine counts;
- RUL cap and feature count;
- validation metrics and training time;
- serialized model artifact.

A separate champion-refit run records the final test metrics. The tracking metadata is stored in local SQLite and model files are stored under `artifacts/mlruns`. This makes experiments auditable instead of relying on memory or copied console output.

On Windows, run `run_mlflow.ps1` after Day 2 and open `http://127.0.0.1:5000`. Press Ctrl+C when finished.

## 11. Files to know

- `ml/split.py`: creates disjoint engine groups and selects test endpoints.
- `ml/features.py`: excludes IDs/targets, drops constants, and caps RUL.
- `ml/metrics.py`: calculates MAE, RMSE, R², and NASA score.
- `ml/train.py`: trains candidates, chooses the champion, logs MLflow runs, refits, evaluates, and saves artifacts.
- `reports/day2_metrics.csv`: complete metric comparison.
- `artifacts/models/champion_rul_model.joblib`: fitted production candidate.
- `artifacts/models/champion_metadata.json`: ordered inputs and modeling decisions.

## 12. Short interview answers

**Why not randomly split rows?**  Adjacent rows from one engine are highly similar. I grouped by engine so validation measures generalization to entirely unseen engines.

**Why use a baseline?**  It proves that model complexity adds measurable value over a trivial average prediction.

**Why cap RUL?**  Exact early-life RUL is weakly observable and less operationally useful, so a 125-cycle cap focuses learning on degradation while preserving late-life targets.

**Why did Random Forest win instead of XGBoost?**  Random Forest had the lowest validation RMSE on the fixed unseen-engine split. I selected from measured generalization, not from a preference for an algorithm.

**Why report both MAE and RMSE?**  MAE gives an intuitive average miss; RMSE highlights occasional large misses that matter in maintenance planning.

**Why is NASA score asymmetric?**  A prediction that says an engine has more life than it really does can delay maintenance, so it receives the larger penalty.

**What is MLflow doing?**  It records parameters, metrics, run IDs, and artifacts so every result can be reproduced and compared.

**Does feature importance prove causation?**  No. It describes how this fitted model used a feature; controlled engineering evidence would be needed to claim causation.

**What remains before production?**  Cross-dataset validation, drift monitoring, probability calibration, domain review, robust deployment, and aviation safety certification. This project is decision support on simulated data.

## 13. One-minute explanation

> I used NASA C-MAPSS FD001 to build a leakage-safe RUL regression workflow. I split the 100 training engines by engine ID into 80 training and 20 validation engines, so sensor rows from one engine could never appear on both sides. I removed the identifier, targets, and constant inputs, then capped early-life RUL at 125 cycles. I compared a mean baseline, Ridge regression, Random Forest, and XGBoost while recording every experiment in MLflow. Random Forest had the lowest validation RMSE at 15.25 cycles, so I refit it on all training engines. On the 100 official test endpoints it achieved 12.30 MAE, 16.77 RMSE, and 0.825 R². I persisted the model with its ordered feature metadata, and I treat feature importance as model evidence rather than causation.
