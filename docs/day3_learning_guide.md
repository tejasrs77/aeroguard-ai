# Day 3 learning guide: sequences, LSTM, and SHAP

Day 3 answers two questions that a row-based model cannot fully answer:

1. Can the model use the **order of recent sensor readings**, not just one snapshot?
2. Can we explain why the deployed model predicted a particular Remaining Useful Life (RUL)?

The implementation creates 30-cycle sequences, trains a PyTorch LSTM, compares it fairly with the Day 2 Random Forest, and explains the Random Forest with SHAP.

## 1. The beginner mental model

Imagine an engine has a temperature reading of 80 today. That number alone is incomplete. A stable history of `80, 80, 80` means something different from a rising history of `60, 70, 80`.

- The Day 2 Random Forest sees one row: a snapshot at the current cycle.
- The Day 3 LSTM sees the last 30 ordered rows: a short movie of the engine.

For an engine at cycle 100, one sequence contains cycles 71 through 100. Its target is the RUL at cycle 100. The model is never allowed to see cycles after 100 when predicting cycle 100.

```text
30 ordered cycles × 18 features
             ↓
        two-layer LSTM
             ↓
        dense prediction head
             ↓
     one non-negative RUL value
```

## 2. How sequence data is created

The source table still has one row per `(engine_id, cycle)`. Sequence construction performs these steps independently for every engine:

1. Sort the engine's rows by cycle.
2. Take every complete rolling window of 30 rows.
3. Store the 18 selected feature values from those rows.
4. Use the RUL on the window's final row as the target.
5. Preserve the final engine ID and cycle so predictions can be traced back.

An engine with 100 observed cycles produces `100 - 30 + 1 = 71` sequences. Cycles 1–30 form the first sequence, cycles 2–31 form the second, and cycles 71–100 form the last.

For the official test set, only the **last** 30-cycle sequence for each engine is used. This matches the real decision point: given everything currently observed, how much life remains?

The real run produced:

| Partition | Engines | Sequences |
|---|---:|---:|
| Training | 80 | 14,241 |
| Validation | 20 | 3,490 |
| Official test | 100 | 100 |

The 80 training engines and 20 validation engines have zero overlap. Splitting after sequences were mixed together would leak behavior from the same engine into both partitions.

## 3. Scaling without leakage

LSTMs learn more reliably when input features are on comparable numerical scales. A `StandardScaler` converts each feature using:

```text
scaled value = (original value - training mean) / training standard deviation
```

During model selection, the scaler is fitted on only the 80 training engines. The already-learned means and standard deviations are then applied to the 20 validation engines. Fitting on validation data would leak information.

After early stopping selects the number of epochs, a new scaler is fitted on all 100 training engines for the final model. The test data is only transformed with this scaler; it is never used to fit it.

The `cycle` value may be scaled as a model feature, but the original cycle is separately preserved as metadata. This prevents a scaled decimal from being mistaken for the real cycle identity during row lookup.

## 4. What an LSTM does

LSTM means **Long Short-Term Memory**. It is a recurrent neural network designed to retain useful information across a sequence and forget information that is no longer useful.

At each cycle, the LSTM reads the current 18-feature vector together with its memory from earlier cycles. Internally, learned gates control the information flow:

- **Forget gate:** what old information should be removed?
- **Input gate:** what new information should be stored?
- **Output gate:** what part of the memory should influence the current output?

You do not manually program rules such as "sensor 11 is rising." Training adjusts the network weights so useful temporal patterns reduce prediction error.

The implemented network uses:

| Component | Configuration | Purpose |
|---|---|---|
| Input | 30 × 18 | Thirty ordered cycles, each with 18 features |
| LSTM | 2 layers, 48 hidden units | Learns temporal sensor patterns |
| Recurrent dropout | 0.15 | Reduces overfitting between LSTM layers |
| Dense layer | 24 units with ReLU | Combines learned sequence information |
| Output | 1 unit with Softplus | Produces one non-negative RUL estimate |

Softplus is used at the output because negative remaining life is not meaningful. Predictions are also clipped to the same 0–125 range used by the capped training target.

## 5. How training is controlled

- **Huber loss:** behaves like squared error for small mistakes but is less dominated by very large errors. This is useful when a few engines are unusually difficult.
- **AdamW optimizer:** adjusts model weights using gradients and adds weight decay to discourage overly complex weights.
- **Mini-batches:** 256 sequences are processed at a time rather than loading the full training set into one update.
- **Gradient clipping:** limits the gradient norm to 5.0 to prevent unstable recurrent-network updates.
- **Fixed random seed:** seeds Python, NumPy, PyTorch, and data shuffling for repeatable CPU runs.
- **Early stopping:** validation RMSE is checked after every epoch. Training stops after four epochs without improvement and restores the weights from the best epoch.

The real validation training ran for 18 epochs and selected epoch 14. The model was then rebuilt from scratch on all 100 training engines for exactly 14 epochs. This uses all available training evidence without tuning on the official test set.

## 6. A fair Random Forest versus LSTM comparison

The models must be evaluated on identical examples. The LSTM cannot make a 30-cycle prediction before cycle 30, so the Day 2 Random Forest is evaluated only on the final row of each validation sequence. Both models therefore predict the same 3,490 endpoints.

### Validation results

| Model | MAE | RMSE | R² | NASA score |
|---|---:|---:|---:|---:|
| Random Forest | 12.00 | 16.42 | 0.845 | 18,345.05 |
| LSTM | **8.95** | **13.00** | **0.903** | **11,345.78** |

Lower MAE, RMSE, and NASA score are better; higher R² is better. The LSTM is the validation accuracy winner.

### One final official-test evaluation

| Model | MAE | RMSE | R² | NASA score |
|---|---:|---:|---:|---:|
| Random Forest | 12.30 | 16.77 | 0.825 | **489.00** |
| LSTM | **10.34** | **14.13** | **0.876** | 536.49 |

The LSTM reduces average error, but the Random Forest has the better official-test NASA score.

This is not a contradiction. RMSE treats equally sized early and late errors symmetrically. NASA's scoring function penalizes **late warnings** more heavily because overestimating remaining life can delay maintenance. A model can therefore improve average accuracy while making a few operationally expensive late predictions.

## 7. Champion and deployment decision

The test set was not used to train models, select hyperparameters, or declare the validation accuracy winner. LSTM won that process using unseen validation engines.

Deployment is a separate safety and governance decision. The project keeps the Random Forest as the **current explainable deployment champion** because:

1. it has the better official-test asymmetric NASA risk score;
2. its individual predictions are directly explained by the implemented SHAP method;
3. promoting the LSTM requires further investigation of its late-warning tail and a sequence-model explanation method.

The LSTM is saved as a promotion candidate rather than discarded. This distinction is realistic: the most accurate model is not automatically the safest or most auditable production choice.

## 8. What SHAP explains

SHAP stands for **SHapley Additive exPlanations**. It breaks one model prediction into a baseline plus feature contributions:

```text
model prediction = base value + contribution 1 + ... + contribution N
```

- A positive SHAP value pushes predicted RUL upward.
- A negative SHAP value pushes predicted RUL downward.
- A larger absolute value means a stronger effect on that prediction.

The global report averages the absolute SHAP values across all 100 official test engines. It answers: "Which features generally influenced Random Forest predictions most?" The five strongest features in this run were `cycle`, `sensor_11`, `sensor_4`, `sensor_9`, and `sensor_12`.

The local report explains the Random Forest's lowest-RUL test prediction, engine 34. It answers: "Which feature values pushed this particular prediction above or below the baseline?"

The recorded SHAP additivity error was approximately `6.24 × 10⁻¹³`, effectively zero, confirming that the baseline and contributions reconstruct the raw Random Forest prediction.

Important limitations:

- SHAP describes the model, not physical causation.
- An influential sensor is not automatically the physical cause of degradation.
- The Day 3 SHAP values explain the Random Forest, **not the LSTM**.
- The public NASA sensor names and engineering units are anonymized, so the project does not invent them.

## 9. Saved, reproducible evidence

Day 3 saves:

- the LSTM checkpoint and exact network configuration;
- the fitted feature scaler and ordered feature list;
- training history for every completed epoch;
- validation and official-test metrics for both models;
- predictions for all 100 test engines;
- global and local SHAP tables;
- four PNG figures;
- parameters, metrics, versions, run ID, and artifacts in MLflow.

The checkpoint stores a PyTorch `state_dict` plus the model configuration. Loading reconstructs the architecture before restoring the learned weights. The scaler and feature order must be reused at inference time; otherwise the inputs would not mean the same thing as they did during training.

## 10. Common interview questions and short answers

**Why use an LSTM?**  
Sensor deterioration is temporal. An LSTM can learn changes across ordered cycles that a single-row model cannot see.

**Why choose a 30-cycle window?**  
It gives the model meaningful recent history while keeping training and inference manageable. It is a tunable design choice, not a universal constant.

**Why split by engine?**  
Rows from one engine are related. Keeping an engine in only one partition prevents identity and lifecycle leakage.

**Why scale features?**  
Neural-network optimization is more stable when features have comparable magnitudes. The scaler is fitted only on training engines to prevent leakage.

**What is early stopping?**  
Stop training when validation performance stops improving, then restore the best epoch's weights. It reduces overfitting.

**Why Huber loss?**  
It learns from ordinary errors like squared loss but reduces the influence of extreme errors.

**Why compare at sequence endpoints?**  
It ensures both models predict the exact same engine-cycle examples, making the comparison fair.

**Did you tune on the test set?**  
No. The grouped validation set selected the epoch and accuracy winner. The official test set was evaluated once afterward.

**Why was the LSTM not automatically deployed?**  
It improved RMSE but had a worse asymmetric NASA risk score, and the current SHAP explanation is tied to the Random Forest. Its late-warning risk needs review.

**What does a negative SHAP value mean?**  
For that prediction, the feature pushed predicted RUL below the model's baseline.

**Does SHAP prove a sensor caused the failure?**  
No. It explains how the model used a feature; it does not establish physical causation.

**How would you improve Day 3?**  
Tune window length and architecture with grouped cross-validation, measure late-warning tail risk, add uncertainty intervals, explain the LSTM with a validated sequence explainer, and test on other C-MAPSS operating conditions.

## 11. One-minute interview explanation

> Day 3 adds temporal modeling and explainability. I create leakage-safe 30-cycle windows from each engine, fit the scaler only on training engines, and train a two-layer PyTorch LSTM with Huber loss, AdamW, gradient clipping, and validation-based early stopping. For a fair comparison, the Random Forest and LSTM predict the same validation sequence endpoints. The LSTM improved validation RMSE from 16.42 to 13.00 cycles and official-test RMSE from 16.77 to 14.13. However, its official NASA asymmetric risk score was slightly worse, so I saved it as a promotion candidate and retained the Random Forest as the explainable deployment champion. TreeSHAP then provides global feature importance and a local additive explanation for the highest-risk engine. This separates model accuracy, operational risk, and deployment governance rather than treating them as the same decision.
