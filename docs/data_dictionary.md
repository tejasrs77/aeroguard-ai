# FD001 data dictionary

## Row grain

One row represents **one simulated engine during one operating cycle**. Therefore, `(engine_id, cycle)` must be unique.

| Column | Meaning |
|---|---|
| `engine_id` | Identifier of the simulated engine/unit |
| `cycle` | Sequential operating-cycle number for that engine |
| `operational_setting_1` | First simulated operating condition |
| `operational_setting_2` | Second simulated operating condition |
| `operational_setting_3` | Third simulated operating condition |
| `sensor_1` ... `sensor_21` | Twenty-one anonymized sensor measurements |

## Day 2 modeling fields

| Field | Role |
|---|---|
| `rul` | Regression target: cycles remaining before simulated failure |
| `failure_within_30_cycles` | Binary target retained for later risk classification |
| `rul_capped` | Conceptual training target `min(rul, 125)`; not written back into the source CSV |
| `predicted_rul` | Champion model's estimated capped RUL |
| `residual` | `predicted_rul - actual_rul_capped`; positive means a late/risky estimate |

`engine_id` identifies groups but is not a model feature. It would encourage memorization of arbitrary engine numbers. Constant settings and sensors are also removed because they cannot separate healthy from degrading observations in FD001.

## Day 3 sequence and explanation fields

| Field | Meaning |
|---|---|
| `sequence_length` | Number of consecutive cycles in one LSTM input; 30 in Day 3 |
| `sequence.values` | Three-dimensional array: examples × 30 cycles × 18 features |
| `sequence.targets` | Capped RUL at the final cycle of each sequence |
| `sequence.engine_ids` | Engine identity retained for audit and grouping, not as a model feature |
| `sequence.cycles` | Original, unscaled endpoint cycle for each sequence |
| `random_forest_predicted_rul` | Random Forest estimate at the engine endpoint |
| `lstm_predicted_rul` | LSTM estimate from the engine's last 30 cycles |
| `mean_absolute_shap` | Average absolute Random Forest feature contribution over explained engines |
| `shap_contribution_cycles` | Signed contribution of one feature to one Random Forest prediction |
| `base_value` | TreeSHAP expected model output before local feature contributions |

For a local explanation, `base_value + sum(shap_contribution_cycles)` reconstructs the raw Random Forest output. Positive contributions increase predicted RUL; negative contributions decrease it. These are model effects, not claims of physical causation.

## Day 4 retrieval and generation fields

| Field | Meaning |
|---|---|
| `chunk_id` | Stable citation ID built from source filename and section heading |
| `sha256` | Fingerprint of exact chunk text for knowledge-version auditing |
| `score` | TF-IDF cosine similarity between the query and a retrieved chunk |
| `rank` | One-based retrieval position; 1 is the highest scoring chunk |
| `hit_at_3` | 1 when the labeled expected chunk appears in the first three results |
| `reciprocal_rank` | `1 / expected_rank`, or 0 when the expected chunk is not retrieved |
| `risk_band` | Review priority derived deterministically from deployment-model RUL |
| `advisory_lstm_rul_cycles` | Secondary sequence-model estimate used as disagreement evidence |
| `model_disagreement_cycles` | Absolute difference between deployment and advisory RUL estimates |
| `generator_provider` | Offline template or explicitly enabled OpenAI Responses provider |
| `generator_can_modify_rul` | Always false; RUL remains authoritative model output |

## Day 5 API and dashboard fields

| Field | Meaning |
|---|---|
| `observed_cycle` | Last available sensor cycle for one test engine |
| `predicted_rul` | Authoritative Random Forest capped-RUL estimate used for triage |
| `benchmark_actual_rul` | NASA-supplied endpoint truth used only for retrospective evaluation |
| `model_disagreement` | Absolute difference between Random Forest and LSTM endpoint estimates |
| `risk_counts` | Number of monitored engines in each deterministic review band |
| `highest_priority_engine` | Engine with the lowest deployment-model RUL in the current queue |
| `ready` | True only when every artifact needed by the app exists |
| `X-Request-ID` | Trace identifier returned with each API response |
| `X-Process-Time-Ms` | Server processing time for one request in milliseconds |

In a live system, `benchmark_actual_rul` would not exist because future failure time is unknown. It is exposed only to make evaluation behavior inspectable.

The official test target is not included in the copilot evidence packet because future failure time would not be known in real operation.

NASA anonymizes the physical meaning of the sensor columns in this public dataset. AeroGuard therefore refers to them by number and does not invent names or engineering units.

## Training versus test labels

The final training observation is the simulated failure point:

```text
training RUL = final cycle for that engine - current cycle
```

The test histories stop before failure. NASA supplies the RUL at the last observed test cycle:

```text
test-row RUL = supplied final RUL + last observed cycle - current cycle
```

That difference is why training and test labels use separate functions.
