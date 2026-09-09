# RUL Interpretation Guide

This is project-authored demonstration guidance for AeroGuard. It is not an approved aircraft maintenance manual.

## RUL risk bands

AeroGuard uses four decision-support bands: critical at 0–15 predicted cycles, high at more than 15–30 cycles, elevated at more than 30–60 cycles, and routine above 60 cycles. A band controls review priority only. It does not independently determine airworthiness, dispatch, grounding, or a maintenance task.

## Model disagreement

Compare the deployed Random Forest estimate with the advisory LSTM estimate. Large disagreement is a reason to inspect data quality, lifecycle coverage, and uncertainty rather than averaging the values without justification. The deployed model remains the numerical source of truth until a documented promotion decision changes it.

## Capped RUL target

The models cap the learning target at 125 cycles because exact early-life RUL is weakly observable and less important for near-term maintenance. A prediction near the cap means "not presently in the modeled degradation region," not a guarantee of exactly 125 healthy cycles.
