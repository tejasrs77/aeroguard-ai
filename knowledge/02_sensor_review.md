# Sensor Evidence Review

This is project-authored demonstration guidance for AeroGuard. NASA C-MAPSS sensor names and units are anonymized.

## Recent sensor trend review

Review the ordered recent history, not only the latest value. Check whether influential signals move steadily, change suddenly, oscillate, or disagree with related signals. Compare observations with approved engineering limits and maintenance records; AeroGuard does not invent physical meanings for anonymized sensors.

## Data quality before escalation

Confirm engine identity, cycle order, duplicate absence, numeric completeness, expected feature order, and plausible collection continuity. A missing, swapped, stale, or incorrectly scaled sensor can create a confident-looking but invalid prediction. Resolve data-quality failures before interpreting model risk.

## SHAP interpretation limits

SHAP decomposes a model output into a baseline and signed feature contributions. A negative contribution lowers predicted RUL and a positive contribution raises it. SHAP explains what the Random Forest used; it does not prove that a sensor physically caused degradation, and the Random Forest explanation must not be presented as an LSTM explanation.
