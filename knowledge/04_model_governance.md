# Model Governance and Monitoring

This guide defines the boundary between an ML experiment and a production safety process.

## Deployment model boundary

The persisted Random Forest is the current explainable deployment champion. The LSTM is an advisory promotion candidate. An LLM or template may describe supplied evidence but must never recalculate, replace, or silently round the authoritative RUL field.

## Late warning risk

Overestimating RUL can delay maintenance, so AeroGuard reports NASA's asymmetric score in addition to MAE and RMSE. Review the late-warning tail, not only average error. A candidate with better RMSE can still be held from promotion when its operational-risk score or auditability is worse.

## Simulation and monitoring limitations

C-MAPSS is simulated and FD001 contains one operating condition and one degradation mode. Performance does not establish safety on physical aircraft or other operating regimes. A production system needs external validation, drift monitoring, uncertainty handling, access control, audit logging, and formal approval before operational use.
