# Fitting and reproducibility scope

## What can be run

The released code constructs candidates and features from supplied upstream arrays, trains conditional and preference heads, applies COVAL/CUS/final selection, and calibrates policies from supplied held-group predictions. Callers supply the upstream arrays and fitting labels, or previously fitted head states for inference. See [Input and output contracts](INTERFACES.md) for the required inputs and API calls.

## Fitting boundaries

Candidate scores and history statistics for downstream training must be generated without using the corresponding held group's labels. Fit text/score transforms, history retrieval banks, personnel candidate sets, and normalization statistics using only the permitted fitting groups. The cosine-history helper rejects query groups that are present in its supplied history population.

Training inputs use these labels:

- `hit[n, a]`: whether candidate `a` has the recorded leader.
- `recall[n, a]`: worker recall against the complete observed roster.
- `observed[n]`: whether a nonempty worker roster is observed.
- `group_ids[n]`: group membership, used for macro weighting and resampling.

An unobserved roster is not an empty roster. Its worker recall may be NaN; its worker and joint targets are excluded. Leader labels can still contribute to the conditional leader loss. A preference-training population with no nonzero utility differences produces an inactive zero-potential head.

Five separately bootstrapped group samples train each ensemble. Default optimization uses float64, AdamW, learning rate 0.003, and 600 updates. Explicit mean-square parameter penalties have coefficients 0.1 for conditional utility and 1.0 for preference learning. The initialization seed is fixed at 20261018; the five default bootstrap seeds are 20261019 through 20261023. Distinguish these within-ensemble seeds from the paper's independently fitted upstream runs.

## Calibration

Policy calibration consumes held-group predictions supplied by the caller. It must not use an outer evaluation population. Native conditional and risk-policy selection retain their archived group-macro utility ranking. Final preference calibration ranks policies by mean group utility gain minus one group standard error and includes the keep-reference policy. A revision policy must improve this criterion by more than 1e-12.

The industrial experiments reused selected upstream policies within an outer fitting population. This release preserves the decision calculations but does not provide the original experiment scheduler or claim complete layer-by-layer isolation of that archived policy-selection procedure. Users requiring fully nested policy selection must construct the corresponding prediction populations.

## Reproducing the paper's numerical results

The released core alone cannot reproduce the paper's tables. That requires the confidential dataset, matching fitting populations, HCR/PRR models and features, COVAL predictors and evidence banks, trained ensemble weights, calibrated policies, and the original evaluation protocol. Paper result tables and the underlying evaluation outputs are outside this core implementation release.
