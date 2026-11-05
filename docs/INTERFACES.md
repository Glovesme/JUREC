# Input and output contracts

All arrays are in-memory NumPy arrays. People are local zero-based integer positions in a caller-owned vocabulary; they are not employee identifiers. All four candidates are retained in the order `a00, a01, a10, a11`.

## Candidate construction

`jurec.candidates.build_candidate_pack` accepts keyword arguments:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `hcr_leader_scores` | N × P | HCR leader scores |
| `prr_leader_scores` | N × P | PRR leader scores |
| `hcr_worker_scores` | N × P | Shared base HCR worker scores |
| `leader_valid`, `worker_valid` | N × P, bool | Eligible personnel for each role |
| `conditional` | N × 2 × P | Association evidence for the HCR/PRR proposals |
| `support` | N × 2 | Distinct observed-roster fitting groups for each proposal |
| `ess` | N × 2 | Effective local-history sample size |
| `fit_group_count` | positive int | Number of distinct fitting groups |

The 47-feature extractor requires at least six personnel positions and two eligible leaders per query. Score arrays must be finite. `conditional` is in `[0,1]` and already includes the `support/(support+10)` shrinkage. Construct it with `conditional_association`, or supply equivalent fitting-only evidence. The four smoothed probability arrays for that helper have shape N × P.

The returned mapping contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x` | N × 47 | Shared numerical evidence |
| `leaders` | N × 4 | Proposed leader index for each candidate |
| `worker_top5` | N × 4 × 5 | Ordered worker indices; `-1` pads unfilled slots |
| `leader_changed` | N, bool | Whether the two leader proposals differ |
| `worker_changed` | N × 2, bool | Whether correction changes each branch's worker set |
| `support`, `ess` | N × 4 | Support and ESS repeated for both scoring variants |

`NUMERICAL_FEATURE_NAMES` in `jurec.features` specifies all 47 columns in exact order. `top_workers` independently supports fewer than five eligible workers. Do not count `-1` padding as a recommended person.

## Frozen-embedding history evidence

`jurec.features.build_cosine_descriptors` accepts query/history embedding matrices, explicit query/history groups, unique history tie-breaking keys, history text prefixes (or equality-preserving local tokens), historical leader/worker indices, and the candidate arrays above. Query groups must be absent from the histories. The supplied histories must already belong to the permitted fitting population.

Vectors are L2-normalized without fitting. Histories are ranked by cosine similarity, deduplicated by group and the first 160 text characters, and limited to the top two per leader, worker, or leader–worker pair. Locally supplied tokens can preserve the prefix equality test without exposing any text to this module. The function makes no external calls and does not log the tokens.

It returns N × 32 by default, or N × 4 × 8 with `flatten=False`. The eight candidate descriptors are leader cosine similarity, mean/minimum worker cosine similarity, mean/minimum paired-history cosine similarity, and leader/worker/pair coverage. Missing evidence has zero similarity and zero coverage. These are the raw eight-dimensional descriptors. The 16-dimensional historical feature block in the manuscript's preference input is assembled inside the model, as described in [Implementation details](METHOD.md#implementation-details); callers continue to supply N × 32 or N × 4 × 8 arrays.

## COVAL reference inputs

`jurec.reference.coval_reference` consumes predictions from fitted upstream models. It does not train these models:

| Argument | Shape and ordering |
| --- | --- |
| `leader_probabilities` | N × 3: neither, PRR correct, HCR correct |
| `base_worker_recall` | N × 2: conditional recall for PRR, then HCR |
| `worker_gain` | N × 2: correction gain for HCR, then PRR |
| `leader_changed` | N, bool |
| `worker_changed` | N × 2, bool |
| `contrast_predictions` | N × 5 × 4: members × cross-leader combinations |

Cross-leader combination order is `(c_HCR,c_PRR) = (0,0),(0,1),(1,0),(1,1)`. Positive contrast means the complete PRR assignment is better. Supply either `local_stats` with `local_net` and `local_uncertainty` vectors, or `x` and a fitting-only `gallery` for `local_signed_statistics`. The gallery contains `mean`, `scale`, `centroids`, `contrasts`, and `absolute_contrasts` arrays; it must be computed from eligible cross-fitted fitting groups. Its absolute contrasts are group means of absolute plan contrasts, not absolute group means.

The function returns `(choice, diagnostics)`. Choice is an N-vector of candidate indices. The COVAL predictor output orders deliberately follow the archived implementation; they differ from the conditional model's HCR/PRR branch order.

## Learning and inference

The following integration sketch uses caller-supplied arrays, labels, and held-group predictions. Prepare these inputs before running the calls; the [fitting boundaries](REPRODUCIBILITY.md#fitting-boundaries) define the labels.

```python
from jurec import JUREC
from jurec.model import inference_pack

# train_pack includes hit, recall, observed, and group_ids from fitting data.
model = JUREC.fit_heads(train_pack, train_descriptors)  # 600 updates per head

# cal_joint[N,5,4], cal_potential[N,5,4] are supplied held-group predictions.
model.calibrate_from_oof(
    calibration_pack, calibration_descriptors, calibration_coval_choice,
    cal_joint, cal_potential,
)
result = model.recommend(
    inference_pack(query_pack), query_descriptors, query_coval_choice,
)
```

`inference_pack` explicitly removes `hit`, `recall`, and `observed`. Inference methods reject inputs that still contain these supervision keys. The caller is responsible for ensuring that upstream features and predictions are also label-free and generated from permitted histories.

For finer control, `learning.fit_conditional` and `learning.fit_preference` fit individual members. `predict_conditional` returns `joint[N,M,4]`, `leader[N,M,2]` (HCR, PRR), and `worker[N,M,4]`; `predict_preference` returns `[N,M,4]`. The JUREC selector requires five members.

`JUREC.recommend` returns `initial_choice`, `reference_choice`, and `choice` vectors, `leader[N]`, and ordered `workers[N,5]`. It accepts no true leader or worker roster. Uncalibrated selector defaults retain the COVAL reference.

`model.save(path)` and `JUREC.load(path)` use JSON rather than pickle. Saving requires an explicit path; nothing is saved by default. Treat real-data fitted states as private. Save them outside a public repository or in an ignored local directory such as `checkpoints/`.
