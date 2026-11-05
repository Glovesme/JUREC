# Core architecture

## Candidate construction

Candidate order is fixed as `[a00, a01, a10, a11]`. The first index selects the HCR or PRR leader proposal; the second selects base or corrected HCR worker scores. Worker rankings always exclude the selected leader. Duplicate indexed assignments are retained. Output-change tests compare worker sets, while the returned shortlist preserves order.

With smoothed distributions supplied from fitting histories, the association is

```text
C = support/(support+10)
    * clip(0.5*(log(P_global/worker_prior)
               + log(P_local/local_worker_marginal)), 0, 2) / 2
```

The correction is `0.75 * worker_score_std * C * ESS/(ESS+3)`, restricted to the top twenty eligible base-ranked workers and to leader histories with at least ten observed-roster groups. The numerical-feature helper retains the additional correction strengths used by the archived shortlist-overlap features. These auxiliary shortlists do not enlarge the four final candidates.

## Evidence and networks

| Component | Dimensions / behavior |
| --- | --- |
| Shared numerical features | 47 = 35 ranking/history/overlap features + 12 leader-score diagnostics |
| Per-candidate cosine-history descriptor | 8; four candidates give 32 values |
| Historical feature block in the preference input | 16 per candidate |
| Conditional leader network | 51 → 16 (tanh) → 3 (softmax) |
| Conditional worker network | 57 → 16 (tanh) → 1 (sigmoid), shared across four candidates |
| Joint estimate | Leader-match probability × conditional worker recall |
| Conditional model parameters | 1,828 per member |
| Preference scorer | 65 → 16 (tanh) → 1, without output bias |
| Preference model parameters | 1,072 per member |
| Ensemble size | Five group-bootstrap members |

The conditional leader input contains the shared numerical vector and the two leader-history similarity/availability summaries. The worker input contains the 47 features, eight candidate history descriptors, and the two branch indicators.

The preference input follows Eq. (8) in the manuscript:

$$
\mathbf z_{ia}=[\mathbf f_i;\mathbf c_{ia};b;c],\qquad a=a_{bc}(i).
$$

Here, $\mathbf f_i$ contains the 47 numerical features, $\mathbf c_{ia}$ denotes the final historical feature block, and $b,c$ are the leader and worker-correction branch indicators.

Conditional training combines leader negative log-likelihood, worker squared error conditional on a correct leader, joint-utility squared error, and an explicit parameter penalty. Preference training uses all ordered unequal-utility candidate pairs and a logistic loss weighted by absolute joint-utility gaps. Group weights and bootstrap multiplicities determine each plan's contribution.

### Implementation details

The historical feature block $\mathbf c_{ia}$ comprises two identical copies of an eight-dimensional cosine-history descriptor, giving 16 historical features. Together with the 47 numerical features and two candidate indicators, this yields the implemented 65-dimensional preference input. Both copies contain the same historical evidence. Normalization statistics are computed from the corresponding fitting groups. The public descriptor interface retains eight values per candidate; `learning.preference_features` constructs the 16-dimensional block internally.

## Reference and final selection

COVAL accepts fitted provisional leader/worker predictions, five cross-leader contrast predictions, and a fitting-group evidence bank (or computed local moments). It preserves the directional ensemble and local net-evidence tests. Training these upstream predictors is outside this package.

CUS evaluates complete alternatives using conditional utility predictions. Leader changes use a calibrated reference-utility penalty; worker-only changes use the ordinary utility contrast. CUS checks history availability at the plan level: at least one of the four candidates must have available leader, worker, or paired-history evidence. The resulting flag is shared across all candidate revisions for that plan. The final preference gate requires both the alternative and its reference to have historical evidence independently; each satisfies this condition when any of its leader, worker, or paired-history coverage values is positive.

Final selection compares each member's potential for an alternative with its potential for the reference. At least four of the five differences must be positive, the output must change, and the mean difference minus `kappa` times the sample standard deviation must exceed the threshold. The largest eligible score wins; ties use candidate order. Otherwise the reference is kept.
