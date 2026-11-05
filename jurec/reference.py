"""COVAL reference inference used before JUREC's conditional/regret selection.

This is an inference interface, not the private upstream fitting pipeline. Supply
cross-fitted provisional head predictions and a fitting-only evidence gallery.
Candidate order is (a00, a01, a10, a11); the first bit selects HCR/PRR and the
second selects the uncorrected/conditionally corrected worker shortlist.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

EPS = 1e-12
GAIN_MARGIN = 0.002


def _finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return value


def local_signed_moments(weights, signed_contrasts, absolute_contrasts):
    """Return COVAL's continuous signed local evidence with two pseudo-groups.

    Inputs are N x K matrices. Absolute contrasts are *group means of absolute
    plan contrasts*, not the absolute values of group means. The pseudo-second
    moment contributes 0.5; positive and negative prior masses each contribute
    0.5. These constants and the effective-group formula match the source.
    """
    q = np.asarray(weights, dtype=float)
    if q.ndim != 2 or q.shape[1] == 0 or not np.isfinite(q).all():
        raise ValueError("weights must be a finite N x K matrix with K > 0")
    d = _finite(signed_contrasts, q.shape, "signed_contrasts")
    a = _finite(absolute_contrasts, q.shape, "absolute_contrasts")
    if np.any(q < 0) or np.any(q.sum(1) <= 0) or np.any(a < np.abs(d) - EPS):
        raise ValueError("invalid weights or absolute contrast moments")
    q = q / q.sum(1, keepdims=True)
    n = 1 / (q * q).sum(1)
    gain = (q * (a + d) * 0.5).sum(1)
    loss = (q * (a - d) * 0.5).sum(1)
    mean = n * (gain - loss) / (n + 2)
    second = (n * (q * d * d).sum(1) + 0.5) / (n + 2)
    uncertainty = np.sqrt(np.maximum(second - mean * mean, 0) / (n + 2))
    return {
        "local_positive": (n * gain + 0.5) / (n + 2),
        "local_negative": (n * loss + 0.5) / (n + 2),
        "local_net": mean,
        "local_uncertainty": uncertainty,
        "local_effective_groups": n,
    }


def local_signed_statistics(x, combo, gallery: Mapping):
    """Retrieve at most 32 fitting-group centroids and calculate local moments.

    ``gallery`` contains mean/scale (47,), standardized centroids (G,47), and
    contrasts/absolute_contrasts (G,4). Column ``2*c_HCR+c_PRR`` compares the
    corresponding final PRR action against the corresponding final HCR action.
    Group aggregation and all gallery statistics must precede held-out inference.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 47 or not np.isfinite(x).all():
        raise ValueError("x must be a finite N x 47 matrix")
    combo = np.asarray(combo)
    if combo.shape != (len(x),) or not np.isin(combo, np.arange(4)).all():
        raise ValueError("combo must contain one integer in [0,3] per query")
    combo = combo.astype(int)
    mean = _finite(gallery["mean"], (47,), "gallery mean")
    scale = _finite(gallery["scale"], (47,), "gallery scale")
    centers = np.asarray(gallery["centroids"], dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 47 or len(centers) == 0:
        raise ValueError("gallery centroids must have shape G x 47 with G > 0")
    if not np.isfinite(centers).all() or np.any(scale <= 0):
        raise ValueError("gallery centroids/scale are invalid")
    contrasts = _finite(gallery["contrasts"], (len(centers), 4), "gallery contrasts")
    absolute = _finite(gallery["absolute_contrasts"], (len(centers), 4), "gallery absolute contrasts")
    z = np.clip((x - mean) / scale, -5, 5)
    k = min(32, len(centers))
    parts = []
    for start in range(0, len(x), 128):
        zz = z[start:start + 128]
        distance = ((zz[:, None, :] - centers[None, :, :]) ** 2).mean(2)
        ix = np.argsort(distance, axis=1, kind="stable")[:, :k]
        nearest = np.take_along_axis(distance, ix, axis=1)
        bandwidth = np.maximum(np.median(nearest, axis=1), 1e-8)
        weights = np.exp(-nearest / bandwidth[:, None])
        c = combo[start:start + len(zz)]
        parts.append(local_signed_moments(
            weights, contrasts[ix, c[:, None]], absolute[ix, c[:, None]],
        ))
    keys = ("local_positive", "local_negative", "local_net", "local_uncertainty", "local_effective_groups")
    return {key: np.concatenate([part[key] for part in parts]) if parts else np.empty(0)
            for key in keys}


def coval_reference(
    leader_probabilities,
    base_worker_recall,
    worker_gain,
    leader_changed,
    worker_changed,
    contrast_predictions,
    *,
    local_stats=None,
    x=None,
    gallery=None,
):
    """Compute the frozen COVAL choice and source-compatible diagnostics.

    ``leader_probabilities`` is N x 3 in [neither, PRR correct, HCR correct]
    order; ``base_worker_recall`` is N x 2 in [PRR, HCR] order. ``worker_gain``
    is N x 2 in [HCR, PRR] order and equals positive-minus-negative predicted
    correction utility. ``contrast_predictions`` is N x 5 x 4: five resampled
    final-action contrast heads and the four cross-leader worker combinations.

    The provisional switch needs positive expected utility and a greater PRR
    correctness probability. COVAL permits either switch direction only with four
    favorable heads and mean advantage > 0.002 + sample SD. Additional switches
    also need local net - 2*local uncertainty > 0.002; vetoes of provisional switches
    do not require that local condition. Insufficient evidence retains the
    provisional choice exactly.
    The history<10 fallback and local continuous worker shrinkage belong to
    candidate construction; they are not reapplied as a new final-choice veto.
    """
    probability = np.asarray(leader_probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError("leader_probabilities must have shape N x 3")
    n = len(probability)
    probability = _finite(probability, (n, 3), "leader_probabilities")
    recall = _finite(base_worker_recall, (n, 2), "base_worker_recall")
    worker_gain = _finite(worker_gain, (n, 2), "worker_gain")
    prediction = _finite(contrast_predictions, (n, 5, 4), "contrast_predictions")
    changed = np.asarray(leader_changed, dtype=bool)
    worker_changed = np.asarray(worker_changed, dtype=bool)
    if changed.shape != (n,) or worker_changed.shape != (n, 2):
        raise ValueError("change masks must have shapes N and N x 2")
    if np.any(probability < 0) or not np.allclose(probability.sum(1), 1, atol=1e-8):
        raise ValueError("leader_probabilities must be normalized probabilities")
    if np.any(recall < 0) or np.any(recall > 1):
        raise ValueError("base_worker_recall must be in [0,1]")
    expected_gain = probability[:, 1] * recall[:, 0]
    expected_harm = probability[:, 2] * recall[:, 1]
    base_pair = ((expected_gain - expected_harm > GAIN_MARGIN + EPS)
                 & changed & (probability[:, 1] > probability[:, 2] + EPS))
    worker_branch = ((worker_gain > GAIN_MARGIN + EPS) & worker_changed).astype(int)
    actions = np.column_stack([worker_branch[:, 0], 2 + worker_branch[:, 1]])
    rows = np.arange(n)
    original = actions[rows, base_pair.astype(int)]
    combo = 2 * worker_branch[:, 0] + worker_branch[:, 1]
    selected_predictions = prediction[rows, :, combo]
    mean = selected_predictions.mean(1)
    sd = selected_predictions.std(1, ddof=1)
    direction = np.where(base_pair, -1.0, 1.0)
    votes = (selected_predictions * direction[:, None] > GAIN_MARGIN + EPS).sum(1)
    eligible = ((direction * mean > GAIN_MARGIN + sd + EPS) & (votes >= 4) & changed)
    if local_stats is None:
        if x is None or gallery is None:
            raise ValueError("provide local_stats, or both x and a fitting-only gallery")
        local_stats = local_signed_statistics(x, combo, gallery)
    local_net = _finite(local_stats["local_net"], (n,), "local_net")
    local_uncertainty = _finite(local_stats["local_uncertainty"], (n,), "local_uncertainty")
    if np.any(local_uncertainty < 0):
        raise ValueError("local_uncertainty must be nonnegative")
    margin_score = direction * local_net - np.where(base_pair, local_uncertainty, 2 * local_uncertainty)
    local_pass = np.where(base_pair, True, margin_score > GAIN_MARGIN + EPS)
    take = eligible & local_pass
    choice = actions[rows, np.where(take, ~base_pair, base_pair).astype(int)]
    diagnostics = {
        "original_b5_choice": original,
        "original_e2_choice": actions[rows, np.where(eligible, ~base_pair, base_pair).astype(int)],
        "j3_final_action": actions[:, 0], "pair_final_action": actions[:, 1],
        "route_actions": actions, "combo": combo,
        "predicted_final_contrast": mean, "resampling_sd": sd,
        "direction_votes": votes, "e2_eligible": eligible,
        "net_evidence_pass": local_pass, "net_margin_score": margin_score,
        "overridden": take, "expected_gain": expected_gain, "expected_harm": expected_harm,
        **{key: np.asarray(value) for key, value in local_stats.items()},
    }
    return choice, diagnostics
