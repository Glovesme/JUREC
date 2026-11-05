"""JUREC's four factorized proposals and bounded conditional worker correction.

This module starts at caller-supplied HCR/PRR scores and fit-only historical
evidence. It does not implement the confidential application's history loader,
HCR/PRR training, or embedding model. Person indices are local integer positions
in the caller's candidate vocabulary; no application identifiers are required.
"""
from __future__ import annotations

import numpy as np

ACTION_NAMES = ("a00", "a01", "a10", "a11")
DIAGNOSTIC_ALPHAS = (0.0, 0.25, 0.75, 1.5)


def _finite(name, value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _mask(name, value, shape):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind != "b":
        raise ValueError(f"{name} must be a Boolean array with shape {shape}")
    return result


def _indices(name, value, shape, people):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with shape {shape}")
    if np.any((result < -1) | (result >= people)):
        raise ValueError(f"{name} indices must be -1 or in [0, {people})")
    return result.astype(np.int64, copy=False)


def predicted_leaders(scores, valid):
    """Stable argmax over eligible leaders; return -1 for an empty row."""
    scores = _finite("scores", scores)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("scores must be a nonempty-vocabulary (N, P) array")
    valid = _mask("valid", valid, scores.shape)
    result = np.argmax(np.where(valid, scores, -np.inf), axis=1)
    result[~valid.any(axis=1)] = -1
    return result


def top_workers(scores, valid, leaders):
    """Return stable descending top five, excluding the leader, padded with -1."""
    scores = _finite("scores", scores)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (N, P)")
    n, p = scores.shape
    valid = _mask("valid", valid, scores.shape)
    leaders = _indices("leaders", leaders, (n,), p)
    eligible = valid.copy()
    rows = np.flatnonzero(leaders >= 0)
    eligible[rows, leaders[rows]] = False
    ids = np.argsort(-np.where(eligible, scores, -np.inf), axis=1, kind="stable")[:, :5]
    result = np.full((n, 5), -1, dtype=np.int64)
    result[:, :ids.shape[1]] = np.where(np.take_along_axis(eligible, ids, 1), ids, -1)
    return result


def conditional_association(*, global_conditional, worker_prior,
                            local_conditional, local_marginal, support,
                            worker_valid):
    """Build the archived positive association evidence from smoothed probabilities.

    The four probability inputs are (N, P); they must be positive at eligible
    workers. ``support[N]`` is the number of distinct fitting groups with an
    observed roster for the proposed leader. The caller supplies distributions
    estimated using permitted fitting histories only; this function cannot audit
    their provenance. The formula is

      support/(support+10) * clip((log(global/prior)+log(local/marginal))/2,0,2)/2.

    In the experiment, global and local distributions used additive smoothing
    with beta=10; local neighbors were the top 20 distinct groups with weights
    exp((similarity-best_similarity)/0.1). All query-held groups were excluded.
    The output ALREADY includes support/(support+10), so do not apply it again
    before calling ``continuous_worker_scores``.
    """
    glob = _finite("global_conditional", global_conditional)
    if glob.ndim != 2:
        raise ValueError("probabilities must have shape (N, P)")
    prior = _finite("worker_prior", worker_prior, glob.shape)
    local = _finite("local_conditional", local_conditional, glob.shape)
    marginal = _finite("local_marginal", local_marginal, glob.shape)
    valid = _mask("worker_valid", worker_valid, glob.shape)
    support = _finite("support", support, (len(glob),))
    if np.any(support < 0):
        raise ValueError("support must be nonnegative")
    for value in (glob, prior, local, marginal):
        if np.any(value[valid] <= 0) or np.any(value[valid] > 1 + 1e-12):
            raise ValueError("eligible-worker probabilities must be in (0, 1]")
    lift = np.zeros_like(glob)
    lift[valid] = .5 * (np.log(glob[valid] / prior[valid]) +
                       np.log(local[valid] / marginal[valid]))
    return support[:, None] / (support[:, None] + 10.) * np.clip(lift, 0., 2.) / 2.


def continuous_worker_scores(anchor, worker_valid, leaders, conditional,
                             support, ess, *, alpha=0.75):
    """Apply the frozen top-20 correction; preserve anchor scores if support < 10.

    Inputs: ``anchor, worker_valid, conditional`` have shape (N,P), and
    ``leaders, support, ess`` have shape (N,). Conditional evidence is in [0,1]
    and includes the support shrinkage produced by ``conditional_association``.
    ESS is (sum weights)^2 / sum squared weights from fit-only leader histories.
    Only the top-20 eligible base workers can be promoted. Their positive
    residual is alpha * std(anchor eligible workers) * conditional * ESS/(ESS+3).
    The leader is excluded when defining the eligible pool. Ineligible scores
    remain unchanged; final ranking must still apply the role mask.
    """
    anchor = _finite("anchor", anchor)
    if anchor.ndim != 2:
        raise ValueError("anchor must have shape (N, P)")
    n, p = anchor.shape
    valid = _mask("worker_valid", worker_valid, anchor.shape)
    leaders = _indices("leaders", leaders, (n,), p)
    conditional = _finite("conditional", conditional, anchor.shape)
    support = _finite("support", support, (n,))
    ess = _finite("ess", ess, (n,))
    if np.any((conditional < 0) | (conditional > 1)):
        raise ValueError("conditional evidence must lie in [0, 1]")
    if np.any(support < 0) or np.any(ess < 0):
        raise ValueError("support and ess must be nonnegative")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    eligible = valid.copy()
    rows = np.flatnonzero(leaders >= 0)
    eligible[rows, leaders[rows]] = False
    order = np.argsort(-np.where(eligible, anchor, -np.inf), axis=1, kind="stable")[:, :20]
    pool = np.zeros_like(eligible)
    pool[np.arange(n)[:, None], order] = True
    pool &= eligible
    count = eligible.sum(axis=1).clip(min=1)
    mean = np.where(eligible, anchor, 0.).sum(axis=1) / count
    scale = np.sqrt(np.where(eligible, (anchor - mean[:, None]) ** 2, 0.).sum(axis=1) / count)
    rho = ess / (ess + 3.)
    delta = alpha * scale[:, None] * conditional * pool * ((support >= 10) * rho)[:, None]
    result = anchor.copy()
    change = delta > 0
    result[change] += delta[change]
    return result


def build_candidate_pack(*, hcr_leader_scores, prr_leader_scores,
                         hcr_worker_scores, leader_valid, worker_valid,
                         conditional, support, ess, fit_group_count):
    """Construct proposals a00,a01,a10,a11 and their archived 47 input features.

    Score/mask arrays have shape (N,P). ``conditional[N,2,P]``, ``support[N,2]``
    and ``ess[N,2]`` correspond to the HCR then PRR proposed leader. The model
    vocabulary needs P>=6 and at least two eligible leaders in every row, as
    required by the archived feature extractor. ``fit_group_count`` is the
    number of distinct fitting groups, used for history-size normalization.

    a00=(HCR leader, original HCR workers); a01=(HCR leader, corrected workers);
    a10=(PRR leader, original HCR workers); a11=(PRR leader, corrected workers).
    Each shortlist excludes its selected leader. Correction alpha is 0.75;
    alphas (0,.25,.75,1.5) are used only to construct overlap diagnostics.

    Returns x[N,47], leaders[N,4], worker_top5[N,4,5], leader_changed[N],
    worker_changed[N,2], support[N,4], ess[N,4]. No ground-truth labels are read.
    """
    from .features import numerical_features

    hcr_l = _finite("hcr_leader_scores", hcr_leader_scores)
    if hcr_l.ndim != 2 or hcr_l.shape[0] == 0 or hcr_l.shape[1] < 6:
        raise ValueError("score arrays require N>0 and P>=6 for archived features")
    n, p = hcr_l.shape
    prr_l = _finite("prr_leader_scores", prr_leader_scores, (n, p))
    base = _finite("hcr_worker_scores", hcr_worker_scores, (n, p))
    leader_mask = _mask("leader_valid", leader_valid, (n, p))
    worker_mask = _mask("worker_valid", worker_valid, (n, p))
    if np.any(leader_mask.sum(axis=1) < 2):
        raise ValueError("archived leader diagnostics require at least two eligible leaders")
    cond = _finite("conditional", conditional, (n, 2, p))
    support = _finite("support", support, (n, 2))
    ess = _finite("ess", ess, (n, 2))
    if isinstance(fit_group_count, bool) or not isinstance(fit_group_count, (int, np.integer)) or fit_group_count <= 0:
        raise ValueError("fit_group_count must be a positive integer")
    if np.any(support > fit_group_count):
        raise ValueError("leader group support cannot exceed fit_group_count")
    leader_pair = np.column_stack([predicted_leaders(hcr_l, leader_mask),
                                   predicted_leaders(prr_l, leader_mask)])
    anchor = np.where(worker_mask, base, 0.)
    diagnostics, corrected = [], []
    for branch in range(2):
        scores = [continuous_worker_scores(anchor, worker_mask, leader_pair[:, branch],
                  cond[:, branch], support[:, branch], ess[:, branch], alpha=alpha)
                  for alpha in DIAGNOSTIC_ALPHAS]
        diagnostics.append(np.stack([top_workers(s, worker_mask, leader_pair[:, branch]) for s in scores], axis=1))
        corrected.append(scores[2])
    leaders = leader_pair[:, [0, 0, 1, 1]]
    workers = np.stack([top_workers(scores, worker_mask, leaders[:, a]) for a, scores
                        in enumerate((anchor, corrected[0], anchor, corrected[1]))], axis=1)
    pack = {
        "leaders": leaders,
        "worker_top5": workers,
        "leader_changed": leader_pair[:, 1] != leader_pair[:, 0],
        "worker_changed": np.stack([np.any(np.sort(workers[:, k + 1], axis=1) !=
                                           np.sort(workers[:, k], axis=1), axis=1)
                                    for k in (0, 2)], axis=1),
        "support": support[:, [0, 0, 1, 1]],
        "ess": ess[:, [0, 0, 1, 1]],
    }
    pack["x"] = numerical_features(
        anchor=anchor, leader_scores=np.stack([hcr_l, prr_l], axis=1),
        leader_valid=leader_mask, worker_valid=worker_mask, leaders=leader_pair,
        conditional=cond, support=support, ess=ess, fit_group_count=fit_group_count,
        diagnostic_top5=np.stack(diagnostics, axis=1), worker_top5=workers)
    return pack
