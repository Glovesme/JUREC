"""JUREC's archived 47 numerical and 4 x 8 cosine semantic descriptors.

Numerical order (zero based): 0:15 HCR branch, 15:30 PRR branch, 30 leader
agreement, 31:33 score-span-normalized proposal gaps (HCR then PRR), 33 overlap
(a00,a01), 34 overlap(a00,a11). Each 15-feature branch contains normalized
log support, ESS shrinkage, leader confidence, its square, worker gap at ranks
5/6, worker span at ranks 1/5, conditional-evidence mean at ranks 1:5, mean at
ranks 6:20, their difference, confidence*shrinkage, confidence*gap,
shrinkage*evidence-difference, and shortlist overlaps of alphas 0/.25/1.5 with
alpha .75. Features 35:47 interleave HCR/PRR values for standardized top-two
leader gap, standardized own-versus-other-proposal gap, other reciprocal rank,
normalized top-ten entropy, top softmax probability, and worker eligibility.

Semantic descriptors are candidate-major (a00,a01,a10,a11). Each block contains
leader cosine; mean/min worker cosine; mean/min leader-worker cosine; leader
coverage; worker coverage; pair coverage. Means/minima ignore missing evidence;
coverage divides by five, including absent shortlist slots.
"""
from __future__ import annotations

import numpy as np

from .candidates import _finite, _mask, _indices

BRANCH_FEATURE_NAMES = (
    "normalized_log_support", "local_ess_shrinkage", "leader_confidence",
    "leader_confidence_squared", "worker_rank5_rank6_gap", "worker_rank1_rank5_span",
    "top5_conditional_mean", "ranks6to20_conditional_mean", "conditional_mean_difference",
    "confidence_times_shrinkage", "confidence_times_worker_gap",
    "shrinkage_times_conditional_difference", "overlap_alpha0_alpha075",
    "overlap_alpha025_alpha075", "overlap_alpha15_alpha075",
)
LEADER_DIAGNOSTIC_NAMES = (
    "standardized_top2_gap", "standardized_proposal_gap", "other_reciprocal_rank",
    "top10_normalized_entropy", "top_softmax_probability", "worker_eligibility",
)
NUMERICAL_FEATURE_NAMES = tuple(
    f"{branch}_{name}" for branch in ("hcr", "prr") for name in BRANCH_FEATURE_NAMES
) + ("leader_agreement", "hcr_proposal_span_gap", "prr_proposal_span_gap",
     "overlap_a00_a01", "overlap_a00_a11") + tuple(
    f"{branch}_{name}" for name in LEADER_DIAGNOSTIC_NAMES for branch in ("hcr", "prr")
)
SEMANTIC_FEATURE_NAMES = (
    "leader_cosine", "worker_cosine_mean", "worker_cosine_min",
    "pair_cosine_mean", "pair_cosine_min", "leader_coverage",
    "worker_coverage", "pair_coverage",
)


def _overlap(a, b, *, include_padding=False):
    if include_padding:
        return np.asarray([len(set(x) & set(y)) / 5 for x, y in zip(a, b)])
    return np.asarray([len(set(x[x >= 0]) & set(y[y >= 0])) / 5 for x, y in zip(a, b)])


def leader_confidence(scores, valid):
    """Top-two gap divided by the top-to-rank-min(10,count) span; not a probability."""
    scores = _finite("scores", scores)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (N, P)")
    valid = _mask("valid", valid, scores.shape)
    result = np.zeros(len(scores))
    for i, (row, mask) in enumerate(zip(scores, valid)):
        values = np.sort(row[mask])[::-1]
        if len(values) == 1:
            result[i] = 1.
        elif len(values) > 1:
            span = values[0] - values[min(len(values), 10) - 1]
            if span > 0:
                result[i] = (values[0] - values[1]) / span
    return result


def numerical_features(*, anchor, leader_scores, leader_valid, worker_valid,
                       leaders, conditional, support, ess, fit_group_count,
                       diagnostic_top5, worker_top5):
    """Return (N,47) in NUMERICAL_FEATURE_NAMES order, from candidate-builder arrays.

    This lower-level function accepts the arrays from build_candidate_pack:
    anchor[N,P], leader_scores[N,2,P], masks[N,P], leaders[N,2], conditional[N,2,P],
    support/ess[N,2], diagnostic_top5[N,2,4,5], worker_top5[N,4,5]. P>=6 and >=2
    eligible leaders per row are required. The original within-branch overlap
    diagnostic counts shared -1 padding once; cross-candidate overlap excludes
    padding. This detail is retained for exact numerical compatibility.
    """
    anchor = _finite("anchor", anchor)
    if anchor.ndim != 2:
        raise ValueError("anchor must have shape (N, P)")
    n, p = anchor.shape
    leader_scores = _finite("leader_scores", leader_scores, (n, 2, p))
    leader_valid = _mask("leader_valid", leader_valid, (n, p))
    worker_valid = _mask("worker_valid", worker_valid, (n, p))
    leaders = _indices("leaders", leaders, (n, 2), p)
    conditional = _finite("conditional", conditional, (n, 2, p))
    support = _finite("support", support, (n, 2))
    ess = _finite("ess", ess, (n, 2))
    diagnostic_top5 = _indices("diagnostic_top5", diagnostic_top5, (n, 2, 4, 5), p)
    worker_top5 = _indices("worker_top5", worker_top5, (n, 4, 5), p)
    if isinstance(fit_group_count, bool) or not isinstance(fit_group_count, (int, np.integer)) or fit_group_count <= 0:
        raise ValueError("fit_group_count must be a positive integer")
    if np.any(support < 0) or np.any(support > fit_group_count) or np.any(ess < 0):
        raise ValueError("support must lie in [0, fit_group_count] and ess must be nonnegative")
    if np.any((conditional < 0) | (conditional > 1)):
        raise ValueError("conditional evidence must lie in [0, 1]")
    if p < 6 or np.any(leader_valid.sum(axis=1) < 2):
        raise ValueError("archived features require P>=6 and >=2 eligible leaders")
    rows = np.arange(n)
    if np.any(leaders < 0) or not leader_valid[rows[:, None], leaders].all():
        raise ValueError("both proposed leaders must be eligible")
    branches, gaps, extra = [], [], []
    for branch in range(2):
        leader = leaders[:, branch]
        score = leader_scores[:, branch]
        eligible = worker_valid.copy()
        eligible[rows, leader] = False
        order = np.argsort(-np.where(eligible, anchor, -np.inf), axis=1, kind="stable")[:, :20]
        values = np.take_along_axis(anchor, order, 1)
        count = eligible.sum(axis=1).clip(min=1)
        mean = np.where(eligible, anchor, 0.).sum(axis=1) / count
        sd = np.sqrt(np.where(eligible, (anchor - mean[:, None]) ** 2, 0.).sum(axis=1) / count).clip(min=1e-8)
        cond = np.take_along_axis(conditional[:, branch], order, 1)
        confidence = leader_confidence(score, leader_valid)
        rho = ess[:, branch] / (ess[:, branch] + 3.)
        gap = np.clip((values[:, 4] - values[:, 5]) / sd, 0, 10)
        span = np.clip((values[:, 0] - values[:, 4]) / sd, 0, 10)
        early, late = cond[:, :5].mean(axis=1), cond[:, 5:].mean(axis=1)
        difference = late - early
        tops = diagnostic_top5[:, branch]
        overlaps = [_overlap(tops[:, k], tops[:, 2], include_padding=True) for k in (0, 1, 3)]
        branches.append(np.column_stack([
            np.log1p(1000 * support[:, branch] / fit_group_count), rho, confidence,
            confidence ** 2, gap, span, early, late, difference, confidence * rho,
            confidence * gap, rho * difference, *overlaps,
        ]))
        leader_order = np.argsort(-np.where(leader_valid, score, -np.inf), axis=1, kind="stable")
        count_l = leader_valid.sum(axis=1)
        other = leaders[:, 1 - branch]
        bottom = leader_order[rows, np.maximum(np.minimum(count_l, 10) - 1, 0)]
        lspan = score[rows, leader] - score[rows, bottom]
        lgap = np.divide(score[rows, leader] - score[rows, other], lspan,
                         out=np.zeros(n), where=lspan > 1e-8)
        gaps.append(np.clip(lgap, 0, 5))
        lmean = np.where(leader_valid, score, 0).sum(axis=1) / count_l
        lsd = np.sqrt(np.where(leader_valid, (score - lmean[:, None]) ** 2, 0).sum(axis=1) / count_l).clip(min=1e-8)
        ranks = np.empty_like(leader_order)
        ranks[rows[:, None], leader_order] = np.arange(p)[None, :] + 1
        z = (np.take_along_axis(score, leader_order[:, :10], 1) - lmean[:, None]) / lsd[:, None]
        keep = np.arange(z.shape[1])[None, :] < np.minimum(count_l, 10)[:, None]
        z = np.where(keep, z, -np.inf)
        z -= np.max(z, axis=1, keepdims=True)
        prob = np.exp(z)
        prob /= prob.sum(axis=1, keepdims=True)
        entropy = -(prob * np.log(prob.clip(min=1e-300))).sum(axis=1) / np.log(np.minimum(count_l, 10))
        extra.append([
            (score[rows, leader] - score[rows, leader_order[:, 1]]) / lsd,
            (score[rows, leader] - score[rows, other]) / lsd,
            1 / ranks[rows, other], entropy, prob[:, 0], worker_valid[rows, leader].astype(float),
        ])
    result = np.column_stack([
        *branches, leaders[:, 0] == leaders[:, 1], *gaps,
        _overlap(worker_top5[:, 0], worker_top5[:, 1]),
        _overlap(worker_top5[:, 0], worker_top5[:, 3]),
        *[extra[branch][k] for k in range(6) for branch in range(2)],
    ])
    if result.shape != (n, 47) or not np.isfinite(result).all():
        raise ValueError("numerical descriptor construction produced invalid features")
    return result


def build_cosine_descriptors(*, query_embeddings, history_embeddings,
                             query_groups, history_groups, history_keys,
                             history_text_prefixes, history_leaders,
                             history_workers, leaders, worker_top5, flatten=True):
    """Build frozen-embedding semantic descriptors from explicit fitting histories.

    Inputs: query_embeddings[N,D], history_embeddings[H,D], query_groups[N],
    history_groups/history_keys/history_text_prefixes/history_leaders[H],
    history_workers is H iterables of integer person indices, leaders[N,4],
    worker_top5[N,4,5]. A -1 denotes a missing candidate. ``history_keys`` must be
    unique strings and define deterministic ascending tie-breaking. Deduplication
    uses distinct history groups and distinct supplied text prefixes truncated to
    160 characters, exactly as the original retrieval. To avoid passing text,
    callers may supply equality-preserving local tokens for these prefixes.

    All query groups must be absent from history; an overlap raises ValueError.
    No held labels are consumed. Vectors are L2-normalized without fitting; zero
    vectors are rejected. For a leader, worker, or leader-worker association, use
    the mean cosine of the top two matching histories after deduplication. Missing
    evidence contributes zero and a false availability flag. Returns [N,32], or
    [N,4,8] with flatten=False, in SEMANTIC_FEATURE_NAMES order.
    """
    query = _finite("query_embeddings", query_embeddings)
    history = _finite("history_embeddings", history_embeddings)
    if query.ndim != 2 or history.ndim != 2 or query.shape[1] != history.shape[1] or query.shape[1] == 0:
        raise ValueError("embedding arrays must share a positive feature dimension")
    n, h = len(query), len(history)
    qg, hg = np.asarray(query_groups), np.asarray(history_groups)
    keys, prefixes = np.asarray(history_keys), np.asarray(history_text_prefixes)
    hl = np.asarray(history_leaders)
    if qg.shape != (n,) or any(x.shape != (h,) for x in (hg, keys, prefixes, hl)) or len(history_workers) != h:
        raise ValueError("group and history metadata lengths must match embeddings")
    if keys.dtype.kind not in "US" or prefixes.dtype.kind not in "US" or len(set(keys.tolist())) != h:
        raise ValueError("history_keys must be unique strings and prefixes must be strings")
    if set(qg.tolist()) & set(hg.tolist()):
        raise ValueError("query-held groups must be excluded from fitting history")
    if hl.dtype.kind not in "iu" or np.any(hl < -1):
        raise ValueError("history_leaders must contain integer positions or -1")
    ls, crews = np.asarray(leaders), np.asarray(worker_top5)
    if ls.shape != (n, 4) or crews.shape != (n, 4, 5) or ls.dtype.kind not in "iu" or crews.dtype.kind not in "iu":
        raise ValueError("candidate leaders/worker_top5 must be integer arrays [N,4]/[N,4,5]")
    if np.any(ls < -1) or np.any(crews < -1):
        raise ValueError("candidate positions cannot be less than -1")
    qnorm, hnorm = np.linalg.norm(query, axis=1), np.linalg.norm(history, axis=1)
    if np.any(qnorm <= 0) or np.any(hnorm <= 0) or not np.isfinite(qnorm).all() or not np.isfinite(hnorm).all():
        raise ValueError("cosine embeddings must have finite nonzero norm")
    for row in crews.reshape(-1, 5):
        present = row[row >= 0]
        if len(np.unique(present)) != len(present):
            raise ValueError("candidate shortlists cannot repeat workers")
    if np.any((crews == ls[:, :, None]) & (crews >= 0)):
        raise ValueError("candidate shortlists must exclude their proposed leader")
    similarities = (query / qnorm[:, None]) @ (history / hnorm[:, None]).T
    by_leader, by_worker, by_pair = {}, {}, {}
    for j, (leader, workers) in enumerate(zip(hl, history_workers)):
        ww = np.asarray(list(workers))
        if len(ww) and (ww.dtype.kind not in "iu" or np.any(ww < 0)):
            raise ValueError("historical workers must be nonnegative integer positions")
        if len(set(ww.tolist())) != len(ww):
            raise ValueError("historical worker rosters cannot contain duplicates")
        if leader >= 0:
            by_leader.setdefault(int(leader), []).append(j)
        for worker in ww:
            by_worker.setdefault(int(worker), []).append(j)
            if leader >= 0:
                by_pair.setdefault((int(leader), int(worker)), []).append(j)
    result = np.zeros((n, 4, 8))
    for i in range(n):
        memo = {}

        def score(indices):
            key = tuple(indices)
            if key not in memo:
                order = sorted(indices, key=lambda j: (-similarities[i, j], str(keys[j])))
                chosen, groups_seen, texts_seen = [], set(), set()
                for j in order:
                    text = str(prefixes[j])[:160]
                    if hg[j] in groups_seen or text in texts_seen:
                        continue
                    chosen.append(j)
                    groups_seen.add(hg[j])
                    texts_seen.add(text)
                    if len(chosen) == 2:
                        break
                memo[key] = (float(np.mean(similarities[i, chosen])), True) if chosen else (0., False)
            return memo[key]

        for a in range(4):
            leader = int(ls[i, a])
            leader_value, leader_known = score(by_leader.get(leader, []))
            worker_values, pair_values = [], []
            for worker in crews[i, a]:
                if worker < 0:
                    continue
                value, known = score(by_worker.get(int(worker), []))
                if known:
                    worker_values.append(value)
                value, known = score(by_pair.get((leader, int(worker)), []))
                if known:
                    pair_values.append(value)
            result[i, a] = [leader_value,
                            np.mean(worker_values) if worker_values else 0.,
                            min(worker_values) if worker_values else 0.,
                            np.mean(pair_values) if pair_values else 0.,
                            min(pair_values) if pair_values else 0.,
                            leader_known, len(worker_values) / 5, len(pair_values) / 5]
    return result.reshape(n, 32) if flatten else result
