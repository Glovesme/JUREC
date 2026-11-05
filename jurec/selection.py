"""JUREC's conditional reference, leader-risk policy, and regret gate.

Policies operate on inference-only diagnostics. Calibration functions take
held-out fitting-group utilities explicitly; never calibrate on evaluation rows.
The source's exact policy grids, strict comparisons, availability scopes,
sample standard deviations, and lexicographic tie rules are retained.
"""

from __future__ import annotations

import numpy as np


NATIVE_POLICIES = [{"enabled": False, "kappa": 0.0, "margin": 0.0}] + [
    {"enabled": True, "kappa": float(k), "margin": float(m)}
    for k in [0, 1, 2] for m in [0.0, 0.002, 0.01, 0.02]
]
OFF = {"rule": "off", "rho": 1.0, "kappa": 0.0, "margin": 0.0}
NATIVE = {"rule": "native", "rho": 1.0, "kappa": 0.0, "margin": 0.0}
KEEP = {"rule": "keep", "kappa": 0.0, "threshold": 0.0}
PREFERENCE_POLICIES = [KEEP] + [
    {"rule": "revise", "kappa": float(k), "threshold": float(t)}
    for k in [0, 1, 2] for t in [0, 0.1, 0.25, 0.5, 1]
]


def _choices(value, n, name):
    value = np.asarray(value)
    if value.shape != (n,) or not np.isin(value, np.arange(4)).all():
        raise ValueError(f"{name} must contain one action index in [0,3] per query")
    return value.astype(int)


def _ensemble(value, n, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (n, 5, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and have shape N x 5 x 4")
    return value


def _mask(value, shape, name):
    value = np.asarray(value, dtype=bool)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return value


def candidate_diagnostics(leaders, worker_top5, descriptors, reference):
    """Build native/risk masks from candidates and N x 4 x 8 descriptors.

    Worker sets, rather than their order, determine a complete-output change.
    Native/risk semantic availability is *query-wide*: any positive history
    coverage in descriptor slots 5:8 among the four candidates is sufficient.
    This deliberately differs from the paired coverage in the final gate.
    """
    leaders = np.asarray(leaders)
    workers = np.asarray(worker_top5)
    semantic = np.asarray(descriptors, dtype=float)
    if leaders.ndim != 2 or leaders.shape[1] != 4:
        raise ValueError("leaders must have shape N x 4")
    n = len(leaders)
    if workers.ndim != 3 or workers.shape[:2] != (n, 4) or workers.shape[2] == 0:
        raise ValueError("worker_top5 must have shape N x 4 x K, K > 0")
    if semantic.shape != (n, 4, 8) or not np.isfinite(semantic).all():
        raise ValueError("descriptors must be finite and have shape N x 4 x 8")
    baseline = _choices(reference, n, "reference")
    rows = np.arange(n)
    leader_switch = leaders != leaders[rows, baseline][:, None]
    sorted_workers = np.sort(workers, axis=2)
    worker_change = np.any(sorted_workers != sorted_workers[rows, baseline][:, None, :], axis=2)
    return {
        "frozen": baseline,
        "full_change": leader_switch | worker_change,
        "leader_switch": leader_switch,
        "semantic_available": semantic[:, :, 5:8].max(axis=(1, 2)) > 0,
    }


def preference_diagnostics(leaders, worker_top5, descriptors, current, potential):
    """Build final-gate evidence, requiring coverage for both compared actions."""
    d = candidate_diagnostics(leaders, worker_top5, descriptors, current)
    n = len(d["frozen"])
    coverage = np.asarray(descriptors)[:, :, 5:8].max(2) > 0
    baseline_coverage = coverage[np.arange(n), d["frozen"]]
    return {
        "current_choice": d["frozen"],
        "change_from_current": d["full_change"],
        "leader_from_current": d["leader_switch"],
        "paired_history_available": coverage & baseline_coverage[:, None],
        "potential": _ensemble(potential, n, "potential"),
    }


def _rank_choice(diff, valid, baseline, kappa, threshold):
    score = diff.mean(1) - float(kappa) * diff.std(1, ddof=1)
    valid = valid & ((diff > 0).sum(1) >= 4) & (score > float(threshold))
    # NumPy argmax resolves equal valid scores to the lowest action index.
    best = np.where(valid, score, -np.inf).argmax(1)
    return np.where(valid.any(1), best, baseline)


def native_choose(d, policy, two_route=False):
    """Select a conditional-utility revision of frozen COVAL (13 policies)."""
    n = len(d["frozen"])
    baseline = _choices(d["frozen"], n, "frozen")
    if not policy["enabled"]:
        return baseline.copy()
    utility = _ensemble(d["pred_joint"], n, "pred_joint")
    diff = utility - utility[np.arange(n), :, baseline][:, :, None]
    valid = (_mask(d["full_change"], (n, 4), "full_change")
             & _mask(d["semantic_available"], (n,), "semantic_available")[:, None])
    if two_route:
        route = np.asarray(d["route_actions"])
        if route.shape != (n, 2) or not np.isin(route, np.arange(4)).all():
            raise ValueError("route_actions must have shape N x 2 with action indices")
        valid &= np.any(np.arange(4)[None, None, :] == route[:, :, None], axis=1)
    return _rank_choice(diff, valid, baseline, policy["kappa"], policy["margin"])


def risk_policies(method="cosine_risk"):
    """Return the source grid (50 policies for cosine_risk/leader_risk)."""
    if method == "direct_cosine":
        prefix, rhos = [OFF], [1.0]
    elif method == "no_risk":
        prefix, rhos = [OFF, NATIVE], [1.0]
    elif method in {"cosine_risk", "leader_risk", "full_mix"}:
        prefix, rhos = [OFF, NATIVE], [1.0, 1.05, 1.1, 1.25]
    else:
        raise ValueError(f"unsupported risk method: {method}")
    return [dict(p) for p in prefix] + [
        {"rule": "risk", "rho": r, "kappa": float(k), "margin": float(m)}
        for r in rhos for k in [0, 1, 2] for m in [0.0, 0.002, 0.01, 0.02]
    ]


def risk_utilities(d, method="cosine_risk"):
    """Factorized cosine utility, or optional source-compatible role fusion."""
    n = len(d["frozen"])
    if method == "direct_cosine":
        return _ensemble(d["direct_joint"], n, "direct_joint")
    cosine = _ensemble(d["cos_joint"], n, "cos_joint")
    if method == "cosine_risk":
        return cosine
    gate = 0.5 * _mask(d["llm_available"], (n,), "llm_available")[:, None, None]
    if method == "full_mix":
        return (1 - gate) * cosine + gate * _ensemble(d["llm_joint"], n, "llm_joint")
    if method not in {"leader_risk", "no_risk"}:
        raise ValueError(f"unsupported risk method: {method}")
    cosine_leader = np.asarray(d["cos_leader"], dtype=float)
    llm_leader = np.asarray(d["llm_leader"], dtype=float)
    if (cosine_leader.shape != (n, 5, 2) or llm_leader.shape != (n, 5, 2)
            or not np.isfinite(cosine_leader).all() or not np.isfinite(llm_leader).all()):
        raise ValueError("leader predictions must be finite N x 5 x 2 arrays")
    leader = (1 - gate) * cosine_leader + gate * llm_leader
    return leader[:, :, [0, 0, 1, 1]] * _ensemble(d["cos_worker"], n, "cos_worker")


def risk_choose(d, policy, method="cosine_risk"):
    """Risk-calibrated CUS choice; rho penalizes only leader-switch actions.

    A leader switch uses q(a)-rho*q(reference). Worker-only choices preserve the
    native cosine contrast. Availability remains query-wide, as in the archive.
    """
    n = len(d["frozen"])
    baseline = _choices(d["frozen"], n, "frozen")
    if policy["rule"] == "off":
        return baseline.copy()
    if policy["rule"] == "native":
        return _choices(d["native_choice"], n, "native_choice").copy()
    if policy["rule"] != "risk":
        raise ValueError("risk policy rule must be off, native, or risk")
    utility = risk_utilities(d, method)
    base = utility[np.arange(n), :, baseline]
    diff = utility - base[:, :, None]
    if method != "direct_cosine":
        cosine = _ensemble(d["cos_joint"], n, "cos_joint")
        cosine_base = cosine[np.arange(n), :, baseline]
        diff = np.where(
            _mask(d["leader_switch"], (n, 4), "leader_switch")[:, None, :],
            utility - float(policy["rho"]) * base[:, :, None],
            cosine - cosine_base[:, :, None],
        )
    valid = (_mask(d["full_change"], (n, 4), "full_change")
             & _mask(d["semantic_available"], (n,), "semantic_available")[:, None])
    return _rank_choice(diff, valid, baseline, policy["kappa"], policy["margin"])


def preference_choose(d, policy, method="regret_cosine"):
    """Final JUREC preference gate (16 policies including keep).

    Compare each action's scalar potential to the selected CUS reference within
    each of five resampled heads. Four positive differences, paired history,
    complete-output change, and mean-kappa*sample-SD > threshold are required.
    There is no extra 0.002 margin here. Ties select the lowest action index;
    no eligible revision retains the reference. ``leader_only`` is an ablation.
    """
    n = len(d["current_choice"])
    current = _choices(d["current_choice"], n, "current_choice")
    if policy["rule"] == "keep":
        return current.copy()
    if policy["rule"] != "revise":
        raise ValueError("preference policy rule must be keep or revise")
    potential = _ensemble(d["potential"], n, "potential")
    diff = potential - potential[np.arange(n), :, current][:, :, None]
    valid = (_mask(d["change_from_current"], (n, 4), "change_from_current")
             & _mask(d["paired_history_available"], (n, 4), "paired_history_available"))
    if method == "leader_only":
        valid &= _mask(d["leader_from_current"], (n, 4), "leader_from_current")
    return _rank_choice(diff, valid, current, policy["kappa"], policy["threshold"])


def _calibration_data(utilities, groups, observed, n):
    utility = np.asarray(utilities, dtype=float)
    groups = np.asarray(groups)
    observed = np.asarray(observed, dtype=bool)
    if utility.shape != (n, 4) or groups.shape != (n,) or observed.shape != (n,):
        raise ValueError("calibration shapes must be utilities N x 4, groups N, observed N")
    if not observed.any() or not np.isfinite(utility[observed]).all():
        raise ValueError("calibration needs finite observed-roster utilities")
    keys, inverse, count = np.unique(groups[observed], return_inverse=True, return_counts=True)
    return utility, observed, keys, inverse, count


def _group_mean(values, observed, inverse, count):
    return np.bincount(inverse, weights=values[observed], minlength=len(count)) / count


def calibrate_native(utilities, groups, observed, d, two_route=False):
    """Select the native grid by group-macro utility; return policy id and log.

    Tie order: fewer changed plans, disabled policy, larger kappa, larger margin.
    Utilities must come from the fitting population's group-held calibration.
    """
    n = len(d["frozen"])
    utility, mask, _, inverse, count = _calibration_data(utilities, groups, observed, n)
    rows = np.arange(n)
    best = None
    logs = []
    for j, policy in enumerate(NATIVE_POLICIES):
        choice = native_choose(d, policy, two_route)
        mean = float(_group_mean(utility[rows, choice], mask, inverse, count).mean())
        changes = int(np.asarray(d["full_change"])[rows, choice].sum())
        rank = (-mean, changes, int(policy["enabled"]), -policy["kappa"], -policy["margin"])
        logs.append({"policy_id": j, **policy, "joint_fraction": mean, "actual_changes": changes})
        if best is None or rank < best[0]:
            best = (rank, j)
    return best[1], logs


def calibrate_risk(utilities, groups, observed, d, method="cosine_risk"):
    """Select the risk grid by macro utility with archived conservative tie order."""
    n = len(d["frozen"])
    utility, mask, _, inverse, count = _calibration_data(utilities, groups, observed, n)
    rows = np.arange(n)
    best = None
    logs = []
    for j, policy in enumerate(risk_policies(method)):
        choice = risk_choose(d, policy, method)
        mean = float(_group_mean(utility[rows, choice], mask, inverse, count).mean())
        changes = int(np.asarray(d["full_change"])[rows, choice].sum())
        complexity = {"off": 0, "native": 1, "risk": 2}[policy["rule"]]
        rank = (-mean, changes, complexity, -policy["rho"], -policy["kappa"], -policy["margin"])
        logs.append({"policy_id": j, **policy, "joint_fraction": mean, "actual_changes": changes})
        if best is None or rank < best[0]:
            best = (rank, j)
    return best[1], logs


def calibrate_preference(utilities, groups, observed, d, method="regret_cosine"):
    """Select the final gate by mean group gain minus its standard error.

    Positive criterion (>1e-12) is required for every revise policy; keep is
    always eligible. Ties prefer greater mean gain, keep, fewer changed plans,
    larger kappa, then larger threshold. ``meanselect`` omits the SE penalty
    only as an explicitly named ablation. At least two observed groups are
    required to define the sample standard error.
    """
    n = len(d["current_choice"])
    utility, mask, keys, inverse, count = _calibration_data(utilities, groups, observed, n)
    if len(keys) < 2:
        raise ValueError("final policy calibration requires at least two observed groups")
    rows = np.arange(n)
    current = _choices(d["current_choice"], n, "current_choice")
    best = None
    logs = []
    for j, policy in enumerate(PREFERENCE_POLICIES):
        choice = preference_choose(d, policy, method)
        gains = _group_mean(utility[rows, choice] - utility[rows, current], mask, inverse, count)
        mean = float(gains.mean())
        se = float(gains.std(ddof=1) / np.sqrt(len(gains)))
        criterion = mean if method == "meanselect" else mean - se
        changes = int(np.asarray(d["change_from_current"])[rows, choice].sum())
        rank = (-criterion, -mean, 0 if policy["rule"] == "keep" else 1,
                changes, -policy["kappa"], -policy["threshold"])
        eligible = policy["rule"] == "keep" or criterion > 1e-12
        logs.append({"policy_id": j, **policy, "mean_group_gain_fraction": mean,
                     "group_standard_error": se, "selection_criterion_fraction": criterion,
                     "actual_revisions_from_current": changes,
                     "known_roster_groups": len(keys), "eligible": eligible})
        if eligible and (best is None or rank < best[0]):
            best = (rank, j)
    return best[1], logs
