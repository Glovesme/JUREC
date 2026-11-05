"""JUREC's conditional utility and regret-weighted preference heads.

The implementation preserves the archived cosine variant's parameterization,
group-bootstrap weighting, fitting-only standardization, and CPU float64
optimization. It consumes caller-provided cross-fitted features; it does not
create those features or establish their group isolation. No data or fitted
weights are bundled. Returned head dictionaries can be saved as JSON.

Candidate order is a00, a01, a10, a11. ``sem`` accepts (N, 4, 8) or (N, 32).
The preference input intentionally contains the same eight cosine features
twice, preserving the evaluated 65-input network rather than a smaller variant.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


INITIALIZATION_SEED = 20261018
PAPER_UPDATES = 600
_LABEL_KEYS = frozenset({"hit", "recall", "observed"})


def _boolean(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape or not np.isin(value, [False, True]).all():
        raise ValueError(f"{name} must have shape {shape} with boolean/0/1 values")
    return value.astype(bool)


def _descriptors(sem: Any, n: int) -> np.ndarray:
    sem = np.asarray(sem, dtype=np.float64)
    if sem.shape == (n, 4, 8):
        sem = sem.reshape(n, 32)
    if sem.shape != (n, 32) or not np.isfinite(sem).all():
        raise ValueError("sem must be finite with shape (N, 4, 8) or (N, 32)")
    return sem


def _inputs(pack: Mapping[str, Any], sem: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(pack["x"], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 47 or not np.isfinite(x).all():
        raise ValueError("x must be a finite (N, 47) matrix")
    n = len(x)
    leaders = np.asarray(pack["leaders"])
    if leaders.shape != (n, 4):
        raise ValueError("leaders must have shape (N, 4)")
    if not (np.array_equal(leaders[:, 0], leaders[:, 1]) and
            np.array_equal(leaders[:, 2], leaders[:, 3])):
        raise ValueError("candidate order requires leaders a00=a01 and a10=a11")
    changed = _boolean(pack["leader_changed"], (n,), "leader_changed")
    agree = leaders[:, 0] == leaders[:, 2]
    if not np.array_equal(~changed, agree):
        raise ValueError("leader_changed disagrees with the proposed leader identities")
    return x, _descriptors(sem, n), agree


def _labels(pack: Mapping[str, Any], n: int, agree: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit = _boolean(pack["hit"], (n, 4), "hit").astype(np.float64)
    observed = _boolean(pack["observed"], (n,), "observed")
    recall = np.asarray(pack["recall"], dtype=np.float64)
    if recall.shape != (n, 4):
        raise ValueError("recall must have shape (N, 4)")
    if not np.array_equal(hit[:, 0], hit[:, 1]) or not np.array_equal(hit[:, 2], hit[:, 3]):
        raise ValueError("leader hits must agree for candidates sharing a leader")
    if not np.array_equal(hit[agree, 0], hit[agree, 2]):
        raise ValueError("identical leader proposals must have identical hit labels")
    if ((hit[:, 0] + hit[:, 2])[~agree] > 1).any():
        raise ValueError("distinct leader proposals cannot both match one recorded leader")
    known_recall = recall[observed]
    if not np.isfinite(known_recall).all() or ((known_recall < 0) | (known_recall > 1)).any():
        raise ValueError("observed recalls must be finite and lie in [0, 1]")
    return hit, recall, observed


def group_macro_weights(group_ids: Any, mask: Any | None = None) -> np.ndarray:
    """Equal eligible-group mass, then equal eligible-plan mass within each group.

    Groups without eligible plans receive no mass. An all-false mask returns
    zeros, allowing callers to explicitly handle the absence of supervision.
    """
    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be one-dimensional")
    if np.issubdtype(group_ids.dtype, np.number) and not np.isfinite(group_ids).all():
        raise ValueError("group_ids cannot contain nonfinite values")
    mask = np.ones(len(group_ids), dtype=bool) if mask is None else _boolean(mask, (len(group_ids),), "mask")
    result = np.zeros(len(group_ids), dtype=np.float64)
    if mask.any():
        keys, inverse, counts = np.unique(group_ids[mask], return_inverse=True, return_counts=True)
        result[mask] = 1.0 / (len(keys) * counts[inverse])
    return result


def _bootstrap(pack: Mapping[str, Any], n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray(pack["group_ids"])
    if groups.shape != (n,) or n == 0:
        raise ValueError("fitting requires nonempty group_ids with shape (N,)")
    group_macro_weights(groups)  # Validate identifiers before sampling.
    keys, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.bincount(rng.integers(len(keys), size=len(keys)), minlength=len(keys))
    return groups, inverse, counts


def conditional_features(pack: Mapping[str, Any], sem: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact 51 leader and four 57-dimensional worker inputs."""
    x, sem, _ = _inputs(pack, sem)
    n = len(x)
    leader = np.column_stack([x, sem[:, 0], sem[:, 16], sem[:, 5], sem[:, 21]])
    worker = np.stack([
        np.column_stack([x, sem[:, 8*a:8*a+8], np.full(n, a // 2), np.full(n, a % 2)])
        for a in range(4)
    ], axis=1)
    return leader, worker


def preference_features(pack: Mapping[str, Any], sem: Any) -> np.ndarray:
    """Return the evaluated 65 inputs: 47 + cosine8 + cosine8 + action2."""
    x, sem, _ = _inputs(pack, sem)
    n = len(x)
    return np.stack([
        np.column_stack([x, sem[:, 8*a:8*a+8], sem[:, 8*a:8*a+8],
                         np.full(n, a // 2), np.full(n, a % 2)])
        for a in range(4)
    ], axis=1)


class ConditionalUtilityNet(torch.nn.Module):
    """1,828 parameters; conditional leader probability times worker recall."""

    def __init__(self) -> None:
        super().__init__()
        self.leader = torch.nn.Sequential(torch.nn.Linear(51, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3))
        self.worker = torch.nn.Sequential(torch.nn.Linear(57, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))

    def forward(self, leader: torch.Tensor, worker: torch.Tensor, agree: torch.Tensor) -> tuple[torch.Tensor, ...]:
        probability = self.leader(leader).softmax(1)
        both = probability[:, 1] + probability[:, 2]
        leader_probability = torch.stack([
            torch.where(agree, both, probability[:, 1]),
            torch.where(agree, both, probability[:, 2]),
        ], dim=1)
        worker_recall = self.worker(worker).squeeze(-1).sigmoid()
        joint = leader_probability[:, [0, 0, 1, 1]] * worker_recall
        return joint, leader_probability, worker_recall, probability


class PreferenceNet(torch.nn.Module):
    """1,072 parameters; shared scalar potentials give cycle-consistent differences."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(65, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1, bias=False)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def conditional_objective(
    net: ConditionalUtilityNet, leader: torch.Tensor, worker: torch.Tensor,
    agree: torch.Tensor, hit: torch.Tensor, recall: torch.Tensor,
    observed: torch.Tensor, leader_weights: torch.Tensor, joint_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Leader NLL + conditional recall MSE + joint MSE; regularization is separate."""
    joint, _, worker_recall, probability = net(leader, worker, agree)
    safe_recall = torch.where(observed[:, None], recall, torch.zeros_like(recall))
    truth = hit * safe_recall
    joint_mse = (joint_weights[:, None] * (joint - truth).square()).sum() / 4
    positive = hit[:, 0] + hit[:, 2] > 0
    positive_probability = torch.where(
        agree, probability[:, 1] + probability[:, 2],
        probability[:, 1] * hit[:, 0] + probability[:, 2] * hit[:, 2],
    )
    likelihood = torch.where(positive, positive_probability, probability[:, 0]).clamp(min=1e-12)
    leader_nll = -(leader_weights * likelihood.log()).sum()
    conditional_weights = joint_weights[:, None] * hit
    conditional_mse = (conditional_weights * (worker_recall - safe_recall).square()).sum()
    conditional_mse = conditional_mse / conditional_weights.sum().clamp(min=1e-12)
    return leader_nll + conditional_mse + joint_mse, {
        "leader_nll": leader_nll,
        "conditional_worker_mse": conditional_mse,
        "joint_mse": joint_mse,
    }


def pair_targets(pack: Mapping[str, Any]) -> np.ndarray:
    """Observed final-combination utility differences; missing rosters yield zeros."""
    hit = np.asarray(pack["hit"], dtype=np.float64)
    recall = np.asarray(pack["recall"], dtype=np.float64)
    observed = np.asarray(pack["observed"], dtype=bool)
    safe_recall = np.where(observed[:, None], recall, 0.0)
    utility = hit * safe_recall
    return utility[:, :, None] - utility[:, None, :]


def preference_objective(values: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted logistic loss over ordered pairs; ties receive zero fitting weight."""
    delta = values[:, :, None] - values[:, None, :]
    return (weights * torch.nn.functional.softplus(-torch.sign(targets) * delta)).sum()


def _validate_updates(updates: int) -> None:
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer (paper default: 600)")


def _state(net: torch.nn.Module) -> dict[str, list]:
    return {key: value.detach().cpu().numpy().tolist() for key, value in net.state_dict().items()}


def _initialize(net_class: type[torch.nn.Module]) -> torch.nn.Module:
    # Preserve the caller's RNG state while matching the original fixed initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(INITIALIZATION_SEED)
        return net_class().double()


def _regularizer(net: torch.nn.Module) -> torch.Tensor:
    return sum(value.square().sum() for value in net.parameters()) / sum(value.numel() for value in net.parameters())


def fit_conditional(pack: Mapping[str, Any], sem: Any, bootstrap_seed: int, updates: int = PAPER_UPDATES) -> dict[str, Any]:
    """Fit one conditional head on cross-fitted fitting plans only.

    A bootstrap with no observed roster is rejected rather than treating missing
    worker labels as empty rosters. This head is only one ensemble member.
    """
    _validate_updates(updates)
    x, _, agree = _inputs(pack, sem)
    hit, recall, observed = _labels(pack, len(x), agree)
    leader, worker = conditional_features(pack, sem)
    groups, inverse, counts = _bootstrap(pack, len(x), bootstrap_seed)
    leader_weights = group_macro_weights(groups) * counts[inverse]
    leader_weights /= leader_weights.sum()
    joint_weights = group_macro_weights(groups, observed) * counts[inverse]
    if joint_weights.sum() <= 0:
        raise ValueError("conditional fit has no observed worker roster in the sampled groups")
    joint_weights /= joint_weights.sum()
    keep = leader_weights > 0
    leader_mean = leader_weights @ leader
    leader_scale = np.sqrt(leader_weights @ ((leader - leader_mean) ** 2)).clip(min=1e-6)
    worker_mean = np.einsum("n,nad->d", leader_weights / 4, worker)
    worker_scale = np.sqrt(np.einsum("n,nad->d", leader_weights / 4, (worker-worker_mean)**2)).clip(min=1e-6)
    leader_z = np.clip((leader[keep] - leader_mean) / leader_scale, -5, 5)
    worker_z = np.clip((worker[keep] - worker_mean) / worker_scale, -5, 5)
    net = _initialize(ConditionalUtilityNet)
    torch.nn.init.zeros_(net.leader[2].weight)
    torch.nn.init.zeros_(net.worker[2].weight)
    none = (hit[:, 0] + hit[:, 2]) == 0
    mass = np.array([
        leader_weights @ none,
        leader_weights @ (hit[:, 0]*(~agree) + .5*hit[:, 0]*agree),
        leader_weights @ (hit[:, 2]*(~agree) + .5*hit[:, 2]*agree),
    ])
    prior = (len(counts)*mass + .5) / (len(counts) + 1.5)
    conditional_weights = joint_weights[:, None] * hit
    safe_recall = np.where(observed[:, None], recall, 0.0)
    average = float((conditional_weights*safe_recall).sum() / max(conditional_weights.sum(), 1e-12))
    average = np.clip(average, 1e-6, 1-1e-6)
    with torch.no_grad():
        net.leader[2].bias.copy_(torch.tensor(np.log(prior)))
        net.worker[2].bias.fill_(np.log(average/(1-average)))
    args = [torch.tensor(value) for value in [
        leader_z, worker_z, agree[keep], hit[keep], recall[keep], observed[keep],
        leader_weights[keep], joint_weights[keep],
    ]]
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=0.)
    trace = []
    started = time.perf_counter()
    for step in range(1, updates+1):
        optimizer.zero_grad()
        loss, parts = conditional_objective(net, *args)
        total = loss + .1 * _regularizer(net)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("nonfinite conditional training loss")
        total.backward()
        optimizer.step()
        if step in {1, 100, 300, updates}:
            trace.append({"step": step, "loss": float(loss.detach()), **{key: float(value.detach()) for key, value in parts.items()}})
    return {
        "schema": "jurec.conditional.v1", "config": "factor_cosine", "active": True,
        "bootstrap_seed": int(bootstrap_seed), "initialization_seed": INITIALIZATION_SEED,
        "parameters": 1828, "updates": updates, "paper_default_updates": PAPER_UPDATES,
        "regularization": .1, "learning_rate": .003,
        "leader_mean": leader_mean.tolist(), "leader_scale": leader_scale.tolist(),
        "worker_mean": worker_mean.tolist(), "worker_scale": worker_scale.tolist(), "state": _state(net),
        "fit_plans": int(keep.sum()), "fit_groups": int((counts > 0).sum()),
        "known_roster_plans": int((keep & observed).sum()),
        "seconds": time.perf_counter() - started, "learning_curve": trace,
    }


def fit_preference(pack: Mapping[str, Any], sem: Any, bootstrap_seed: int, updates: int = PAPER_UPDATES) -> dict[str, Any]:
    """Fit one regret-magnitude-weighted potential head; zero evidence gives zero potentials."""
    _validate_updates(updates)
    raw, _, agree = _inputs(pack, sem)
    _, _, observed = _labels(pack, len(raw), agree)
    features = preference_features(pack, sem)
    targets = pair_targets(pack)
    groups, inverse, counts = _bootstrap(pack, len(raw), bootstrap_seed)
    row = group_macro_weights(groups, observed) * counts[inverse]
    weights = row[:, None, None] * np.abs(targets) / 12
    total_mass = weights.sum()
    net = _initialize(PreferenceNet)
    torch.nn.init.zeros_(net.layers[2].weight)
    result = {
        "schema": "jurec.preference.v1", "config": "regret_cosine", "input": 65, "hidden": 16,
        "parameters": 1072, "bootstrap_seed": int(bootstrap_seed), "initialization_seed": INITIALIZATION_SEED,
        "requested_updates": updates, "paper_default_updates": PAPER_UPDATES, "regularization": 1., "learning_rate": .003,
        "bootstrap_nonzero_groups": int((counts > 0).sum()), "duplicated_cosine_block": True,
    }
    if total_mass <= 0:
        return {**result, "active": False, "reason": "no_nonzero_observed_regret_in_bootstrap",
                "updates": 0, "mean": [0.]*65, "scale": [1.]*65, "state": _state(net),
                "fit_nonzero_regret_plans": 0, "ordered_nonzero_pairs": 0, "seconds": 0., "learning_curve": []}
    weights /= total_mass
    keep = weights.sum((1, 2)) > 0
    row_weights = row * keep
    row_weights /= row_weights.sum()
    mean = np.einsum("n,nad->d", row_weights / 4, features)
    scale = np.sqrt(np.einsum("n,nad->d", row_weights / 4, (features-mean)**2)).clip(min=1e-6)
    z = np.clip((features[keep] - mean) / scale, -5, 5)
    xx, yy, ww = torch.tensor(z), torch.tensor(targets[keep]), torch.tensor(weights[keep])
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=0.)
    trace = []
    started = time.perf_counter()
    for step in range(1, updates+1):
        optimizer.zero_grad()
        loss = preference_objective(net(xx), yy, ww)
        penalty = _regularizer(net)
        total = loss + penalty
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("nonfinite preference training loss")
        total.backward()
        optimizer.step()
        if step in {1, 100, 300, updates}:
            trace.append({"step": step, "pair_logistic_loss": float(loss.detach()), "mean_parameter_square": float(penalty.detach())})
    return {
        **result, "active": True, "updates": updates, "mean": mean.tolist(), "scale": scale.tolist(), "state": _state(net),
        "fit_nonzero_regret_plans": int(keep.sum()), "ordered_nonzero_pairs": int((weights[keep] > 0).sum()),
        "seconds": time.perf_counter()-started, "learning_curve": trace,
    }


def _inference_only(pack: Mapping[str, Any], heads: Sequence[Mapping[str, Any]]) -> None:
    if _LABEL_KEYS.intersection(pack):
        raise ValueError("prediction pack must omit hit, recall, and observed")
    if len(heads) == 0:
        raise ValueError("at least one fitted head is required")


def _normalization(head: Mapping[str, Any], mean_key: str, scale_key: str, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(head[mean_key], dtype=np.float64)
    scale = np.asarray(head[scale_key], dtype=np.float64)
    if mean.shape != (dimension,) or scale.shape != (dimension,):
        raise ValueError("fitted normalization has an incompatible dimension")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("fitted normalization must be finite with positive scales")
    return mean, scale


def _restore(net_class: type[torch.nn.Module], head: Mapping[str, Any]) -> torch.nn.Module:
    net = _initialize(net_class)
    state = {key: torch.tensor(value, dtype=torch.float64) for key, value in head["state"].items()}
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("fitted weights contain nonfinite values")
    net.load_state_dict(state)
    net.eval()
    return net


def predict_conditional(pack: Mapping[str, Any], sem: Any, heads: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Predict joint (N,M,4), leader (N,M,2), and worker (N,M,4) arrays."""
    _inference_only(pack, heads)
    _, _, agree = _inputs(pack, sem)
    leader, worker = conditional_features(pack, sem)
    joint_out, leader_out, worker_out = [], [], []
    for head in heads:
        if head.get("config") != "factor_cosine" or head.get("parameters") != 1828:
            raise ValueError("expected a JUREC conditional cosine head")
        net = _restore(ConditionalUtilityNet, head)
        lm, ls = _normalization(head, "leader_mean", "leader_scale", 51)
        wm, ws = _normalization(head, "worker_mean", "worker_scale", 57)
        ll = torch.tensor(np.clip((leader-lm)/ls, -5, 5))
        ww = torch.tensor(np.clip((worker-wm)/ws, -5, 5))
        with torch.no_grad():
            joint, lp, wr, _ = net(ll, ww, torch.tensor(agree))
        joint_out.append(joint.numpy())
        leader_out.append(lp.numpy())
        worker_out.append(wr.numpy())
    return {"joint": np.stack(joint_out, axis=1), "leader": np.stack(leader_out, axis=1), "worker": np.stack(worker_out, axis=1)}


def predict_preference(pack: Mapping[str, Any], sem: Any, heads: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Predict centered candidate potentials with shape (N,M,4)."""
    _inference_only(pack, heads)
    features = preference_features(pack, sem)
    output = []
    for head in heads:
        if head.get("config") != "regret_cosine" or head.get("parameters") != 1072 or head.get("regularization") != 1.:
            raise ValueError("expected a JUREC regret-weighted cosine preference head")
        net = _restore(PreferenceNet, head)
        mean, scale = _normalization(head, "mean", "scale", 65)
        z = np.clip((features-mean)/scale, -5, 5)
        with torch.no_grad():
            values = net(torch.tensor(z)).numpy()
        output.append(values-values.mean(1, keepdims=True))
    return np.stack(output, axis=1)
