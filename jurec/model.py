"""Composition of the released JUREC heads and calibrated selection rules."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .learning import fit_conditional, fit_preference, predict_conditional, predict_preference
from .selection import (
    NATIVE_POLICIES, PREFERENCE_POLICIES, candidate_diagnostics,
    preference_diagnostics, native_choose, risk_choose, preference_choose,
    risk_policies, calibrate_native, calibrate_risk, calibrate_preference,
)

BOOTSTRAP_SEEDS = tuple(range(20261019, 20261024))


def inference_pack(pack):
    """Explicitly remove supervision from a pack before any inference call."""
    return {key: value for key, value in pack.items()
            if key not in {"hit", "recall", "observed"}}


def _semantic(sem, n):
    sem = np.asarray(sem, dtype=np.float64)
    if sem.shape == (n, 32):
        sem = sem.reshape(n, 4, 8)
    if sem.shape != (n, 4, 8) or not np.isfinite(sem).all():
        raise ValueError("descriptors must be finite N x 4 x 8 or N x 32")
    return sem


class JUREC:
    """Five-member JUREC core, with explicit upstream and calibration interfaces.

    Call ``fit_heads`` on cross-fitted training features. Supply COVAL choices
    from ``reference.coval_reference`` to inference and calibration. The three
    selectors default to keeping their reference until calibrated. This class
    does not fit HCR, PRR, or COVAL's upstream predictors, nor create CV splits.
    """

    def __init__(self, conditional_heads, preference_heads, *, native_policy=None,
                 risk_policy=None, preference_policy=None):
        if len(conditional_heads) != 5 or len(preference_heads) != 5:
            raise ValueError("JUREC requires five members in each ensemble")
        self.conditional_heads = deepcopy(list(conditional_heads))
        self.preference_heads = deepcopy(list(preference_heads))
        self.native_policy = deepcopy(NATIVE_POLICIES[0] if native_policy is None else native_policy)
        self.risk_policy = deepcopy(risk_policies()[0] if risk_policy is None else risk_policy)
        self.preference_policy = deepcopy(PREFERENCE_POLICIES[0] if preference_policy is None else preference_policy)

    @classmethod
    def fit_heads(cls, pack, descriptors, *, updates=600, bootstrap_seeds=BOOTSTRAP_SEEDS):
        """Fit conditional/preference ensembles only, not the whole industrial pipeline."""
        seeds = tuple(bootstrap_seeds)
        if len(seeds) != 5 or len(set(seeds)) != 5:
            raise ValueError("provide five distinct group-bootstrap seeds")
        conditional = [fit_conditional(pack, descriptors, s, updates=updates) for s in seeds]
        preference = [fit_preference(pack, descriptors, s, updates=updates) for s in seeds]
        return cls(conditional, preference)

    def predict_heads(self, pack, descriptors):
        """Return predictions without accepting labels in the inference pack."""
        return (predict_conditional(pack, descriptors, self.conditional_heads),
                predict_preference(pack, descriptors, self.preference_heads))

    def calibrate_from_oof(self, pack, descriptors, initial_choice,
                           conditional_joint, preference_potential):
        """Calibrate from caller-supplied group-held predictions, not in-sample fits.

        The caller owns prediction provenance and must exclude outer evaluation
        groups. Upstream policies are selected first, then reused to construct
        final-reference choices on this population, matching the archived rule.
        """
        n = len(pack["x"])
        sem = _semantic(descriptors, n)
        observed = np.asarray(pack["observed"], dtype=bool)
        if observed.shape != (n,):
            raise ValueError("observed must have shape N")
        utility = np.asarray(pack["hit"]) * np.where(observed[:, None], pack["recall"], 0.)
        d = candidate_diagnostics(pack["leaders"], pack["worker_top5"], sem, initial_choice)
        d["pred_joint"] = np.asarray(conditional_joint)
        groups = pack["group_ids"]
        index, native_log = calibrate_native(utility, groups, observed, d)
        self.native_policy = deepcopy(NATIVE_POLICIES[index])
        d.update(cos_joint=d["pred_joint"], native_choice=native_choose(d, self.native_policy))
        index, risk_log = calibrate_risk(utility, groups, observed, d)
        self.risk_policy = deepcopy(risk_policies()[index])
        reference = risk_choose(d, self.risk_policy)
        pd = preference_diagnostics(pack["leaders"], pack["worker_top5"], sem,
                                    reference, preference_potential)
        index, preference_log = calibrate_preference(utility, groups, observed, pd)
        self.preference_policy = deepcopy(PREFERENCE_POLICIES[index])
        return {"native": native_log, "risk": risk_log, "preference": preference_log}

    def recommend(self, pack, descriptors, initial_choice):
        """Return action indices and ordered local personnel indices for each plan."""
        conditional, potential = self.predict_heads(pack, descriptors)
        sem = _semantic(descriptors, len(pack["x"]))
        d = candidate_diagnostics(pack["leaders"], pack["worker_top5"], sem, initial_choice)
        d["pred_joint"] = conditional["joint"]
        d.update(cos_joint=conditional["joint"], native_choice=native_choose(d, self.native_policy))
        reference = risk_choose(d, self.risk_policy)
        pd = preference_diagnostics(pack["leaders"], pack["worker_top5"], sem, reference, potential)
        choice = preference_choose(pd, self.preference_policy)
        rows = np.arange(len(choice))
        return {
            "initial_choice": np.asarray(initial_choice).copy(),
            "reference_choice": reference,
            "choice": choice,
            "leader": np.asarray(pack["leaders"])[rows, choice],
            "workers": np.asarray(pack["worker_top5"])[rows, choice],
        }

    def to_dict(self):
        """Portable fitted state. Real-data fitted states are not public artifacts."""
        return deepcopy({
            "format": "jurec-core-1", "model_name": "JUREC",
            "conditional_heads": self.conditional_heads,
            "preference_heads": self.preference_heads,
            "native_policy": self.native_policy, "risk_policy": self.risk_policy,
            "preference_policy": self.preference_policy,
        })

    @classmethod
    def from_dict(cls, state):
        if state.get("format") != "jurec-core-1" or state.get("model_name") != "JUREC":
            raise ValueError("unsupported JUREC state format")
        return cls(state["conditional_heads"], state["preference_heads"],
                   native_policy=state["native_policy"], risk_policy=state["risk_policy"],
                   preference_policy=state["preference_policy"])

    def save(self, path):
        """Save to an explicitly supplied controlled path; never runs automatically."""
        Path(path).write_text(json.dumps(self.to_dict(), allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
