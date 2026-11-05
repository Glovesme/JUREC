# JUREC

Core implementation of **Joint Utility and Role Evidence for Complete Recommendations (JUREC)** for joint leader–worker recommendation in power-grid work planning. Each recommendation contains one responsible leader and an ordered shortlist of up to five workers. The selected leader is excluded from the worker shortlist.

## Release scope

This repository contains the core model code, dependency and installation instructions, input/output and usage documentation, and availability statements. It supports inspection, adaptation, and training of the released downstream heads on user-supplied features. It is **not a complete reproduction package for the industrial experiments**.

| Included | Outside this release |
| --- | --- |
| Four indexed candidates: two leader proposals × two worker rankings | Confidential Minjiang Power Grid work plans and personnel records |
| History-support backoff and continuous local-evidence shrinkage | Real identifiers, row-level predictions, retrieval banks, and embeddings |
| 47-dimensional numerical features and cosine-history descriptors | Fitted industrial checkpoints and calibration settings |
| COVAL reference-selection calculations | HCR/PRR training, industrial data loaders, and the text encoder |
| Conditional joint-utility heads and training objective | Training of COVAL's provisional and cross-leader predictors |
| Utility-gap-weighted preference learning and final selection | The complete nested cross-validation and baseline experiment runners |
| Group-based policy calibration | Industrial evaluation data and per-record outputs |

Upstream scores, smoothed association probabilities, frozen history embeddings, and COVAL predictor outputs are explicit interfaces. They must come from the user's own permitted fitting data. The core modules do not download models or access remote services.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "from jurec import JUREC; print('JUREC imported')"
```

## Usage

Prepare upstream scores, fitting-history evidence, and COVAL predictor outputs according to the [input/output contracts](docs/INTERFACES.md). That document provides the calls for fitting the released heads, calibrating from held-group predictions, and recommending a leader and worker shortlist. Training requires caller-supplied labels; inference requires fitted heads and upstream inputs. No data or fitted weights are bundled.

## Model flow

```text
Caller-supplied HCR and PRR scores + fitting-history evidence
    → four candidate assignments [a00, a01, a10, a11]
    → numerical and cosine-history features
    → COVAL initial assignment (caller-supplied fitted predictor outputs)
    → conditional joint-utility estimation and CUS reference selection
    → utility-gap-weighted preference ensemble
    → keep the reference or select an eligible alternative
    → leader + ordered worker shortlist
```

The worker correction is zero when the proposed leader has fewer than ten observed-roster fitting groups. Otherwise, the local contribution is scaled by `ESS / (ESS + 3)`. These controls operate in candidate construction; they do not prohibit every later choice of a different leader.

The conditional utility model estimates leader correctness and worker recall conditional on a correct leader. Their product estimates joint utility. The preference ensemble learns candidate score differences weighted by the magnitude of the observed joint-utility difference. Final selection uses agreement from at least four of five members, a mean-minus-dispersion score, and a calibrated threshold, retaining the reference when no alternative qualifies.

## Documentation

- [Input/output contracts](docs/INTERFACES.md)
- [Architecture and implementation details](docs/METHOD.md)
- [Fitting, calibration, and reproducibility scope](docs/REPRODUCIBILITY.md)

## Data availability

The Minjiang Power Grid dataset is confidential and is not distributed. This repository does not contain anonymized derivatives of those records. Fitted weights and the full industrial experiment workflow are also not included.

