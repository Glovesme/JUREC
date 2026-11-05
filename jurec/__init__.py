"""JUREC core model components; no datasets or fitted weights are bundled."""

__version__ = "0.1.0"


def __getattr__(name):
    # Candidate-only users need not import PyTorch until requesting the model.
    if name == "JUREC":
        from .model import JUREC
        return JUREC
    raise AttributeError(name)
