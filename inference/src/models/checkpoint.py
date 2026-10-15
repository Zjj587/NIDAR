"""Load tensor checkpoints without unrestricted pickle execution."""

import argparse

import numpy as np
import torch
from numpy.core.multiarray import scalar


def load_checkpoint(path):
    # Legacy cs-stereo-format checkpoints include a Namespace and NumPy scalars.
    allowed = [
        argparse.Namespace,
        scalar,
        # NumPy 2 moved this callable; older checkpoints retain its old name.
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float32")),
        type(np.dtype("float64")),
    ]
    if not hasattr(torch.serialization, "safe_globals"):
        raise RuntimeError("NIDAR checkpoint loading requires PyTorch >= 2.6.")
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)
