import os

# Several OpenMP runtimes end up in one process (pip torch, conda finufft/numpy, itk-elastix);
# with more than one thread, elastix/transformix segfaults. Must be set before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import torch


def make_phantom(shape=(48, 48, 48), seed=0):
    """Smooth asymmetric 3D blobs, easy to register and with enough k-space content."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    vol = np.zeros(shape, dtype=np.float32)
    for _ in range(6):
        center = rng.uniform(-0.5, 0.5, 3)
        sigma = rng.uniform(0.15, 0.3, 3)
        amp = rng.uniform(0.5, 1.0)
        vol += amp * np.exp(-sum(((g - c) / s) ** 2 for g, c, s in zip(grids, center, sigma)))
    return torch.from_numpy(vol / vol.max())


@pytest.fixture(scope="session")
def phantom():
    return make_phantom()


@pytest.fixture(scope="session")
def phantom_factory():
    return make_phantom
