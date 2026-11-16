r"""Artefacts applied on the acquired k-space, between the acquisition (NUFFT) and the reconstruction.

Each artefact is a small callable ``artifact(kspace, context) -> kspace`` working on the complex,
*centred* k-space of the volume (the zero frequency sits at index ``n // 2`` of every axis).
They are passed to :class:`~kspace_motion.MotionGPU` with ``kspace_artifacts=[...]`` and applied in
the order of their ``stage`` (acquisition geometry, line corruption, interference, receiver, noise),
whatever the order of the list.

Axes are named by their role in the acquisition: ``"readout"`` (fastest), ``"phase2"`` (second
phase-encoding axis) and ``"phase"`` (slowest one, where the motion takes place). The roles come from
the ``kspace_order`` of the transform, ``(readout, phase2, phase)``.

Amplitudes are expressed relative to the *signal level* of the image (mean of the voxels above 10 % of
the maximum), so that the settings do not depend on the intensity scale.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

ROLES = ("readout", "phase2", "phase")

# application order: what shapes the acquisition first, the receiver and the noise last
STAGE_ACQUISITION, STAGE_LINES, STAGE_INTERFERENCE, STAGE_RECEIVER, STAGE_NOISE = 10, 20, 30, 40, 50


@dataclass
class KSpaceContext:
    """What an artefact needs to know about the acquisition."""
    kspace_order: Tuple[int, int, int]   # (readout, phase2, phase) axes of the volume
    signal_level: float                  # reference intensity of the image

    def axis(self, role: str) -> int:
        if role not in ROLES:
            raise ValueError(f"unknown axis role {role!r}, expected one of {ROLES}")
        return int(self.kspace_order[ROLES.index(role)])


def signal_level(image: torch.Tensor) -> float:
    """Mean intensity of the voxels above 10 % of the maximum (the 'tissue')."""
    magnitude = image.abs()
    top = float(magnitude.max())
    if top <= 0:
        return 0.0
    return float(magnitude[magnitude > 0.1 * top].mean())


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def normalized_frequency(n: int, device="cpu") -> torch.Tensor:
    """Frequency of every sample of an axis, in units of the maximum frequency: 0 at ``n // 2``."""
    return (torch.arange(n, device=device, dtype=torch.float32) - n // 2) / (n / 2)


def _along(values: torch.Tensor, axis: int, ndim: int = 3) -> torch.Tensor:
    """Reshape a 1D tensor so that it broadcasts along `axis`."""
    shape = [1] * ndim
    shape[axis] = values.numel()
    return values.reshape(shape)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _check_axes(axes: Sequence[str]) -> Tuple[str, ...]:
    axes = tuple(axes)
    for role in axes:
        if role not in ROLES:
            raise ValueError(f"unknown axis role {role!r}, expected one of {ROLES}")
    if not axes:
        raise ValueError("at least one axis is needed")
    return axes


def _check_fraction(name: str, value: float, low: float = 0.0, high: float = 1.0, closed_low: bool = False):
    ok = (low <= value if closed_low else low < value) and value <= high
    if not ok:
        bracket = "[" if closed_low else "]"
        raise ValueError(f"{name} must be in {bracket}{low}, {high}], got {value}")


class KSpaceArtifact:
    """Base class: subclasses set `stage` and implement `__call__`."""
    stage = STAGE_ACQUISITION

    def __call__(self, kspace: torch.Tensor, context: KSpaceContext) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({fields})"


# ----------------------------------------------------------------------------------------------
# acquisition geometry / weighting (multiplicative, deterministic)
# ----------------------------------------------------------------------------------------------
class Truncation(KSpaceArtifact):
    """Gibbs ringing: the k-space is cut beyond ``fraction`` of the maximum frequency.

    Args:
        fraction: part of the k-space extent that is kept, in ]0, 1].
        axes: roles of the truncated axes.
    """
    stage = STAGE_ACQUISITION

    def __init__(self, fraction: float = 0.6, axes: Sequence[str] = ROLES):
        _check_fraction("fraction", fraction)
        self.fraction = fraction
        self.axes = _check_axes(axes)

    def __call__(self, kspace, context):
        for role in self.axes:
            axis = context.axis(role)
            keep = normalized_frequency(kspace.shape[axis], kspace.device).abs() <= self.fraction
            kspace = kspace * _along(keep, axis)
        return kspace


class LowPassFilter(KSpaceArtifact):
    """Loss of resolution: the k-space is multiplied by a low-pass window.

    Args:
        cutoff: width of the window, in units of the maximum frequency (]0, 1]); for the Gaussian
            it is the standard deviation, for the Hamming window the frequency where it reaches 0.
        kind: ``"gaussian"`` or ``"hamming"``.
        axes: roles of the filtered axes.
    """
    stage = STAGE_ACQUISITION

    def __init__(self, cutoff: float = 0.5, kind: str = "gaussian", axes: Sequence[str] = ROLES):
        _check_fraction("cutoff", cutoff)
        if kind not in ("gaussian", "hamming"):
            raise ValueError(f"kind must be 'gaussian' or 'hamming', got {kind!r}")
        self.cutoff = cutoff
        self.kind = kind
        self.axes = _check_axes(axes)

    def window(self, frequency: torch.Tensor) -> torch.Tensor:
        if self.kind == "gaussian":
            return torch.exp(-0.5 * (frequency / self.cutoff) ** 2)
        inside = frequency.abs() <= self.cutoff
        return torch.where(inside, 0.54 + 0.46 * torch.cos(math.pi * frequency / self.cutoff),
                           torch.zeros_like(frequency))

    def __call__(self, kspace, context):
        for role in self.axes:
            axis = context.axis(role)
            frequency = normalized_frequency(kspace.shape[axis], kspace.device)
            kspace = kspace * _along(self.window(frequency), axis)
        return kspace


class T2StarDecay(KSpaceArtifact):
    """Signal decay during the readout: ``exp(-|k| / ratio)`` along the readout axis, which blurs
    the image in that direction.

    Args:
        ratio: T2* divided by the half duration of the readout (the edge of the k-space is reached at
            ``|k| = 1``). Small values: strong blur.
    """
    stage = STAGE_ACQUISITION

    def __init__(self, ratio: float = 1.0):
        if not ratio > 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        self.ratio = ratio

    def __call__(self, kspace, context):
        axis = context.axis("readout")
        decay = torch.exp(-normalized_frequency(kspace.shape[axis], kspace.device).abs() / self.ratio)
        return kspace * _along(decay, axis)


class PartialFourier(KSpaceArtifact):
    """Only ``fraction`` of the k-space is acquired along an axis (the part that contains the centre
    and the positive frequencies), the rest is zero-filled: the image loses resolution and gets ringing.

    Args:
        fraction: acquired part, in ]0, 1] (typically 0.55 to 0.8).
        axis: role of the axis.
    """
    stage = STAGE_ACQUISITION

    def __init__(self, fraction: float = 0.65, axis: str = "phase"):
        _check_fraction("fraction", fraction)
        self.fraction = fraction
        self.axis = _check_axes([axis])[0]

    def __call__(self, kspace, context):
        axis = context.axis(self.axis)
        n = kspace.shape[axis]
        kept = min(n, int(math.ceil(self.fraction * n)))
        keep = torch.arange(n, device=kspace.device) >= n - kept
        return kspace * _along(keep, axis)


class Undersampling(KSpaceArtifact):
    """Aliasing: only one k-space line out of ``factor`` is kept along an axis (no parallel
    reconstruction). The image is folded: it shows ``factor`` copies, each at 1/``factor`` of the
    intensity, shifted by the field of view divided by ``factor``.

    Args:
        factor: acceleration factor (integer >= 1).
        axis: role of the undersampled axis.
    """
    stage = STAGE_ACQUISITION

    def __init__(self, factor: int = 2, axis: str = "phase"):
        if int(factor) != factor or factor < 1:
            raise ValueError(f"factor must be an integer >= 1, got {factor}")
        self.factor = int(factor)
        self.axis = _check_axes([axis])[0]

    def __call__(self, kspace, context):
        axis = context.axis(self.axis)
        n = kspace.shape[axis]
        keep = (torch.arange(n, device=kspace.device) - n // 2) % self.factor == 0
        return kspace * _along(keep, axis)


# ----------------------------------------------------------------------------------------------
# corrupted lines
# ----------------------------------------------------------------------------------------------
class LineDropout(KSpaceArtifact):
    """Missing or corrupted data: a part of the k-space lines (or planes) is set to ``gain`` times
    its value (0: lost).

    Args:
        fraction: part of the candidate lines that is corrupted, in ]0, 1].
        unit: ``"planes"`` (whole planes perpendicular to the phase axis: stripes) or ``"lines"``
            (single readout lines of the phase / phase2 plane: diffuse corruption).
        mode: ``"random"`` positions or one ``"grouped"`` block of consecutive ones.
        gain: factor applied to the corrupted data (0: dropout, 0.5: attenuated, 2: amplified).
        keep_center: part of the central k-space that is never corrupted, in [0, 1[.
        seed: seed of the random draw.
    """
    stage = STAGE_LINES

    def __init__(self, fraction: float = 0.05, unit: str = "planes", mode: str = "random",
                 gain: float = 0.0, keep_center: float = 0.0, seed: int = 0):
        _check_fraction("fraction", fraction)
        if unit not in ("planes", "lines"):
            raise ValueError(f"unit must be 'planes' or 'lines', got {unit!r}")
        if mode not in ("random", "grouped"):
            raise ValueError(f"mode must be 'random' or 'grouped', got {mode!r}")
        if not 0 <= keep_center < 1:
            raise ValueError(f"keep_center must be in [0, 1[, got {keep_center}")
        self.fraction, self.unit, self.mode = fraction, unit, mode
        self.gain, self.keep_center, self.seed = gain, keep_center, seed

    def _choose(self, candidates: torch.Tensor) -> torch.Tensor:
        """Pick the corrupted ones among the flat indices `candidates`."""
        count = min(len(candidates), max(1, round(self.fraction * len(candidates))))
        generator = _generator(self.seed)
        if self.mode == "random":
            return candidates[torch.randperm(len(candidates), generator=generator)[:count]]
        start = int(torch.randint(0, len(candidates) - count + 1, (1,), generator=generator))
        return candidates[start:start + count]

    def __call__(self, kspace, context):
        phase, phase2 = context.axis("phase"), context.axis("phase2")
        n_phase = kspace.shape[phase]
        k_phase = normalized_frequency(n_phase).abs()
        if self.unit == "planes":
            candidates = torch.nonzero(k_phase >= self.keep_center).flatten()
            if len(candidates) == 0:
                return kspace
            weights = torch.ones(n_phase)
            weights[self._choose(candidates)] = self.gain
            return kspace * _along(weights.to(kspace.device), phase)

        n_phase2 = kspace.shape[phase2]
        k_phase2 = normalized_frequency(n_phase2).abs()
        radius = torch.sqrt(k_phase[:, None] ** 2 + k_phase2[None, :] ** 2)
        candidates = torch.nonzero((radius >= self.keep_center).flatten()).flatten()
        if len(candidates) == 0:
            return kspace
        weights = torch.ones(n_phase * n_phase2)
        weights[self._choose(candidates)] = self.gain
        weights = weights.reshape(n_phase, n_phase2)
        if phase > phase2:   # the broadcast shape lists the axes in increasing order
            weights = weights.T
        shape = [1, 1, 1]
        shape[phase], shape[phase2] = n_phase, n_phase2
        return kspace * weights.reshape(shape).to(kspace.device)


# ----------------------------------------------------------------------------------------------
# interference (additive)
# ----------------------------------------------------------------------------------------------
class Spikes(KSpaceArtifact):
    """Spike noise: isolated k-space samples with a large random-phase value. Each one adds a
    sinusoidal fringe pattern to the whole image (herringbone artefact), whose spatial frequency is
    given by the position of the spike.

    Args:
        count: number of spikes.
        amplitude: amplitude of each fringe pattern relative to the signal level of the image.
        radius: (min, max) distance of the spikes from the k-space centre, in units of the maximum
            frequency. Close to the centre: wide bands, close to the edge: fine fringes.
        seed: seed of the random positions and phases.
    """
    stage = STAGE_INTERFERENCE

    def __init__(self, count: int = 3, amplitude: float = 0.1, radius: Tuple[float, float] = (0.2, 0.5),
                 seed: int = 0):
        if int(count) != count or count < 1:
            raise ValueError(f"count must be an integer >= 1, got {count}")
        if not amplitude >= 0:
            raise ValueError(f"amplitude must be >= 0, got {amplitude}")
        if not 0 < radius[0] <= radius[1] <= 1.5:
            raise ValueError(f"radius must satisfy 0 < min <= max <= 1.5, got {radius}")
        self.count, self.amplitude, self.radius, self.seed = int(count), amplitude, tuple(radius), seed

    def positions(self, shape) -> torch.Tensor:
        """(count, 3) integer sample indices of the spikes (never the centre)."""
        generator = _generator(self.seed)
        direction = torch.randn(self.count, 3, generator=generator)
        direction = direction / direction.norm(dim=1, keepdim=True)
        radius = self.radius[0] + (self.radius[1] - self.radius[0]) * torch.rand(self.count, generator=generator)
        sizes = torch.tensor(shape, dtype=torch.float32)
        index = torch.round(sizes // 2 + direction * radius[:, None] * sizes / 2).long()
        index = torch.minimum(torch.clamp(index, min=0), torch.tensor(shape) - 1)
        centre = torch.tensor(shape) // 2
        at_centre = (index == centre).all(dim=1)
        index[at_centre, 0] += 1 if shape[0] > 1 else 0
        return index

    def __call__(self, kspace, context):
        shape = tuple(kspace.shape)
        index = self.positions(shape)
        phases = torch.rand(self.count, generator=_generator(self.seed + 7919)) * 2 * math.pi
        # a plane wave of amplitude A in the image needs A * (number of voxels) in one k-space sample
        value = self.amplitude * context.signal_level * kspace.numel()
        kspace = kspace.clone()
        for (i, j, k), phase in zip(index.tolist(), phases.tolist()):
            kspace[i, j, k] += complex(value * math.cos(phase), value * math.sin(phase))
        return kspace


class Zipper(KSpaceArtifact):
    """Zipper: a constant signal at one readout frequency (RF leak), seen as a line of dashes along
    the readout axis through the centre of the image.

    Args:
        amplitude: amplitude of the line relative to the signal level of the image.
        position: readout frequency of the leak, in ]0, 1] of the maximum frequency (sign +).
            It sets the spacing of the dashes.
    """
    stage = STAGE_INTERFERENCE

    def __init__(self, amplitude: float = 0.3, position: float = 0.35):
        if not amplitude >= 0:
            raise ValueError(f"amplitude must be >= 0, got {amplitude}")
        _check_fraction("position", position)
        self.amplitude, self.position = amplitude, position

    def __call__(self, kspace, context):
        axis = context.axis("readout")
        n = kspace.shape[axis]
        index = min(n - 1, n // 2 + max(1, round(self.position * n / 2)))
        kspace = kspace.clone()
        # the whole plane at this readout frequency: a delta in the other axes, a sinusoid along `axis`
        kspace.select(axis, index).add_(self.amplitude * context.signal_level * n)
        return kspace


class DCOffset(KSpaceArtifact):
    """Receiver offset: a constant added to every sample, which is a single bright voxel at the
    centre of the image (central point artefact).

    Args:
        amplitude: intensity of the voxel relative to the signal level of the image.
    """
    stage = STAGE_INTERFERENCE

    def __init__(self, amplitude: float = 1.0):
        if not amplitude >= 0:
            raise ValueError(f"amplitude must be >= 0, got {amplitude}")
        self.amplitude = amplitude

    def __call__(self, kspace, context):
        return kspace + self.amplitude * context.signal_level


# ----------------------------------------------------------------------------------------------
# receiver and noise
# ----------------------------------------------------------------------------------------------
class Clipping(KSpaceArtifact):
    """Receiver saturation: the magnitude of the largest samples (the centre of the k-space) is
    clipped, keeping their phase.

    Args:
        saturated_pct: percentage of the samples that are clipped, in ]0, 100[.
    """
    stage = STAGE_RECEIVER

    def __init__(self, saturated_pct: float = 0.5):
        if not 0 < saturated_pct < 100:
            raise ValueError(f"saturated_pct must be in ]0, 100[, got {saturated_pct}")
        self.saturated_pct = saturated_pct

    def __call__(self, kspace, context):
        magnitude = kspace.abs()
        rank = max(1, int(magnitude.numel() * (1 - self.saturated_pct / 100)))
        threshold = torch.kthvalue(magnitude.flatten(), rank).values
        if threshold <= 0:
            return kspace
        return kspace * torch.clamp(threshold / magnitude.clamp_min(1e-30), max=1.0)


class RicianNoise(KSpaceArtifact):
    """Complex Gaussian noise added to every k-space sample. The reconstruction takes the magnitude, so
    the noise of the image is Rician (Rayleigh in the background).

    Args:
        snr_db: signal to noise ratio in dB: signal level of the image over the standard deviation of
            the noise of the real (or imaginary) part of the image.
        seed: seed of the noise.
    """
    stage = STAGE_NOISE

    def __init__(self, snr_db: float = 25.0, seed: int = 0):
        self.snr_db = snr_db
        self.seed = seed

    def sigma(self, context: KSpaceContext) -> float:
        """Standard deviation of the noise of the image (real or imaginary part)."""
        return context.signal_level / 10 ** (self.snr_db / 20)

    def __call__(self, kspace, context):
        # with the 1/N normalisation of the inverse FFT, white k-space noise of std s gives image
        # noise of std s / sqrt(N)
        std = self.sigma(context) * math.sqrt(kspace.numel())
        generator = _generator(self.seed)
        real = torch.randn(kspace.shape, generator=generator)
        imag = torch.randn(kspace.shape, generator=generator)
        noise = torch.complex(real, imag).to(device=kspace.device, dtype=kspace.dtype)
        return kspace + std * noise


ARTIFACTS = {
    "truncation": Truncation,
    "lowpass": LowPassFilter,
    "t2star": T2StarDecay,
    "partial_fourier": PartialFourier,
    "undersampling": Undersampling,
    "line_dropout": LineDropout,
    "spikes": Spikes,
    "zipper": Zipper,
    "dc_offset": DCOffset,
    "clipping": Clipping,
    "noise": RicianNoise,
}


def apply_kspace_artifacts(kspace: torch.Tensor, artifacts: Optional[Sequence[KSpaceArtifact]],
                           context: KSpaceContext) -> torch.Tensor:
    """Apply the artefacts ordered by `stage` (the order of the list breaks ties)."""
    for artifact in sorted(artifacts or [], key=lambda a: a.stage):
        kspace = artifact(kspace, context)
    return kspace
