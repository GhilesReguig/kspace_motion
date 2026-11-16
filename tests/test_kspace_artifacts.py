import numpy as np
import pytest
import torch
import torchio as tio

pytest.importorskip("finufft")
pytest.importorskip("pytorch_finufft")

from kspace_motion import MotionGPU, RandomMotionGPU
from kspace_motion import kspace_artifacts as ka

ORDER = (0, 2, 1)            # readout x, phase2 z, phase y
SHAPE = (32, 40, 48)
CONTEXT = ka.KSpaceContext(kspace_order=ORDER, signal_level=1.0)


def _subject(volume):
    return tio.Subject(img=tio.ScalarImage(tensor=volume[None]))


def simulate(volume, artifacts, params=None, order=ORDER):
    """MotionGPU output (float64 (X, Y, Z)); no motion unless `params` is given."""
    if params is None:
        params = np.zeros((6, volume.shape[order[-1]]))
    transform = MotionGPU(euler_motion_params=params, kspace_order=order, kspace_artifacts=artifacts)
    return transform(_subject(volume))["img"].data[0].double()


def kspace_of(volume):
    return MotionGPU._fft_kspace(volume)


def reconstruct(kspace):
    return torch.abs(torch.fft.ifftshift(torch.fft.ifftn(kspace)))


def image_of(kspace):
    """Complex image of a k-space difference, centred like the reconstruction."""
    return torch.fft.ifftshift(torch.fft.ifftn(kspace))


@pytest.fixture(scope="module")
def volume(phantom_factory):
    return phantom_factory(shape=SHAPE)


@pytest.fixture(scope="module")
def block():
    """Bright block in a zero background, signal level exactly 1."""
    vol = torch.zeros(SHAPE)
    vol[8:24, 10:30, 12:36] = 1.0
    return vol


def centre_axis(kspace, axis):
    return kspace.shape[axis] // 2


# ----------------------------------------------------------------------------------------------
# helpers and plumbing
# ----------------------------------------------------------------------------------------------
def test_context_axis_roles():
    context = ka.KSpaceContext(kspace_order=(2, 0, 1), signal_level=1.0)
    assert (context.axis("readout"), context.axis("phase2"), context.axis("phase")) == (2, 0, 1)
    with pytest.raises(ValueError):
        context.axis("nope")


def test_signal_level_is_the_mean_of_the_tissue(block):
    assert ka.signal_level(block) == pytest.approx(1.0)
    assert ka.signal_level(torch.zeros(4, 4, 4)) == 0.0
    assert ka.signal_level(2 * block + 0.05 * (block == 0)) == pytest.approx(2.0)


def test_normalized_frequency_has_zero_at_n_over_2():
    for n in (8, 9):
        frequency = ka.normalized_frequency(n)
        assert frequency[n // 2] == 0
        assert frequency[0] == pytest.approx(-(n // 2) / (n / 2))


def test_without_artifacts_no_motion_is_identity(volume):
    out = simulate(volume, None)
    torch.testing.assert_close(out, volume.double(), atol=1e-4, rtol=0)


def test_fft_shortcut_equals_nufft_without_motion(volume):
    """The plain FFT used when there is no motion gives the same image as the NUFFT."""
    tiny = np.zeros((6, volume.shape[1]))
    tiny[0, :] = 1e-9   # forces the NUFFT path
    torch.testing.assert_close(simulate(volume, None, params=tiny), simulate(volume, None), atol=1e-4, rtol=0)


def test_artifacts_are_applied_in_stage_order_not_list_order(volume):
    noise, gibbs = ka.RicianNoise(20, seed=1), ka.Truncation(0.5)
    torch.testing.assert_close(simulate(volume, [noise, gibbs]), simulate(volume, [gibbs, noise]))


def test_artifacts_work_with_motion(volume):
    params = np.zeros((6, volume.shape[1]))
    params[1, :] = np.linspace(0, 3, volume.shape[1])
    moved = simulate(volume, None, params=params)
    both = simulate(volume, [ka.RicianNoise(30, seed=0), ka.Truncation(0.8)], params=params)
    assert both.shape == moved.shape
    assert not torch.allclose(both, moved, atol=1e-3)


def test_random_motion_forwards_the_artifacts(volume):
    artifacts = [ka.Truncation(0.5)]
    transform = RandomMotionGPU(euler_motion_params=np.zeros((6, volume.shape[1])), phase_encoding_choice=[1],
                                kspace_artifacts=artifacts)
    out = transform(_subject(volume))["img"].data[0].double()
    expected = simulate(volume, artifacts, order=tuple(transform.kspace_order))
    torch.testing.assert_close(out, expected)


def test_apply_with_empty_list_or_none_returns_input():
    k = torch.ones(4, 4, 4, dtype=torch.complex64)
    assert ka.apply_kspace_artifacts(k, None, CONTEXT) is k
    assert ka.apply_kspace_artifacts(k, [], CONTEXT) is k


def test_registry_builds_every_artifact():
    assert set(ka.ARTIFACTS) == {"truncation", "lowpass", "t2star", "partial_fourier", "undersampling",
                                 "line_dropout", "spikes", "zipper", "dc_offset", "clipping", "noise"}
    for cls in ka.ARTIFACTS.values():
        assert cls().stage in (ka.STAGE_ACQUISITION, ka.STAGE_LINES, ka.STAGE_INTERFERENCE,
                               ka.STAGE_RECEIVER, ka.STAGE_NOISE)


@pytest.mark.parametrize("factory", [
    lambda: ka.Truncation(0), lambda: ka.Truncation(1.2), lambda: ka.Truncation(0.5, axes=["z"]),
    lambda: ka.Truncation(0.5, axes=[]), lambda: ka.LowPassFilter(0), lambda: ka.LowPassFilter(0.5, kind="x"),
    lambda: ka.T2StarDecay(0), lambda: ka.PartialFourier(0), lambda: ka.PartialFourier(0.5, axis="x"),
    lambda: ka.Undersampling(0), lambda: ka.Undersampling(1.5), lambda: ka.LineDropout(0),
    lambda: ka.LineDropout(0.1, unit="x"), lambda: ka.LineDropout(0.1, mode="x"),
    lambda: ka.LineDropout(0.1, keep_center=1), lambda: ka.Spikes(0), lambda: ka.Spikes(1, amplitude=-1),
    lambda: ka.Spikes(1, radius=(0.5, 0.2)), lambda: ka.Zipper(-1), lambda: ka.Zipper(0.3, position=0),
    lambda: ka.DCOffset(-1), lambda: ka.Clipping(0), lambda: ka.Clipping(100),
])
def test_invalid_parameters_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


# ----------------------------------------------------------------------------------------------
# Gibbs, blur, T2*
# ----------------------------------------------------------------------------------------------
def test_truncation_zeroes_the_high_frequencies_only(volume):
    k = kspace_of(volume)
    out = ka.Truncation(0.5)(k, CONTEXT)
    for axis in range(3):
        frequency = ka.normalized_frequency(k.shape[axis]).abs()
        index = [slice(None)] * 3
        index[axis] = frequency > 0.5
        assert (out[tuple(index)] == 0).all()
        index[axis] = frequency <= 0.5
    inside = (ka.normalized_frequency(SHAPE[0]).abs() <= 0.5)[:, None, None] \
        & (ka.normalized_frequency(SHAPE[1]).abs() <= 0.5)[None, :, None] \
        & (ka.normalized_frequency(SHAPE[2]).abs() <= 0.5)[None, None, :]
    torch.testing.assert_close(out[inside], k[inside])


def test_truncation_fraction_one_is_identity_and_error_grows_when_truncating(volume):
    assert torch.equal(ka.Truncation(1.0)(kspace_of(volume), CONTEXT), kspace_of(volume))
    errors = [(simulate(volume, [ka.Truncation(f)]) - volume.double()).abs().mean() for f in (0.9, 0.6, 0.3)]
    assert errors[0] < errors[1] < errors[2]


def test_truncation_gives_ringing_next_to_an_edge(block):
    """Gibbs: the truncated block overshoots its plateau and rings (non monotonic profile) outside."""
    out = simulate(block, [ka.Truncation(0.4, axes=["phase"])])
    profile = out[16, :, 24]            # along y (the phase axis), through the block
    assert profile.max() > 1.02         # overshoot
    outside = profile[:10]
    assert (outside.diff().sign().diff().abs() > 0).any()   # oscillates


def test_truncation_only_on_the_requested_axis(block):
    out = simulate(block, [ka.Truncation(0.3, axes=["phase"])])
    # along x (readout) the block edges stay sharp: the plateau is flat
    row = out[:, 20, 24]
    assert row[10:22].std() < 0.01
    # along y (phase) the profile is distorted
    assert out[16, 12:28, 24].std() > 0.01


def sharpness(volume, dim):
    """Steepest edge along `dim`."""
    return float(volume.double().diff(dim=dim).abs().max())


def test_gaussian_lowpass_blurs_and_hamming_cuts(volume, block):
    k = kspace_of(volume)
    assert (ka.LowPassFilter(0.3, "gaussian")(k, CONTEXT).abs() <= k.abs() + 1e-6).all()
    out = ka.LowPassFilter(0.3, "hamming")(k, CONTEXT)
    cut = ka.normalized_frequency(SHAPE[0]).abs() > 0.3
    assert (out[cut] == 0).all()
    assert sharpness(simulate(block, [ka.LowPassFilter(0.3)]), 0) < 0.5 * sharpness(block, 0)


def test_lowpass_axes(block):
    out = simulate(block, [ka.LowPassFilter(0.2, axes=["phase"])])
    assert sharpness(out, 1) < 0.5 * sharpness(block, 1)             # smoothing along y (phase)...
    assert sharpness(out, 0) > 0.9 * sharpness(block, 0)             # ...only


def test_t2star_blurs_along_the_readout_only(block):
    out = simulate(block, [ka.T2StarDecay(0.3)])
    assert sharpness(out, 0) < 0.5 * sharpness(block, 0)             # readout = x
    assert sharpness(out, 1) > 0.9 * sharpness(block, 1)


def test_t2star_decay_is_an_exponential_of_the_frequency(volume):
    k = kspace_of(volume)
    out = ka.T2StarDecay(0.5)(k, CONTEXT)
    n = SHAPE[0]
    frequency = ka.normalized_frequency(n).abs()
    ratio = (out / k)[:, 5, 7].real
    torch.testing.assert_close(ratio, torch.exp(-frequency / 0.5), atol=1e-5, rtol=1e-4)


def test_longer_t2star_means_less_blur(volume):
    errors = [(simulate(volume, [ka.T2StarDecay(r)]) - volume.double()).abs().mean() for r in (0.3, 1, 5)]
    assert errors[0] > errors[1] > errors[2]


# ----------------------------------------------------------------------------------------------
# partial Fourier, aliasing
# ----------------------------------------------------------------------------------------------
def test_partial_fourier_keeps_the_centre_and_the_positive_side(volume):
    k = kspace_of(volume)
    out = ka.PartialFourier(0.6, axis="phase")(k, CONTEXT)
    axis, n = 1, SHAPE[1]
    kept = int(np.ceil(0.6 * n))
    assert (out.narrow(axis, 0, n - kept) == 0).all()
    torch.testing.assert_close(out.narrow(axis, n - kept, kept), k.narrow(axis, n - kept, kept))
    assert n - kept < n // 2          # the centre is acquired
    assert torch.equal(ka.PartialFourier(1.0)(k, CONTEXT), k)


def test_partial_fourier_degrades_the_phase_direction_only(block):
    out = simulate(block, [ka.PartialFourier(0.55, axis="phase")])
    assert sharpness(out, 1) < 0.8 * sharpness(block, 1)
    assert sharpness(out, 0) > 0.95 * sharpness(block, 0)


@pytest.mark.parametrize("role,axis", [("readout", 0), ("phase2", 2), ("phase", 1)])
def test_partial_fourier_axis_roles(volume, role, axis):
    out = ka.PartialFourier(0.6, axis=role)(kspace_of(volume), CONTEXT)
    n = SHAPE[axis]
    assert (out.narrow(axis, 0, n - int(np.ceil(0.6 * n))) == 0).all()


def test_undersampling_folds_the_image(volume):
    """Keeping one line out of two (even size) = sum of the image and its copy shifted by half the FOV, halved."""
    out = simulate(volume, [ka.Undersampling(2, axis="phase")])
    ref = volume.double()
    expected = 0.5 * (ref + torch.roll(ref, SHAPE[1] // 2, dims=1))
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=0)


def test_undersampling_keeps_one_line_out_of_r(volume):
    out = ka.Undersampling(3, axis="readout")(kspace_of(volume), CONTEXT)
    n = SHAPE[0]
    kept = (torch.arange(n) - n // 2) % 3 == 0
    assert (out[~kept] == 0).all()
    torch.testing.assert_close(out[kept], kspace_of(volume)[kept])


def test_undersampling_factor_one_is_identity(volume):
    assert torch.equal(ka.Undersampling(1)(kspace_of(volume), CONTEXT), kspace_of(volume))


# ----------------------------------------------------------------------------------------------
# missing lines
# ----------------------------------------------------------------------------------------------
def _zero_planes(out, axis):
    other = tuple(a for a in range(3) if a != axis)
    return torch.nonzero((out.abs().sum(dim=other) == 0)).flatten()


def test_dropout_planes_zeroes_the_requested_number_of_planes(volume):
    k = kspace_of(volume)
    out = ka.LineDropout(0.1, unit="planes", seed=3)(k, CONTEXT)
    planes = _zero_planes(out, axis=1)
    assert len(planes) == round(0.1 * SHAPE[1])
    kept = torch.ones(SHAPE[1], dtype=torch.bool)
    kept[planes] = False
    torch.testing.assert_close(out[:, kept], k[:, kept])


def test_dropout_is_reproducible_and_depends_on_the_seed(volume):
    k = kspace_of(volume)
    a = ka.LineDropout(0.2, seed=1)(k, CONTEXT)
    assert torch.equal(a, ka.LineDropout(0.2, seed=1)(k, CONTEXT))
    assert not torch.equal(a, ka.LineDropout(0.2, seed=2)(k, CONTEXT))


def test_dropout_grouped_planes_are_consecutive(volume):
    planes = _zero_planes(ka.LineDropout(0.25, mode="grouped", seed=4)(kspace_of(volume), CONTEXT), axis=1)
    assert len(planes) == round(0.25 * SHAPE[1])
    assert (planes.diff() == 1).all()


def test_dropout_keep_center_protects_the_centre(volume):
    k = kspace_of(volume)
    out = ka.LineDropout(0.5, keep_center=0.3, seed=0)(k, CONTEXT)
    central = ka.normalized_frequency(SHAPE[1]).abs() < 0.3
    torch.testing.assert_close(out[:, central], k[:, central])
    assert (out != k).any()


def test_dropout_gain_attenuates_instead_of_zeroing(volume):
    k = kspace_of(volume)
    out = ka.LineDropout(0.1, gain=0.5, seed=0)(k, CONTEXT)
    changed = (out != k).any(dim=0).any(dim=1) if False else (out != k).any(dim=2).any(dim=0)
    planes = torch.nonzero(changed).flatten()
    assert len(planes) == round(0.1 * SHAPE[1])
    torch.testing.assert_close(out[:, planes], 0.5 * k[:, planes])


def test_dropout_lines_removes_single_readout_lines(volume):
    k = kspace_of(volume)
    out = ka.LineDropout(0.05, unit="lines", seed=5)(k, CONTEXT)
    zero_line = (out == 0).all(dim=0)                  # (y, z): readout is x
    assert int(zero_line.sum()) == round(0.05 * SHAPE[1] * SHAPE[2])
    assert torch.equal(out[:, ~zero_line], k[:, ~zero_line])


def test_dropout_lines_with_swapped_phase_axes(volume):
    """With the phase axis before the second phase axis in the array, the 2D mask is transposed."""
    context = ka.KSpaceContext(kspace_order=(0, 1, 2), signal_level=1.0)   # phase = z, phase2 = y
    k = kspace_of(volume)
    out = ka.LineDropout(0.05, unit="lines", seed=5)(k, context)
    zero_line = (out == 0).all(dim=0)
    assert int(zero_line.sum()) == round(0.05 * SHAPE[1] * SHAPE[2])
    assert torch.equal(out[:, ~zero_line], k[:, ~zero_line])


def test_dropout_makes_stripes_across_the_phase_direction(volume):
    out = simulate(volume, [ka.LineDropout(0.1, seed=0)])
    assert (out - volume.double()).abs().mean() > 0.003


# ----------------------------------------------------------------------------------------------
# interference
# ----------------------------------------------------------------------------------------------
def test_spikes_change_exactly_count_samples_outside_the_centre(volume):
    k = kspace_of(volume)
    spikes = ka.Spikes(count=5, amplitude=0.1, radius=(0.2, 0.6), seed=2)
    out = spikes(k, CONTEXT)
    changed = torch.nonzero(out != k)
    assert len(changed) == 5
    centre = torch.tensor(SHAPE) // 2
    assert not (changed == centre).all(dim=1).any()
    for index in changed:
        frequency = (index - centre).float() / (torch.tensor(SHAPE) / 2)
        assert 0.2 - 0.1 <= float(frequency.norm()) <= 0.6 + 0.1     # rounding to the grid


def test_spikes_positions_are_reproducible_and_seeded():
    a = ka.Spikes(4, seed=1).positions(SHAPE)
    assert torch.equal(a, ka.Spikes(4, seed=1).positions(SHAPE))
    assert not torch.equal(a, ka.Spikes(4, seed=2).positions(SHAPE))


def test_a_spike_is_a_plane_wave_of_the_requested_amplitude(volume):
    k = kspace_of(volume)
    out = ka.Spikes(count=1, amplitude=0.25, seed=0)(k, CONTEXT)
    wave = image_of(out - k)
    modulus = wave.abs()
    expected = 0.25 * CONTEXT.signal_level
    torch.testing.assert_close(modulus, torch.full_like(modulus, expected), rtol=1e-3, atol=1e-5)


def test_spike_radius_sets_the_distance_to_the_centre():
    """Close to the centre: wide bands, close to the edge: fine fringes (a spike at k gives a wave of frequency k)."""
    centre, half = torch.tensor(SHAPE) // 2, torch.tensor(SHAPE) / 2

    def mean_radius(radius):
        positions = ka.Spikes(10, 0.2, radius=radius, seed=0).positions(SHAPE)
        return float(((positions - centre) / half).norm(dim=1).mean())

    assert mean_radius((0.05, 0.15)) < 0.25 < 0.6 < mean_radius((0.6, 0.8))


def test_spikes_show_in_the_image(volume):
    out = simulate(volume, [ka.Spikes(2, amplitude=0.2, seed=1)])
    assert (out - volume.double()).abs().mean() > 0.01


def test_zipper_is_a_line_along_the_readout_axis(volume):
    k = kspace_of(volume)
    out = ka.Zipper(amplitude=0.5, position=0.4)(k, CONTEXT)
    wave = image_of(out - k)
    modulus = wave.abs()
    expected = 0.5 * CONTEXT.signal_level
    # non zero only on the line x free, y = z = centre; there the modulus is the amplitude
    line = modulus[:, SHAPE[1] // 2, SHAPE[2] // 2]
    torch.testing.assert_close(line, torch.full_like(line, expected), rtol=1e-3, atol=1e-5)
    mask = torch.ones(SHAPE, dtype=torch.bool)
    mask[:, SHAPE[1] // 2, SHAPE[2] // 2] = False
    assert modulus[mask].max() < 1e-4 * expected
    # position sets the number of dashes along the line
    assert wave[:, SHAPE[1] // 2, SHAPE[2] // 2].real.diff().sign().diff().ne(0).sum() > 4


def test_zipper_follows_the_readout_role(volume):
    context = ka.KSpaceContext(kspace_order=(2, 0, 1), signal_level=1.0)   # readout = z
    k = kspace_of(volume)
    modulus = image_of(ka.Zipper(0.5)(k, context) - k).abs()
    along_z = modulus[SHAPE[0] // 2, SHAPE[1] // 2, :]
    assert along_z.min() > 0.4 * context.signal_level


def test_dc_offset_is_one_bright_voxel_at_the_centre(volume):
    k = kspace_of(volume)
    modulus = image_of(ka.DCOffset(1.5)(k, CONTEXT) - k).abs()
    centre = tuple(s // 2 for s in SHAPE)
    assert modulus[centre] == pytest.approx(1.5 * CONTEXT.signal_level, rel=1e-3)
    modulus[centre] = 0
    assert modulus.max() < 1e-4


# ----------------------------------------------------------------------------------------------
# receiver, noise
# ----------------------------------------------------------------------------------------------
def test_clipping_limits_the_largest_samples_and_keeps_the_phase(volume):
    k = kspace_of(volume)
    out = ka.Clipping(saturated_pct=1.0)(k, CONTEXT)
    threshold = out.abs().max()
    saturated = (out.abs() >= threshold * (1 - 1e-4)).float().mean()
    assert 0.009 <= float(saturated) <= 0.02
    assert (out.abs() <= k.abs() * (1 + 1e-5)).all()
    big = k.abs() > threshold * 1.01
    torch.testing.assert_close(torch.angle(out[big]), torch.angle(k[big]), atol=1e-5, rtol=0)
    # the samples below the threshold are untouched
    small = k.abs() < threshold
    torch.testing.assert_close(out[small], k[small])


def test_more_clipping_means_a_larger_error(volume):
    errors = [(simulate(volume, [ka.Clipping(p)]) - volume.double()).abs().mean() for p in (0.1, 1, 10)]
    assert errors[0] < errors[1] < errors[2]


@pytest.mark.parametrize("snr_db", [10.0, 20.0, 30.0])
def test_noise_in_the_background_is_rayleigh_with_the_requested_snr(block, snr_db):
    out = simulate(block, [ka.RicianNoise(snr_db, seed=0)])
    sigma = 1.0 / 10 ** (snr_db / 20)                  # signal level of the block is 1
    background = torch.ones(SHAPE, dtype=torch.bool)
    background[4:28, 6:34, 8:40] = False
    values = out[background]
    assert float(values.mean()) == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.03)
    assert float((values ** 2).mean()) == pytest.approx(2 * sigma ** 2, rel=0.05)


def test_noise_in_the_signal_is_gaussian_around_the_signal(block):
    out = simulate(block, [ka.RicianNoise(25.0, seed=0)])
    inside = out[12:20, 14:26, 18:30]
    sigma = 10 ** (-25 / 20)
    assert float(inside.mean()) == pytest.approx(1.0, abs=0.01)
    assert float(inside.std()) == pytest.approx(sigma, rel=0.1)


def test_noise_is_reproducible_and_seeded(volume):
    a = simulate(volume, [ka.RicianNoise(20, seed=1)])
    torch.testing.assert_close(a, simulate(volume, [ka.RicianNoise(20, seed=1)]))
    assert not torch.allclose(a, simulate(volume, [ka.RicianNoise(20, seed=2)]))


def test_lower_snr_means_more_error(volume):
    errors = [(simulate(volume, [ka.RicianNoise(s, seed=0)]) - volume.double()).abs().mean() for s in (40, 25, 10)]
    assert errors[0] < errors[1] < errors[2]


def test_noise_sigma_uses_the_signal_level():
    noise = ka.RicianNoise(20.0)
    assert noise.sigma(ka.KSpaceContext(ORDER, 2.0)) == pytest.approx(0.2)
