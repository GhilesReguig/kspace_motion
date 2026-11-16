import warnings

import numpy as np
import pytest
import torch
import torchio as tio
from scipy.spatial.distance import pdist

pytest.importorskip("finufft")
pytest.importorskip("pytorch_finufft")

from kspace_motion import (MotionFromTimeCourse, MotionGPU,
                           RandomMotionFromTimeCourse, RandomMotionGPU)
from kspace_motion import kspace_motion as gpu_module
from kspace_motion import motion as cpu_module

ATOL = 1e-4
ORDERS = [(0, 2, 1), (0, 1, 2), (1, 2, 0), (2, 0, 1)]


def _subject(volume):
    return tio.Subject(img=tio.ScalarImage(tensor=volume[None]))


def _run(kind, volume, params, order=(0, 2, 1), **kwargs):
    """Apply MotionGPU ('gpu') or MotionFromTimeCourse ('cpu'), return a float64 (X, Y, Z) tensor."""
    cls = MotionGPU if kind == "gpu" else MotionFromTimeCourse
    transform = cls(euler_motion_params=np.array(params, copy=True), kspace_order=order, **kwargs)
    return transform(_subject(volume))["img"].data[0].double()


def _constant(axis, value, n):
    params = np.zeros((6, n))
    params[axis, :] = value
    return params


@pytest.fixture(scope="module")
def volume(phantom_factory):
    # non-cubic on purpose, so that mixed-up axes are detected
    return phantom_factory(shape=(24, 32, 40))


# ----------------------------------------------------------------------------------------------
# physical behaviour of the simulated motion
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["gpu", "cpu"])
@pytest.mark.parametrize("order", ORDERS)
def test_no_motion_is_identity(kind, order, volume):
    out = _run(kind, volume, np.zeros((6, volume.shape[order[-1]])), order)
    assert out.shape == volume.shape
    torch.testing.assert_close(out, volume.double(), atol=ATOL, rtol=0)


@pytest.mark.parametrize("kind", ["gpu", "cpu"])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_constant_translation_shifts_the_object(kind, order, axis, volume):
    """A translation of +2 voxels, constant over the whole acquisition, moves the object by +2 voxels."""
    out = _run(kind, volume, _constant(axis, 2.0, volume.shape[order[-1]]), order)
    expected = torch.roll(volume.double(), 2, axis)
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("kind", ["gpu", "cpu"])
def test_motion_in_half_of_the_phase_lines_gives_an_artefact(kind, volume):
    n = volume.shape[1]
    params = np.zeros((6, n))
    params[0, n // 2:] = 2.0
    out = _run(kind, volume, params)
    still, moved = volume.double(), torch.roll(volume.double(), 2, 0)
    assert (out - still).abs().max() > 0.05
    assert (out - moved).abs().max() > 0.05


def test_gpu_and_cpu_implementations_agree(volume):
    n = volume.shape[1]
    params = np.zeros((6, n))
    params[0, n // 2:] = 2.0
    params[1, n // 4:] = -1.5
    params[3, n // 2:] = 4.0   # rotations
    params[5, 3 * n // 4:] = -3.0
    torch.testing.assert_close(_run("gpu", volume, params), _run("cpu", volume, params), atol=ATOL, rtol=0)


def test_nufft_type1_matches_type2_for_translations(volume):
    n = volume.shape[1]
    params = np.zeros((6, n))
    params[0, n // 2:] = 2.0
    params[1, n // 4:] = -1.5
    type2 = _run("cpu", volume, params)
    type1 = _run("cpu", volume, params, nufft_type="1D_type1")
    torch.testing.assert_close(type1, type2, atol=ATOL, rtol=0)


def test_nufft_type1_is_only_approximately_type2_for_rotations(volume):
    """type1 is kept for testing only (the code warns that type2 is the correct one)."""
    n = volume.shape[1]
    params = np.zeros((6, n))
    params[4, n // 2:] = 3.0
    type2 = _run("cpu", volume, params)
    type1 = _run("cpu", volume, params, nufft_type="1D_type1")
    torch.testing.assert_close(type1, type2, atol=5e-3, rtol=0)


@pytest.mark.parametrize("kind", ["gpu", "cpu"])
def test_motion_time_course_is_interpolated_to_the_phase_dimension(kind, volume):
    short = np.random.default_rng(1).normal(size=(6, 9))
    n = volume.shape[1]
    x_new, x_old = np.linspace(0, 1, n), np.linspace(0, 1, 9)
    dense = np.stack([np.interp(x_new, x_old, row) for row in short])
    torch.testing.assert_close(_run(kind, volume, short), _run(kind, volume, dense), atol=ATOL, rtol=0)


def test_cpu_oversampling_keeps_shape_and_identity(volume):
    out = _run("cpu", volume, np.zeros((6, volume.shape[1])), oversampling_pct=0.3)
    assert out.shape == volume.shape
    torch.testing.assert_close(out, volume.double(), atol=ATOL, rtol=0)


def test_cpu_center_zero_makes_the_central_position_the_reference(volume):
    n = volume.shape[1]
    params = np.zeros((6, n))
    params[0, :] = 1.0
    params[0, n // 2:] = 3.0
    transform = MotionFromTimeCourse(euler_motion_params=params.copy(), kspace_order=(0, 2, 1),
                                     displacement_shift_strategy="center_zero")
    transform(_subject(volume))
    assert np.all(transform.euler_motion_params[:, n // 2] == 0)
    np.testing.assert_allclose(transform.euler_motion_params + transform.to_subtract[:, None], params)


@pytest.mark.parametrize("strategy", ["1D_wFT", "1D_wFT2"])
def test_cpu_weighted_demean_strategies(strategy, volume):
    n = volume.shape[1]
    params = np.zeros((6, n))
    params[0, n // 2:] = 2.0
    transform = MotionFromTimeCourse(euler_motion_params=params.copy(), kspace_order=(0, 2, 1),
                                     displacement_shift_strategy=strategy)
    out = transform(_subject(volume))["img"].data
    assert torch.isfinite(out).all()
    np.testing.assert_allclose(transform.euler_motion_params + transform.to_subtract[:, None], params)
    assert 0 < transform.to_subtract[0] < 2.0


# ----------------------------------------------------------------------------------------------
# random transforms
# ----------------------------------------------------------------------------------------------
def test_random_motion_gpu_end_to_end(volume):
    torch.manual_seed(0)
    transform = RandomMotionGPU(nT=50)
    out = transform(_subject(volume))["img"].data
    assert out.shape == volume[None].shape
    assert torch.isfinite(out).all()
    assert tuple(transform.euler_motion_params.shape) == (6, 50)
    assert (out[0] - volume).abs().max() > 0.01


def test_random_motion_cpu_end_to_end(volume):
    torch.manual_seed(0)
    out = RandomMotionFromTimeCourse(nT=50)(_subject(volume))["img"].data
    assert out.shape == volume[None].shape
    assert torch.isfinite(out).all()
    assert (out[0] - volume).abs().max() > 0.01


def test_random_motion_is_reproducible_with_the_torch_seed(volume):
    def run(seed):
        torch.manual_seed(seed)
        return RandomMotionGPU(nT=40)(_subject(volume))["img"].data

    torch.testing.assert_close(run(3), run(3))
    assert (run(3) - run(4)).abs().max() > 1e-3


@pytest.mark.parametrize("choice", [[1], [0, 2]])
def test_random_kspace_order_ends_with_an_allowed_phase_axis(choice):
    for seed in range(10):
        torch.manual_seed(seed)
        order = RandomMotionGPU(phase_encoding_choice=choice).kspace_order
        assert sorted(order) == [0, 1, 2]
        assert order[-1] in choice


def test_random_motion_gpu_with_given_params_is_deterministic(volume):
    params = np.random.default_rng(0).normal(size=(6, 40))
    transform = RandomMotionGPU(euler_motion_params=params, phase_encoding_choice=[1])
    assert transform.simulate_displacement is False
    out = transform(_subject(volume))["img"].data[0].double()
    expected = _run("gpu", volume, params, tuple(transform.kspace_order))
    torch.testing.assert_close(out, expected)


def test_random_motion_cpu_accepts_given_params_as_array(volume):
    params = np.random.default_rng(0).normal(size=(6, 40))
    transform = RandomMotionFromTimeCourse(euler_motion_params=params, phase_encoding_choice=[1],
                                           oversampling_pct=0)
    out = transform(_subject(volume))["img"].data[0].double()
    expected = _run("cpu", volume, params, tuple(transform.kspace_order))
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("cls", [RandomMotionGPU, RandomMotionFromTimeCourse])
def test_global_amplitude_is_rescaled_to_the_requested_value(cls):
    for seed in range(5):
        torch.manual_seed(seed)
        transform = cls(nT=60, maxGlobalDisp=(5, 5), maxGlobalRot=(3, 3))
        transform._simulate_random_trajectory()
        params = np.asarray(transform.euler_motion_params)
        assert pdist(params[:3].T).max() == pytest.approx(5.0, rel=1e-6)
        assert pdist(params[3:].T).max() == pytest.approx(3.0, rel=1e-6)


@pytest.mark.parametrize("cls", [RandomMotionGPU, RandomMotionFromTimeCourse])
def test_no_global_rescaling_when_not_requested(cls):
    torch.manual_seed(0)
    transform = cls(nT=60, maxGlobalDisp=None, maxGlobalRot=None)
    transform._simulate_random_trajectory()
    params = np.asarray(transform.euler_motion_params)
    assert params.shape == (6, 60)
    assert np.isfinite(params).all()
    assert np.abs(params).max() > 0


@pytest.mark.parametrize("cls", [RandomMotionGPU, RandomMotionFromTimeCourse])
def test_disabled_noise_types_are_really_disabled(cls):
    transform = cls()
    # third value of the tuples is the probability of using that kind of noise
    only_perlin = transform.random_params(noiseBasePars=(5, 15, 1.0), swallowFrequency=(1, 5, 0.0),
                                          suddenFrequency=(1, 5, 0.0))
    assert only_perlin[2] > 0 and only_perlin[5] == 0 and only_perlin[6] == 0
    no_perlin = transform.random_params(noiseBasePars=(5, 15, 0.0), swallowFrequency=(1, 5, 1.0),
                                        suddenFrequency=(1, 5, 1.0))
    assert no_perlin[2] == 0 and no_perlin[5] >= 1 and no_perlin[6] >= 1


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def test_read_euler_motion_params(tmp_path):
    read = cpu_module.read_euler_motion_params
    good = np.arange(24, dtype=float).reshape(6, 4)
    np.testing.assert_array_equal(read(good), good)
    np.testing.assert_array_equal(read(good.tolist()), good)

    csv = tmp_path / "motion.csv"
    np.savetxt(csv, good, delimiter=",")
    np.testing.assert_allclose(read(str(csv)), good)

    # a trailing comma at the end of each csv line gives a NaN last column
    with_nan = np.hstack([good, np.full((6, 1), np.nan)])
    np.testing.assert_array_equal(read(with_nan), good)

    with pytest.warns(UserWarning, match="Expected 6"):
        assert read(np.zeros((5, 4))) is None
    with pytest.warns(UserWarning, match="shape \\(6, N\\)"):
        assert read(np.zeros(6)) is None


@pytest.mark.parametrize("rot_order", ["xyz", "yxz"])
def test_euler_matrix_is_a_rigid_transform(rot_order):
    params = torch.tensor([1.0, -2.0, 3.0, 10.0, -20.0, 30.0], dtype=torch.float64)
    cpu = cpu_module.get_matrix_from_euler_and_trans(params.numpy(), rot_order=rot_order)
    gpu = gpu_module.get_matrix_from_euler_and_trans(params, rot_order=rot_order).numpy()
    np.testing.assert_allclose(cpu, gpu, atol=1e-6)
    np.testing.assert_allclose(cpu[:3, 3], [1, -2, 3])
    np.testing.assert_allclose(cpu[3], [0, 0, 0, 1])
    rotation = cpu[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_euler_matrix_identity_for_no_motion():
    np.testing.assert_allclose(cpu_module.get_matrix_from_euler_and_trans(np.zeros(6)), np.eye(4), atol=1e-12)


def test_unknown_rotation_order_raises():
    with pytest.raises(NotImplementedError):
        cpu_module.get_matrix_from_euler_and_trans(np.zeros(6), rot_order="zzz")
    with pytest.raises(NotImplementedError):
        gpu_module.get_matrix_from_euler_and_trans(torch.zeros(6, dtype=torch.float64), rot_order="zzz")


def test_torch_linear_interpolation_matches_numpy():
    xp = torch.linspace(0, 1, 7)
    fp = torch.rand(7)
    x = torch.linspace(-0.2, 1.2, 25)   # includes points outside the range
    expected = np.interp(x.numpy(), xp.numpy(), fp.numpy())
    np.testing.assert_allclose(gpu_module.torch_1d_interp(x, xp, fp).numpy(), expected, atol=1e-6)


def test_interpolate_torch_resamples_each_parameter():
    params = torch.rand(6, 11, dtype=torch.float64).float()
    out = gpu_module._interpolate_torch(params, len_output=33)
    assert out.shape == (6, 33)
    torch.testing.assert_close(out[:, 0], params[:, 0])
    torch.testing.assert_close(out[:, -1], params[:, -1])


def test_periodic_motion_puts_energy_outside_the_object_in_proportion_to_its_amplitude():
    """The ghosts of a periodic motion (here 4 cycles along the phase axis) are signal outside the object."""
    volume = torch.zeros(24, 96, 24)
    volume[6:18, 40:56, 6:18] = 1.0
    outside = torch.ones(volume.shape, dtype=torch.bool)
    outside[:, 24:72, :] = False

    def leaked(amplitude):
        params = _constant(1, 0.0, 96)
        params[1] = amplitude * np.sin(2 * np.pi * 4 * np.linspace(0, 1, 96))
        return float(_run("gpu", volume, params, order=(0, 2, 1))[outside].pow(2).sum())

    still, small, large = leaked(0.0), leaked(0.5), leaked(2.0)
    assert still < 1e-3 * small
    assert small < large
