import numpy as np
import pytest
import torch
import torchio as tio
from scipy.ndimage import shift as nd_shift

itk = pytest.importorskip("itk")

from kspace_motion.coregister_itk import Coregister
from kspace_motion.utils import nib_to_itk, itk_to_nib

SHIFT = (3.0, -2.0, 2.0)


def _corr(a, b):
    # numpy on purpose: torch.corrcoef segfaulted with pip torch next to conda numpy
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _affine():
    return np.diag([1.0, 1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def moving_data(phantom):
    return torch.from_numpy(nd_shift(phantom.numpy(), SHIFT, order=1, mode="constant"))


def test_nib_itk_roundtrip(phantom_factory):
    # non-cubic on purpose, so that a swapped axis order is detected
    volume = phantom_factory(shape=(20, 28, 36))
    affine = np.array([[2, 0, 0, 5], [0, 1.5, 0, -3], [0, 0, 1, 7], [0, 0, 0, 1]], dtype=float)
    image = nib_to_itk(volume[None].numpy(), affine, force_3d=True)
    assert tuple(itk.size(image)) == tuple(volume.shape)
    data, affine_back = itk_to_nib(image)
    np.testing.assert_allclose(data[0], volume.numpy(), atol=1e-6)
    np.testing.assert_allclose(affine_back, affine, atol=1e-6)


def test_coregister_to_subject_key_improves_alignment(phantom, moving_data):
    subject = tio.Subject(
        ref=tio.ScalarImage(tensor=phantom[None], affine=_affine()),
        moving=tio.ScalarImage(tensor=moving_data[None], affine=_affine()),
    )
    before = _corr(subject["ref"].data, subject["moving"].data)
    out = Coregister(target="ref")(subject)
    after = _corr(out["ref"].data, out["moving"].data)
    assert out["moving"].data.shape == phantom[None].shape
    assert after > before
    assert after > 0.98


def test_coregister_target_image_instance(phantom, moving_data):
    ref = tio.ScalarImage(tensor=phantom[None], affine=_affine())
    subject = tio.Subject(moving=tio.ScalarImage(tensor=moving_data[None], affine=_affine()))
    before = _corr(ref.data, subject["moving"].data)
    out = Coregister(target=ref)(subject)
    assert _corr(ref.data, out["moving"].data) > max(before, 0.98)


def test_multichannel_every_channel_is_coregistered(phantom, moving_data):
    """Both channels must come out aligned (they used to be the same whole-image copy)."""
    other = moving_data * 0.5
    two_channels = torch.stack([moving_data, other])
    subject = tio.Subject(
        ref=tio.ScalarImage(tensor=phantom[None], affine=_affine()),
        moving=tio.ScalarImage(tensor=two_channels, affine=_affine()),
    )
    out = Coregister(target="ref")(subject)
    data = out["moving"].data
    assert data.shape == two_channels.shape
    assert _corr(data[0], phantom) > 0.98
    assert _corr(data[1], phantom) > 0.98
    # second channel keeps its own intensity scaling
    assert not torch.allclose(data[0], data[1])


def test_estimation_mapping_applies_transform_to_other_image(phantom, moving_data):
    subject = tio.Subject(
        ref=tio.ScalarImage(tensor=phantom[None], affine=_affine()),
        moving=tio.ScalarImage(tensor=moving_data[None], affine=_affine()),
        other=tio.ScalarImage(tensor=moving_data[None] * 2, affine=_affine()),
    )
    out = Coregister(target="ref", estimation_mapping="moving")(subject)
    assert _corr(out["other"].data, phantom) > 0.98


def test_unrecognized_target_warns_and_skips(phantom, moving_data):
    subject = tio.Subject(moving=tio.ScalarImage(tensor=moving_data[None], affine=_affine()))
    with pytest.warns(UserWarning, match="Unrecognized target"):
        out = Coregister(target={"a": 1})(subject)
    assert torch.equal(out["moving"].data, moving_data[None])
