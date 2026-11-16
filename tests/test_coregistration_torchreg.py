import numpy as np
import pytest
import torch
import torchio as tio
from scipy.ndimage import shift as nd_shift

pytest.importorskip("torchreg")

from kspace_motion.coregister_torchreg import Coregister

SHIFT = (3.0, -2.0, 2.0)
KWARGS = dict(scales=(4, 2), iterations=(300, 100))


def _corr(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def moving_data(phantom):
    return torch.from_numpy(nd_shift(phantom.numpy(), SHIFT, order=1, mode="constant"))


def _subject(phantom, moving, **extra):
    images = dict(ref=tio.ScalarImage(tensor=phantom[None]),
                  moving=tio.ScalarImage(tensor=moving[None]))
    images.update({k: tio.ScalarImage(tensor=v) for k, v in extra.items()})
    return tio.Subject(**images)


def test_coregister_to_subject_key_improves_alignment(phantom, moving_data):
    subject = _subject(phantom, moving_data)
    before = _corr(phantom, moving_data)
    out = Coregister(target="ref", registration_kwargs=KWARGS)(subject)
    after = _corr(out["ref"].data, out["moving"].data)
    assert out["moving"].data.shape == phantom[None].shape
    assert after > before
    assert after > 0.98
    assert torch.equal(out["ref"].data, phantom[None])  # target is left untouched


def test_estimated_transform_is_close_to_the_applied_shift(phantom, moving_data):
    transform = Coregister(target="ref", registration_kwargs=KWARGS)
    transform(_subject(phantom, moving_data))
    affine = transform.estimated_parameter_maps["moving"].get_affine()[0]  # (3, 4)
    # translations are in normalised coordinates in [-1, 1] -> convert to voxels
    voxels = affine[:, 3].abs() * (phantom.shape[0] - 1) / 2
    np.testing.assert_allclose(np.sort(voxels.numpy()), np.sort(np.abs(SHIFT)), atol=0.7)
    np.testing.assert_allclose(affine[:, :3].numpy(), np.eye(3), atol=0.05)


def test_target_image_instance(phantom, moving_data):
    ref = tio.ScalarImage(tensor=phantom[None])
    subject = tio.Subject(moving=tio.ScalarImage(tensor=moving_data[None]))
    out = Coregister(target=ref, registration_kwargs=KWARGS)(subject)
    assert _corr(ref.data, out["moving"].data) > 0.98


def test_target_tensor(phantom, moving_data):
    subject = tio.Subject(moving=tio.ScalarImage(tensor=moving_data[None]))
    out = Coregister(target=phantom, registration_kwargs=KWARGS)(subject)
    assert _corr(phantom, out["moving"].data) > 0.98


def test_multichannel_every_channel_uses_the_transform(phantom, moving_data):
    two = torch.stack([moving_data, moving_data * 0.5])
    subject = tio.Subject(ref=tio.ScalarImage(tensor=phantom[None]),
                          moving=tio.ScalarImage(tensor=two))
    out = Coregister(target="ref", registration_kwargs=KWARGS)(subject)
    data = out["moving"].data
    assert data.shape == two.shape
    assert _corr(data[0], phantom) > 0.98
    assert _corr(data[1], phantom) > 0.98
    assert not torch.allclose(data[0], data[1])


def test_estimation_mapping_applies_transform_to_other_image(phantom, moving_data):
    subject = _subject(phantom, moving_data, other=moving_data[None] * 2)
    out = Coregister(target="ref", estimation_mapping="moving", registration_kwargs=KWARGS)(subject)
    assert _corr(out["other"].data, phantom) > 0.98


def test_unrecognized_target_warns_and_skips(moving_data):
    subject = tio.Subject(moving=tio.ScalarImage(tensor=moving_data[None]))
    with pytest.warns(UserWarning, match="Unrecognized target"):
        out = Coregister(target={"a": 1})(subject)
    assert torch.equal(out["moving"].data, moving_data[None])


def test_exclude_list_is_not_shared_with_caller(phantom):
    exclude = ["foo"]
    Coregister(target="ref", exclude=exclude)
    assert exclude == ["foo"]
