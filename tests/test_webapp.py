import base64

import numpy as np
import pytest

pytest.importorskip("flask")

from webapp import app as webapp


@pytest.fixture(scope="module")
def client():
    return webapp.app.test_client()


def _decode(b64, shape):
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(shape)


def test_config_describes_every_form_field(client):
    config = client.get("/api/config").get_json()["config"]
    keys = [f["key"] for group in config for f in group["fields"]]
    assert {"seed", "nT", "maxDisp", "maxGlobalDisp", "phase_axis", "resolution"} <= set(keys)
    assert len(keys) == len(set(keys))


def test_defaults_are_valid():
    clean = webapp.validate({})
    assert clean["nT"] == 200 and clean["maxDisp"] == [2, 5]
    simulation = webapp.validate({}, scope="simulation")
    assert {k: simulation[k] for k in ("seed", "phase_axis", "resolution", "oversampling_pct", "coregistration")} \
        == {"seed": 0, "phase_axis": 1, "resolution": 2.0, "oversampling_pct": 30, "coregistration": "none"}
    assert "nT" not in simulation and "maxDisp" not in simulation       # trajectory only
    assert clean["seed"] == 0 and clean["ghost_enabled"] is False        # the seed and ghosting are in both
    assert not any(v for k, v in simulation.items() if k.endswith("_enabled"))    # no artefact by default


def test_trajectory_shape_and_stats(client):
    reply = client.post("/api/trajectory", json={"nT": 80, "seed": 1}).get_json()
    params = np.array(reply["params"])
    assert params.shape == (6, 80)
    assert np.isfinite(params).all()
    # default global amplitudes are rescaled to [3, 6] mm / degrees
    assert 3 - 1e-3 <= reply["stats"]["max_translation"] <= 6 + 1e-3
    assert 3 - 1e-3 <= reply["stats"]["max_rotation"] <= 6 + 1e-3


def test_trajectory_is_reproducible_with_the_seed(client):
    get = lambda seed: np.array(client.post("/api/trajectory", json={"seed": seed}).get_json()["params"])
    np.testing.assert_array_equal(get(5), get(5))
    assert np.abs(get(5) - get(6)).max() > 0.1


def test_trajectory_without_global_rescaling(client):
    reply = client.post("/api/trajectory", json={"maxGlobalDisp": None, "maxGlobalRot": None}).get_json()
    assert np.isfinite(np.array(reply["params"])).all()


def test_noise_types_can_be_switched_off(client):
    reply = client.post("/api/trajectory", json={
        "noiseProba": 0, "swallowProba": 1, "suddenProba": 0, "swallowFrequency": [1, 2],
        "maxGlobalDisp": None, "maxGlobalRot": None}).get_json()
    params = np.array(reply["params"])
    # only swallowing: translation along z and pitch, nothing else
    assert np.abs(params[[0, 1, 4, 5]]).max() == 0
    assert np.abs(params[[2, 3]]).max() > 0


@pytest.mark.parametrize("payload, message", [
    ({"nT": 3}, "Points de la trajectoire"),
    ({"maxDisp": [5, 2]}, "min <= max"),
    ({"swallowFrequency": [2, 2]}, "min < max"),
    ({"noiseProba": 1.5}, "Probabilité"),
    ({"seed": "abc"}, "Graine"),
    ({"maxRot": [1]}, "intervalle"),
])
def test_invalid_trajectory_parameters_are_rejected(client, payload, message):
    response = client.post("/api/trajectory", json=payload)
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_simulation_without_motion_returns_the_original(client):
    reply = client.post("/api/simulate", json={
        "trajectory": np.zeros((6, 50)).tolist(), "resolution": 3.0, "phase_axis": 1}).get_json()
    shape = reply["shape"]
    assert shape == [61, 73, 61]
    original, corrupted = _decode(reply["original"], shape), _decode(reply["corrupted"], shape)
    assert np.abs(original.astype(int) - corrupted.astype(int)).max() <= 1
    assert reply["metrics"]["correlation"] > 0.9999
    assert reply["metrics"]["nrmse"] < 1e-3


def test_simulation_with_motion_degrades_the_image(client):
    trajectory = client.post("/api/trajectory", json={"nT": 60, "seed": 2}).get_json()["params"]
    reply = client.post("/api/simulate", json={
        "trajectory": trajectory, "resolution": 3.0, "phase_axis": 0}).get_json()
    assert reply["metrics"]["correlation"] < 0.999
    assert reply["metrics"]["nrmse"] > 1e-3
    assert reply["metrics"]["seconds"] > 0


@pytest.mark.parametrize("payload", [
    {"trajectory": [[0, 0]] * 5},
    {"trajectory": [[0.0]] * 6},
    {"trajectory": [[0.0, float("nan")]] * 6},
    {"trajectory": np.zeros((6, 20)).tolist(), "resolution": 7},
    {"trajectory": np.zeros((6, 20)).tolist(), "phase_axis": 3},
    {},
])
def test_invalid_simulation_requests_are_rejected(client, payload):
    assert client.post("/api/simulate", json=payload).status_code == 400


# ----------------------------------------------------------------------------------------------
# coregistration with the original image after the simulation
# ----------------------------------------------------------------------------------------------
def _simulate(client, **options):
    trajectory = client.post("/api/trajectory", json={"seed": 0}).get_json()["params"]   # clear motion
    return client.post("/api/simulate", json={"trajectory": trajectory, "resolution": 3.0, "phase_axis": 1,
                                              **options}).get_json()


def test_no_coregistration_by_default(client):
    reply = _simulate(client)
    assert reply["coregistered"] is None and reply["coregistration"] is None
    assert reply["coregistration_error"] is None


@pytest.mark.parametrize("method, module", [("elastix", "itk"), ("torchreg", "torchreg")])
def test_coregistration_brings_the_image_back_to_the_original(client, method, module):
    pytest.importorskip(module)
    reply = _simulate(client, coregistration=method)
    shape = reply["shape"]
    assert reply["coregistration_error"] is None
    assert _decode(reply["coregistered"], shape).shape == tuple(shape)
    raw, registered = reply["metrics"], reply["coregistration"]
    assert registered["method"] == webapp.COREGISTRATION_NAMES[method]
    assert registered["correlation"] > raw["correlation"] + 0.05
    assert registered["nrmse"] < raw["nrmse"]
    assert registered["seconds"] > 0
    # the raw image is still returned untouched
    assert not np.array_equal(_decode(reply["corrupted"], shape), _decode(reply["coregistered"], shape))


def test_coregistration_failure_keeps_the_simulation(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(webapp, "run_coregistration", broken)
    reply = _simulate(client, coregistration="elastix")
    assert reply["coregistered"] is None and reply["coregistration"] is None
    assert "boom" in reply["coregistration_error"]
    assert reply["metrics"]["correlation"] < 0.999   # the simulation itself is there


def test_unknown_coregistration_method_is_rejected(client):
    response = client.post("/api/simulate", json={"trajectory": np.zeros((6, 20)).tolist(),
                                                  "coregistration": "ants"})
    assert response.status_code == 400


# ----------------------------------------------------------------------------------------------
# oversampling (zero padding) before the simulation
# ----------------------------------------------------------------------------------------------
def test_padding_and_crop_are_inverse():
    volume = np.random.default_rng(0).random((10, 11, 12)).astype(np.float32)
    pads = webapp.padding_for(volume.shape, 30)
    assert pads == (2, 2, 2)                       # ceil(size * 0.3 / 2)
    padded = np.pad(volume, [(p, p) for p in pads])
    assert padded.shape == (14, 15, 16)
    np.testing.assert_array_equal(webapp.crop(padded, pads), volume)
    assert webapp.padding_for(volume.shape, 0) == (0, 0, 0)
    np.testing.assert_array_equal(webapp.crop(volume, (0, 0, 0)), volume)


@pytest.mark.parametrize("pct", [0, 30, 100])
def test_simulation_without_motion_is_identity_whatever_the_padding(client, pct):
    reply = client.post("/api/simulate", json={"trajectory": np.zeros((6, 40)).tolist(), "resolution": 3.0,
                                               "oversampling_pct": pct}).get_json()
    assert reply["shape"] == [61, 73, 61]          # always cropped back to the original size
    original, corrupted = _decode(reply["original"], reply["shape"]), _decode(reply["corrupted"], reply["shape"])
    assert np.abs(original.astype(int) - corrupted.astype(int)).max() <= 1


def _edge_error(original, other, thickness=4):
    mask = np.zeros(original.shape, bool)
    for axis in range(3):
        for side in (slice(0, thickness), slice(original.shape[axis] - thickness, None)):
            index = [slice(None)] * 3
            index[axis] = side
            mask[tuple(index)] = True
    return float(np.abs(original - other)[mask].mean())


@pytest.mark.parametrize("method", ["none", "elastix"])
def test_padding_reduces_the_errors_at_the_edges_of_the_volume(client, method):
    pytest.importorskip("itk")
    trajectory = np.array(client.post("/api/trajectory", json={"seed": 0}).get_json()["params"])
    errors = {}
    for pct in (0, 30):
        original, moved, pads, _ = webapp.run_motion(3.0, 1, trajectory, pct)
        image = moved
        if method == "elastix":
            image, _ = webapp.run_coregistration(method, webapp.colin27(3.0), original, moved)
        errors[pct] = _edge_error(webapp.crop(original, pads), webapp.crop(image, pads))
    assert errors[30] < errors[0]


def test_padding_works_with_every_coregistration_method(client):
    pytest.importorskip("itk")
    pytest.importorskip("torchreg")
    for method in ("elastix", "torchreg"):
        reply = _simulate(client, coregistration=method, oversampling_pct=30)
        assert reply["coregistration_error"] is None
        assert _decode(reply["coregistered"], reply["shape"]).shape == tuple(reply["shape"])
        assert reply["coregistration"]["correlation"] > reply["metrics"]["correlation"] + 0.05


@pytest.mark.parametrize("pct", [-5, 150, "abc"])
def test_invalid_padding_is_rejected(client, pct):
    response = client.post("/api/simulate", json={"trajectory": np.zeros((6, 20)).tolist(), "oversampling_pct": pct})
    assert response.status_code == 400


# ----------------------------------------------------------------------------------------------
# configuration of the simple / advanced form
# ----------------------------------------------------------------------------------------------
def _fields():
    return {f["key"]: f for g in webapp.CONFIG for f in g["fields"]}


def test_config_returns_the_field_of_view(client):
    fov = client.get("/api/config").get_json()["fov_mm"]
    assert fov == pytest.approx([181, 217, 181])


def test_every_default_is_valid_in_both_scopes():
    for scope in ("trajectory", "simulation"):
        webapp.validate({}, scope=scope)


def test_toggles_are_boolean_fields_of_their_group_and_off_by_default():
    for group in webapp.CONFIG:
        if "toggle" not in group:
            continue
        keys = [f["key"] for f in group["fields"]]
        assert group["toggle"] in keys
        toggle = _fields()[group["toggle"]]
        assert toggle["kind"] == "boolean" and toggle["default"] is False
        assert group.get("summary"), group["group"]


def test_summaries_only_use_existing_fields():
    import re
    for group in webapp.CONFIG:
        for key in re.findall(r"{(\w+)}", group.get("summary", "")):
            assert key in _fields(), (group["group"], key)


def test_presets_target_existing_fields_with_valid_values():
    fields = _fields()
    seen = 0
    for field in fields.values():
        if "presets" not in field:
            continue
        values = [o[0] for o in field["options"]]
        assert "custom" in values
        for name, preset in field["presets"].items():
            seen += 1
            assert name in values
            assert field["default"] in field["presets"] or field["default"] == "custom"
            for key, value in preset.items():
                assert key in fields and "presets" not in fields[key]
                payload = {key: value}
                scope = webapp.scopes_of(fields[key])[0]
                webapp.validate(payload, scope=scope)
    assert seen >= 8


def test_default_motion_preset_is_the_default_motion():
    fields = _fields()
    moderate = fields["motion_level"]["presets"]["moderate"]
    assert fields["motion_level"]["default"] == "moderate"
    for key, value in moderate.items():
        assert fields[key]["default"] == value, key


def test_advanced_flags_hide_the_detailed_motion_and_keep_the_main_controls():
    advanced_groups = {g["group"] for g in webapp.CONFIG if g.get("advanced")}
    assert {"Dérive lente (bruit de Perlin)", "Déglutitions", "Mouvements brusques"} <= advanced_groups
    assert not _fields()["motion_level"].get("advanced")
    assert not _fields()["ghost_amplitude"].get("advanced") and not _fields()["noise_snr"].get("advanced")


@pytest.mark.parametrize("payload, message", [
    ({"ghost_enabled": "yes"}, "vrai ou faux"),
    ({"ghost_cycles": 31}, "Cycles"),
    ({"ghost_amplitude": 0}, "Amplitude"),
    ({"motion_level": "extreme"}, "non autorisée"),
])
def test_invalid_trajectory_fields_are_rejected(client, payload, message):
    response = client.post("/api/trajectory", json=payload)
    assert response.status_code == 400
    assert message in response.get_json()["error"]


@pytest.mark.parametrize("payload", [
    {"noise_snr": 99}, {"noise_enabled": 1}, {"spikes_radius": [50, 20]}, {"spikes_count": 0},
    {"gibbs_fraction": 5}, {"alias_factor": 1}, {"pf_axis": "x"}, {"dropout_unit": "cells"},
    {"clip_pct": 50}, {"blur_filter": "box"},
])
def test_invalid_artifact_fields_are_rejected(client, payload):
    response = client.post("/api/simulate", json={"trajectory": np.zeros((6, 20)).tolist(), **payload})
    assert response.status_code == 400


# ----------------------------------------------------------------------------------------------
# motion level and ghosting in the trajectory
# ----------------------------------------------------------------------------------------------
def _trajectory(client, **payload):
    reply = client.post("/api/trajectory", json=payload).get_json()
    return np.array(reply["params"]), reply["stats"]


def test_no_motion_level_gives_a_flat_trajectory(client):
    params, stats = _trajectory(client, motion_level="none", nT=60)
    assert params.shape == (6, 60) and not params.any()
    assert stats["max_translation"] == 0 and stats["max_rotation"] == 0


def test_motion_level_only_matters_for_none(client):
    """The detailed fields are the truth: the other levels are just presets of these fields."""
    base, _ = _trajectory(client, seed=3)
    for level in ("custom", "mild", "strong"):
        np.testing.assert_array_equal(_trajectory(client, seed=3, motion_level=level)[0], base)


def test_ghosting_alone_is_a_sinusoid_on_the_chosen_axis(client):
    params, stats = _trajectory(client, motion_level="none", nT=200, ghost_enabled=True, ghost_amplitude=2.0,
                                ghost_cycles=5, ghost_axis=1)
    assert params.shape == (6, 200)
    assert not params[[0, 2, 3, 4, 5]].any()
    assert np.abs(params[1]).max() == pytest.approx(2.0, abs=0.02)
    crossings = np.sum(np.diff(np.sign(params[1][1:-1])) != 0)
    assert crossings == 2 * 5 - 1 or crossings == 2 * 5 - 2     # 5 periods, the ends are zeros
    assert stats["max_translation"] == pytest.approx(4.0, abs=0.05)


def test_ghosting_is_added_to_the_random_motion(client):
    random_motion, _ = _trajectory(client, seed=4, nT=100)
    both, _ = _trajectory(client, seed=4, nT=100, ghost_enabled=True, ghost_amplitude=1.0, ghost_cycles=3, ghost_axis=2)
    expected = random_motion.copy()
    expected[2] += np.sin(2 * np.pi * 3 * np.linspace(0, 1, 100))
    np.testing.assert_allclose(both, expected, atol=1e-3)


@pytest.mark.parametrize("axis", range(6))
def test_ghosting_axis_can_be_any_translation_or_rotation(client, axis):
    params, stats = _trajectory(client, motion_level="none", nT=100, ghost_enabled=True, ghost_amplitude=2.0,
                                ghost_cycles=3, ghost_axis=axis)
    others = [i for i in range(6) if i != axis]
    assert not params[others].any()
    assert np.abs(params[axis]).max() == pytest.approx(2.0, abs=0.05)
    assert stats["max_translation" if axis < 3 else "max_rotation"] == pytest.approx(4.0, abs=0.1)


def test_ghosting_axis_is_a_main_control_listing_translations_and_rotations():
    field = _fields()["ghost_axis"]
    assert not field.get("advanced")
    assert [o[0] for o in field["options"]] == list(range(6))


def test_ghosting_with_many_cycles_resamples_the_trajectory(client):
    params, _ = _trajectory(client, motion_level="none", nT=10, ghost_enabled=True, ghost_cycles=30)
    assert params.shape[1] == webapp.GHOST_POINTS_PER_CYCLE * 30
    assert np.abs(params[1]).max() > 1.4


def test_ghosting_off_leaves_the_trajectory_alone(client):
    off, _ = _trajectory(client, seed=2, ghost_enabled=False, ghost_amplitude=5)
    np.testing.assert_array_equal(off, _trajectory(client, seed=2)[0])


# ----------------------------------------------------------------------------------------------
# k-space artefacts built from the form
# ----------------------------------------------------------------------------------------------
def _artifacts(**payload):
    return webapp.build_artifacts(webapp.validate(payload, scope="simulation"))


def test_no_artifact_by_default():
    assert _artifacts() == []


@pytest.mark.parametrize("flag, cls", [
    ("noise_enabled", "RicianNoise"), ("spikes_enabled", "Spikes"), ("gibbs_enabled", "Truncation"),
    ("blur_enabled", "LowPassFilter"), ("t2star_enabled", "T2StarDecay"), ("dropout_enabled", "LineDropout"),
    ("pf_enabled", "PartialFourier"), ("alias_enabled", "Undersampling"), ("zipper_enabled", "Zipper"),
    ("dc_enabled", "DCOffset"), ("clip_enabled", "Clipping"),
])
def test_each_switch_builds_its_artifact(flag, cls):
    built = _artifacts(**{flag: True})
    assert [type(a).__name__ for a in built] == [cls]


def test_form_values_are_converted_to_the_library_units():
    (gibbs,) = _artifacts(gibbs_enabled=True, gibbs_fraction=70, gibbs_axes="phase")
    assert gibbs.fraction == pytest.approx(0.7) and gibbs.axes == ("phase",)
    (blur,) = _artifacts(blur_enabled=True, blur_cutoff=25, blur_filter="hamming")
    assert blur.cutoff == pytest.approx(0.25) and blur.kind == "hamming"
    (spikes,) = _artifacts(spikes_enabled=True, spikes_count=4, spikes_amplitude=20, spikes_radius=[10, 30])
    assert spikes.count == 4 and spikes.amplitude == pytest.approx(0.2) and spikes.radius == pytest.approx((0.1, 0.3))
    (noise,) = _artifacts(noise_enabled=True, noise_snr=18)
    assert noise.snr_db == 18
    (dropout,) = _artifacts(dropout_enabled=True, dropout_pct=10, dropout_keep_center=20, dropout_gain=0.5,
                            dropout_unit="lines", dropout_mode="grouped")
    assert (dropout.fraction, dropout.keep_center, dropout.gain) == (pytest.approx(0.1), pytest.approx(0.2), 0.5)
    assert (dropout.unit, dropout.mode) == ("lines", "grouped")
    (clip,) = _artifacts(clip_enabled=True, clip_pct=2)
    assert clip.saturated_pct == 2
    (alias,) = _artifacts(alias_enabled=True, alias_factor=3, alias_axis="readout")
    assert alias.factor == 3 and alias.axis == "readout"
    (zipper,) = _artifacts(zipper_enabled=True, zipper_amplitude=50, zipper_position=20)
    assert zipper.amplitude == pytest.approx(0.5) and zipper.position == pytest.approx(0.2)
    (dc,) = _artifacts(dc_enabled=True, dc_amplitude=250)
    assert dc.amplitude == pytest.approx(2.5)
    (pf,) = _artifacts(pf_enabled=True, pf_fraction=60, pf_axis="phase2")
    assert pf.fraction == pytest.approx(0.6) and pf.axis == "phase2"
    (t2,) = _artifacts(t2star_enabled=True, t2star_ratio=2)
    assert t2.ratio == 2


def test_random_artifacts_have_distinct_seeds_derived_from_the_form_seed():
    built = {type(a).__name__: a for a in _artifacts(seed=10, noise_enabled=True, spikes_enabled=True,
                                                      dropout_enabled=True)}
    seeds = {built["RicianNoise"].seed, built["Spikes"].seed, built["LineDropout"].seed}
    assert len(seeds) == 3 and all(s >= 10 for s in seeds)


def test_the_gibbs_axes_choice_all_means_the_three_axes():
    (gibbs,) = _artifacts(gibbs_enabled=True)
    assert set(gibbs.axes) == {"readout", "phase2", "phase"}


def _simulate_flat(client, **options):
    """Simulation with no motion at all: only the artefacts change the image."""
    return client.post("/api/simulate", json={"trajectory": np.zeros((6, 40)).tolist(), "resolution": 3.0,
                                              "oversampling_pct": 0, **options}).get_json()


@pytest.mark.parametrize("options", [
    {"noise_enabled": True, "noise_snr": 15},
    {"spikes_enabled": True, "spikes_amplitude": 20},
    {"gibbs_enabled": True, "gibbs_fraction": 40},
    {"blur_enabled": True, "blur_cutoff": 15},
    {"t2star_enabled": True, "t2star_ratio": 0.2},
    {"dropout_enabled": True, "dropout_pct": 20},
    {"pf_enabled": True, "pf_fraction": 55},
    {"alias_enabled": True, "alias_factor": 2},
    {"zipper_enabled": True, "zipper_amplitude": 100},
    {"dc_enabled": True, "dc_amplitude": 1000},
    {"clip_enabled": True, "clip_pct": 5},
])
def test_each_artifact_degrades_the_image_through_the_api(client, options):
    reply = _simulate_flat(client, **options)
    assert reply["shape"] == [61, 73, 61]
    original, corrupted = _decode(reply["original"], reply["shape"]), _decode(reply["corrupted"], reply["shape"])
    assert np.abs(original.astype(int) - corrupted.astype(int)).max() > 2
    assert reply["metrics"]["correlation"] < 0.9999


def test_artifacts_are_not_applied_unless_switched_on(client):
    reply = _simulate_flat(client, noise_snr=5, spikes_amplitude=90, gibbs_fraction=20)
    assert reply["metrics"]["correlation"] > 0.9999


def test_noise_through_the_api_is_seeded_and_scales_with_the_snr(client):
    a = _simulate_flat(client, noise_enabled=True, noise_snr=20, seed=1)
    b = _simulate_flat(client, noise_enabled=True, noise_snr=20, seed=1)
    c = _simulate_flat(client, noise_enabled=True, noise_snr=20, seed=2)
    assert a["corrupted"] == b["corrupted"] and a["corrupted"] != c["corrupted"]
    quiet = _simulate_flat(client, noise_enabled=True, noise_snr=40, seed=1)
    assert quiet["metrics"]["nrmse"] < a["metrics"]["nrmse"] < _simulate_flat(
        client, noise_enabled=True, noise_snr=5, seed=1)["metrics"]["nrmse"]


def test_artifacts_combine_with_motion_padding_and_coregistration(client):
    pytest.importorskip("itk")
    reply = _simulate(client, coregistration="elastix", oversampling_pct=30, noise_enabled=True, noise_snr=25,
                      gibbs_enabled=True, gibbs_fraction=70, spikes_enabled=True)
    assert reply["coregistration_error"] is None
    assert _decode(reply["corrupted"], reply["shape"]).shape == tuple(reply["shape"])
    assert reply["coregistration"]["correlation"] > reply["metrics"]["correlation"]


# ----------------------------------------------------------------------------------------------
# the trajectory is in mm, whatever the resolution
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("resolution, axis", [(3.0, 0), (3.0, 1), (2.0, 2)])
def test_a_translation_in_mm_moves_the_object_by_that_distance(resolution, axis):
    trajectory = np.zeros((6, 40))
    trajectory[axis] = 2 * resolution                 # 2 voxels, expressed in mm
    original, moved, _, _ = webapp.run_motion(resolution, 1 if axis != 1 else 0, trajectory, oversampling_pct=100)
    np.testing.assert_allclose(moved, np.roll(original, 2, axis=axis), atol=2e-3)


def test_the_same_trajectory_gives_the_same_displacement_in_mm_at_any_resolution():
    """6 mm along x: 6 voxels at 1 mm would be 2 voxels at 3 mm, i.e. 3 times less than the old behaviour."""
    trajectory = np.zeros((6, 40))
    trajectory[0] = 6.0
    original, moved, _, _ = webapp.run_motion(3.0, 1, trajectory, oversampling_pct=100)
    assert np.abs(moved - np.roll(original, 2, axis=0)).max() < 2e-3
    assert np.abs(moved - np.roll(original, 6, axis=0)).max() > 0.05


def test_rotations_are_not_rescaled_by_the_resolution():
    trajectory = np.zeros((6, 40))
    trajectory[5] = 5.0
    a = webapp.run_motion(3.0, 1, trajectory, 0)[1]
    b = webapp.run_motion(3.0, 1, trajectory.copy(), 0)[1]
    np.testing.assert_array_equal(a, b)
    assert np.abs(a - webapp.run_motion(3.0, 1, np.zeros((6, 40)), 0)[1]).max() > 0.05


def test_run_motion_does_not_modify_its_trajectory():
    trajectory = np.full((6, 20), 3.0)
    before = trajectory.copy()
    webapp.run_motion(3.0, 1, trajectory, 0)
    np.testing.assert_array_equal(trajectory, before)
