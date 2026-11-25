import json

import numpy as np
import pytest

from whydrift.common import STORED_DT
from whydrift.physics.generate import (ENERGY_TOL, SPLITS, TEST_TIME, TRAIN_TIME, generate,
                                       load_split, relative_energy_error)
from whydrift.physics.systems import make_system


@pytest.fixture(scope="module", params=["s1", "s2"])
def small_data(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(request.param)
    meta = generate(request.param, out, small=True)
    return request.param, out, meta


def test_files_and_shapes(small_data):
    name, out, meta = small_data
    assert json.loads((out / f"{name}_meta.json").read_text()) == meta
    for split, spec in SPLITS[name].items():
        d = load_split(name, split, out)
        n, t_len = meta["splits"][split]["n"], int(round(spec["time"] / STORED_DT)) + 1
        assert d["states"].shape == (n, t_len, 2)
        assert d["energy"].shape == (n, t_len)
        assert d["x0"].shape == (n, 2)
        assert d["states"].dtype == np.float64
        np.testing.assert_array_equal(d["x0"], d["states"][:, 0])


def test_lengths_cover_largest_dt(small_data):
    name, out, _ = small_data
    assert load_split(name, "train", out)["t"][-1] == pytest.approx(TRAIN_TIME)  # 100 steps at 0.4
    assert load_split(name, "test", out)["t"][-1] == pytest.approx(TEST_TIME)  # 1000 steps at 0.4


def test_energy_conserved(small_data):
    name, out, _ = small_data
    system = make_system(name)
    for split in SPLITS[name]:
        states = load_split(name, split, out)["states"]
        assert relative_energy_error(system, states).max() <= ENERGY_TOL


def test_splits_disjoint_and_in_range(small_data):
    name, out, meta = small_data
    x0s = {s: load_split(name, s, out)["x0"] for s in SPLITS[name]}
    all_x0 = np.concatenate(list(x0s.values()))
    assert len(np.unique(all_x0, axis=0)) == len(all_x0)
    system = make_system(name)
    for split, x0 in x0s.items():
        lo, hi = meta["splits"][split]["e_range"]
        e = np.asarray(system.energy(x0))
        assert np.all((e >= lo - 1e-9) & (e <= hi + 1e-9))


def test_regenerates_bit_identically(small_data, tmp_path):
    name, out, _ = small_data
    generate(name, tmp_path, small=True)
    for split in SPLITS[name]:
        a, b = load_split(name, split, out), load_split(name, split, tmp_path)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


def test_s1_matches_exact_solution(tmp_path_factory):
    """Plan 1.2: spring data matches q = q0 cos t + p0 sin t to 1e-8, over the full 400 time units."""
    out = tmp_path_factory.mktemp("s1_exact")
    generate("s1", out, small=True)
    for split in SPLITS["s1"]:
        d = load_split("s1", split, out)
        q0, p0, t = d["x0"][:, :1], d["x0"][:, 1:], d["t"]
        exact = np.stack([q0 * np.cos(t) + p0 * np.sin(t), -q0 * np.sin(t) + p0 * np.cos(t)], axis=-1)
        assert np.abs(d["states"] - exact).max() < 1e-8


def test_load_split_subsamples_to_dt(small_data):
    name, out, _ = small_data
    full = load_split(name, "train", out)
    for dt, k in [(0.05, 1), (0.1, 2), (0.2, 4), (0.4, 8)]:
        d = load_split(name, "train", out, dt=dt, n_steps=100)
        assert d["states"].shape[1] == 101
        np.testing.assert_allclose(np.diff(d["t"]), dt, atol=1e-12)
        np.testing.assert_array_equal(d["states"], full["states"][:, : 100 * k + 1 : k])
        np.testing.assert_array_equal(d["energy"], full["energy"][:, : 100 * k + 1 : k])


def test_load_split_refuses_bad_requests(small_data):
    name, out, _ = small_data
    with pytest.raises(ValueError, match="multiple"):
        load_split(name, "train", out, dt=0.07)
    with pytest.raises(ValueError, match="stops at"):
        load_split(name, "train", out, dt=0.4, n_steps=1000)  # train covers 40, not 400
    with pytest.raises(ValueError, match="needs dt"):
        load_split(name, "train", out, n_steps=100)


def test_audit_passes_on_generated_data(small_data):
    from whydrift.physics.checks import audit
    name, out, _ = small_data
    rows = audit(name, out)
    assert rows and all(r["ok"] for r in rows)


def test_s3_has_no_split_plan_yet(tmp_path):
    with pytest.raises(ValueError, match="OPEN"):
        generate("s3", tmp_path, small=True, system_kwargs=dict(alpha=0.25, beta=0.0))
