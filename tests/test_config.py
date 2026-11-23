from pathlib import Path

import pytest

from whydrift.train.config import ConfigError, load_config, validate

EXAMPLE = Path(__file__).parents[1] / "whydrift" / "train" / "configs" / "example_s1.yaml"


@pytest.fixture
def cfg():
    return load_config(EXAMPLE)


def test_example_loads(cfg):
    assert cfg["model"] == "M3"


def test_example_is_not_complete_yet():
    with pytest.raises(ConfigError, match="still null"):
        load_config(EXAMPLE, complete=True)


def test_missing_and_unknown_keys(cfg):
    bad = dict(cfg)
    del bad["K"]
    with pytest.raises(ConfigError, match="missing"):
        validate(bad)
    with pytest.raises(ConfigError, match="unknown"):
        validate({**cfg, "lr": 1e-3})


@pytest.mark.parametrize("change, msg", [
    (dict(model="M1", integrator="leapfrog", rollout_integrator="leapfrog"), "separable"),
    (dict(model="M2s", integrator="leapfrog", rollout_integrator="rk4"), "match"),
    (dict(integrator="euler", rollout_integrator="euler"), "oracle-only"),
    (dict(model="B1"), "B1"),
    (dict(model="G3"), "s3 only"),
    (dict(dt=0.07), "multiple"),
    (dict(dt=0.8), "multiple"),
    (dict(K=0), "K"),
    (dict(seeds=[0, 0]), "distinct"),
])
def test_guards(cfg, change, msg):
    with pytest.raises(ConfigError, match=msg):
        validate({**cfg, **change})


def test_mismatch_cell_allowed(cfg):
    validate({**cfg, "model": "M3s", "integrator": "rk4", "rollout_integrator": "leapfrog"})


def test_b1_config(cfg):
    validate({**cfg, "model": "B1", "integrator": "none", "rollout_integrator": "none"})
