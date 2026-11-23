import jax.numpy as jnp
import numpy as np
from fakes import TrueFieldModel

from whydrift.eval.rollout import rollout
from whydrift.integrators import make_stepper


def test_rollout_shape_and_start(s1):
    x0 = s1.sample_x0(3, (0.1, 1.6), seed=0)
    traj = rollout(make_stepper(TrueFieldModel(s1), {}, 0.1, "rk4"), x0, 50)
    assert traj.shape == (3, 51, 2)
    np.testing.assert_array_equal(traj[:, 0], x0)


def test_true_field_small_dt_has_near_zero_drift(s1):
    x0 = s1.sample_x0(4, (0.1, 1.6), seed=1)
    traj = rollout(make_stepper(TrueFieldModel(s1), {}, 0.05, "rk4"), x0, 1000)
    e = np.asarray(s1.energy(traj))
    assert np.max(np.abs(e - e[:, :1]) / e[:, :1]) < 1e-5


def test_blowup_is_frozen_as_nan_not_dropped():
    explode = lambda x: x * 1e200  # noqa: E731  overflows to inf after 2 steps
    x0 = jnp.array([[1.0, 1.0], [0.0, 0.0]])
    traj = np.asarray(rollout(explode, x0, 5))
    assert traj.shape == (2, 6, 2)
    assert np.isnan(traj[0, 2:]).all() and np.isfinite(traj[0, :2]).all()
    assert np.isfinite(traj[1]).all()  # zeros never blow up
