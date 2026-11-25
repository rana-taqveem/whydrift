import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fakes import ExactStepModel, TrueFieldModel, s1_exact

from whydrift.eval.rollout import rollout
from whydrift.integrators import Field, make_stepper, oracle_stepper, step, system_field


def one_step_error(system, method, h, x):
    return float(jnp.max(jnp.abs(oracle_stepper(system, h, method)(x) - s1_exact(x, h))))


@pytest.mark.parametrize("method, local_order", [("rk4", 5), ("leapfrog", 3), ("euler", 2)])
def test_order_on_s1(s1, method, local_order):
    """Halving h cuts the one-step error by 2^(p+1): 32 for RK4, 8 for leapfrog."""
    x = s1.sample_x0(8, (0.5, 1.0), seed=0)
    ratio = one_step_error(s1, method, 0.1, x) / one_step_error(s1, method, 0.05, x)
    assert ratio == pytest.approx(2**local_order, rel=0.1)


def global_error(system, method, h, x0, T=1.0):
    xT = rollout(oracle_stepper(system, h, method), x0, int(round(T / h)))[:, -1]
    return float(jnp.max(jnp.abs(xT - s1_exact(x0, T))))


@pytest.mark.parametrize("method, order", [("rk4", 4), ("leapfrog", 2), ("euler", 1)])
def test_global_order_on_s1(s1, method, order):
    """Plan 1.3 / CI: halving dt over a fixed time cuts the error ~16x (RK4), ~4x (leapfrog), ~2x (Euler).

    T = 1 keeps Euler in its asymptotic range; at T = 10 its energy growth pushes the ratio to ~2.3.
    """
    x0 = s1.sample_x0(8, (0.5, 1.0), seed=0)
    ratio = global_error(s1, method, 0.1, x0) / global_error(s1, method, 0.05, x0)
    assert ratio == pytest.approx(2**order, rel=0.05)


def test_euler_s1_energy_factor(s1):
    """Euler on S1 multiplies energy by exactly 1 + h^2 per step."""
    x = s1.sample_x0(8, (0.1, 1.6), seed=1)
    x1 = oracle_stepper(s1, 0.1, "euler")(x)
    np.testing.assert_allclose(s1.energy(x1) / s1.energy(x), 1.01, rtol=1e-13)


@pytest.mark.parametrize("name", ["s1", "s2"])
def test_leapfrog_energy_bounded(name, s1, s2):
    system = {"s1": s1, "s2": s2}[name]
    x0 = system.sample_x0(4, (0.5, 1.2), seed=2)
    traj = rollout(oracle_stepper(system, 0.1, "leapfrog"), x0, 10_000)
    e = np.asarray(system.energy(traj))
    rel = np.abs(e - e[:, :1]) / e[:, :1]
    first, last = rel[:, :5000].max(axis=1), rel[:, 5000:].max(axis=1)
    assert rel.max() < 1e-2  # bounded, O(h^2)
    np.testing.assert_allclose(last, first, rtol=0.2)  # oscillates, no trend


def test_rk4_drifts_slowly_not_bounded(s1):
    """RK4 energy error on S1 trends (slowly) in one direction; a sanity contrast to leapfrog."""
    x0 = s1.sample_x0(4, (0.5, 1.2), seed=2)
    traj = rollout(oracle_stepper(s1, 0.1, "rk4"), x0, 2000)
    e = np.asarray(s1.energy(traj))
    assert np.all(e[:, -1] < e[:, 1000]) and np.all(e[:, 1000] < e[:, 0])


def test_guards():
    learned = Field(rhs=lambda p, x: x, acceleration=None, learned=True)
    x = jnp.ones((2, 2))
    with pytest.raises(ValueError, match="leapfrog"):
        step(learned, None, x, 0.1, "leapfrog")
    with pytest.raises(ValueError, match="euler"):
        step(learned, None, x, 0.1, "euler")
    with pytest.raises(ValueError, match="unknown method"):
        step(learned, None, x, 0.1, "rk2")


def test_bare_callable_counts_as_learned_and_non_separable():
    with pytest.raises(ValueError):
        step(lambda p, x: x, None, jnp.ones(2), 0.1, "leapfrog")


def test_make_stepper_field_model_matches_oracle(s1, fake_model):
    x = s1.sample_x0(5, (0.1, 1.6), seed=3)
    for method in ("rk4", "leapfrog"):
        np.testing.assert_array_equal(
            make_stepper(fake_model, {}, 0.1, method)(x), oracle_stepper(s1, 0.1, method)(x))


def test_make_stepper_b1_dt_lock(s1):
    model = ExactStepModel(dt=0.1)
    x = s1.sample_x0(3, (0.1, 1.0), seed=4)
    np.testing.assert_allclose(make_stepper(model, {}, 0.1, "rk4")(x), s1_exact(x, 0.1), atol=1e-14)
    with pytest.raises(ValueError, match="B1"):
        make_stepper(model, {}, 0.2, "rk4")


def test_make_stepper_refuses_leapfrog_for_non_separable(s1):
    model = TrueFieldModel(s1)
    model.separable = False
    with pytest.raises(ValueError, match="leapfrog"):
        make_stepper(model, {}, 0.1, "leapfrog")


def test_step_is_differentiable_through_rk4():
    """Gradient through 5 RK4 steps of a small MLP field matches a finite difference."""
    mlp = eqx.nn.MLP(2, 2, width_size=8, depth=2, activation=jnp.tanh, key=jax.random.PRNGKey(0))
    params, static = eqx.partition(mlp, eqx.is_array)
    field = Field(rhs=lambda p, x: jax.vmap(eqx.combine(p, static))(x))
    x0 = jnp.array([[0.3, -0.2], [1.0, 0.5]])

    def loss(p):
        x = x0
        for _ in range(5):
            x = step(field, p, x, 0.1, "rk4")
        return jnp.sum(x**2)

    g = jax.grad(loss)(params)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    direction = jax.tree_util.tree_unflatten(
        treedef, [jax.random.normal(jax.random.PRNGKey(i), l.shape) for i, l in enumerate(leaves)])
    analytic = sum(jnp.vdot(a, b) for a, b in zip(jax.tree_util.tree_leaves(g),
                                                  jax.tree_util.tree_leaves(direction)))
    eps = 1e-6
    shift = lambda s: jax.tree_util.tree_map(lambda p, d: p + s * d, params, direction)  # noqa: E731
    numeric = (loss(shift(eps)) - loss(shift(-eps))) / (2 * eps)
    assert float(analytic) == pytest.approx(float(numeric), rel=1e-6)


def test_oracle_field_is_not_learned(s1):
    assert system_field(s1).learned is False
