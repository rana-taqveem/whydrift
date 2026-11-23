"""Test doubles shared by every lane."""

import jax.numpy as jnp

from whydrift.contracts import System


class TrueFieldModel:
    """Wraps a System as a zero-parameter 'field' model.

    Sami's evaluator can run on it before any model is trained, and it is a
    shape-test target for the contracts. It is separable, so leapfrog works.
    """

    kind = "field"
    separable = True
    trained_dt = None

    def __init__(self, system: System):
        self.system = system
        self.name = f"TRUE_{system.name}"

    def init(self, key, system, cfg):
        return {}

    def vector_field(self, params, x):
        return self.system.true_rhs(x)

    def acceleration(self, params, q):
        return self.system.true_accel(q)

    def direct_step(self, params, x):
        raise NotImplementedError("field model")

    def n_params(self, params):
        return 0


class ExactStepModel:
    """A 'direct' (B1-like) model that steps S1 exactly; used to test B1 handling."""

    name = "B1"
    kind = "direct"
    separable = False

    def __init__(self, dt):
        self.trained_dt = dt

    def init(self, key, system, cfg):
        return {}

    def vector_field(self, params, x):
        raise NotImplementedError("direct model")

    def acceleration(self, params, q):
        raise NotImplementedError("direct model")

    def direct_step(self, params, x):
        c, s = jnp.cos(self.trained_dt), jnp.sin(self.trained_dt)
        q, p = x[..., :1], x[..., 1:]
        return jnp.concatenate([c * q + s * p, -s * q + c * p], axis=-1)

    def n_params(self, params):
        return 0


def s1_exact(x, t):
    """Exact S1 flow: a rotation in phase space."""
    q, p = x[..., :1], x[..., 1:]
    c, s = jnp.cos(t), jnp.sin(t)
    return jnp.concatenate([c * q + s * p, -s * q + c * p], axis=-1)
