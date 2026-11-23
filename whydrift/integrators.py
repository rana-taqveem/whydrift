"""Differentiable one-step maps. Owner: Rana.

Training backpropagates through step(), so only jax.numpy ops live here:
no SciPy, no in-place updates.

The README's rules are enforced in code, not by convention:
  * leapfrog needs a separable field (one that exposes acceleration(q));
  * euler is for the B0 oracle only, never a learned model;
  * B1 runs only at the dt it was trained at.
"""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

import whydrift  # noqa: F401  (float64)
from whydrift.common import DIRECT_MODELS, METHODS
from whydrift.contracts import Model, Stepper, System, is_separable


@dataclass(frozen=True)
class Field:
    """A vector field in the shape step() expects.

    rhs(params, x) -> dx/dt [..., d]
    acceleration(params, q) -> dp/dt [..., n], or None if the field is not separable
    learned: True for a model, False for the true equations (the B0 oracle)
    """

    rhs: Callable
    acceleration: Callable | None = None
    learned: bool = True

    def __call__(self, params, x):
        return self.rhs(params, x)


def system_field(system: System) -> Field:
    """The true equations in the model slots, so B0 runs through the same code as the models."""
    return Field(
        rhs=lambda params, x: system.true_rhs(x),
        acceleration=lambda params, q: system.true_accel(q),
        learned=False,
    )


def model_field(model: Model) -> Field:
    return Field(
        rhs=model.vector_field,
        acceleration=model.acceleration if is_separable(model) else None,
        learned=True,
    )


def _rk4(f, params, x, h):
    k1 = f(params, x)
    k2 = f(params, x + 0.5 * h * k1)
    k3 = f(params, x + 0.5 * h * k2)
    k4 = f(params, x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _leapfrog(accel, params, x, h):
    """Velocity Verlet: p_half = p + h/2 a(q); q' = q + h p_half; p' = p_half + h/2 a(q')."""
    n = x.shape[-1] // 2
    q, p = x[..., :n], x[..., n:]
    p_half = p + 0.5 * h * accel(params, q)
    q_new = q + h * p_half
    p_new = p_half + 0.5 * h * accel(params, q_new)
    return jnp.concatenate([q_new, p_new], axis=-1)


def _euler(f, params, x, h):
    return x + h * f(params, x)


def step(field, params, x, dt, method):
    """One step of size dt. method: 'rk4' | 'leapfrog' | 'euler'.

    `field` is a Field, or any callable (params, x) -> dx/dt; a bare callable
    counts as learned and non-separable.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not isinstance(field, Field):
        field = Field(rhs=field)
    if method == "leapfrog":
        if field.acceleration is None:
            raise ValueError("leapfrog needs a separable field (acceleration(q)); "
                             "M1, M2 and M3 must use rk4")
        return _leapfrog(field.acceleration, params, x, dt)
    if method == "euler":
        if field.learned:
            raise ValueError("euler is for the B0 oracle only, never a learned model")
        return _euler(field.rhs, params, x, dt)
    return _rk4(field.rhs, params, x, dt)


def make_stepper(model: Model, params, dt: float, method: str) -> Stepper:
    """x -> x_next. Hides the B1 vs field-model difference from rollouts and the trainer."""
    if model.kind == "direct" or model.name in DIRECT_MODELS:
        if model.trained_dt is None or abs(model.trained_dt - dt) > 1e-12:
            raise ValueError(f"B1 was trained at dt={model.trained_dt} and cannot run at dt={dt}")
        return lambda x: model.direct_step(params, x)
    field = model_field(model)
    if method == "leapfrog" and field.acceleration is None:
        raise ValueError(f"leapfrog not allowed for non-separable model {model.name}")
    return lambda x: step(field, params, x, dt, method)


def oracle_stepper(system: System, dt: float, method: str) -> Stepper:
    """B0: the true equations stepped with an integrator."""
    field = system_field(system)
    return lambda x: step(field, None, x, dt, method)
