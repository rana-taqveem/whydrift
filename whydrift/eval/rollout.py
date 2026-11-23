"""Rollouts. Owner: Sami (this is the contract-level baseline from the Day-1 foundations).

A rollout feeds each prediction back in. A trajectory that goes non-finite
is frozen as NaN from that step on and counted as a blow-up, never dropped.
"""

import jax
import jax.numpy as jnp

import whydrift  # noqa: F401  (float64)
from whydrift.contracts import Stepper


def rollout(stepper: Stepper, x0, n_steps: int):
    """x0 [n_ic, d] -> traj [n_ic, n_steps + 1, d], physical units."""
    x0 = jnp.asarray(x0)

    def body(carry, _):
        x, dead = carry
        x_next = stepper(x)
        dead = dead | ~jnp.all(jnp.isfinite(x_next), axis=-1)
        x_next = jnp.where(dead[..., None], jnp.nan, x_next)
        return (x_next, dead), x_next

    dead0 = ~jnp.all(jnp.isfinite(x0), axis=-1)
    _, xs = jax.lax.scan(body, (x0, dead0), None, length=n_steps)
    traj = jnp.concatenate([x0[None], xs], axis=0)  # [T+1, n_ic, d]
    return jnp.swapaxes(traj, 0, 1)
