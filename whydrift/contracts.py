"""Shared contracts. Changing this file needs approval from all three owners.

State convention: x = [q, p], positions first, then momenta. d = 2n.
Every function takes arrays shaped [..., d] and broadcasts over leading dims.
All values outside a model are in physical units; standardisation lives
inside the model.
"""

from typing import Any, Callable, Protocol, runtime_checkable

import jax

Array = jax.Array
Params = Any  # a pytree of arrays
Stepper = Callable[[Array], Array]  # x -> x_next, physical units


@runtime_checkable
class System(Protocol):
    """A true Hamiltonian system with unit mass: H = 1/2 |p|^2 + V(q)."""

    name: str  # 's1' | 's2' | 's3'
    n: int  # number of positions
    dim: int  # 2n

    def true_rhs(self, x: Array) -> Array:
        """dx/dt, shape [..., d]."""
        ...

    def true_accel(self, q: Array) -> Array:
        """dp/dt as a function of q only, shape [..., n]. All systems are separable."""
        ...

    def energy(self, x: Array) -> Array:
        """True energy, shape [...]. The only energy ever used for drift and b."""
        ...

    def sample_x0(self, n_ic: int, e_range: tuple[float, float], seed: int) -> Array:
        """n_ic states, each with an exact energy drawn uniformly from e_range. [n_ic, d]."""
        ...


@runtime_checkable
class Model(Protocol):
    """A learned simulator. Owner: Saad."""

    name: str  # one of common.MODELS
    kind: str  # 'field' (all but B1) | 'direct' (B1)
    separable: bool  # True -> leapfrog allowed (M2s, M3s, G3)
    trained_dt: float | None  # B1 only: the dt it was trained at

    def init(self, key: Array, system: System, cfg: dict) -> Params: ...

    def vector_field(self, params: Params, x: Array) -> Array:
        """dx/dt [..., d]. kind == 'field' only."""
        ...

    def acceleration(self, params: Params, q: Array) -> Array:
        """dp/dt [..., n]. Separable models only."""
        ...

    def direct_step(self, params: Params, x: Array) -> Array:
        """x_next [..., d]. B1 only."""
        ...

    def n_params(self, params: Params) -> int: ...


def is_separable(model: Model) -> bool:
    """True -> leapfrog allowed."""
    return bool(getattr(model, "separable", False))


# ---- metrics rows (one per seed x checkpoint); owner: Sami ----

METRIC_ID_COLUMNS = (
    "run_id", "system", "model", "K", "noise_sigma", "integrator",
    "rollout_integrator", "dt", "seed", "ckpt_step", "n_params", "val_loss",
)
METRIC_VALUE_COLUMNS = (
    "vpt",
    "drift_t10", "drift_t100", "drift_t500", "drift_t1000", "drift_final",
    "mse_1step",
    "err_t10", "err_t100", "err_t500", "err_t1000", "err_auc",
    "phase_err", "blowup",
    "b", "s", "b_abs", "b_val",
)
# Plus one 'b_bin{k}' column per energy bin; the bin edges are [OPEN].
METRIC_COLUMNS = METRIC_ID_COLUMNS + METRIC_VALUE_COLUMNS
B0_MODEL_NAME = "B0"  # the oracle's rows use this in the 'model' column
