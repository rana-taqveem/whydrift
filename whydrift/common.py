"""Project-wide constants and settings. Owner: Rana (shared; changes need all three owners).

Import order matters: JAX must get the float64 flag before any array is created,
so `whydrift/__init__.py` imports this module first.
"""

import jax

jax.config.update("jax_enable_x64", True)

# Integrators. 'euler' is for the B0 oracle only (README section 6).
METHODS = ("rk4", "leapfrog", "euler")
LEARNED_METHODS = ("rk4", "leapfrog")

# Models (README section 5).
MODELS = ("B1", "M1", "M2", "M3", "M2s", "M3s", "G3")
SEPARABLE_MODELS = ("M2s", "M3s", "G3")  # leapfrog allowed
DIRECT_MODELS = ("B1",)  # predict x_next directly, no vector field
S3_ONLY_MODELS = ("G3",)

SYSTEMS = ("s1", "s2", "s3")

# Data is stored at this interval; every swept dt must be a multiple of it.
STORED_DT = 0.05
DT_SWEEP = (0.05, 0.1, 0.2, 0.4)  # [PROPOSED] grid inside 0.05-0.4

# Lengths are set in time, not steps, so the largest dt still has full horizons.
TRAIN_STEPS = 100
TEST_STEPS = 1000


def prng_key(seed: int) -> jax.Array:
    """The one way to turn an integer seed into a JAX key."""
    return jax.random.PRNGKey(int(seed))


def dt_is_valid(dt: float) -> bool:
    """True if dt is a positive multiple of STORED_DT within the sweep range."""
    k = round(dt / STORED_DT)
    return k >= 1 and abs(k * STORED_DT - dt) < 1e-12 and dt <= max(DT_SWEEP) + 1e-12
