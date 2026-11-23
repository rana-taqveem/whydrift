"""True systems S1-S3. Owner: Rana.

All three are separable with unit mass: H(q, p) = 1/2 |p|^2 + V(q), so
q' = p and p' = -dV/dq. Everything is float64 and broadcasts over [..., d].
Methods accept NumPy or JAX arrays and return JAX arrays.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

import whydrift  # noqa: F401  (float64)


def split_state(x):
    """[..., 2n] -> (q [..., n], p [..., n])."""
    n = x.shape[-1] // 2
    return x[..., :n], x[..., n:]


class _Separable:
    """Shared parts. Subclasses define potential(q) -> [...] and true_accel(q)."""

    name: str
    n: int

    @property
    def dim(self) -> int:
        return 2 * self.n

    def energy(self, x):
        x = jnp.asarray(x)
        q, p = split_state(x)
        return 0.5 * jnp.sum(p**2, axis=-1) + self.potential(q)

    def true_rhs(self, x):
        x = jnp.asarray(x)
        q, p = split_state(x)
        return jnp.concatenate([p, self.true_accel(q)], axis=-1)

    def trajectory_ok(self, states) -> np.ndarray:
        """Per-trajectory physical validity check used by generate.py. [N, T, d] -> [N] bool."""
        return np.ones(states.shape[0], dtype=bool)


@dataclass(frozen=True)
class MassSpring(_Separable):
    """S1: q' = p, p' = -q, E = 1/2 p^2 + 1/2 q^2."""

    name: str = "s1"
    n: int = 1

    def potential(self, q):
        return 0.5 * jnp.sum(jnp.asarray(q) ** 2, axis=-1)

    def true_accel(self, q):
        return -jnp.asarray(q)

    def sample_x0(self, n_ic, e_range, seed):
        rng = np.random.default_rng(seed)
        e = rng.uniform(*e_range, size=n_ic)
        phi = rng.uniform(0.0, 2 * np.pi, size=n_ic)
        r = np.sqrt(2 * e)
        return jnp.asarray(np.stack([r * np.sin(phi), r * np.cos(phi)], axis=-1))


@dataclass(frozen=True)
class Pendulum(_Separable):
    """S2: q' = p, p' = -sin q, E = 1/2 p^2 + (1 - cos q). Goes over the top at E = 2."""

    name: str = "s2"
    n: int = 1

    def potential(self, q):
        return jnp.sum(1.0 - jnp.cos(jnp.asarray(q)), axis=-1)

    def true_accel(self, q):
        return -jnp.sin(jnp.asarray(q))

    def sample_x0(self, n_ic, e_range, seed):
        lo, hi = e_range
        if not (0 < lo <= hi < 2):
            raise ValueError(f"S2 energies must lie in (0, 2) to stay below the separatrix, got {e_range}")
        rng = np.random.default_rng(seed)
        e = rng.uniform(lo, hi, size=n_ic)
        q_max = np.arccos(1.0 - e)
        q = rng.uniform(-1.0, 1.0, size=n_ic) * q_max
        sign = rng.choice([-1.0, 1.0], size=n_ic)
        p = sign * np.sqrt(np.maximum(2.0 * (e - 1.0 + np.cos(q)), 0.0))
        return jnp.asarray(np.stack([q, p], axis=-1))


@dataclass(frozen=True)
class FPUTChain(_Separable):
    """S3: FPUT chain, fixed ends q_0 = q_{n+1} = 0, so n + 1 bonds.

    Bond potential V(r) = 1/2 r^2 + (alpha/3) r^3 + (beta/4) r^4, r_i = q_{i+1} - q_i.
    alpha-FPUT: beta = 0. beta-FPUT: alpha = 0. The form and values are [OPEN];
    both are explicit arguments so no hidden default creeps in.
    The state holds q_1..q_n then p_1..p_n; the fixed ends are not stored.
    """

    alpha: float
    beta: float
    name: str = "s3"
    n: int = 8

    def __post_init__(self):
        if self.beta < 0:
            raise ValueError("beta < 0 makes the potential unbounded below")

    # ---- geometry ----
    def bonds(self, q):
        q = jnp.asarray(q)
        zeros = jnp.zeros(q.shape[:-1] + (1,), dtype=q.dtype)
        padded = jnp.concatenate([zeros, q, zeros], axis=-1)
        return padded[..., 1:] - padded[..., :-1]  # [..., n + 1]

    def bond_potential(self, r):
        return 0.5 * r**2 + (self.alpha / 3.0) * r**3 + (self.beta / 4.0) * r**4

    def potential(self, q):
        return jnp.sum(self.bond_potential(self.bonds(q)), axis=-1)

    def true_accel(self, q):
        q = jnp.asarray(q)
        return -jax.grad(lambda qq: jnp.sum(self.potential(qq)))(q)

    # ---- escape barrier ----
    @property
    def barrier_r(self) -> float | None:
        """Bond length of the nearest local maximum of V(r), or None if V has none.

        V'(r) = r (1 + alpha r + beta r^2). For alpha-FPUT this is r = -1/alpha,
        with barrier height 1 / (6 alpha^2). Beyond it the true chain escapes.
        """
        roots = np.roots([self.beta, self.alpha, 1.0]) if self.beta else (
            np.array([-1.0 / self.alpha]) if self.alpha else np.array([])
        )
        real = [float(r.real) for r in np.atleast_1d(roots) if abs(np.imag(r)) < 1e-12]
        return min(real, key=abs) if real else None

    def bonds_safe(self, q) -> np.ndarray:
        """[..., n] -> [...] bool: every bond stays within half the distance to the barrier."""
        rb = self.barrier_r
        r = np.asarray(self.bonds(q))
        if rb is None:
            return np.ones(r.shape[:-1], dtype=bool)
        return np.all(r * np.sign(rb) < abs(rb) / 2, axis=-1)

    def trajectory_ok(self, states) -> np.ndarray:
        q, _ = split_state(np.asarray(states))
        return np.all(self.bonds_safe(q), axis=-1)

    # ---- sampling ----
    def sample_x0(self, n_ic, e_range, seed):
        """Random direction u in state space, then solve E(lambda u) = E_target for lambda > 0."""
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n_ic:
            e_t = rng.uniform(*e_range)
            u = rng.normal(size=self.dim)
            u /= np.linalg.norm(u)
            x = self._scale_to_energy(u, e_t)
            if x is not None:
                out.append(x)
        return jnp.asarray(np.stack(out))

    def _scale_to_energy(self, u, e_t):
        def f(lam):
            return float(self.energy(lam * u)) - e_t

        hi = 0.1
        while f(hi) < 0:
            if not self.bonds_safe(hi * u[: self.n]):
                return None  # would reach the barrier before the target energy
            hi *= 2.0
            if hi > 1e6:
                return None
        lam = brentq(f, 0.0, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)
        x = lam * u
        return x if self.bonds_safe(x[: self.n]) else None


def make_system(name: str, **kwargs):
    """'s1' | 's2' | 's3'. S3 needs alpha and beta explicitly."""
    if name == "s1":
        return MassSpring()
    if name == "s2":
        return Pendulum()
    if name == "s3":
        return FPUTChain(**kwargs)
    raise ValueError(f"unknown system {name!r}")
