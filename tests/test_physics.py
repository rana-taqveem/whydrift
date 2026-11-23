import jax
import jax.numpy as jnp
import numpy as np
import pytest

from whydrift.contracts import System
from whydrift.physics.systems import FPUTChain

E_RANGE = {"s1": (0.1, 1.6), "s2": (0.1, 1.6), "s3": (0.05, 0.5)}


def test_satisfies_contract(any_system):
    assert isinstance(any_system, System)
    assert any_system.dim == 2 * any_system.n


@pytest.mark.parametrize("batch", [(), (5,), (3, 4)])
def test_shapes_and_dtype(any_system, batch):
    sys_ = any_system
    x = jnp.ones(batch + (sys_.dim,)) * 0.1
    assert sys_.true_rhs(x).shape == batch + (sys_.dim,)
    assert sys_.true_accel(x[..., : sys_.n]).shape == batch + (sys_.n,)
    assert sys_.energy(x).shape == batch
    assert sys_.true_rhs(x).dtype == jnp.float64
    assert sys_.energy(x).dtype == jnp.float64


def test_rhs_is_hamiltonian(any_system):
    """q' = dE/dp and p' = -dE/dq, checked against autodiff of the energy."""
    sys_ = any_system
    x = sys_.sample_x0(6, E_RANGE[sys_.name], seed=0)
    grad = jax.vmap(jax.grad(sys_.energy))(x)
    n = sys_.n
    expected = jnp.concatenate([grad[:, n:], -grad[:, :n]], axis=-1)
    np.testing.assert_allclose(sys_.true_rhs(x), expected, rtol=1e-12, atol=1e-12)


def test_true_accel_is_p_part_of_rhs(any_system):
    sys_ = any_system
    x = sys_.sample_x0(6, E_RANGE[sys_.name], seed=1)
    np.testing.assert_array_equal(sys_.true_accel(x[:, : sys_.n]), sys_.true_rhs(x)[:, sys_.n:])


def test_sample_x0_hits_energy_and_is_reproducible(any_system):
    sys_ = any_system
    lo, hi = E_RANGE[sys_.name]
    x = sys_.sample_x0(20, (lo, hi), seed=7)
    assert x.shape == (20, sys_.dim) and x.dtype == jnp.float64
    e = np.asarray(sys_.energy(x))
    assert np.all((e >= lo - 1e-12) & (e <= hi + 1e-12))
    np.testing.assert_array_equal(x, sys_.sample_x0(20, (lo, hi), seed=7))
    assert not np.array_equal(x, sys_.sample_x0(20, (lo, hi), seed=8))


def test_sample_x0_exact_energy_s1(s1):
    """S1 draws the target energy first, so a fixed range collapses to that energy exactly."""
    x = s1.sample_x0(10, (0.7, 0.7), seed=0)
    np.testing.assert_allclose(s1.energy(x), 0.7, rtol=1e-14)


def test_s2_rejects_energies_over_the_top(s2):
    with pytest.raises(ValueError):
        s2.sample_x0(3, (1.5, 2.1), seed=0)


def test_s3_barrier_alpha():
    chain = FPUTChain(alpha=0.25, beta=0.0)
    assert chain.barrier_r == pytest.approx(-4.0)
    r = chain.barrier_r
    assert float(chain.bond_potential(r)) == pytest.approx(1 / (6 * 0.25**2))


def test_s3_beta_has_no_barrier():
    assert FPUTChain(alpha=0.0, beta=1.0).barrier_r is None


def test_s3_bonds_include_fixed_ends(s3):
    q = jnp.arange(1.0, 9.0)  # q_1..q_8
    r = s3.bonds(q)
    assert r.shape == (9,)
    assert float(r[0]) == 1.0 and float(r[-1]) == -8.0  # q_1 - 0 and 0 - q_8


def test_s3_samples_stay_below_barrier(s3):
    x = s3.sample_x0(20, (0.05, 0.5), seed=3)
    assert np.all(s3.bonds_safe(x[:, : s3.n]))
