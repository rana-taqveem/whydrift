"""Generate ground-truth trajectories. Owner: Rana.

    python -m whydrift.physics.generate --system s1 [--small] [--out data]

Writes data/{system}_{split}.npz (states [N, T+1, d], energy [N, T+1],
x0 [N, d], t [T+1]) and data/{system}_meta.json.

All trajectories of a split are integrated in ONE batched solve_ivp call
(DOP853, rtol = atol = 1e-12; see RTOL below for why not 1e-10). Any trajectory whose relative energy error
exceeds 1e-8, or that fails the system's validity check (S3 escape barrier),
is replaced by a fresh draw until the split is full. Splits use distinct
seeds, so they are disjoint. Lengths are set in time, so every dt in the
sweep (up to 0.4) has its full 100 training / 1,000 test steps.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from scipy.integrate import solve_ivp

import whydrift  # noqa: F401  (float64)
from whydrift.common import DT_SWEEP, STORED_DT, TEST_STEPS, TRAIN_STEPS, dt_is_valid
from whydrift.physics.systems import make_system

# Tighter than README's original ~1e-10: with test references 400 time units long,
# 1e-10 let S1 stray ~1.05e-8 from the exact solution (over the 1e-8 check).
# 1e-12 gives ~1e-10 at t = 400 and costs ~2 s more per split. Recorded in README section 3.
RTOL = ATOL = 1e-12
ENERGY_TOL = 1e-8
MAX_DT = max(DT_SWEEP)
TRAIN_TIME = TRAIN_STEPS * MAX_DT  # 40 time units
TEST_TIME = TEST_STEPS * MAX_DT  # 400 time units

# [PROPOSED] split plan. S1's energy range is not in the proposal; it mirrors S2's.
# S3's range and sizes are [OPEN] until the FPUT form is chosen.
SPLITS = {
    "s1": {
        "train": dict(n=500, time=TRAIN_TIME, e_range=(0.1, 1.6), seed=101),
        "val": dict(n=100, time=TRAIN_TIME, e_range=(0.1, 1.6), seed=102),
        "test": dict(n=100, time=TEST_TIME, e_range=(0.1, 1.6), seed=103),
    },
    "s2": {
        "train": dict(n=500, time=TRAIN_TIME, e_range=(0.1, 1.6), seed=201),
        "val": dict(n=100, time=TRAIN_TIME, e_range=(0.1, 1.6), seed=202),
        "test": dict(n=100, time=TEST_TIME, e_range=(0.1, 1.6), seed=203),
        "test_ood": dict(n=100, time=TEST_TIME, e_range=(1.7, 1.95), seed=204),
    },
}
SMALL_N = {"train": 8, "val": 4, "test": 4, "test_ood": 4}
MAX_ATTEMPTS = 50


def integrate(system, x0: np.ndarray, t_end: float, dt: float = STORED_DT):
    """Batched DOP853 solve of all initial conditions at once. Returns states [N, T+1, d], t [T+1]."""
    n_ic, d = x0.shape
    n_t = int(round(t_end / dt))
    t = np.arange(n_t + 1) * dt

    f = jax.jit(system.true_rhs)  # SciPy calls this thousands of times; eager dispatch is slow

    def rhs(_, y):
        return np.asarray(f(y.reshape(n_ic, d))).ravel()

    sol = solve_ivp(rhs, (0.0, t[-1]), x0.ravel(), method="DOP853",
                    t_eval=t, rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.y.reshape(n_ic, d, n_t + 1).transpose(0, 2, 1), t


def relative_energy_error(system, states: np.ndarray) -> np.ndarray:
    """[N, T+1, d] -> [N]: max_t |E(t) - E(0)| / |E(0)|."""
    e = np.asarray(system.energy(states))
    return np.max(np.abs(e - e[:, :1]), axis=1) / np.abs(e[:, 0])


def generate_split(system, n: int, time: float, e_range, seed: int):
    """Sample, integrate, reject and resample until n valid trajectories exist."""
    kept, n_rejected = [], 0
    for attempt in range(MAX_ATTEMPTS):
        need = n - sum(len(k) for k in kept)
        if need == 0:
            break
        x0 = np.asarray(system.sample_x0(need, e_range, seed * 1000 + attempt))
        states, t = integrate(system, x0, time)
        ok = (relative_energy_error(system, states) <= ENERGY_TOL) & system.trajectory_ok(states)
        kept.append(states[ok])
        n_rejected += int((~ok).sum())
    else:
        raise RuntimeError(f"could not fill split after {MAX_ATTEMPTS} attempts")
    states = np.concatenate(kept)[:n]
    return states, t, n_rejected


def generate(system_name: str, out_dir="data", small: bool = False, system_kwargs=None):
    if system_name not in SPLITS:
        raise ValueError(f"no split plan for {system_name!r} yet (S3 sizes and energy range are [OPEN])")
    system = make_system(system_name, **(system_kwargs or {}))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(system=system_name, system_kwargs=system_kwargs or {}, stored_dt=STORED_DT,
                rtol=RTOL, atol=ATOL, energy_tol=ENERGY_TOL, small=small, splits={})
    for split, spec in SPLITS[system_name].items():
        n = SMALL_N[split] if small else spec["n"]
        states, t, n_rej = generate_split(system, n, spec["time"], spec["e_range"], spec["seed"])
        energy = np.asarray(system.energy(states))
        np.savez(out / f"{system_name}_{split}.npz",
                 states=states, energy=energy, x0=states[:, 0], t=t)
        meta["splits"][split] = dict(n=n, time=spec["time"], n_samples=len(t),
                                     e_range=list(spec["e_range"]), seed=spec["seed"],
                                     n_rejected=n_rej)
    (out / f"{system_name}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_split(system_name: str, split: str, data_dir="data", dt: float | None = None,
               n_steps: int | None = None) -> dict:
    """Load one split. With dt, keep every k-th stored sample (k = dt / 0.05);
    with n_steps, also cut to the first n_steps steps at that dt.

    The one shared way to get data at a run's dt, so nobody hand-rolls the stride.
    """
    with np.load(Path(data_dir) / f"{system_name}_{split}.npz") as f:
        d = {k: f[k] for k in f.files}
    if dt is None:
        if n_steps is not None:
            raise ValueError("n_steps needs dt")
        return d
    if not dt_is_valid(dt):
        raise ValueError(f"dt={dt} must be a multiple of the stored {STORED_DT}, at most {MAX_DT}")
    k = int(round(dt / STORED_DT))
    end = None if n_steps is None else n_steps * k + 1
    if end is not None and end > len(d["t"]):
        raise ValueError(f"{n_steps} steps at dt={dt} need t={n_steps * dt}, "
                         f"but this split stops at t={d['t'][-1]}")
    sl = slice(0, end, k)
    return dict(states=d["states"][:, sl], energy=d["energy"][:, sl], x0=d["x0"], t=d["t"][sl])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True, choices=sorted(SPLITS))
    ap.add_argument("--out", default="data")
    ap.add_argument("--small", action="store_true", help="a few trajectories per split, for tests and stubs")
    args = ap.parse_args()
    meta = generate(args.system, args.out, args.small)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
