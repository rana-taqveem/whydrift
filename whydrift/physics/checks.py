"""Audit generated data. Owner: Rana.

    python -m whydrift.physics.checks --system s1 [--data data]

Prints one line per split and exits non-zero if any check fails:
energy conserved to 1e-8, energies inside the split's range, x0 equal to
the first stored state, no initial condition shared between splits, and
(S1 only) agreement with the exact solution to 1e-8.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import whydrift  # noqa: F401  (float64)
from whydrift.physics.generate import ENERGY_TOL, load_split, relative_energy_error
from whydrift.physics.systems import make_system

EXACT_TOL = 1e-8


def s1_exact_error(d) -> float:
    q0, p0, t = d["x0"][:, :1], d["x0"][:, 1:], d["t"]
    exact = np.stack([q0 * np.cos(t) + p0 * np.sin(t), -q0 * np.sin(t) + p0 * np.cos(t)], axis=-1)
    return float(np.abs(d["states"] - exact).max())


def audit(system_name: str, data_dir="data") -> list[dict]:
    meta = json.loads((Path(data_dir) / f"{system_name}_meta.json").read_text())
    system = make_system(system_name, **meta.get("system_kwargs", {}))
    rows, all_x0 = [], []
    for split, spec in meta["splits"].items():
        d = load_split(system_name, split, data_dir)
        e0 = np.asarray(system.energy(d["x0"]))
        lo, hi = spec["e_range"]
        row = dict(split=split, n=len(d["x0"]), n_rejected=spec["n_rejected"],
                   t_end=float(d["t"][-1]),
                   max_rel_energy_err=float(relative_energy_error(system, d["states"]).max()),
                   e_min=float(e0.min()), e_max=float(e0.max()))
        row["ok"] = bool(row["max_rel_energy_err"] <= ENERGY_TOL
                         and lo - 1e-9 <= row["e_min"] and row["e_max"] <= hi + 1e-9
                         and np.array_equal(d["x0"], d["states"][:, 0])
                         and row["n"] == spec["n"])
        if system_name == "s1":
            row["exact_err"] = s1_exact_error(d)
            row["ok"] &= row["exact_err"] < EXACT_TOL
        rows.append(row)
        all_x0.append(d["x0"])
    stacked = np.concatenate(all_x0)
    disjoint = len(np.unique(stacked, axis=0)) == len(stacked)
    for row in rows:
        row["ok"] &= disjoint
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", required=True)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    rows = audit(args.system, args.data)
    for r in rows:
        extra = f"  exact_err={r['exact_err']:.1e}" if "exact_err" in r else ""
        print(f"{args.system} {r['split']:<9} n={r['n']:<4} rejected={r['n_rejected']:<3} "
              f"t_end={r['t_end']:<5g} E=[{r['e_min']:.3f}, {r['e_max']:.3f}]  "
              f"max_rel_dE={r['max_rel_energy_err']:.1e}{extra}  {'OK' if r['ok'] else 'FAIL'}")
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


if __name__ == "__main__":
    main()
