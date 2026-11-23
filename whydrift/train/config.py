"""Run configs. Owner: Saad (Day-1 foundations version).

One config file plus its seed list fully defines a run. The loader refuses
missing or unknown keys, so no hidden default can sneak in. Values still
[OPEN] may be null while the team decides; load with complete=True to
refuse nulls before a real run.
"""

from pathlib import Path

import yaml

from whydrift.common import (DIRECT_MODELS, LEARNED_METHODS, MODELS, S3_ONLY_MODELS,
                             SEPARABLE_MODELS, SYSTEMS, dt_is_valid)

KEYS = {
    "system", "model", "width", "target_params", "activation", "integrator",
    "rollout_integrator", "dt", "K", "noise_sigma", "optimizer", "updates",
    "ckpt_every", "seeds", "precision",
}
OPTIMIZER_KEYS = {"name", "lr", "batch_size"}
OPEN_KEYS = {"target_params", "updates", "ckpt_every"}  # may be null until decided
OPEN_OPTIMIZER_KEYS = {"lr", "batch_size"}


class ConfigError(ValueError):
    pass


def validate(cfg: dict, complete: bool = False) -> dict:
    def fail(msg):
        raise ConfigError(msg)

    missing, unknown = KEYS - cfg.keys(), cfg.keys() - KEYS
    if missing or unknown:
        fail(f"missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    opt = cfg["optimizer"]
    if not isinstance(opt, dict) or opt.keys() != OPTIMIZER_KEYS:
        fail(f"optimizer must have exactly {sorted(OPTIMIZER_KEYS)}")

    if cfg["system"] not in SYSTEMS:
        fail(f"system must be one of {SYSTEMS}")
    model = cfg["model"]
    if model not in MODELS:
        fail(f"model must be one of {MODELS}")
    if model in S3_ONLY_MODELS and cfg["system"] != "s3":
        fail(f"{model} runs on s3 only")
    if not dt_is_valid(cfg["dt"]):
        fail(f"dt={cfg['dt']} must be a multiple of the stored 0.05, at most 0.4")
    if not (isinstance(cfg["K"], int) and cfg["K"] >= 1):
        fail("K must be an integer >= 1")
    if not (isinstance(cfg["noise_sigma"], (int, float)) and cfg["noise_sigma"] >= 0):
        fail("noise_sigma must be a number >= 0")
    seeds = cfg["seeds"]
    if not (isinstance(seeds, list) and seeds and all(isinstance(s, int) for s in seeds)
            and len(set(seeds)) == len(seeds)):
        fail("seeds must be a non-empty list of distinct integers")
    if cfg["precision"] not in ("float32", "float64"):
        fail("precision must be float32 or float64")

    # Integrator rules (README section 7 and guardrails).
    tr, ro = cfg["integrator"], cfg["rollout_integrator"]
    if model in DIRECT_MODELS:
        if tr != "none" or ro != "none":
            fail("B1 predicts the next state directly: integrator and rollout_integrator must be 'none'")
    else:
        if tr not in LEARNED_METHODS or ro not in LEARNED_METHODS:
            fail(f"integrators for learned field models must be in {LEARNED_METHODS} (euler is oracle-only)")
        if "leapfrog" in (tr, ro) and model not in SEPARABLE_MODELS:
            fail(f"leapfrog needs a separable model {SEPARABLE_MODELS}, got {model}")
        if tr != ro and not (tr == "rk4" and ro == "leapfrog"):
            fail("training and rollout integrators must match, except the single rk4 -> leapfrog mismatch cell")

    if complete:
        nulls = [k for k in OPEN_KEYS if cfg[k] is None] + \
                [f"optimizer.{k}" for k in OPEN_OPTIMIZER_KEYS if opt[k] is None]
        if nulls:
            fail(f"still null (decide before a real run): {sorted(nulls)}")
    return cfg


def load_config(path, complete: bool = False) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ConfigError("config file must hold a mapping")
    return validate(cfg, complete=complete)
