# whydrift — Why Do Learned Simulators Drift?

AI651 (Deep Learning for Time, Space, and Graphs), LUMS, Fall 2026 · Track E / Idea 1
Team: Saad Ali · Rana Taqveem Ul Hassan · Muhammad Sami Ul Basit

> **Status: pre-implementation.** The proposal is approved in outline. Nothing is built yet.
> This README is the working spec. Refine it with the team before writing code.
> Items marked **[PROPOSED]** are our own suggestions, not in the proposal. Items marked **[OPEN]** are undecided.

---

## 1. Problem statement

Neural networks trained to predict one time step of a physical system are often accurate for that one step. When their own predictions are fed back in for hundreds of steps, they **drift**: energy creeps up or down and the motion falls out of phase.

**Goal:** explain **why** learned simulators drift, not only which choices reduce drift.

**Central hypothesis (H5):** drift is driven by a *systematic* per-step energy error, the energy bias **b**. Random per-step errors mostly cancel and grow like √n·s over n steps. A systematic bias adds up like n·b. If that is right, b can be measured from **one step, with no rollout**. It should predict 1,000-step drift better than one-step error does, and selecting checkpoints by lowest |b| should give more stable models.

**What success means**
1. Every hypothesis H1–H5 gets a clear pass or fail under rules fixed **before** the main runs.
2. We establish whether b predicts long-run drift better than one-step MSE.
3. We establish whether |b|-based checkpoint selection gives models that drift less.

A clean negative result still counts. The explanation is the deliverable. We are **not** inventing a new drift-correction method.

## 2. Scope

**In (core)**
- Systems: S1 mass–spring, S2 ideal pendulum, S3 8-mass nonlinear FPUT chain.
- Models: B1, M1, M2, M3, M2-s, M3-s, G3 (see §5).
- Training factors: loss horizon K = 1–20 steps, input noise on/off, integrator matched between training and rollout (plus one mismatch cell).
- Integrators: RK4 and leapfrog, Δt swept 0.05–0.4. Euler for the oracle only.
- 10 seeds per configuration, bootstrap CIs, every test reported per system.
- ≈660 core runs.

**Optional (cut first if time is short):** extended sweeps (≈1,500 runs total).

**Out:** new drift-correction methods, external or real-world data, dissipative systems (friction), claims about chaotic long-run statistics.

## 3. Data — generated, not downloaded

There is nothing to buy or download. We integrate the true equations with SciPy, so the exact answer is always available as an oracle. Mass and inertia are 1, so p = dq/dt.

| System | Dynamics | Energy |
|---|---|---|
| S1 mass–spring | q̇ = p, ṗ = −q | E = ½p² + ½q² |
| S2 pendulum | q̇ = p, ṗ = −sin q | E = ½p² + (1 − cos q); goes over the top at E = 2 |
| S3 FPUT chain (8 masses, 16-number state) | **[PROPOSED]** α-FPUT, fixed ends q₀ = q₉ = 0 | E = Σ ½pᵢ² + Σ [½(q_{i+1}−q_i)² + (α/3)(q_{i+1}−q_i)³] |

- Solver: `scipy.integrate.solve_ivp`, method `DOP853`, rtol = atol = **1e-12**. *Changed from ≈1e-10 on Nov 24:* with 400-time-unit test references, 1e-10 let S1 stray 1.05e-8 from the exact solution, over the 1e-8 check. 1e-12 gives 1.0e-10 and costs a few seconds.
- Sampling: stored at Δt = 0.05 (implemented Nov 24). `load_split(system, split, dt=…, n_steps=…)` subsamples to a run's Δt; it is the one shared way to do this.
- **Lengths are set in time, not steps** (Nov 24), so every Δt in the sweep up to 0.4 has its full horizon: training and validation trajectories cover 40 time units (100 steps at Δt = 0.4), test references cover 400 (1,000 steps at Δt = 0.4).
- Per system (S1, S2): 500 training trajectories, 100 validation, 100 test initial conditions. Splits use distinct seeds and share no initial condition.
- Test rollouts run 1,000 steps (10× the training horizon).
- Energy ranges: S2 train/val/test on E ∈ [0.1, 1.6], plus an out-of-range test set `test_ood` on E ∈ [1.7, 1.95]. **[PROPOSED]** S1 uses the same [0.1, 1.6], because the proposal gives no S1 range.
- S3 sizes and energy range: **[OPEN]**. `generate.py` refuses S3 until they are set.
- Reject any reference trajectory whose relative energy error exceeds 1e-8, and redraw until the split is full; the count is in `meta.json`. On the Nov 24 data, no trajectory was rejected.
- Format: `data/{system}_{split}.npz` with `states [N, T+1, d]`, `energy [N, T+1]`, `x0 [N, d]` and `t [T+1]`, plus `data/{system}_meta.json` (system, stored Δt, tolerances, energy ranges, seeds, rejection counts). Data is generated, not committed: `python -m whydrift.physics.generate --system s1`, then audit it with `python -m whydrift.physics.checks --system s1`.

## 4. Pipeline

1. **Generate** ground-truth trajectories.
2. **Train** each configuration: model × K × noise × integrator × Δt × 10 seeds.
3. **Roll out** each frozen checkpoint for 1,000 steps, feeding predictions back in.
4. **Evaluate** against the oracle. Rollout error is an *evaluation score*, never a training loss.
5. **Compare factors** one at a time (an ablation-style controlled study).
6. **Test the mechanism (H5):** compute b from one step and correlate it with drift.
7. **Practical test:** choose checkpoints by lowest |b| vs lowest validation loss and compare their drift.

## 5. Models

Every continuous-time model outputs a vector field dx/dt, and an integrator turns it into a step. Only B1 predicts the next state directly.

| Model | Learns | Dynamics | Energy built in | Leapfrog OK | Used for |
|---|---|---|---|---|---|
| B1 residual MLP | the state change per step | x_{t+1} = x_t + h_θ(x_t) | no | n/a | black-box reference |
| M1 Neural ODE | the full vector field | ẋ = f_θ(x) | no | no | H1, H2, H3 |
| M2 acceleration ODE | the acceleration | q̇ = p, ṗ = g_θ(q, p) | no | no | H2a (replicates Gruver 2022) |
| M3 HNN | one energy function | ẋ = J∇H_θ(x) | yes | no (non-separable) | H2 |
| M2-s | acceleration from q only | q̇ = p, ṗ = g_θ(q) | on S1/S2 by construction, not on S3 | yes | H2b, H4 |
| M3-s | a potential | H_θ = ½p² + V_θ(q) | yes | yes | H2b, H4 |
| G3 edge-potential GNN (S3 only) | one shared spring potential | H_θ = Σ½pᵢ² + Σ V_θ(q_{i+1} − q_i) | yes | yes | H2b, graph prior |

**Why M2-s vs M3-s only matters on S3:** with one position variable, any force g(q) is the slope of some potential. So on S1 and S2 the two models are the same function class. On the chain, g_θ(q) can be non-conservative, and that is what H2b tests.

Rules:
- Two-hidden-layer MLPs, parameter counts matched and reported. Optimiser: Adam.
- M3, M3-s and G3 get their field by autodiff of H_θ.
- **Never** use a learned H_θ as the energy metric. Drift is always measured with the true energy.

## 6. Integrators

| Integrator | Order | Long-run energy | Runs on | Role |
|---|---|---|---|---|
| RK4 | 4 | slow drift | all continuous-time models | default |
| Leapfrog (velocity Verlet) | 2 | bounded (symplectic) | separable only: M2-s, M3-s, G3 | H4 |
| Euler | 1 | grows fast (×1.01 per step on S1 at Δt = 0.1) | oracle only | diagnostic |

- Integrators must be differentiable, because training backpropagates through them.
- **B0 oracle** = the true equations stepped with each integrator at each Δt. It separates integrator error from learning error. Build it before reading any model result.
- **[PROPOSED]** retrain at each rollout Δt (matched-integrator rule). **[OPEN]** whether to do this.

## 7. Training

K-step unrolled loss, with Φ_θ = integrator ∘ model:

```
L(θ) = (1/K) Σ_{k=1..K} || x̂_{t+k} − x_{t+k} ||²,   x̂_{t+k} = Φ_θ(x̂_{t+k−1}),   x̂_t = x_t + ε
```

- K = 1 is ordinary one-step training. **[PROPOSED]** K ∈ {1, 5, 10, 20}.
- Noise ε ~ N(0, σ²) on the starting state only, scaled per dimension. σ = 0 is the no-noise arm; the nonzero σ is **[OPEN]**.
- The integrator and Δt used in training match those used at rollout. One mismatch cell: train with RK4, roll out with leapfrog.
- 10 seeds trained in **one vectorised call** (`vmap`).
- Save checkpoints at fixed intervals, not just the best one. H5 needs all of them.
- Standardise inputs and targets with training statistics. Save them, and undo them before computing energy.

## 8. Evaluation

| Metric | Definition | Status |
|---|---|---|
| Valid prediction time | first step where standardised state RMSE > 0.25; on S3, divided by the oracle's | primary, tested |
| Energy drift | \|E(t) − E(0)\| / max(\|E(0)\|, ε), ε = 1% of median training energy | primary, tested |
| One-step MSE | single-step error on test states | input to H1, H5 |
| Rollout error @10/100/500/1,000 + AUC | normalised state error | reported, not tested (saturates) |
| Phase error, blow-up rate | — | reported |
| Gap-closure r | (e_M1 − e_M2) / (e_M1 − e_M3) | H2a |
| Energy bias b, spread s | below | H5 |

**[PROPOSED] energy bias b**, over N test states, with Φ* the exact one-step map:

```
b = mean_i [ E(Φ_θ(x_i)) − E(Φ*(x_i)) ]      s = std_i [ same ]
```

Compute b across the test energy range and also per energy bin. Use Φ* as the reference, not the next data sample. **[OPEN]** confirm this definition against the revised proposal *before* writing evaluator code.

## 9. Hypotheses and pass rules

Rules are encoded in `analysis/hypotheses.py` and **frozen before the main runs**.

| ID | Question | Pass rule |
|---|---|---|
| H1 | Does one-step accuracy predict long-run stability? | Spearman \|ρ\|(one-step MSE, long-run metric) < 0.5, per system (original) |
| H2a | How much of the HNN gain comes from modelling acceleration? | r > 0.5 (original) |
| H2b | When does energy structure start to matter? | **[PROPOSED]** no gap on S1/S2, CI-separated drift gap on S3 |
| H3 | Do multi-step loss and noise reduce drift for every model? | better for every model, largest gain for M1 (original, extended to noise) |
| H4 | At what Δt does integrator error take over? Does leapfrog stay bounded? | **[PROPOSED]** crossover = smallest Δt where B0 drift ≥ 10% of the best learned model's |
| H5 | Does b predict drift better than one-step MSE? Does \|b\| selection help? | **[PROPOSED]** ρ(\|b\|, drift) > ρ(MSE, drift) with a bootstrap CI excluding 0; \|b\|-chosen checkpoints drift less, paired over seeds |

## 10. Architecture

```
 physics/ (S1,S2,S3, oracle) ──train/val──▶ train/ (K-step loss, noise, 10 seeds) ──checkpoints──▶ eval/ (rollouts, metrics, b) ──rows──▶ analysis/ (H1–H5)
          │                                   ▲                                                    ▲
          └────────────────test set, Φ*──────┼────────────────────────────────────────────────────┘
 models/ + integrators.py ── stepper Φθ ──────┴──────────────────────────────────────────────────────┘
```

### Interfaces **[PROPOSED]** — agree these first

```python
# physics/systems.py
class System:
    dim: int                                    # 2 for S1/S2, 16 for S3
    def true_rhs(self, x) -> "dxdt": ...
    def energy(self, x) -> "E": ...
    def sample_x0(self, n, e_range, seed) -> "x0": ...

# models/base.py
def vector_field(params, x) -> "dxdt": ...      # M1, M2, M3, M2-s, M3-s, G3
def is_separable(model) -> bool: ...            # True -> leapfrog allowed

# integrators.py
def step(field, params, x, dt, method) -> "x_next": ...   # 'rk4' | 'leapfrog' | 'euler'

# eval/rollout.py
def rollout(stepper, x0, n_steps) -> "traj [n_ic, n_steps+1, dim]": ...
def evaluate(ckpt, system, split) -> dict: ...  # one metrics row, incl. b and s
```

### Repo layout **[PROPOSED]**

```
whydrift/
  physics/       systems.py, generate.py, checks.py
  models/        b1.py, m1.py, m2.py, m3.py, separable.py, g3.py, base.py
  integrators.py
  train/         loop.py, losses.py, configs/*.yaml
  eval/          rollout.py, metrics.py, bias.py, oracle.py
  analysis/      stats.py, hypotheses.py, figures.py
data/            generated .npz + meta.json (gitignored)
runs/            one folder per config: config, checkpoints, metrics.csv (gitignored)
tests/
```

## 11. Engineering conventions

- **Framework (decided Nov 23):** JAX + Equinox + Optax. `vmap` over seeds (parameters and Adam state together) and gradients through integrators come easily, and `jit` removes Python overhead on these tiny models. Do not mix in PyTorch. Versions are pinned in `pyproject.toml`.
- **Setup:** `python -m venv .venv`, then `.venv\Scripts\python -m pip install -e .[dev]` (Windows) or `.venv/bin/python -m pip install -e .[dev]`. Run `pytest -q` before every pull request. Generate data with `python -m whydrift.physics.generate --system s1` (add `--small` for a quick test set).
- Float64 for the physics and oracle. **[OPEN]** float32 vs float64 for training.
- Every run is fully defined by one config file plus a seed list. No hidden defaults.
- Fixed seeds everywhere. A rerun must reproduce its numbers exactly.
- CPU only. No GPU needed.
- Tests that protect results: the oracle conserves energy to tolerance; RK4 and leapfrog show their expected order on S1; every model returns the right shapes on S1 and S3; fixed-seed runs are reproducible.

## 12. Compute and effort (estimates — replace with pilot timings)

- Models are tiny MLPs and GNNs on 2- to 16-number states, so each run takes minutes on CPU.
- ≈660 core runs ≈ 20–50 CPU-hours, much less with seeds vectorised. One laptop (8–16 GB RAM) is enough.
- People are the bottleneck: ≈115–155 person-hours total, about 40–50 each.
- **Week-1 pilot:** time real runs, confirm data sizes, and check that b is stable across seeds and checkpoints.

## 13. Ownership and timeline

| Owner | Modules |
|---|---|
| Rana | physics and data, integrators, B0 oracle |
| Saad | models, trainer |
| Sami | evaluator, experiments and analysis, figures |

Everyone reviews experiments and writes and defends the report.

| Date | Milestone |
|---|---|
| ~Nov 25 | interfaces and data format frozen **[PROPOSED]** |
| Nov 30 | mid-project update: simulators, clean splits, oracle, simple baseline, first long-rollout curves |
| Dec 1–14 | structured models, matched training comparisons, integrator analysis |
| Dec 15–22 | replicated runs, figures, checks, 5-page report |
| Dec 22 | deliverables: code with fixed seeds, results tables, figures, a verdict per hypothesis |

## 14. Reading list (only what changes a decision)

Must read:
1. Greydanus, Dzamba & Yosinski (2019), *Hamiltonian Neural Networks*, NeurIPS — M3 and the S1/S2 setups.
2. Gruver, Finzi, Stanton & Wilson (2022), *Deconstructing the Inductive Biases of Hamiltonian Neural Networks*, ICLR — closest prior work; H2a replicates it.
3. Chen, Zhang, Arjovsky & Bottou (2020), *Symplectic Recurrent Neural Networks*, ICLR — training through leapfrog, multi-step training.
4. Sanchez-Gonzalez et al. (2020), *Learning to Simulate Complex Physics with Graph Networks*, ICML — training noise (noise section only).
5. Sanchez-Gonzalez, Bapst, Cranmer & Battaglia (2019), *Hamiltonian Graph Networks with ODE Integrators* — prior work for G3.

Skim: Hairer, Lubich & Wanner (2006), *Geometric Numerical Integration*, ch. 1 (H4); Chen et al. (2018), *Neural ODEs*, intro (M1).
Optional: Botev et al. (2021), physics-prior benchmark.

Positioning: our new contributions are H2b, the Δt crossover and H5. H2a is a controlled replication of Gruver et al.

## 15. Open questions (resolve before or during the pilot)

- [ ] Exact definition of b in the revised proposal.
- [ ] Pass rules for H2b, H3 (with noise), H4 and H5.
- [ ] FPUT form (α or β), coupling, boundary conditions, energy range, S3 data sizes.
- [ ] Noise σ and the exact K grid.
- [ ] Δt sweep: retrain at each Δt, or roll out models trained at 0.1?
- [ ] H1's long-run metric: 1,000-step error, or valid prediction time and energy drift?
- [ ] Are there graph models besides G3 (the name suggests G1 and G2)?
- [ ] How the 660 core runs break down.
- [x] JAX or PyTorch: JAX + Equinox + Optax (Nov 23).
- [ ] float32 or float64 for training.

---

## Notes for Claude Code

**How we will work**
1. **Refine first, then build.** Begin by reviewing this spec with the user: challenge the hypotheses, pass rules and the definition of b, and help close the open questions in §15. Update this README as decisions are made. Mark resolved items and remove their **[PROPOSED]**/**[OPEN]** tags.
2. **Explain the why before the how.** Before implementing a module, state the design choice and its trade-offs, then implement.
3. **Build in this order:** `physics/` + tests → `integrators.py` + B0 oracle → `models/` → `train/` → `eval/` (b last, after its definition is agreed) → `analysis/`.
4. **Small, testable steps.** Each module ships with tests before the next one starts.

**Guardrails — do not break these**
- Never change a pass rule after main-run results exist. If one must change, record the change and the reason in this README.
- Never measure drift with a learned energy. Always use `System.energy`.
- Never train on test initial conditions. Keep splits disjoint by energy and seed.
- Never use rollout error as a training loss. It is an evaluation score.
- Keep the integrator used in training identical to the one used at rollout, except in the single designated mismatch cell.
- Mark any number that comes from an estimate rather than a run as an estimate.
