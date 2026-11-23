# Wildfire Containment: MDP-Based Resource Allocation

An AI "fire chief" that decides where to place firebreaks while a wildfire spreads
randomly across a forest grid. The project models wildfire containment as a **Markov
Decision Process (MDP)** and compares simple heuristics against planning and exact
methods.

> **CS272: Artificial Intelligence** · BSAI-2k25 · Fall 2026
> School of Electrical Engineering and Computer Science, NUST

---

## Overview

A forest is a small grid of cells (grass, shrub or forest). A fire starts in one cell and
spreads to neighbouring cells **stochastically**: the chance of ignition depends on the
vegetation of the target cell and on the wind direction. Each turn the agent may build
`k` firebreaks (cells that can never burn) on unburnt cells, or do nothing.

**Goal:** maximise the number of cells saved while accounting for the cost of the
firebreaks used.

The project focuses on AI reasoning (sequential decisions under uncertainty), not graphics.

## Methods

| Method | Role | Status |
|---|---|---|
| Random (B0) | weakest baseline | ✅ done |
| Nearest-to-fire-front (B1) | fixed heuristic rule | ✅ done |
| One-step greedy (B2) | maximises exact expected immediate reward | ✅ done |
| Value iteration / backward DP | exactly optimal policy on tiny grids (gold standard) | ✅ done |
| Monte Carlo Tree Search (UCT) | scalable planner using rollouts | ✅ done |

Experiments will measure decision quality, optimality gap, time per decision, memory
(states stored or tree nodes), scalability (up to 10x10), and robustness to a wrong wind
model or a mid-episode wind shift.

## MDP formulation

| Element | Definition |
|---|---|
| **State** | status of every cell: `U` unburnt, `B` burning, `X` burnt, `F` firebreak |
| **Action** | place `k` firebreaks on distinct unburnt cells, or no-op (`C(n_U, k) + 1` actions) |
| **Transition** | each unburnt cell next to fire ignites independently with `P = 1 − ∏(1 − p_veg × wind)` |
| **Burning time** | one turn, then the cell becomes burnt, so the state graph is acyclic and every episode ends |
| **Reward** | `−(cells that caught fire) − λ × (firebreaks placed)`, with λ = 0.2 |
| **Terminal** | no cell is burning |

Default ignition probabilities: grass 0.6, shrub 0.45, forest 0.3. Wind multipliers:
downwind ×1.5, crosswind ×1.0, upwind ×0.5, with each edge probability capped at 0.95.

## Getting started

```bash
git clone https://github.com/Tayybaa07/Wildfire-Containment.git
cd Wildfire-Containment
pip install -r requirements.txt
python -m pytest -q          # run the unit tests
```

Run everything from the project root so `import wildfire` works.

### Run Stage 4 experiment runner & scalability benchmarks

```bash
python run_stage4.py --quick          # fast smoke test (~30s)
python run_stage4.py                  # full benchmark suites A, B, and C
```
Runs small-grid optimality gaps ($3\times 3, 3\times 4$), MCTS budget sweeps ($N \in \{50 \dots 5000\}$), and scalability runs up to $10\times 10$. Results are exported as raw and summary CSV/JSON files in `results/stage4/`.

### Run Stage 2 validation & benchmarks

```bash
python run_stage2.py
```
Validates theoretical $V^*(s_0)$ against 1,500 Monte Carlo rollouts, runs grid scaling ($3\times 3$, $3\times 4$, $4\times 4$), and exports Stage 3 handover reference tables to `results/`.

### Compare the baselines

```bash
python demo.py --rows 5 --cols 5 --wind E --episodes 500
```

Example output (5x5 mixed vegetation, east wind, 500 episodes, seed 0):

```
agent                       return     cells saved  firebreaks
NoOp                  -7.27 ±0.55     16.73 ±0.55        0.00
B0-Random             -5.81 ±0.41     18.97 ±0.37        3.92
B1-NearestFront       -2.79 ±0.25     21.74 ±0.23        2.62
B2-Greedy             -2.21 ±0.23     22.25 ±0.20        2.26
```
(means with 95% confidence intervals)

### Watch a single episode

```bash
python demo.py --show greedy --seed 3 --rows 4 --cols 4 --wind E
```

### Use it from Python

```python
from wildfire import make_config, WildfireEnv, GreedyAgent, run_episode

cfg = make_config(5, 5, vegetation="mixed", wind="E", k=1, lam=0.2)
env = WildfireEnv(cfg)
result = run_episode(env, GreedyAgent(), seed=0, verbose=True)
print(result)   # return, burnt, saved, firebreaks, steps
```

## Project structure

```
wildfire/
  config.py       scenario configuration (grid, vegetation, wind, lambda, k)
  env.py          the MDP: reset, legal_actions, step, transition_probs
  agents.py       NoOp, Random, NearestFront, Greedy
  exact_solver.py exact solver (Backward DP & Value Iteration), ExactAgent
  mcts.py         Monte Carlo Tree Search (UCT) agent
  experiments.py  experiment runner, timing, gap tracking, structured CSV/JSON
  simulate.py     episode runner, evaluation, confidence intervals
  visualize.py    console rendering of the grid
demo.py           baseline comparison script
run_stage2.py     exact solver validation & scaling benchmarks
run_stage4.py     master experiment runner & scalability benchmark suites
tests/            pytest unit tests (env, agents, exact_solver, mcts, experiments)
docs/             design notes (STAGE1_NOTES, STAGE2_NOTES, STAGE4_NOTES)
results/          reference tables and stage4 benchmark data
```

## Interface (for solver and agent code)

- **State:** flat tuple of length `R*C` with entries `U=0, B=1, X=2, F=3`
- **Action:** sorted tuple of flat cell indices; `()` is the no-op
- **`env.step(state, action, rng)`** → `(next_state, reward, done)`
- **`env.transition_probs(state, action)`** → exact list of `(prob, next_state, reward)`
  (used by value iteration)
- **`agent.select_action(state, env, rng)`** → a legal action (same signature for every agent)

More detail and the modelling decisions are in [`docs/STAGE1_NOTES.md`](docs/STAGE1_NOTES.md).

## Reproducibility

- Every episode is determined by an integer seed.
- Common random numbers: the same seed gives every agent the same ignition cell and the
  same environment randomness, so comparisons are fair.
- All scenario settings live in one `Config` object, which makes runs easy to log.
- Unit tests check the dynamics, the legality of agent actions, and that the exact
  transition model matches the sampled simulator.

## Roadmap

- [x] **Stage 1 (Zainab):** environment, simulator, baselines B0 to B2, tests
- [x] **Stage 2 (Tayyba):** exact solver (value iteration) on 3x3 and 3x4 grids
- [x] **Stage 3 (Basma):** MCTS (UCT) agent and optimality gap against value iteration
- [x] **Stage 4 (Tayyba):** experiment runner, rollout-budget sweeps, scalability
- [x] **Stage 5 (Zainab, Basma):** 5a robustness experiments; 5b failure-case analysis
- [x] **Stage 6 (Basma, Zainab):** 6a statistics and plots; 6b report and presentation

## Team

| Member | CMS ID | Stages |
|---|---|---|
| Zainab Parveen | 550827 | 1, 5a, 6b: environment and baselines; robustness experiments; report and presentation |
| Tayyba Bashir | 541682 | 2, 4: exact solver (value iteration); experiment runner and scaling |
| Basma Hussain | 569690 | 3, 5b, 6a: MCTS agent; failure-case analysis; statistics and plots |


## AI assistance declaration

Generative-AI tools were used for brainstorming the proposal. All code is written, tested and understood by the group members.
It has been once reviewed by AI_Tool too. 

## References

- Russell, S. and Norvig, P. *Artificial Intelligence: A Modern Approach* (MDPs, value iteration, Monte Carlo methods)
- Kocsis, L. and Szepesvári, C. "Bandit Based Monte-Carlo Planning" (UCT)
