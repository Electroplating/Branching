# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language convention

The project's working language is **Chinese**. All docstrings, code comments, `README.md`, `plan.md`, and `AGENTS.md` are written in Chinese; only `__all__`, identifiers, and type annotations are English. Preserve this style when editing or adding code. A full Chinese deep-dive lives in `AGENTS.md` — consult it for exhaustive module-by-module detail beyond this file.

## What this is

`omt_branching` is a research prototype: a GNN policy framework for **OMT (Optimization Modulo Theories) branching selection**. It predicts which Boolean variable to branch on, its phase, integer B&B split choice/direction, plus auxiliary signals (conflict probability, unsat-core membership, objective improvement, subtree size). It is **solver-agnostic** — no real solver integration code exists yet; everything is exercised through a synthetic demo and synthetic experiments.

## Commands

PyTorch is **not** in the base environment. Use the `omt` conda env (or a venv with `requirements.txt`) or commands will fail with `ModuleNotFoundError: No module named 'torch'`.

```bash
conda activate omt                      # or: pip install -r requirements.txt  (torch>=2.0, numpy>=1.21)

python -m examples.demo                 # end-to-end smoke test (build graph -> infer -> decode -> 1 train step + 1 REINFORCE step)
python -m experiments.run_experiments   # full synthetic ablation suite -> experiments/results/*.json
python -m experiments.run_experiments <exp_name>   # single named experiment
python -m experiments.run_quick_experiments        # fast sanity run (small hidden/layers/epochs)
python -m experiments.analyze_results   # print result tables from saved JSON
python -m experiments.plot_results      # training curves + comparison bars
```

There is **no test suite, no `pyproject.toml`/`setup.py`, no lint/CI config, no Makefile**. `python -m examples.demo` is the primary correctness check — it should print a `HeteroGraph.summary()`, a populated `BranchingAdvice`, and run training steps without error.

## Architecture

The codebase is split into three decoupled parts joined by two **stable dataclass contracts**. The solver only ever touches the contracts, never the model internals.

```
SolverSnapshot ──▶ GraphBuilder ──▶ HeteroGraph ──▶ InferenceEngine ──▶ PolicyOutput ──▶ AdviceDecoder ──▶ BranchingAdvice
   (input/                                            (model/                              (output/
    solver_state.py)                                   policy.py,inference.py)              decoder.py)
```

- **Input contract** `SolverSnapshot` (`omt_branching/input/solver_state.py`): the solver fills this once per consultation. `GraphBuilder` (`graph_builder.py`) encodes it into a `HeteroGraph` and keeps `id_maps` translating solver IDs ↔ local indices.
- **Model** (`omt_branching/model/`): `HeteroEncoder` (R-GCN-style, `gnn.py`) + task `heads.py`, combined in `BranchingPolicy` (`policy.py`) producing `PolicyOutput`. `trainer.py` = offline imitation learning (ListNet-style ranking loss); `finetune.py` = solver-in-the-loop DAgger/REINFORCE; `inference.py` = deployment-time gating.
- **Output contract** `BranchingAdvice` (`omt_branching/output/advice.py`): activity priors, candidate ranking, phase, integer split, confidence, and a `fallback` flag. `AdviceDecoder` maps local indices back to solver IDs.
- **Facade** `BranchingPolicyService.advise(snapshot)` (`service.py`) is the single entry point the solver holds; it wires GraphBuilder → InferenceEngine → AdviceDecoder. Top-level exports are in `omt_branching/__init__.py`.

Recommended integration (not implemented here) is **VSIDS refocus**: call `advise()` periodically (after restart / every N conflicts), mix `advice.activity_priors` into native SAT activity, and fall back to native heuristics when `advice.fallback is True`. This preserves solver completeness and keeps inference cost low.

## Invariants that span multiple files (read before editing)

- **Feature-dimension coupling.** In `graph_builder.py`, `_NODE_DIMS` / `_EDGE_DIMS`, the `_encode_*` functions, and `FeatureSpec` must agree. `_check()` asserts dimensions at runtime per node. Changing any feature requires updating all three in lockstep.
- **Enum/schema coupling.** Node/edge/atom/clause/search-mode enums live in `interfaces.py` with `EDGE_SCHEMA`; each enum's `all()` classmethod guarantees stable one-hot indices. Adding a type means updating the enum, `EDGE_SCHEMA`, and the corresponding encoder/dims.
- **Local indices, not solver IDs.** Training labels (`RankingExample`) and `PolicyOutput` use **graph-local indices**. Only `AdviceDecoder` restores solver IDs. Don't mix the two.
- **No PyG/DGL.** All graph ops (`index_add_` + mean aggregation, `HeteroGraph` tensor buckets) are hand-written for minimal deps. Migrating to PyG/DGL would require rewriting the `HeteroEncoder`↔`HeteroGraph` interface.
- **Missing-value encoding conventions.** Optional scalars → `[value, present_flag]`; tri-state booleans → one-hot `[none, true, false]`; large-scale numerics pass through `log1p` / signed-log / clamping.

## Config-as-dataclass

Behavior is controlled by dataclasses, not globals: `PolicyConfig`, `TrainConfig`, `FinetuneConfig`, `InferenceConfig`, `DecoderConfig`, `ServiceConfig`, `FeatureSpec`. `InferenceEngine` gates on `max_total_nodes` (size), `time_budget_ms` (time), and `min_confidence`; on failure it returns `None`, writes diagnostics to `graph.meta["inference"]`, and the decoder emits `use_gnn=False` for fallback. Device is a config string threaded through `HeteroGraph.to()`, `BranchingPolicy.to()`, and the engine.

## Pre-edit checklist

1. Synced the node/edge dims in `FeatureSpec` / `_NODE_DIMS` / `_EDGE_DIMS` with the encoders?
2. Updated `interfaces.py` enums and `EDGE_SCHEMA` together?
3. Kept docstrings/comments in Chinese?
4. Verified end-to-end with `python -m examples.demo`?
5. Exported any new public API in `omt_branching/__init__.py`?
