# LaF-MCTS Final Generated Solver

This folder provides the **final executable decomposition-enhanced HGS solver selected by LaF-MCTS**. It is intended to accompany the paper as a reproducibility artifact. The released solver does **not** invoke an LLM or MCTS at deployment time: the three learned components are fixed and directly assembled into the HGS solver.

## What is included

- `final_solver/generated/main_configuration.py` — final Tier-1 HGS/global configuration (`MC_0.py`).
- `final_solver/generated/decomposition_strategy.py` — final Tier-2 decomposition strategy (`DE_0.py`).
- `final_solver/generated/subsolver_configuration.py` — final Tier-3 recursive sub-solver configuration (`SC_0.py`).
- `final_solver/core/dehgs.py` — decomposition-enhanced HGS implementation extracted from the authors' experimental code.
- `final_solver/core/decomposition.py` — minimal adapter required by the learned decomposition strategy.
- `run_final_solver.py` — command-line entry point that assembles and runs all three generated components as one solver.

The generated component logic and parameter values are preserved from the selected final LaF-MCTS solver. The only integration-level change in `core/dehgs.py` is that the solver now respects the runtime passed by the command line instead of the source-code debugging constant.

## Installation

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

The implementation uses the genetic-algorithm PyVRP API used by the original experimental code. `requirements.txt` therefore constrains PyVRP to a pre-0.13 release because PyVRP 0.13 removed the genetic-algorithm internals used here. For the archival GitHub release, it is still best to replace this range with the **exact PyVRP version from the authors' experiment environment** (e.g., from `pip freeze`).

## Run on a CVRPLIB instance

If a `.sol` file is available, it is used to determine the number of vehicles and optionally the BKS:

```bash
python run_final_solver.py \
  --instance path/to/X-n101-k25.vrp \
  --solution path/to/X-n101-k25.sol \
  --runtime 242.4 \
  --output X-n101-k25_result.json
```

If no solution file is supplied, specify the vehicle count explicitly:

```bash
python run_final_solver.py \
  --instance examples/X-n501-k38.vrp \
  --vehicles 38 \
  --runtime 1202.4
```

When `--runtime` is omitted, the default is `N * 2.4` seconds, where `N` is the number of customers, matching the evaluation budget described in the manuscript.

## Exact experimental distance matrices

For strict reproduction, the recommended interface is an NPZ file containing the exact arrays used by the experiment:

- `coordinates`
- `dist_matrix`
- `demands`
- `capacity`
- optional `num_vehicles`
- optional `opt_cost`

Run it as:

```bash
python run_final_solver.py --instance-npz instance.npz
```

This avoids any ambiguity from benchmark parsing or distance rounding. The lightweight `.vrp` loader supports standard coordinate-based `EUC_2D` instances.

## Dry run

To verify that the input and final learned components are loaded correctly without importing PyVRP:

```bash
python run_final_solver.py \
  --instance examples/X-n501-k38.vrp \
  --vehicles 38 \
  --dry-run
```

