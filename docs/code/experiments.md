## Experiment System Guide

This document explains how to organize configs, define experiment specs, run sequential sweeps, and use background execution.

### What it does
- Runs a series of training or simulation runs sequentially (prevents VRAM overload)
- Supports parameter sweeps (full grid) or explicit lists of configs
- Saves per-run resolved configs for reproducibility
- Tracks metadata (status/timestamps/return codes) and links produced `run_dir`s
- Supports resume and background execution

### Key scripts and files
- `scripts/run_experiment.py`: Orchestrator (foreground)
- `bg_experiment.sh`: Background launcher for experiments
- `scripts/run_background.py`: Background manager (now supports `--experiment`)

Outputs:
- `experiments/<name>_<timestamp>/`
  - `configs/run_XXX.yaml`: resolved per-run configs
  - `experiment.json`: metadata (status, timestamps, return codes, discovered `run_dir`s)

## Organizing configs and specs
- Keep small single-run base configs in `configs/`:
  - `configs/mlp/mlp_mnist.yaml`
  - `configs/greedy/greedy_mnist.yaml`
  - etc.
- Keep experiment specs in `experiments/specs/`:
  - `experiments/specs/lr_sweep_mnist_mlp.yaml`
  - `experiments/specs/lambda_ace_sweep_greedy.yaml`

Tips:
- Use base configs for shared defaults (batch size, workers, etc.)
- Keep model/dataset-specific keys in model folders (e.g., `greedy/*` with `lambda_ace`, `ace_variant`)
- Avoid duplicating many config variants; the orchestrator writes per-run configs for you

## Experiment spec schema
Provide either a base config with a `sweep` or a list of explicit configs.

Required fields:
- `name` (string): experiment name
- `mode` (string): `training` or `simulation`

One of:
- `base_config` (string path) + optional `sweep` + optional `fixed_overrides`
- `configs` (list of string paths)

Optional:
- `fixed_overrides` (dict): applied on top of base before sweep

### Examples
Learning-rate sweep (simulation):
```yaml
name: lr_sweep_mnist_mlp
mode: simulation
base_config: configs/mlp/mlp_mnist.yaml
sweep:
  lr: [0.0003, 0.001, 0.003, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2]
fixed_overrides:
  seed: 0
```

Grid sweep over strings:
```yaml
name: sched_sim_mnist_mlp
mode: simulation
base_config: configs/mlp/mlp_mnist.yaml
sweep:
  scheduler: [cosine, constant]
  similarity: [cosine, l2]
```

Explicit list of configs (useful when changing model/dataset):
```yaml
name: mixed_configs
mode: training
configs:
  - configs/mlp/mlp_mnist.yaml
  - configs/greedy/greedy_mnist.yaml
```

## How sweeps work
- The orchestrator builds a full cartesian product (grid) across keys under `sweep:`
- Values are treated literally (numbers, strings, booleans, lists, etc.)
- Overrides replace whole top-level keys

### Strings
Put literal strings in the sweep list:
```yaml
sweep:
  scheduler: [cosine, constant]
```
Note: Quote YAML-reserved tokens if needed (`"on"`, `"off"`, `"yes"`, `"no"`).

### Lists in base configs
- Lists in your base configs (e.g., `lambda_ace`) remain plain lists and are not treated as sweeps
- They change only if you explicitly override them in the sweep

### Sweeping list-valued parameters (list-of-lists)
Provide a list-of-lists under the parameter key. Each inner list is one candidate value:
```yaml
name: lambda_sweep_greedy
mode: training
base_config: configs/greedy/greedy_mnist.yaml
sweep:
  lambda_ace:
    - [0.001, 0.001, 0.001]
    - [0.01, 0.01, 0.01]
  lr: [0.0005, 0.001]
```
Produces 2 × 2 = 4 runs. Ensure `lambda_ace` length equals `layers - 1` for the greedy model.

### Coupled parameters
The sweep is a full grid, not a zip. If values must be paired (e.g., specific `layers` with matching `lambda_ace` lengths), either:
- Use `configs:` with explicit files for each pairing, or
- Run separate experiment specs per pairing

## Running experiments
Foreground:
```bash
python3 scripts/run_experiment.py --exp experiments/specs/lr_sweep_mnist_mlp.yaml
```

Background (detach from SSH):
```bash
./bg_experiment.sh experiments/specs/lr_sweep_mnist_mlp.yaml simulation
python3 scripts/run_background.py --list
```

Resume and limit:
```bash
python3 scripts/run_experiment.py --exp experiments/specs/lr_sweep_mnist_mlp.yaml --resume --limit 3
```

Simulation flags (forwarded when `mode: simulation`): `--show`, `--research_only`, `--no_confusion`, `--advanced`.

## Metadata and outputs
Inside `experiments/<name>_<timestamp>/experiment.json` the orchestrator stores for each run:
- `status`: `pending` | `running` | `completed` | `failed`
- `started_at`, `completed_at`, `return_code`
- `config_path`: resolved per-run YAML
- `run_dir`: discovered directory under `runs/<dataset>/<model>/...` (after successful run)

Per-run configs are written to `experiments/<name>_<timestamp>/configs/` and are the exact inputs used for each run.

## Background monitoring
Use the existing background manager:
```bash
python3 scripts/run_background.py --list
tail -f background_runs/<...>/nohup.log
```
You can also stop a background process by its run directory with `--stop`.

## Common pitfalls
- Changing `model` in sweeps can break validators (e.g., greedy requires `lambda_ace` length = `layers - 1`). Prefer `configs:` for model/dataset switches.
- For YAML, quote reserved tokens when you intend strings.
- Overrides replace entire keys; partial list element updates aren’t supported—provide the full desired list.


