# Project organization

## Main evaluation pipeline

- `main.py`: evaluates a selected model/quantization checkpoint on the current multilingual benchmark suite.
- `tools.py`: shared dataset dispatch, model loading, and prompt/sample construction.
- `dataset_loader/`: dataset-specific parsing code.
- `run_gptq.py` and `run_awq.py`: offline quantization entry points.
- `recovery_experiment.py`: separate neuron/layer recovery experiments.
- `analysis_notebook/`: exploratory and post-hoc analysis notebooks.

## Current benchmark scope

The maintained evaluation scope is KLAR, MCLM, and BELEBELE. Legacy INCLUDE and PIQA datasets and their dedicated loaders have been removed.

The main evaluator and recovery workflow do not have identical dataset support. Check each script's `--dataset_type` choices before launching an experiment.

## Data and generated files

- Keep source datasets and calibration corpora under `data/`.
- Keep model weights under `models/` (ignored by Git).
- Keep generated evaluation/recovery outputs under `results/` or `result/` (ignored by Git).
- Keep exploratory notebooks under `analysis_notebook/`, rather than mixing them with raw dataset files.
