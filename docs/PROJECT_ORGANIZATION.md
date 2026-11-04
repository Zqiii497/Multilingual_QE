# Project organization

## Main evaluation pipeline

- `main.py`: evaluates a selected model/quantization checkpoint on the current multilingual benchmark suite.
- `tools.py`: shared dataset dispatch, model loading, and prompt/sample construction.
- `dataset_loader/`: dataset-specific parsing code.
- `run_gptq.py` and `run_awq.py`: offline quantization entry points.
- `recovery_experiment.py`: separate neuron/layer recovery experiments.
- `analysis_notebook/`: seven research notebooks covering quantization-error subspaces, held-out validation, Transfer/LAPE neurons, functional-neuron properties, answer margins, and layer-wise error sensitivity. These are exploratory research workflows rather than additional benchmark entry points.


## Analysis methods and attribution

- The subspace notebooks use project-designed SVD discovery and intervention protocols; OPUS/Tatoeba is the external corpus source.
- Transfer-Neuron identification is based on Tezuka and Inoue (EMNLP 2025), with project-specific model/layer restrictions and recovery experiments.
- LAPE identification is based on Tang et al. (ACL 2024), with project-specific cross-language pooling and quantization-recovery extensions.
- The answer-margin notebook uses a Logit Lens-style readout; the layer-wise sensitivity notebook uses an activation-intervention design informed by activation-patching literature.

Each notebook's opening section contains method-specific citations and clarifies which parts are adaptations or original project analyses.

## Current benchmark scope

The maintained evaluation scope is KLAR, MCLM, and BELEBELE. Legacy INCLUDE and PIQA datasets and their dedicated loaders have been removed.

The main evaluator and recovery workflow do not have identical dataset support. Check each script's `--dataset_type` choices before launching an experiment.

## Data and generated files

- Keep source datasets and calibration corpora under `data/`.
- Keep model weights under `models/` (ignored by Git).
- Keep generated evaluation/recovery outputs under `results/` or `result/` (ignored by Git).
- Keep exploratory notebooks under `analysis_notebook/`, rather than mixing them with raw dataset files.
