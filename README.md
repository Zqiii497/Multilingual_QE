# Multilingual Quantization Evaluation

An independently designed framework for evaluating the impact of post-training quantization on multilingual language-model performance.

## Evaluation setup

**Models**
- Llama-3.1-8B-Instruct
- Qwen-3-8B
- Gemma-2-9B-it

**Datasets**

| Dataset | Task | Evaluation |
| --- | --- | --- |
| [KLAR](https://aclanthology.org/2025.acl-long.253/) | Multilingual factual knowledge probing | Answer-based accuracy |
| [MCLM](https://aclanthology.org/2025.acl-long.699/) | Multilingual mathematical reasoning | Generated-answer evaluation |
| [BELEBELE](https://github.com/facebookresearch/belebele) | Multilingual reading comprehension | Four-choice accuracy |

The current language configuration is: `ar, ca, el, en, es, fr, he, hu, ja, ko, nl, tr, zh`. Available languages and sample counts depend on the dataset files.

**Quantization methods**
- FP16 baseline
- BitsAndBytes INT8 / NF4
- GPTQ INT8 / INT4
- AWQ INT4

GPTQ and AWQ use the multilingual Wikipedia calibration corpus under `data/quantization/`.

## Repository layout

```text
.
├── analysis_notebook/   # Analysis and neuron/recovery notebooks
├── data/
│   ├── klar/            # KLAR evaluation data
│   ├── mclm/            # MCLM evaluation data
│   ├── belebele/        # BELEBELE evaluation data
│   └── quantization/    # Quantization calibration data
├── dataset_loader/      # Dataset readers
├── docs/                # Project notes
├── main.py              # Quantization evaluation
├── recovery_experiment.py
├── run_awq.py           # AWQ quantization
├── run_gptq.py          # GPTQ quantization
└── tools.py             # Shared loading and sample utilities
```

## Quick start

Run an evaluation with `main.py`:

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type fp16 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

Set `--dataset_type` to `klar`, `mclm`, or `belebele`. Change `--checkpoint` and `--quant_type` to evaluate another model or quantization configuration. See the argument definitions in each script for optional settings.

## Recovery experiments

`recovery_experiment.py` evaluates whether restoring selected parts of a quantized model to FP16 improves multilingual performance. It compares the quantized baseline with layer-group and component-level recovery experiments.

**Supported datasets:** `klar`, `mclm`. BELEBELE is not supported by this workflow.

### 1. Prepare checkpoints

You need both:
- **Quantized checkpoint**: the GPTQ/AWQ/BitsAndBytes model to evaluate.
- **FP16 checkpoint**: the corresponding original full-precision model used as the source of recovered modules.

The two checkpoints must have the same model architecture and layer layout. The FP16 reference is loaded on CPU; make sure the machine has enough system RAM as well as GPU memory for the quantized model and evaluation.

### 2. Run directly

Example for Llama 3.1 8B with GPTQ INT4:

```bash
python recovery_experiment.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-gptq-int4 \
  --fp16_checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type gptq-int4 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --max_new_tokens 8 \
  --stages 1,2 \
  --save_path result/recovery \
  --resume true \
  --calculate_ppl false \
  --seed 42
```

Change `--dataset_type` to `mclm` to run the MCLM evaluation. Set both checkpoint paths to the matching quantized and FP16 checkpoints for the model being tested, and set `--quant_type` / `--model_type` to match that model and quantization method.

### 3. Recovery stages

- **Stage 1** (`--stages 1`): restore Attention and MLP together in each quarter of the model (0–25%, 25–50%, 50–75%, and 75–100%).
- **Stage 2** (`--stages 2`): separately restore Attention or MLP in the first quarter (0–25%).
- **Both** (`--stages 1,2`): run the baseline and all six recovery settings.

The script first evaluates the quantized baseline. Each recovery intervention is then applied to the same quantized model and rolled back before the next experiment. AWQ uses coupled-component recovery to account for its folded activation-aware scales.


## Analysis notebooks

The notebooks in `analysis_notebook/` contain exploratory analyses beyond the main benchmark and recovery scripts:

- `01_subspace_discovery_recovery.ipynb`: Tatoeba-derived quantization-error subspaces and KLAR oracle-projection experiments.
- `02_subspace_heldout_validation.ipynb`: fact-disjoint KLAR discovery and held-out subspace validation.
- `03_transfer_neuron_discovery_recovery.ipynb`: Transfer-Neuron identification and quantized contribution-replacement experiments.
- `04_lape_neuron_discovery_recovery.ipynb`: LAPE/Transfer neuron recovery comparisons on quantization-error cases.
- `05_functional_neuron_analysis_full_layer_lape_updated.ipynb`: functional-neuron properties and paired FP16/quantized comparisons.
- `06_answer_margin_analysis.ipynb`: answer-margin and layer-wise vocabulary-readout analysis.
- `07_layerwise_error_sensitivity.ipynb`: local block-level quantization-error and answer-margin sensitivity analysis.

Some notebooks adapt methods from prior work; their introductions identify the relevant papers and distinguish the original methods from this project's experimental extensions. These notebooks are research analyses, not additional benchmark entry points.

## Notes

- Place model checkpoints under `models/` or provide their paths through command-line arguments.
- Evaluation outputs are written to the directory specified by `--save_path`.
- The three datasets above define the maintained evaluation scope.
