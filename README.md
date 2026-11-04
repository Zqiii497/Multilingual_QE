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

`recovery_experiment.py` is a separate experimental workflow for neuron/layer recovery analysis. Its currently supported dataset choices are narrower than the main evaluator; check its command-line arguments before running. Do not assume BELEBELE is supported by this workflow.

## Notes

- Place model checkpoints under `models/` or provide their paths through command-line arguments.
- Evaluation outputs are written to the directory specified by `--save_path`.
- The three datasets above define the maintained evaluation scope.
