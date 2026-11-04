# Multilingual Quantization Evaluation

This repository compares several quantization methods on **Llama-3.1-8B-Instruct**, **Qwen-3-8B**, and **Gemma-2-9B-it** under a multilingual evaluation setting.

The evaluation focuses on how different post-training quantization methods affect multilingual capabilities across several tasks, including language understanding, knowledge-intensive multiple-choice question answering, and mathematical reasoning.

---

## Evaluation Scope and Datasets

This repository presents an **independently designed multilingual quantization evaluation framework**. Its purpose is to compare how post-training quantization methods affect language-model performance across different multilingual capabilities. The evaluation protocol and experimental comparisons in this repository are project-specific; the datasets and quantization algorithms are established resources from prior work and are cited below.

The selected evaluation suite contains exactly three datasets:

| Dataset | Evaluation task | Answer format | Capability measured |
| ------- | --------------- | ------------- | ------------------- |
| KLAR | Multilingual factual knowledge probing | Free-form answer | Factual knowledge recall across languages |
| MCLM | Multilingual mathematical reasoning | Free-form answer | Mathematical reasoning across languages |
| BELEBELE | Multilingual multiple-choice reading comprehension | A / B / C / D | Reading comprehension across languages |

### KLAR

**KLAR** is a multilingual knowledge-probing dataset designed for autoregressive language models. It uses factual questions whose expected answer appears at the end of the prompt, supporting evaluation of multilingual factual knowledge recall. See the original [ACL 2025 paper](https://aclanthology.org/2025.acl-long.253/).

### MCLM

**MCLM** (Multilingual Competition-Level Math) evaluates mathematical reasoning using competition-level problems across multiple languages. The implementation in this repository evaluates generated answers against the provided reference answers. See [Son et al. (2025)](https://aclanthology.org/2025.acl-long.699/).

### BELEBELE

**BELEBELE** is a parallel multiple-choice machine-reading-comprehension benchmark spanning 122 language variants. Each question is based on a passage and has four candidate answers. This repository evaluates the selected language files available in `data/belebele/`. See [Bandarkar et al. (2024)](https://aclanthology.org/2024.acl-long.44/) and the [official dataset repository](https://github.com/facebookresearch/belebele).

### Languages

The current evaluation configuration covers the following 13 language codes:

```text
ar, ca, el, en, es, fr, he, hu, ja, ko, nl, tr, zh
```

| Code | Language |
| ---- | -------- |
| ar | Arabic |
| ca | Catalan |
| el | Greek |
| en | English |
| es | Spanish |
| fr | French |
| he | Hebrew |
| hu | Hungarian |
| ja | Japanese |
| ko | Korean |
| nl | Dutch |
| tr | Turkish |
| zh | Chinese |

The available sample count can differ by dataset and language. The three datasets cover complementary capabilities, so results should be interpreted per dataset and language rather than treated as interchangeable task scores.

### Evaluation entry points

- `main.py` supports the three selected datasets: `klar`, `mclm`, and `belebele`.
- `recovery_experiment.py` currently does **not** support BELEBELE; its argument choices have not yet been aligned with the selected three-dataset suite. This README does not claim that recovery experiments are available for BELEBELE.

---

## Quantization Methods

The following model precision and quantization methods are used:

| Method            | Precision | Calibration Required |
| ----------------- | --------: | -------------------- |
| FP16              |    16-bit | No                   |
| BitsAndBytes INT8 |     8-bit | No                   |
| BitsAndBytes NF4  |     4-bit | No                   |
| GPTQ INT8         |     8-bit | Yes                  |
| GPTQ INT4         |     4-bit | Yes                  |
| AWQ INT4          |     4-bit | Yes                  |

### FP16

The original FP16 model is used as the full-precision baseline.

### BitsAndBytes

BitsAndBytes performs quantization during model loading and does not require a separate calibration stage.

The following settings are used:

* `bnb-int8`: 8-bit quantization
* `bnb-nf4`: 4-bit NF4 quantization

### GPTQ

GPTQ is a post-training quantization method that uses calibration data.

The following settings are used:

* GPTQ INT8
* GPTQ INT4

### AWQ

AWQ is an activation-aware post-training weight quantization method.

The current experiment uses:

* AWQ INT4

Both GPTQ and AWQ use the multilingual Wikipedia calibration dataset.

---

## Environment

The experiments are conducted in a Linux-based GPU environment.

| Component    | Version          |
| ------------ | ---------------- |
| Python       | 3.12             |
| PyTorch      | 2.12.1+cu130     |
| Transformers | 5.16.1           |
| GPTQ         | GPTQModel        |
| AWQ          | AutoAWQ          |
| Quantization | BitsAndBytes     |
| GPU          | NVIDIA A100 40GB |

---

# Usage

The evaluation dataset is selected through:

```bash
--dataset_type
```

The selected dataset values for the main evaluation runner are:

```text
klar
mclm
belebele
```

## 1. FP16

### KLAR

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

### BELEBELE

```bash
python main.py \
  --dataset_type belebele \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type fp16 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

### MCLM

```bash
python main.py \
  --dataset_type mclm \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type fp16 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 2. BitsAndBytes NF4

BitsAndBytes NF4 directly loads the original model in 4-bit NF4 format.

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type bnb-nf4 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 3. BitsAndBytes INT8

BitsAndBytes INT8 directly loads the original model in 8-bit format.

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-instruct \
  --quant_type bnb-int8 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 4. GPTQ INT4

### Quantize the model

```bash
python run_gptq.py \
  --model_dir models/llama3.1-8b/llama3.1-8b-instruct \
  --save_dir models/llama3.1-8b/llama3.1-8b-gptq-int4-calib \
  --calib_file data/quantization/calibration_wikipedia_multilingual.jsonl \
  --bits 4
```

### Evaluate the quantized model

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-gptq-int4-calib \
  --quant_type gptq-int4 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 5. GPTQ INT8

### Quantize the model

```bash
python run_gptq.py \
  --model_dir models/llama3.1-8b/llama3.1-8b-instruct \
  --save_dir models/llama3.1-8b/llama3.1-8b-gptq-int8-calib \
  --calib_file data/quantization/calibration_wikipedia_multilingual.jsonl \
  --bits 8
```

### Evaluate the quantized model

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-gptq-int8-calib \
  --quant_type gptq-int8 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

---

## 6. AWQ INT4

### Quantize the model

```bash
python run_awq.py \
  --model_dir models/llama3.1-8b/llama3.1-8b-instruct \
  --save_dir models/llama3.1-8b/llama3.1-8b-awq-int4-calib \
  --calib_file data/quantization/calibration_wikipedia_multilingual.jsonl
```

### Evaluate the quantized model

```bash
python main.py \
  --dataset_type klar \
  --checkpoint models/llama3.1-8b/llama3.1-8b-awq-int4-calib \
  --quant_type awq-int4 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 32 \
  --save_path results
```

For all quantization methods, `--dataset_type` can be switched among `klar`, `mclm`, and `belebele` without changing the quantized model.


---

## References

### Evaluation datasets

- Wang, M., Adel, H., Lange, L., Liu, Y., Nie, E., Strötgen, J., and Schütze, H. (2025). [Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models](https://aclanthology.org/2025.acl-long.253/). ACL 2025. (KLAR)
- Son, G., Hong, J., Ko, H., and Thorne, J. (2025). [Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](https://aclanthology.org/2025.acl-long.699/). ACL 2025. (MCLM)
- Bandarkar, L., Liang, D., Muller, B., Artetxe, M., Shukla, S. N., Husa, D., Goyal, N., Krishnan, A., Zettlemoyer, L., and Khabsa, M. (2024). [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants](https://aclanthology.org/2024.acl-long.44/). ACL 2024. (BELEBELE)

### Quantization methods

- Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. (2023). [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323).
- Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, Z., Gan, C., and Han, S. (2024). [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). MLSys 2024.
- Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). (Includes the NF4 quantization format.)

The evaluation framework and the way experiments are organized in this repository are project-specific. Citations above acknowledge the origins of the benchmark datasets and quantization techniques; they do not imply that this repository authored those datasets or quantization algorithms.
