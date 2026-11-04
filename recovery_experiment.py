"""Layer/module precision-recovery experiments for supported datasets."""

import argparse
import copy
import gc
import json
import os
import random
from types import SimpleNamespace

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import main as evaluation
from tools import build_samples, get_dataset, load_model as load_default_model


ATTENTION_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)

MLP_MODULES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

BLOCK_MODULES = ATTENTION_MODULES + MLP_MODULES

# AutoAWQ folds its activation-aware scales into coupled operators.  For
# Llama-like blocks, q/k/v are coupled to input_layernorm, o_proj is coupled
# to v_proj, gate/up are coupled to post_attention_layernorm, and down_proj is
# coupled to up_proj.  Replacing only an individual FP16 projection therefore
# mixes two incompatible parameterizations.  Recover these complete coupled
# components instead.
AWQ_ATTENTION_COMPONENTS = ("input_layernorm", "self_attn")
AWQ_MLP_COMPONENTS = ("post_attention_layernorm", "mlp")
AWQ_RECOVERY_STRATEGY = "awq_coupled_components_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FP16 module-recovery experiments on a quantized model."
    )
    parser.add_argument(
        "--dataset_type",
        required=True,
        choices=("klar", "mclm"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fp16_checkpoint", required=True)
    parser.add_argument("--quant_type", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--languages", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--enable_thinking",
        type=evaluation.str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--calculate_ppl",
        type=evaluation.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Calculate answer-token PPL for KLAR (default: False).",
    )
    parser.add_argument(
        "--stages",
        default="1,2",
        help=(
            "Comma-separated stages: 1 restores Attention+MLP in each "
            "quarter; 2 compares Attention and MLP within the first "
            "quarter (default: 1,2)."
        ),
    )
    parser.add_argument(
        "--save_path",
        default="results",
        help="Root result directory used by the recovery runner.",
    )
    parser.add_argument(
        "--resume",
        type=evaluation.str2bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Reuse valid per-language JSON files and continue from the first "
            "unfinished language (default: True). Use --resume false to rerun "
            "and overwrite every language."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_attribute(root, dotted_path):
    current = root
    for name in dotted_path.split("."):
        if not hasattr(current, name):
            raise AttributeError(
                f"{type(current).__name__} has no attribute {name!r} "
                f"while resolving {dotted_path!r}."
            )
        current = getattr(current, name)
    return current


def get_parent_and_name(root, dotted_path):
    parts = dotted_path.split(".")
    parent = root
    for name in parts[:-1]:
        if not hasattr(parent, name):
            raise AttributeError(
                f"Cannot resolve recovery module {dotted_path!r}: "
                f"missing {name!r}."
            )
        parent = getattr(parent, name)
    return parent, parts[-1]


def find_decoder_layers(model):
    candidate_paths = (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "transformer.blocks",
    )
    for path in candidate_paths:
        try:
            layers = get_attribute(model, path)
        except AttributeError:
            continue
        if hasattr(layers, "__len__") and len(layers) > 0:
            return layers, path
    raise ValueError(
        "Cannot locate decoder layers. Add this model architecture to "
        "find_decoder_layers()."
    )


def make_quarter_layer_groups(num_layers):
    """Split layer indices at 0%, 25%, 50%, 75%, and 100%."""
    if num_layers < 4:
        raise ValueError(
            f"Cannot split {num_layers} layers into four non-empty groups."
        )

    boundaries = [num_layers * index // 4 for index in range(5)]
    return [
        list(range(boundaries[index], boundaries[index + 1]))
        for index in range(4)
    ]


def make_experiments(groups, stages):
    experiments = [{
        "name": "quant_baseline",
        "stage": 0,
        "recovery_level": "baseline",
        "layers": [],
        "modules": [],
    }]

    if 1 in stages:
        percentage_ranges = ("0_25", "25_50", "50_75", "75_100")
        for index, (group, percentage_range) in enumerate(
            zip(groups, percentage_ranges)
        ):
            experiments.append({
                "name": f"layer_group_{percentage_range}_attention_mlp",
                "stage": 1,
                "recovery_level": "layer_group",
                "layer_group_index": index,
                "percentage_range": percentage_range.replace("_", "-") + "%",
                "layers": group,
                "modules": list(BLOCK_MODULES),
            })

    if 2 in stages:
        first_quarter = groups[0]
        for component_name, modules in (
            ("attention", ATTENTION_MODULES),
            ("mlp", MLP_MODULES),
        ):
            experiments.append({
                "name": f"component_0_25_{component_name}",
                "stage": 2,
                "recovery_level": "component",
                "layer_group_index": 0,
                "percentage_range": "0-25%",
                "component": component_name,
                "layers": first_quarter,
                "modules": list(modules),
            })

    return experiments


def load_fp16_reference(checkpoint):
    print(f"Loading FP16 reference model on CPU: {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def module_device(module):
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def clone_fp16_module(source_module, target_device):
    recovered_module = copy.deepcopy(source_module)
    recovered_module.to(device=target_device, dtype=torch.float16)
    recovered_module.eval()
    return recovered_module


def module_parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def recover_awq_coupled_components(
    quant_layers,
    reference_layers,
    layer_indices,
    module_paths,
    swapped_modules,
):
    """Restore scale-consistent Llama-like components in an AWQ model."""
    requested = set(module_paths)
    restore_attention = bool(requested.intersection(ATTENTION_MODULES))
    restore_mlp = bool(requested.intersection(MLP_MODULES))
    if not restore_attention and not restore_mlp:
        raise ValueError(
            "AWQ recovery did not request an Attention or MLP projection."
        )

    recovered_parameters = 0
    recovered_modules = 0
    recovered_module_paths = []

    for layer_index in layer_indices:
        quant_layer = quant_layers[layer_index]
        reference_layer = reference_layers[layer_index]
        target_device = module_device(quant_layer)

        if restore_attention and restore_mlp:
            # Restoring the whole block is both simpler and safer: it preserves
            # every scale-coupled boundary introduced by AutoAWQ.
            recovered_layer = clone_fp16_module(reference_layer, target_device)
            recovered_parameters += module_parameter_count(reference_layer)
            recovered_modules += 1
            swapped_modules.append(
                ("item", quant_layers, layer_index, quant_layer)
            )
            quant_layers[layer_index] = recovered_layer
            recovered_module_paths.append(f"layers.{layer_index}")
            continue

        component_paths = (
            AWQ_ATTENTION_COMPONENTS
            if restore_attention
            else AWQ_MLP_COMPONENTS
        )
        for component_path in component_paths:
            source_module = get_attribute(reference_layer, component_path)
            target_parent, target_name = get_parent_and_name(
                quant_layer, component_path
            )
            target_module = getattr(target_parent, target_name)
            component_device = module_device(target_module)
            # Parameter-free modules can report CPU even when the containing
            # block is dispatched to a GPU.
            if component_device.type == "cpu" and target_device.type != "cpu":
                component_device = target_device
            recovered_module = clone_fp16_module(
                source_module, component_device
            )
            recovered_parameters += module_parameter_count(source_module)
            recovered_modules += 1
            swapped_modules.append(
                ("attribute", target_parent, target_name, target_module)
            )
            setattr(target_parent, target_name, recovered_module)
            del target_module
            recovered_module_paths.append(
                f"layers.{layer_index}.{component_path}"
            )

    return {
        "recovered_modules_count": recovered_modules,
        "recovered_parameters": recovered_parameters,
        "recovered_module_paths": recovered_module_paths,
        "recovery_strategy": AWQ_RECOVERY_STRATEGY,
    }


def recover_modules(
    quant_model,
    reference_model,
    layer_indices,
    module_paths,
    quant_type,
    swapped_modules,
):
    quant_layers, quant_layer_path = find_decoder_layers(quant_model)
    reference_layers, reference_layer_path = find_decoder_layers(reference_model)
    if len(quant_layers) != len(reference_layers):
        raise ValueError(
            "Quantized and FP16 models have different layer counts: "
            f"{len(quant_layers)} vs {len(reference_layers)}."
        )

    if "awq" in quant_type.lower():
        recovery = recover_awq_coupled_components(
            quant_layers,
            reference_layers,
            layer_indices,
            module_paths,
            swapped_modules,
        )
        recovery.update({
            "quant_layer_path": quant_layer_path,
            "reference_layer_path": reference_layer_path,
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return recovery

    recovered_parameters = 0
    recovered_modules = 0
    for layer_index in layer_indices:
        quant_layer = quant_layers[layer_index]
        reference_layer = reference_layers[layer_index]

        for module_path in module_paths:
            target_parent, target_name = get_parent_and_name(
                quant_layer, module_path
            )
            if not hasattr(target_parent, target_name):
                raise AttributeError(
                    f"Layer {layer_index} under {quant_layer_path} does not "
                    f"contain {module_path!r}."
                )

            source_module = get_attribute(reference_layer, module_path)
            target_module = getattr(target_parent, target_name)
            target_device = module_device(target_module)

            recovered_module = clone_fp16_module(
                source_module, target_device
            )

            recovered_parameters += sum(
                parameter.numel()
                for parameter in source_module.parameters()
            )
            recovered_modules += 1
            swapped_modules.append(
                ("attribute", target_parent, target_name, target_module)
            )
            setattr(target_parent, target_name, recovered_module)
            del target_module

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "quant_layer_path": quant_layer_path,
        "reference_layer_path": reference_layer_path,
        "recovered_modules_count": recovered_modules,
        "recovered_parameters": recovered_parameters,
        "recovery_strategy": "projection_swap_v1",
    }


def restore_quantized_modules(swapped_modules):
    """Undo one recovery intervention without reloading the quantized model."""
    for swap_kind, owner, key, original_module in reversed(swapped_modules):
        if swap_kind == "attribute":
            current_module = getattr(owner, key)
            setattr(owner, key, original_module)
        elif swap_kind == "item":
            current_module = owner[key]
            owner[key] = original_module
        else:
            raise ValueError(f"Unknown module swap kind: {swap_kind!r}")
        del current_module
    swapped_modules.clear()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_model_args(args):
    """Create the argument namespace expected by tools.load_model()."""
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        quant_type=args.quant_type,
        model_type=args.model_type,
        fp16_checkpoint=args.fp16_checkpoint,
    )


def load_recovery_model(args):
    """Load BNB explicitly for recovery without changing the main runner."""
    model_args = make_model_args(args)
    if "bnb" not in args.quant_type:
        return load_default_model(model_args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.quant_type == "bnb-nf4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif args.quant_type == "bnb-int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Unsupported bitsandbytes quantization: {args.quant_type}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    if args.quant_type == "bnb-nf4":
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Linear4bit":
                print(
                    "bitsandbytes recovery runtime:",
                    f"module={name}",
                    f"quant_type={getattr(module, 'quant_type', None)}",
                    f"compute_dtype={getattr(module, 'compute_dtype', None)}",
                )
                break
    return tokenizer, model


def atomic_json_dump(data, output_path):
    """Write JSON atomically so an interrupted write is never considered done."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def validate_language_result(result, args):
    """Return whether a cached language JSON is complete for this run."""
    if not isinstance(result, dict):
        return False
    if not {"accuracy", "samples", "records"}.issubset(result):
        return False
    if not isinstance(result["records"], (dict, list)):
        return False
    if not isinstance(result["samples"], int) or result["samples"] < 0:
        return False

    if args.calculate_ppl and args.dataset_type == "klar":
        if "ppl" not in result or "ppl_metrics" not in result:
            return False

    if args.dataset_type == "mclm":
        metrics = result.get("generation_length_metrics")
        required_metrics = {
            "total_generated_tokens",
            "mean_generated_tokens",
            "instances",
        }
        if not isinstance(metrics, dict):
            return False
        if not required_metrics.issubset(metrics):
            return False

    return True


def load_cached_language_result(result_path, args):
    if not args.resume or not os.path.exists(result_path):
        return None
    try:
        with open(result_path, "r", encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Invalid cached result, rerunning {result_path}: {error}")
        return None

    if not validate_language_result(result, args):
        print(f"Incomplete cached result, rerunning {result_path}")
        return None
    return result


def load_cached_language_results(output_dir, languages, args):
    cached_results = {}
    for lang in languages:
        result_path = os.path.join(output_dir, f"{lang}.json")
        result = load_cached_language_result(result_path, args)
        if result is not None:
            cached_results[lang] = result
    return cached_results


def load_existing_metadata(metadata_path):
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def aggregate_language_results(language_results, dataset_type, languages):
    if not language_results:
        raise ValueError("No completed language results are available.")

    ordered_results = {
        lang: language_results[lang]
        for lang in languages
        if lang in language_results
    }
    if len(ordered_results) != len(languages):
        missing = [lang for lang in languages if lang not in ordered_results]
        raise ValueError(f"Missing language results: {missing}")

    macro_accuracy = sum(
        result["accuracy"] for result in ordered_results.values()
    ) / len(ordered_results)
    aggregate = {
        "macro_accuracy": macro_accuracy,
        "languages": {
            lang: {
                "accuracy": result["accuracy"],
                "samples": result["samples"],
                **(
                    {"ppl": result["ppl"]}
                    if "ppl" in result
                    else {}
                ),
                **(
                    {
                        "mean_generated_tokens": result[
                            "generation_length_metrics"
                        ]["mean_generated_tokens"]
                    }
                    if "generation_length_metrics" in result
                    else {}
                ),
            }
            for lang, result in ordered_results.items()
        },
    }

    if dataset_type == "mclm":
        total_tokens = sum(
            result["generation_length_metrics"]["total_generated_tokens"]
            for result in ordered_results.values()
        )
        total_instances = sum(
            result["generation_length_metrics"]["instances"]
            for result in ordered_results.values()
        )
        aggregate.update({
            "total_generated_tokens": total_tokens,
            "mean_generated_tokens": total_tokens / total_instances,
            "generation_instances": total_instances,
        })
    return aggregate


def evaluate_experiment(
    args,
    experiment,
    model,
    tokenizer,
    dataset_full,
    dataset_prompt,
    output_dir,
    languages,
    cached_results=None,
):
    language_results = dict(cached_results or {})

    model.generation_config.do_sample = False
    os.makedirs(output_dir, exist_ok=True)

    for lang in languages:
        if lang in language_results:
            print(
                f"[{experiment['name']}] Skipping completed language: {lang}"
            )
            continue

        print(f"[{experiment['name']}] Evaluating {lang}")
        samples = build_samples(dataset_full, lang, dataset_prompt)
        records = evaluation.inference(
            samples=samples,
            tokenizer=tokenizer,
            model=model,
            model_type=args.model_type,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            record_output_tokens=args.dataset_type == "mclm",
        )
        ppl_result = (
            evaluation.calculate_ppl(
                args.dataset_type,
                samples,
                records,
                tokenizer,
                model,
                args.model_type,
                args.batch_size,
            )
            if args.calculate_ppl
            else None
        )

        accuracy = evaluation.evaluate(records)
        result = {
            "accuracy": accuracy,
            "samples": len(records),
            "records": records,
        }
        if ppl_result is not None:
            result["ppl"] = ppl_result["ppl"]
            result["ppl_metrics"] = ppl_result
        if args.dataset_type == "mclm":
            length_metrics = evaluation.calculate_mclm_length_metrics(records)
            result["generation_length_metrics"] = length_metrics

        language_results[lang] = result
        atomic_json_dump(
            result,
            os.path.join(output_dir, f"{lang}.json"),
        )

    return aggregate_language_results(
        language_results,
        args.dataset_type,
        languages,
    )


def main():
    args = parse_args()
    if args.quant_type == "fp16":
        raise ValueError(
            "Recovery experiments require a quantized --checkpoint; "
            "--quant_type cannot be fp16."
        )

    stages = {
        int(value.strip())
        for value in args.stages.split(",")
        if value.strip()
    }
    if not stages or not stages.issubset({1, 2}):
        raise ValueError("--stages must contain only 1 and/or 2.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    languages = [
        value.strip()
        for value in args.languages.split(",")
        if value.strip()
    ]
    if not languages:
        raise ValueError("--languages must contain at least one language.")
    dataset_full, dataset_prompt = get_dataset(args.dataset_type, languages)
    reference_model = load_fp16_reference(args.fp16_checkpoint)
    reference_layers, layer_path = find_decoder_layers(reference_model)
    num_layers = len(reference_layers)
    groups = make_quarter_layer_groups(num_layers)
    experiments = make_experiments(groups, stages)
    total_reference_parameters = sum(
        parameter.numel() for parameter in reference_model.parameters()
    )

    recovery_root = os.path.join(
        args.save_path,
        "recovery",
        args.dataset_type,
        args.model_type,
        args.quant_type,
    )
    os.makedirs(recovery_root, exist_ok=True)

    summary = {
        "checkpoint": args.checkpoint,
        "fp16_checkpoint": args.fp16_checkpoint,
        "dataset_type": args.dataset_type,
        "quant_type": args.quant_type,
        "model_type": args.model_type,
        "languages": languages,
        "num_layers": num_layers,
        "decoder_layer_path": layer_path,
        "num_layer_groups": 4,
        "layer_group_percentages": ["0-25%", "25-50%", "50-75%", "75-100%"],
        "layer_groups": groups,
        "stages": sorted(stages),
        "experiments": {},
    }

    quant_baseline = None
    # Load the quantized checkpoint lazily and keep one clean copy alive for
    # the entire run.  Every FP16 intervention is rolled back after its
    # experiment, so the next experiment starts from the same quantized model.
    quant_model = None
    tokenizer = None
    for experiment_index, experiment in enumerate(experiments, start=1):
        print(
            f"\nExperiment {experiment_index}/{len(experiments)}: "
            f"{experiment['name']}"
        )
        experiment_dir = os.path.join(recovery_root, experiment["name"])
        os.makedirs(experiment_dir, exist_ok=True)
        cached_results = load_cached_language_results(
            experiment_dir,
            languages,
            args,
        )
        metadata_path = os.path.join(experiment_dir, "metadata.json")
        existing_metadata = load_existing_metadata(metadata_path)

        expected_recovery_strategy = (
            AWQ_RECOVERY_STRATEGY
            if experiment["modules"] and "awq" in args.quant_type.lower()
            else None
        )
        if (
            expected_recovery_strategy is not None
            and (
                existing_metadata is None
                or existing_metadata.get("recovery_strategy")
                != expected_recovery_strategy
            )
        ):
            # Per-language JSON files produced by the old projection-only AWQ
            # intervention are structurally valid, so the generic resume check
            # would otherwise reuse their (often zero-accuracy) outputs.
            if cached_results:
                print(
                    f"[{experiment['name']}] Ignoring legacy AWQ cache; "
                    f"expected recovery strategy "
                    f"{expected_recovery_strategy}."
                )
            cached_results = {}

        metadata_has_recovery_counts = (
            existing_metadata is not None
            and "recovered_modules_count" in existing_metadata
            and "recovered_parameters" in existing_metadata
        )
        all_languages_complete = len(cached_results) == len(languages)
        can_skip_model = (
            all_languages_complete
            and (not experiment["modules"] or metadata_has_recovery_counts)
        )

        if can_skip_model:
            print(
                f"[{experiment['name']}] All languages are complete; "
                "skipping model loading and recovery."
            )
            recovery_metadata = {
                "recovered_modules_count": existing_metadata.get(
                    "recovered_modules_count", 0
                ) if existing_metadata else 0,
                "recovered_parameters": existing_metadata.get(
                    "recovered_parameters", 0
                ) if existing_metadata else 0,
            }
            for key in (
                "quant_layer_path",
                "reference_layer_path",
                "recovery_strategy",
                "recovered_module_paths",
            ):
                if existing_metadata and key in existing_metadata:
                    recovery_metadata[key] = existing_metadata[key]
            aggregate = aggregate_language_results(
                cached_results,
                args.dataset_type,
                languages,
            )
        else:
            if cached_results:
                print(
                    f"[{experiment['name']}] Resuming with "
                    f"{len(cached_results)}/{len(languages)} languages complete."
                )

            if quant_model is None:
                tokenizer, quant_model = load_recovery_model(args)
            recovery_metadata = {
                "recovered_modules_count": 0,
                "recovered_parameters": 0,
            }
            swapped_modules = []
            try:
                if experiment["modules"]:
                    recovery_metadata = recover_modules(
                        quant_model,
                        reference_model,
                        experiment["layers"],
                        experiment["modules"],
                        args.quant_type,
                        swapped_modules,
                    )

                aggregate = evaluate_experiment(
                    args,
                    experiment,
                    quant_model,
                    tokenizer,
                    dataset_full,
                    dataset_prompt,
                    experiment_dir,
                    languages,
                    cached_results=cached_results,
                )
            finally:
                # This is empty for the quantized baseline.  For recovery
                # experiments it reinstalls the exact original GPTQ/AWQ/BNB
                # modules before the next intervention starts.
                if swapped_modules:
                    restore_quantized_modules(swapped_modules)

        recovered_parameters = recovery_metadata["recovered_parameters"]
        recovered_parameter_ratio = (
            recovered_parameters / total_reference_parameters
        )

        result_summary = {
            **experiment,
            **recovery_metadata,
            "recovered_parameter_ratio": recovered_parameter_ratio,
            **aggregate,
        }
        if experiment["name"] == "quant_baseline":
            quant_baseline = aggregate
            result_summary["accuracy_difference"] = 0.0
            if args.dataset_type == "mclm":
                result_summary["mean_generated_tokens_difference"] = 0.0
        else:
            result_summary["accuracy_difference"] = (
                aggregate["macro_accuracy"]
                - quant_baseline["macro_accuracy"]
            )
            if args.dataset_type == "mclm":
                result_summary["mean_generated_tokens_difference"] = (
                    aggregate["mean_generated_tokens"]
                    - quant_baseline["mean_generated_tokens"]
                )

        summary["experiments"][experiment["name"]] = result_summary
        atomic_json_dump(result_summary, metadata_path)
        atomic_json_dump(
            summary,
            os.path.join(recovery_root, "recovery_summary.json"),
        )

    if quant_model is not None:
        del quant_model
    if tokenizer is not None:
        del tokenizer
    del reference_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "\nRecovery experiments completed. Summary: "
        f"{os.path.join(recovery_root, 'recovery_summary.json')}"
    )


if __name__ == "__main__":
    main()
