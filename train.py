#!/usr/bin/env python
"""
GRPO fine-tuning script for GPT-OSS 120B using Unsloth.

This script mirrors the SFT setup (adapter-continued-10) and adheres to the Unsloth
GRPO workflow: dataset streaming with token caps, fast inference kernels, and the
reward pipeline defined in reward.py.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from argparse import BooleanOptionalAction
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import unsloth  # ensure Unsloth applies optimizations before torch/trl/peft
from unsloth import FastLanguageModel

import torch
from datasets import Dataset, load_from_disk
from peft import PeftModel
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import HfApi, login

# ---------------------------------------------------------------------------
# Environment & constants
# ---------------------------------------------------------------------------

BASE_MODEL_NAME = "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_R = 32

SFT_ADAPTER_PATH = "grpo-adapter-step-20"  # Resume from checkpoint 20
REWARD_FILE = "reward.py"
DATASET_PATH = "dataset.jsonl"
DEFAULT_OUTPUT_DIR = "grpo_outputs"
GRPO_ADAPTER_NAME = "grpo"
DEFAULT_CACHE_ROOT = "cache/grpo_dataset"

# Hugging Face configuration
HF_TOKEN = os.getenv("HF_TOKEN")  # Set via: export HF_TOKEN=your_token_here
HF_REPO = "orlandowhite/adapters"

if not HF_TOKEN:
    print("[WARNING] HF_TOKEN environment variable not set. Uploads will be skipped.")

MAX_SEQ_LENGTH = 6_144
PROMPT_TOKEN_CAP = 4_096
COMPLETION_TOKEN_CAP = 384

SUPPORTED_TASKS = {
    "select_chapters",
    "select_candidates",
    "score_candidate",
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO training for GPT-OSS 120B (Unsloth)."
    )
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sft-adapter-path", type=str, default=SFT_ADAPTER_PATH)
    parser.add_argument("--reward-path", type=str, default=REWARD_FILE)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)  # -1 means unlimited
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=("low", "medium", "high"),
    )
    parser.add_argument("--cache-root", type=str, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the cached GRPO dataset even if a cached copy exists.",
    )
    parser.add_argument(
        "--reverse-order",
        action=BooleanOptionalAction,
        default=True,
        help="Process the dataset from the end toward the start (default: True).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _is_sequence(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def resolve_cache_path(
    cache_root: str,
    dataset_path: str,
    start_index: int,
    max_rows: Optional[int],
    reasoning_effort: str,
    reverse_order: bool,
) -> Path:
    cache_root_path = Path(cache_root)
    cache_key = {
        "dataset": os.path.abspath(dataset_path),
        "start_index": start_index,
        "max_rows": max_rows,
        "reasoning_effort": reasoning_effort,
        "reverse_order": reverse_order,
        "prompt_cap": PROMPT_TOKEN_CAP,
        "completion_cap": COMPLETION_TOKEN_CAP,
    }
    digest = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root_path / f"grpo_{digest}"


def build_grpo_dataset(
    tokenizer,
    dataset_path: str,
    start_index: int,
    max_rows: Optional[int],
    reasoning_effort: str,
    reverse_order: bool,
) -> Dataset:
    rows: List[Dict[str, Any]] = []
    total = kept = skipped_length = skipped_task = skipped_parse = 0

    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx < start_index:
                continue
            if max_rows is not None and kept >= max_rows:
                break

            line = line.strip()
            if not line:
                continue

            total += 1
            record = json.loads(line)
            messages = record.get("messages", [])
            if len(messages) < 2:
                skipped_parse += 1
                continue

            prompt_messages = messages[:-1]
            reference_message = messages[-1]

            try:
                user_payload = next(
                    _safe_json_load(msg["content"])
                    for msg in reversed(prompt_messages)
                    if msg.get("role") == "user"
                )
            except StopIteration:
                user_payload = None

            targets_payload = record.get("targets")
            if user_payload is None or not isinstance(user_payload, dict):
                skipped_parse += 1
                continue

            task = user_payload.get("task")
            if task not in SUPPORTED_TASKS:
                skipped_task += 1
                continue

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            if len(prompt_ids) > PROMPT_TOKEN_CAP:
                skipped_length += 1
                continue

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            rows.append(
                {
                    "prompt": prompt_text,
                    "user_data": json.dumps(user_payload, ensure_ascii=False),
                    "targets": json.dumps(targets_payload, ensure_ascii=False),
                    "reasoning_effort": reasoning_effort,
                    "reference": reference_message.get("content", ""),
                    "task": task,
                }
            )
            kept += 1

    print(
        "[dataset] scanned={} kept={} skipped_length={} skipped_task={} skipped_parse={}".format(
            total, kept, skipped_length, skipped_task, skipped_parse
        )
    )
    if kept == 0:
        raise ValueError("No dataset rows available after filtering.")

    if reverse_order:
        rows.reverse()

    return Dataset.from_list(rows).with_format("python")


# ---------------------------------------------------------------------------
# Reward function loading
# ---------------------------------------------------------------------------


def load_reward_fn(path: str):
    spec = importlib.util.spec_from_file_location("grpo_reward_module", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load reward module from {path}")
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    if not hasattr(module, "evaluate"):
        raise AttributeError("reward.py must expose an 'evaluate' function.")
    return getattr(module, "evaluate")


def make_reward_wrapper(evaluate_fn, num_generations: int, zero_reward_log_path: Optional[str] = None):
    def _coerce_json(candidate: Any) -> Optional[Dict[str, Any]]:
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, str):
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None

    def _normalize_text(sample: Any) -> str:
        if isinstance(sample, str):
            return sample
        if isinstance(sample, dict):
            return str(sample.get("content", ""))
        if _is_sequence(sample):
            parts: List[str] = []
            for chunk in sample:
                normalized = _normalize_text(chunk)
                if normalized:
                    parts.append(normalized)
            return "\n".join(parts)
        return str(sample)

    def _extract_json_snippet(answer_text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        stripped = answer_text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    return obj, stripped
            except Exception:
                pass
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(answer_text):
            if ch != "{":
                continue
            try:
                obj, end = decoder.raw_decode(answer_text[idx:])
                if isinstance(obj, dict):
                    snippet = answer_text[idx : idx + end]
                    return obj, snippet
            except Exception:
                continue
        return None, None

    def reward_fn(completions: List[Any], **kwargs) -> List[float]:
        user_entries = kwargs.get("user_data") or []
        target_entries = kwargs.get("targets") or []
        metadata_entries = kwargs.get("metadata") or kwargs.get("metadatas") or []

        grouped = bool(completions) and _is_sequence(completions[0])
        rewards: List[float] = []
        preview_count = getattr(reward_fn, "_preview_count", 0)
        zero_preview_count = getattr(reward_fn, "_zero_preview_count", 0)
        zero_total = getattr(reward_fn, "_zero_total", 0)

        if not grouped and num_generations > 1:
            total = len(completions)
            if total % num_generations != 0:
                print(
                    f"[reward] WARNING: completions({total}) not divisible by num_generations({num_generations})."
                )

        batch_count = (
            len(completions)
            if grouped
            else max(1, len(completions) // max(1, num_generations))
        )

        for batch_idx in range(batch_count):
            if grouped:
                generation_group = completions[batch_idx]
            else:
                start = batch_idx * num_generations
                generation_group = completions[start : start + num_generations]
                if not generation_group:
                    continue

            user_item = user_entries[batch_idx] if batch_idx < len(user_entries) else None
            target_item = (
                target_entries[batch_idx] if batch_idx < len(target_entries) else None
            )

            if isinstance(user_item, str):
                user_item = _coerce_json(user_item)
            if isinstance(target_item, str):
                target_item = _coerce_json(target_item)

            if (user_item is None or target_item is None) and metadata_entries:
                meta = metadata_entries[batch_idx] if batch_idx < len(metadata_entries) else {}
                if user_item is None:
                    user_item = meta.get("user_data")
                if target_item is None:
                    target_item = meta.get("targets")

            if not isinstance(user_item, dict):
                user_item = {}
            if target_item is None:
                target_item = {}

            for completion in generation_group:
                answer_text = _normalize_text(completion).strip()
                parsed_json, json_snippet = _extract_json_snippet(answer_text)
                if isinstance(parsed_json, dict) and json_snippet:
                    sanitized_json = json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":"))
                    answer_for_reward = json_snippet
                else:
                    sanitized_json = None
                    answer_for_reward = answer_text
                try:
                    result = evaluate_fn(user_item, answer_for_reward, target_item)
                    score = float(result.get("score", 0.0))
                except Exception as exc:
                    print(f"[reward] exception while scoring: {exc}")
                    score = 0.0
                clamped_score = max(0.0, min(1.0, score))

                if preview_count < 5:
                    print(
                        f"[reward-preview] task={user_item.get('task', 'UNKNOWN')} target={str(target_item)[:160]!r} answer={answer_text[:160]!r} score={clamped_score:.4f}"
                    )
                    preview_count += 1

                if clamped_score == 0.0:
                    record = {
                        "task": user_item.get("task", "UNKNOWN"),
                        "user_data": user_item,
                        "target": target_item,
                        "answer": answer_text,
                        "json_snippet": json_snippet,
                        "sanitized": sanitized_json,
                    }
                    if zero_preview_count < 5:
                        print(
                            f"[reward-zero] task={record['task']} answer={answer_text[:200]!r} target={str(target_item)[:200]!r}"
                        )
                        zero_preview_count += 1
                    if zero_reward_log_path:
                        with open(zero_reward_log_path, "a", encoding="utf-8") as log_file:
                            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    zero_total += 1

                rewards.append(clamped_score)

        if zero_total and zero_total % 25 == 0:
            print(f"[reward-zero-summary] zero_reward_samples={zero_total}")

        reward_fn._preview_count = preview_count
        reward_fn._zero_preview_count = zero_preview_count
        reward_fn._zero_total = zero_total

        return rewards

    reward_fn._preview_count = 0
    reward_fn._zero_preview_count = 0
    reward_fn._zero_total = 0
    return reward_fn


class DebugMetricsCallback(TrainerCallback):
    """Print full metric dictionaries whenever reward or loss collapses."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        loss = logs.get("loss")
        reward = logs.get("reward")
        reward_std = logs.get("reward_std")
        if (
            (loss is not None and float(loss) == 0.0)
            or (reward is not None and float(reward) == 0.0)
            or (reward_std is not None and float(reward_std) == 0.0)
        ):
            try:
                logs_json = json.dumps(logs, default=lambda x: float(x) if isinstance(x, torch.Tensor) else x)
            except TypeError:
                logs_json = str(logs)
            print(f"[metrics-zero] step={state.global_step} metrics={logs_json}")
        return control


class HuggingFaceUploadCallback(TrainerCallback):
    """Upload GRPO adapter checkpoints to Hugging Face periodically."""
    
    def __init__(self, upload_steps=50, hf_token=None, hf_repo=None, adapter_name="grpo"):
        self.upload_steps = upload_steps
        self.hf_token = hf_token
        self.hf_repo = hf_repo
        self.adapter_name = adapter_name
        self.hf_api = None
        
        if hf_token and hf_repo:
            try:
                login(token=hf_token, add_to_git_credential=False)
                self.hf_api = HfApi()
                print(f"[huggingface] Logged in, will upload to {hf_repo}")
            except Exception as e:
                print(f"[huggingface] Failed to login: {e}")
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.upload_steps == 0 and state.global_step > 0:
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")
            
            if model is not None and self.hf_api and self.hf_repo:
                # Local save - use v2 prefix to avoid conflicts
                save_path = f"grpo-v2-step-{state.global_step}"
                try:
                    # Save the adapter
                    try:
                        model.save_pretrained(save_path, selected_adapters=[self.adapter_name])
                    except TypeError:
                        model.save_pretrained(save_path)
                    
                    if tokenizer is not None:
                        tokenizer.save_pretrained(save_path)
                    
                    print(f"\n[checkpoint] Saved GRPO adapter to {save_path}")
                    
                    # Clear CUDA cache to prevent OOM
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("[memory] Cleared CUDA cache")
                    
                    # Upload to Hugging Face
                    try:
                        hf_path = f"grpo-v2/step-{state.global_step}"
                        self.hf_api.upload_folder(
                            folder_path=save_path,
                            repo_id=self.hf_repo,
                            path_in_repo=hf_path,
                            repo_type="model",
                            commit_message=f"GRPO adapter checkpoint at step {state.global_step}"
                        )
                        print(f"[huggingface] Uploaded to {self.hf_repo}/{hf_path}")
                    except Exception as e:
                        print(f"[huggingface] Upload failed: {e}")
                    
                except Exception as e:
                    print(f"\n[checkpoint] Failed to save: {e}")
        
        return control


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def activate_lora_trainable(model, adapter_name: str) -> None:
    """Enable training on the specified LoRA adapter and report trainable parameter counts."""
    if hasattr(model, "train_adapter"):
        model.train_adapter(adapter_name)
    else:
        for name, param in model.named_parameters():
            param.requires_grad = f".{adapter_name}." in name
    total_params = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
    print(f"[lora] trainable parameters = {trainable_params:,} / {total_params:,}")


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA device required for GRPO training."
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        dtype=None,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        full_finetuning=False,
        offload_embedding=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    print(f"[lora] loading adapter from {args.sft_adapter_path}")
    
    # Check if adapter exists locally with 'grpo' subdirectory
    import os.path as osp
    adapter_path = args.sft_adapter_path
    if osp.exists(osp.join(adapter_path, "grpo", "adapter_config.json")):
        adapter_path = osp.join(adapter_path, "grpo")
        print(f"[lora] Found adapter in subdirectory: {adapter_path}")
    
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        adapter_name=GRPO_ADAPTER_NAME,
        is_trainable=True,
    )
    model.set_adapter(GRPO_ADAPTER_NAME)
    activate_lora_trainable(model, GRPO_ADAPTER_NAME)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    
    # Clear CUDA cache after loading checkpoint to reclaim memory
    torch.cuda.empty_cache()
    gc.collect()
    print("[memory] Cleared CUDA cache after checkpoint load")

    gen_config = model.generation_config
    gen_config.max_length = MAX_SEQ_LENGTH  # Explicitly set to 6144 (not 131072!)
    gen_config.max_new_tokens = COMPLETION_TOKEN_CAP
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id
    gen_config.do_sample = True
    gen_config.temperature = 0.95
    gen_config.top_p = 0.95
    gen_config.num_beams = 1
    gen_config.use_cache = True

    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    cache_path = resolve_cache_path(
        cache_root=args.cache_root,
        dataset_path=args.dataset_path,
        start_index=args.start_index,
        max_rows=args.max_rows,
        reasoning_effort=args.reasoning_effort,
        reverse_order=args.reverse_order,
    )
    if cache_path.exists() and not args.refresh_cache:
        print(f"[dataset-cache] loading from {cache_path}")
        grpo_dataset = load_from_disk(cache_path)
    else:
        print(f"[dataset-cache] building dataset → {cache_path}")
        grpo_dataset = build_grpo_dataset(
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            start_index=args.start_index,
            max_rows=args.max_rows,
            reasoning_effort=args.reasoning_effort,
            reverse_order=args.reverse_order,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        grpo_dataset.save_to_disk(cache_path)
    grpo_dataset = grpo_dataset.with_format("python")
    if len(grpo_dataset):
        preview = grpo_dataset[0]
        prompt_preview = preview.get("prompt", "")
        print(f"[dataset] preview task={preview.get('task')} prompt_snippet={prompt_preview[:160]!r}")
        print(f"[dataset] preview targets={preview.get('targets', '')[:160]}")

    os.makedirs(args.output_dir, exist_ok=True)
    zero_reward_log_path = os.path.join(args.output_dir, "zero_rewards.jsonl")
    if os.path.exists(zero_reward_log_path):
        os.remove(zero_reward_log_path)

    reward_eval = load_reward_fn(args.reward_path)
    reward_fn = make_reward_wrapper(
        reward_eval,
        num_generations=args.num_generations,
        zero_reward_log_path=zero_reward_log_path,
    )
    
    # One final cache clear before starting training
    torch.cuda.empty_cache()
    gc.collect()
    print("[memory] Final CUDA cache clear before training")

    grpo_config = GRPOConfig(
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        weight_decay=0.01,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        num_train_epochs=args.num_train_epochs,
        max_prompt_length=PROMPT_TOKEN_CAP,
        max_completion_length=COMPLETION_TOKEN_CAP,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",
        
        # TensorBoard logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="tensorboard",
        
        remove_unused_columns=False,
        output_dir=args.output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        train_dataset=grpo_dataset,
    )
    trainer.add_callback(DebugMetricsCallback())
    trainer.add_callback(HuggingFaceUploadCallback(
        upload_steps=10,  # Save and upload every 10 steps
        hf_token=HF_TOKEN,
        hf_repo=HF_REPO,
        adapter_name=GRPO_ADAPTER_NAME
    ))

    print("[training] starting GRPO")
    print(f"[training] TensorBoard logs: {args.output_dir}/logs")
    print(f"[training] Monitor with: tensorboard --logdir {args.output_dir}/logs")
    train_result = trainer.train()
    print(f"[training] completed - metrics: {train_result.metrics}")

    adapter_out = os.path.join(args.output_dir, "grpo-v2-final")
    try:
        model.save_pretrained(adapter_out, selected_adapters=[GRPO_ADAPTER_NAME])
    except TypeError:
        model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    print(f"[output] saved GRPO adapter to {adapter_out}")
    
    # Upload final adapter to Hugging Face
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        hf_final_path = "grpo-v2/final"
        api.upload_folder(
            folder_path=adapter_out,
            repo_id=HF_REPO,
            path_in_repo=hf_final_path,
            repo_type="model",
            commit_message="GRPO final adapter"
        )
        print(f"[huggingface] Uploaded final GRPO adapter to {HF_REPO}/{hf_final_path}")
    except Exception as e:
        print(f"[huggingface] Failed to upload final adapter: {e}")


if __name__ == "__main__":
    main()
