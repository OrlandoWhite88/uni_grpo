#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO for GPT-OSS 120B with Unsloth + colocated vLLM (Standby), **native MXFP4**.

- Loads openai/gpt-oss-120b (no BitsAndBytes).
- vLLM runs in-process (fast_inference=True), Standby enabled via env.
- LoRA adapter is trained and used for generation (on-policy).
- 4 generations per prompt; rollout batch kept tiny for stability.
"""

import os
# ---- env must be set BEFORE importing unsloth/torch ----
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import argparse
import gc
import hashlib
import importlib.util
import json
from argparse import BooleanOptionalAction
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import unsloth  # keep first
from unsloth import FastLanguageModel
from datasets import Dataset, load_from_disk
from peft import PeftModel
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
try:
    from unsloth import vLLMSamplingParams
except Exception:
    vLLMSamplingParams = None

from huggingface_hub import HfApi, login

# -------------------- constants --------------------
BASE_MODEL_NAME = "openai/gpt-oss-120b"       # <— native MXFP4, NOT bnb-4bit
GRPO_ADAPTER_NAME = "grpo"

SFT_ADAPTER_PATH = "grpo-adapter-step-20"     # resume adapter
REWARD_FILE = "reward.py"
DATASET_PATH = "dataset.jsonl"
DEFAULT_OUTPUT_DIR = "grpo_outputs_colocate_native"
DEFAULT_CACHE_ROOT = "cache/grpo_dataset"

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = "orlandowhite/adapters"             # optional upload

MAX_SEQ_LENGTH = 6_144
PROMPT_TOKEN_CAP = 4_096
COMPLETION_TOKEN_CAP = 384

SUPPORTED_TASKS = {"select_chapters","select_candidates","score_candidate"}

# -------------------- args --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GRPO + Unsloth colocated vLLM (native GPT-OSS)")

    p.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)

    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sft-adapter-path", type=str, default=SFT_ADAPTER_PATH)
    p.add_argument("--reward-path", type=str, default=REWARD_FILE)

    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.10)

    # per-device batch MUST be a multiple of num-generations
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=2)
    p.add_argument("--num-generations", type=int, default=4)

    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--logging-steps", type=int, default=1)

    p.add_argument("--reasoning-effort", type=str, default="medium",
                   choices=("low","medium","high"))
    p.add_argument("--cache-root", type=str, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--reverse-order", action=BooleanOptionalAction, default=True)

    # sampling
    p.add_argument("--temperature", type=float, default=0.95)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-p", type=float, default=None)

    # colocated vLLM memory share
    p.add_argument("--vllm-gpu-mem", type=float, default=0.60,
                   help="gpu_memory_utilization for colocated vLLM (0.0-1.0)")

    # rollout concurrency (safe default 1; tune up after stable)
    p.add_argument("--prompts-per-step", type=int, default=1,
                   help="TRL generation_batch_size equivalent")
    return p.parse_args()

# -------------------- dataset utils --------------------
def _safe_json_load(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None

def _is_sequence(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))

def resolve_cache_path(cache_root: str, dataset_path: str, start_index: int,
                       max_rows: Optional[int], reasoning_effort: str, reverse_order: bool) -> Path:
    cache_root_path = Path(cache_root)
    key = {
        "dataset": os.path.abspath(dataset_path),
        "start_index": start_index,
        "max_rows": max_rows,
        "reasoning_effort": reasoning_effort,
        "reverse_order": reverse_order,
        "prompt_cap": PROMPT_TOKEN_CAP,
        "completion_cap": COMPLETION_TOKEN_CAP,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root_path / f"grpo_{digest}"

def build_grpo_dataset(tokenizer, dataset_path: str, start_index: int,
                       max_rows: Optional[int], reasoning_effort: str,
                       reverse_order: bool) -> Dataset:
    rows: List[Dict[str, Any]] = []
    total = kept = skipped_length = skipped_task = skipped_parse = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_index: continue
            if max_rows is not None and kept >= max_rows: break
            line = line.strip()
            if not line: continue

            total += 1
            rec = json.loads(line)
            messages = rec.get("messages", [])
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

            targets_payload = rec.get("targets")
            if user_payload is None or not isinstance(user_payload, dict):
                skipped_parse += 1
                continue

            task = user_payload.get("task")
            if task not in SUPPORTED_TASKS:
                skipped_task += 1
                continue

            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=True,
            )
            if len(prompt_ids) > PROMPT_TOKEN_CAP:
                skipped_length += 1
                continue

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=False,
            )
            rows.append({
                "prompt": prompt_text,
                "user_data": json.dumps(user_payload, ensure_ascii=False),
                "targets": json.dumps(targets_payload, ensure_ascii=False),
                "reasoning_effort": reasoning_effort,
                "reference": reference_message.get("content", ""),
                "task": task,
            })
            kept += 1

    print(f"[dataset] scanned={total} kept={kept} skipped_length={skipped_length} "
          f"skipped_task={skipped_task} skipped_parse={skipped_parse}")
    if kept == 0:
        raise ValueError("No dataset rows available after filtering.")
    if reverse_order:
        rows.reverse()
    return Dataset.from_list(rows).with_format("python")

# -------------------- reward --------------------
def load_reward_fn(path: str):
    spec = importlib.util.spec_from_file_location("grpo_reward_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    assert hasattr(module, "evaluate"), "reward.py must expose an 'evaluate' function."
    return getattr(module, "evaluate")

def make_reward_wrapper(evaluate_fn, num_generations: int, zero_reward_log_path: Optional[str] = None):
    def _coerce_json(x: Any) -> Optional[Dict[str, Any]]:
        if isinstance(x, dict): return x
        if isinstance(x, str):
            try: return json.loads(x)
            except Exception: return None
        return None

    def _normalize_text(sample: Any) -> str:
        if isinstance(sample, str): return sample
        if isinstance(sample, dict): return str(sample.get("content",""))
        if _is_sequence(sample):
            return "\n".join([_normalize_text(s) for s in sample])
        return str(sample)

    def _extract_json_snippet(text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        s = text.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict): return obj, s
            except Exception: pass
        dec = json.JSONDecoder()
        for i,ch in enumerate(text):
            if ch != "{": continue
            try:
                obj, end = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj, text[i:i+end]
            except Exception:
                continue
        return None, None

    def reward_fn(completions: List[Any], **kwargs) -> List[float]:
        user_entries = kwargs.get("user_data") or []
        target_entries = kwargs.get("targets") or []
        meta_entries = kwargs.get("metadata") or kwargs.get("metadatas") or []
        grouped = bool(completions) and _is_sequence(completions[0])

        out: List[float] = []
        prev = getattr(reward_fn, "_prev", 0)
        zprev = getattr(reward_fn, "_zprev", 0)
        ztot = getattr(reward_fn, "_ztot", 0)

        if not grouped and num_generations > 1:
            total = len(completions)
            if total % num_generations != 0:
                print(f"[reward] WARNING: completions({total}) not divisible by num_generations({num_generations}).")

        batch_count = len(completions) if grouped else max(1, len(completions)//max(1, num_generations))
        for b in range(batch_count):
            group = completions[b] if grouped else completions[b*num_generations:(b+1)*num_generations]
            if not group: continue

            user_item = user_entries[b] if b < len(user_entries) else {}
            target_item = target_entries[b] if b < len(target_entries) else {}

            if isinstance(user_item, str): user_item = _coerce_json(user_item) or {}
            if isinstance(target_item, str): target_item = _coerce_json(target_item) or {}

            if meta_entries and (not user_item or not target_item):
                meta = meta_entries[b] if b < len(meta_entries) else {}
                user_item = user_item or meta.get("user_data", {})
                target_item = target_item or meta.get("targets", {})

            for comp in group:
                text = _normalize_text(comp).strip()
                parsed, snip = _extract_json_snippet(text)
                answer = snip if isinstance(parsed, dict) and snip else text
                try:
                    score = float(evaluate_fn(user_item, answer, target_item).get("score", 0.0))
                except Exception as e:
                    print(f"[reward] exception while scoring: {e}")
                    score = 0.0
                score = max(0.0, min(1.0, score))

                if prev < 5:
                    print(f"[reward-preview] task={user_item.get('task','?')} score={score:.4f} answer={text[:160]!r}")
                    prev += 1
                if score == 0.0:
                    if zprev < 5:
                        print(f"[reward-zero] answer={text[:200]!r} target={str(target_item)[:160]!r}")
                        zprev += 1
                    if zero_reward_log_path:
                        with open(zero_reward_log_path, "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({
                                "task": user_item.get("task","UNKNOWN"),
                                "user_data": user_item,
                                "target": target_item,
                                "answer": text,
                            }, ensure_ascii=False) + "\n")
                    ztot += 1

                out.append(score)

        if ztot and ztot % 25 == 0:
            print(f"[reward-zero-summary] zero_reward_samples={ztot}")

        reward_fn._prev = prev
        reward_fn._zprev = zprev
        reward_fn._ztot = ztot
        return out

    reward_fn._prev = 0
    reward_fn._zprev = 0
    reward_fn._ztot = 0
    return reward_fn

# -------------------- callbacks --------------------
class DebugMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return control
        loss = logs.get("loss"); reward = logs.get("reward"); rstd = logs.get("reward_std")
        if (loss is not None and float(loss) == 0.0) or \
           (reward is not None and float(reward) == 0.0) or \
           (rstd is not None and float(rstd) == 0.0):
            try:
                jj = json.dumps(logs, default=lambda x: float(x) if isinstance(x, torch.Tensor) else x)
            except TypeError:
                jj = str(logs)
            print(f"[metrics-zero] step={state.global_step} metrics={jj}")
        return control

class HuggingFaceUploadCallback(TrainerCallback):
    def __init__(self, upload_steps=10, hf_token=None, hf_repo=None, adapter_name="grpo"):
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
                print(f"[huggingface] login failed: {e}")

    def _is_main(self, trainer=None, state=None):
        if trainer is not None and hasattr(trainer, "is_world_process_zero"):
            return trainer.is_world_process_zero()
        if state is not None and hasattr(state, "is_world_process_zero"):
            return state.is_world_process_zero
        return True

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.upload_steps != 0:
            return control
        trainer = kwargs.get("trainer")
        if not self._is_main(trainer, state):
            return control
        model = kwargs.get("model"); tok = kwargs.get("tokenizer")
        if model is None or self.hf_api is None or not self.hf_repo:
            return control
        save_path = f"grpo-v2-step-{state.global_step}"
        try:
            torch.cuda.empty_cache(); gc.collect()
            base = getattr(model, "module", model)
            try:
                base.save_pretrained(save_path, selected_adapters=[GRPO_ADAPTER_NAME])
            except TypeError:
                base.save_pretrained(save_path)
            if tok is not None:
                tok.save_pretrained(save_path)
            print(f"[checkpoint] saved adapter to {save_path}")
            torch.cuda.empty_cache(); gc.collect()
            try:
                self.hf_api.upload_folder(
                    folder_path=save_path,
                    repo_id=self.hf_repo,
                    path_in_repo=f"grpo-v2/step-{state.global_step}",
                    repo_type="model",
                    commit_message=f"GRPO adapter step {state.global_step}",
                )
                print(f"[huggingface] uploaded step {state.global_step}")
            except Exception as e:
                print(f"[huggingface] upload failed: {e}")
        except Exception as e:
            print(f"[checkpoint] save failed: {e}")
        return control

# -------------------- training helpers --------------------
def activate_lora_trainable(model, adapter_name: str) -> None:
    if hasattr(model, "train_adapter"):
        model.train_adapter(adapter_name)
    else:
        for n, p in model.named_parameters():
            p.requires_grad = f".{adapter_name}." in n
    tot = tr = 0
    for _, p in model.named_parameters():
        n = p.numel(); tot += n
        if p.requires_grad: tr += n
    print(f"[lora] trainable parameters = {tr:,} / {tot:,}")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

# -------------------- main --------------------
def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required."

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    # ---- Load native GPT-OSS (MXFP4), no bnb-4bit ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        dtype=None,                    # Unsloth will choose (bf16 compute)
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,            # <— critical
        full_finetuning=False,
        offload_embedding=True,
        fast_inference=True,           # <— colocated vLLM
        gpu_memory_utilization=args.vllm_gpu_mem,
        float8_kv_cache=True,          # Hopper: halves KV mem
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
    import os.path as osp
    adapter_path = args.sft_adapter_path
    if osp.exists(osp.join(adapter_path, "grpo", "adapter_config.json")):
        adapter_path = osp.join(adapter_path, "grpo")
        print(f"[lora] found adapter subdir: {adapter_path}")

    model = PeftModel.from_pretrained(
        model, adapter_path, adapter_name=GRPO_ADAPTER_NAME, is_trainable=True,
    )
    model.set_adapter(GRPO_ADAPTER_NAME)
    activate_lora_trainable(model, GRPO_ADAPTER_NAME)

    torch.cuda.empty_cache(); gc.collect()
    print("[memory] cleared CUDA cache after checkpoint load")

    gen = model.generation_config
    gen.max_length = MAX_SEQ_LENGTH
    gen.max_new_tokens = COMPLETION_TOKEN_CAP
    gen.pad_token_id = tokenizer.pad_token_id
    gen.eos_token_id = tokenizer.eos_token_id
    gen.do_sample = True
    gen.temperature = args.temperature
    gen.top_p = args.top_p
    gen.num_beams = 1
    gen.use_cache = True

    try:
        model.config.use_cache = False
    except Exception:
        pass

    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    # ---- Dataset
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
        print(f"[dataset-cache] building → {cache_path}")
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
        print(f"[dataset] preview task={preview.get('task')} "
              f"prompt_snippet={preview.get('prompt','')[:160]!r}")
        print(f"[dataset] preview targets={preview.get('targets','')[:160]}")

    os.makedirs(args.output_dir, exist_ok=True)
    zero_log = os.path.join(args.output_dir, "zero_rewards.jsonl")
    if os.path.exists(zero_log): os.remove(zero_log)

    reward_eval = load_reward_fn(args.reward_path)
    reward_fn = make_reward_wrapper(reward_eval, num_generations=args.num_generations, zero_reward_log_path=zero_log)

    torch.cuda.empty_cache(); gc.collect()
    print("[memory] final CUDA cache clear before training")

    # ---- vLLM SamplingParams (optional)
    vsp = None
    if vLLMSamplingParams is not None:
        kwargs = {}
        if args.min_p is not None:
            kwargs["min_p"] = float(args.min_p)
        if kwargs:
            vsp = vLLMSamplingParams(**kwargs)

    # Batch sanity
    if args.per_device_batch_size % args.num_generations != 0:
        fixed = max(args.num_generations, (args.per_device_batch_size // args.num_generations) * args.num_generations)
        print(f"[batch] adjusting per_device_batch_size {args.per_device_batch_size} -> {fixed} "
              f"(multiple of num_generations={args.num_generations})")
        args.per_device_batch_size = fixed

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    grpo_config = GRPOConfig(
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.01,

        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        num_generations=args.num_generations,             # 4
        num_train_epochs=args.num_train_epochs,
        max_prompt_length=PROMPT_TOKEN_CAP,
        max_completion_length=COMPLETION_TOKEN_CAP,

        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_strategy="no",
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="tensorboard",
        remove_unused_columns=False,
        output_dir=args.output_dir,

        bf16=use_bf16,
        fp16=not use_bf16,

        # colocated vLLM
        use_vllm=True,
        vllm_mode="colocate",
        vllm_sampling_params=vsp,

        # serialize rollouts first; then you can try 2
        generation_batch_size=max(1, args.prompts_per_step),

        temperature=args.temperature,
        top_p=args.top_p,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        train_dataset=grpo_dataset,
    )
    trainer.add_callback(DebugMetricsCallback())
    if HF_TOKEN and HF_REPO:
        trainer.add_callback(HuggingFaceUploadCallback(
            upload_steps=10,
            hf_token=HF_TOKEN,
            hf_repo=HF_REPO,
            adapter_name=GRPO_ADAPTER_NAME,
        ))

    print("[training] starting GRPO (colocated vLLM + Standby, native GPT-OSS)")
    print(f"[training] TensorBoard logs: {args.output_dir}/logs")

    res = trainer.train()
    print(f"[training] completed - metrics: {res.metrics}")

    out_dir = os.path.join(args.output_dir, "grpo-v2-final")
    base = getattr(model, "module", model)
    try:
        base.save_pretrained(out_dir, selected_adapters=[GRPO_ADAPTER_NAME])
    except TypeError:
        base.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[output] saved GRPO adapter to {out_dir}")

    if HF_TOKEN and HF_REPO:
        try:
            login(token=HF_TOKEN, add_to_git_credential=False)
            api = HfApi()
            api.upload_folder(
                folder_path=out_dir,
                repo_id=HF_REPO,
                path_in_repo="grpo-v2/final",
                repo_type="model",
                commit_message="GRPO final adapter",
            )
            print(f"[huggingface] uploaded final adapter to {HF_REPO}/grpo-v2/final")
        except Exception as e:
            print(f"[huggingface] final upload failed: {e}")

if __name__ == "__main__":
    main()
