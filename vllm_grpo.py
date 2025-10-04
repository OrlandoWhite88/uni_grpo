#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRPO-style RL fine-tuning for GPT-OSS 120B using Unsloth for training (GPU-0)
and a separate vLLM OpenAI server for fast generation (GPU-1) with 4 completions.

- Uses your dataset.jsonl format from the earlier script (messages[-1] is targets JSON).
- Hot-loads LoRA to vLLM every `HOTSWAP_STEPS` so sampling uses the latest policy.
- Safe default generation batch size = 1 (increase once stable).
"""

import os, re, json, time, math, gc, shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Tuple
import requests
import numpy as np
import torch
from torch.optim import AdamW
from peft import PeftModel
from transformers import PreTrainedTokenizerBase
from unsloth import FastLanguageModel

# -------------------
# Config
# -------------------
BASE_MODEL_NAME      = "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
SFT_ADAPTER_PATH     = "grpo-adapter-step-20"         # your starting adapter
ADAPTER_NAME         = "grpo-live"           # LoRA name we publish to vLLM server
OUTPUT_DIR           = "grpo_outputs_vllm"
DATASET_PATH         = "dataset.jsonl"
REWARD_FILE          = "reward.py"           # must expose evaluate(user, answer, targets) -> {"score": float}

MAX_PROMPT_TOKENS    = 4096
MAX_ANSWER_TOKENS    = 384
MAX_SEQ_LEN          = 6144                   # training-side max_len (tokens)
REASONING_TAG        = False                  # set True if you want <think>..</think> staging

# Sampling on vLLM (GPU-1)
VLLM_BASE_URL        = "http://localhost:8000/v1"
NUM_GENS_PER_PROMPT  = 4            # requested generations per question
TEMPERATURE          = 0.95
TOP_P                = 0.95
MIN_P                = None          # e.g., 0.1 if you like nucleus floor

# Trainer / Optim
LEARNING_RATE        = 5e-5
WEIGHT_DECAY         = 0.01
GRAD_ACCUM_STEPS     = 2             # tune as needed
PROMPTS_PER_STEP     = 2             # "generation batch size" at the trainer side; raise after stable
MAX_STEPS            = 20000          # or set number of epochs/rows instead
LOG_EVERY            = 1
HOTSWAP_STEPS        = 10            # how often to push LoRA to vLLM
CLIP_GRAD_NORM       = 1.0

# Environment safety
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # train on GPU-0
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# -------------------
# Utilities
# -------------------
def _load_reward():
    import importlib.util
    spec = importlib.util.spec_from_file_location("reward_module", REWARD_FILE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    assert hasattr(mod, "evaluate"), "reward.py must expose evaluate(user, answer, targets)->{'score': float}"
    return mod.evaluate

def _extract_json(text: str) -> Optional[str]:
    # Strip <think> blocks if any, then find first valid JSON object
    t = text
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            json.loads(t); return t
        except Exception:
            pass
    brace, start = 0, -1
    for i,ch in enumerate(t):
        if ch == "{":
            if brace == 0: start = i
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0 and start >= 0:
                cand = t[start:i+1]
                try:
                    json.loads(cand)
                    return cand
                except Exception:
                    start = -1
    return None

def _post_vllm(path: str, payload: Dict[str, Any]) -> requests.Response:
    url = f"{VLLM_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    return requests.post(url, json=payload, timeout=600)

def vllm_chat_generate(adapter_name: str, tokenizer: PreTrainedTokenizerBase,
                       messages: List[Dict[str, str]], max_new_tokens: int) -> List[str]:
    """Ask vLLM (OpenAI chat.completions) for NUM_GENS_PER_PROMPT completions."""
    payload = {
        "model": adapter_name,            # << vLLM routes to this LoRA
        "messages": messages,
        "max_tokens": max_new_tokens,
        "n": NUM_GENS_PER_PROMPT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    if MIN_P is not None:
        payload["min_p"] = MIN_P
    r = _post_vllm("/chat/completions", payload)
    r.raise_for_status()
    out = r.json()
    return [c["message"]["content"] for c in out["choices"]]

def vllm_unload_lora(adapter_name: str) -> None:
    _post_vllm("/unload_lora_adapter", {"lora_name": adapter_name})

def vllm_load_lora(adapter_name: str, path: str, lora_scale: Optional[float] = None) -> None:
    body = {"lora_name": adapter_name, "lora_path": os.path.abspath(path)}
    if lora_scale is not None:
        body["lora_scale"] = float(lora_scale)
    _post_vllm("/load_lora_adapter", body)

# -------------------
# Data
# -------------------
def stream_dataset(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            if len(msgs) < 2: continue
            try:
                targets = json.loads(msgs[-1]["content"])
                user_payload = json.loads(next(m["content"] for m in reversed(msgs[:-1]) if m.get("role")=="user"))
            except Exception:
                continue
            # cut long prompts early
            yield {
                "messages": msgs[:-1],
                "targets": targets,
                "user_data": user_payload,
            }

# -------------------
# Loss (simple GRPO-style token-weighted CE with reward)
# -------------------
def compute_loss(model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizerBase,
                 items: List[Dict[str, Any]],
                 dtype: torch.dtype) -> Tuple[torch.Tensor, int]:
    """
    items[i] contains:
      input_ids: List[int] (prompt tokens)
      answer_ids: List[int] (completion tokens)
      reward: float (normalized across the G group)
      count: int (dedup multiplier)
    """
    device = model.device
    total_tokens = 0
    losses = []
    for it in items:
        inp = it["input_ids"] + it["answer_ids"]
        # pad to MAX_SEQ_LEN
        if len(inp) > MAX_SEQ_LEN:
            inp = inp[:MAX_SEQ_LEN]
        pad_len = MAX_SEQ_LEN - len(inp)
        input_ids = torch.tensor([inp + [tokenizer.pad_token_id]*pad_len], device=device, dtype=torch.long)
        # region to score = the answer (plus EOS)
        target = it["answer_ids"] + [tokenizer.eos_token_id]
        target = target[:MAX_SEQ_LEN]  # safety
        input_len = len(it["input_ids"]) + len(it["answer_ids"])
        answer_len = min(len(target), MAX_SEQ_LEN)
        total_tokens += answer_len

        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(input_ids=input_ids, use_cache=False).logits[0]  # [T, V]
            start = input_len - answer_len + 1  # shift because targets are next-token
            if start < 0: start = 0
            slice_logits = logits[start:start+answer_len, :]
            target_ids = torch.tensor(target[:answer_len], device=device, dtype=torch.long)
            ce = torch.nn.functional.cross_entropy(slice_logits, target_ids, reduction="mean")
            losses.append(ce * float(it["reward"]) * float(it.get("count", 1)))

        del logits, slice_logits
    if not losses:
        return torch.tensor(0.0, device=device), 0
    loss = torch.stack(losses).mean()
    return loss, total_tokens

# -------------------
# Main
# -------------------
def main():
    assert torch.cuda.is_available(), "CUDA required"
    # ---- Load model + tokenizer (Unsloth) on GPU-0
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        full_finetuning=False,
        offload_embedding=True,
        dtype=None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    # Attach your existing LoRA
    print(f"[lora] loading adapter from {SFT_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH, adapter_name=ADAPTER_NAME, is_trainable=True)
    model.set_adapter(ADAPTER_NAME)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reward_eval = _load_reward()

    # Optimizer
    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Make sure vLLM knows about the initial adapter
    lora_dir = os.path.abspath(os.path.join(OUTPUT_DIR, "adapter-step-0"))
    if os.path.isdir(lora_dir): shutil.rmtree(lora_dir)
    model.save_pretrained(lora_dir, selected_adapters=[ADAPTER_NAME] if hasattr(model, "save_pretrained") else None)
    try:
        vllm_unload_lora(ADAPTER_NAME)
    except Exception:
        pass
    vllm_load_lora(ADAPTER_NAME, lora_dir)

    step = 0
    torch.cuda.empty_cache()
    gc.collect()

    stream = stream_dataset(DATASET_PATH)
    running = True
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    while running and step < MAX_STEPS:
        # ----- Sample a small batch of prompts to generate (PROMPTS_PER_STEP)
        batch_items: List[Tuple[Dict[str, Any], List[str]]] = []
        for _ in range(PROMPTS_PER_STEP):
            try:
                rec = next(stream)
            except StopIteration:
                running = False
                break

            # Build chat prompt with the tokenizer’s chat template (to stay consistent with training)
            prompt_txt = tokenizer.apply_chat_template(
                rec["messages"],
                add_generation_prompt=True,
                tokenize=False,
            )
            # Ask vLLM for NUM_GENS_PER_PROMPT completions
            answers = vllm_chat_generate(ADAPTER_NAME, tokenizer, rec["messages"], MAX_ANSWER_TOKENS)
            batch_items.append((rec, answers))

        if not batch_items:
            break

        # ----- Compute rewards + build training items
        train_minibatch: List[Dict[str, Any]] = []
        for rec, answers in batch_items:
            # normalize rewards per-prompt group (GRPO)
            rewards = []
            for ans in answers:
                if REASONING_TAG and "</think>" not in ans:
                    ans = "<think>" + ans + "\n</think>\n"
                j = _extract_json(ans)
                try:
                    s = float(reward_eval(rec["user_data"], j if j is not None else ans, rec["targets"]).get("score", 0.0))
                except Exception:
                    s = 0.0
                rewards.append(s)
            rewards = np.array(rewards, dtype=np.float32)
            # avoid div-by-zero; standard GRPO normalization
            norm = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

            # tokenization for the loss
            prompt_ids = tokenizer.apply_chat_template(rec["messages"], add_generation_prompt=True, tokenize=True)
            for ans, r in zip(answers, norm.tolist()):
                answer_ids = tokenizer(ans, add_special_tokens=False)["input_ids"]
                train_minibatch.append({
                    "input_ids": prompt_ids[:MAX_PROMPT_TOKENS],
                    "answer_ids": answer_ids[:MAX_ANSWER_TOKENS],
                    "reward": float(r),
                    "count": 1,
                })

        # ----- Train step with grad accumulation
        model.train()
        total_tok = 0
        opt.zero_grad(set_to_none=True)

        chunks = [train_minibatch[i:i+GRAD_ACCUM_STEPS] for i in range(0, len(train_minibatch), GRAD_ACCUM_STEPS)]
        for ci, chunk in enumerate(chunks):
            loss, n_tok = compute_loss(model, tokenizer, chunk, dtype)
            total_tok += n_tok
            loss.backward()
            if (ci + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        step += 1
        if step % LOG_EVERY == 0:
            print(f"[step {step}] tokens={total_tok} minibatch={len(train_minibatch)}")

        # ----- Hot-swap LoRA on vLLM so sampling keeps up with latest policy
        if step % HOTSWAP_STEPS == 0:
            save_dir = os.path.abspath(os.path.join(OUTPUT_DIR, f"adapter-step-{step}"))
            if os.path.isdir(save_dir): shutil.rmtree(save_dir)
            model.save_pretrained(save_dir, selected_adapters=[ADAPTER_NAME] if hasattr(model, "save_pretrained") else None)
            try:
                vllm_unload_lora(ADAPTER_NAME)
            except Exception:
                pass
            vllm_load_lora(ADAPTER_NAME, save_dir)
            # small cooldown to let server finish loading
            time.sleep(1.0)

    # final save
    final_dir = os.path.abspath(os.path.join(OUTPUT_DIR, "adapter-final"))
    if os.path.isdir(final_dir): shutil.rmtree(final_dir)
    model.save_pretrained(final_dir, selected_adapters=[ADAPTER_NAME] if hasattr(model, "save_pretrained") else None)
    print(f"[done] saved final adapter at: {final_dir}")

if __name__ == "__main__":
    main()
