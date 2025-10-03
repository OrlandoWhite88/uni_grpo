# train_unsloth_gptoss120b_16k.py
# -*- coding: utf-8 -*-
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Avoid Inductor/AOT BW compilation issues
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"   # leave on for stability

import json, gc, torch
from tqdm import tqdm
from torch.optim import AdamW
from torch import amp
from unsloth import FastLanguageModel
from peft import PeftModel
from huggingface_hub import HfApi, login
 
MODEL_NAME = "unsloth/gpt-oss-120b-unsloth-bnb-4bit"  # 4-bit base, Unsloth-optimized
LATEST_ADAPTER = "lora_adapter/adapter-410"  # Load this to continue training

# Hugging Face configuration
HF_TOKEN = os.getenv("HF_TOKEN")  # Set via: export HF_TOKEN=your_token_here
HF_REPO = "orlandowhite/adapters"

if not HF_TOKEN:
    print("[WARNING] HF_TOKEN environment variable not set. Uploads will be skipped.")

# ---- TRAINING KNOBS ----
LEARNING_RATE       = 2e-4         # Higher LR as requested
GRADIENT_ACC_STEP   = 6
START_INDEX         = 10000        # Start from position 10k
NUM_STEPS           = 25           # Run another 25 steps
MAX_EXAMPLES        = START_INDEX + NUM_STEPS * GRADIENT_ACC_STEP
MAX_TOKENS          = 16_384
WARMUP_STEPS        = 5            # Small warmup for short run

# LoRA config with lm_head for JSON tokens
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "modules_to_save": ["lm_head"],  # Train linear on lm_head for better JSON
}

def read_data(path="dataset.jsonl", start=START_INDEX, limit=MAX_EXAMPLES - START_INDEX):
    with open(path, "r") as fi:
        data = [json.loads(line) for line in fi]
    return data[start:start + limit]

def _truncate_pair(prompt_ids, answer_ids, max_len=MAX_TOKENS):
    # keep full answer; trim prompt from front if needed
    keep_prompt = max(0, max_len - len(answer_ids))
    prompt_ids  = prompt_ids[-keep_prompt:]
    return (prompt_ids + answer_ids)[:max_len]

def train_step(model, tokenizer, optimizer, batch, step_idx, scheduler):
    model.train()
    # loss only on the answer (+EOS)
    num_target_tokens = sum(len(x["answer_ids"]) + 1 for x in batch)
    print("num_target_tokens=", num_target_tokens)
    batch_loss = 0.0

    amp_ctx = amp.autocast("cuda", dtype=torch.bfloat16)

    for item in batch:
        ans_ids = item["answer_ids"]
        inp_ids = _truncate_pair(item["input_ids"], ans_ids, MAX_TOKENS)

        answer_len = torch.tensor([len(ans_ids)], dtype=torch.int64, device=model.device)
        input_len  = torch.tensor([len(inp_ids)], dtype=torch.int64, device=model.device)

        input_ids  = torch.tensor([inp_ids], dtype=torch.int64, device=model.device)
        target_ids = torch.tensor(ans_ids + [tokenizer.eos_token_id], dtype=torch.int64, device=model.device)

        with amp_ctx:
            out    = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, input_len - answer_len - 1 : input_len, :].float()
            loss   = torch.nn.functional.cross_entropy(logits, target_ids, reduction="sum") / num_target_tokens

        batch_loss += float(loss.detach())
        loss.backward()

        del logits, out
        gc.collect()
        torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # LR schedule: linear warmup -> cosine
    if step_idx < WARMUP_STEPS:
        warm_lr = LEARNING_RATE * (step_idx + 1) / max(1, WARMUP_STEPS)
        for g in optimizer.param_groups:
            g["lr"] = warm_lr
    else:
        scheduler.step()

    return batch_loss / len(batch)

def upload_to_huggingface(local_path, hf_path, step_num=None):
    """Upload adapter to HuggingFace Hub."""
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi()
        commit_msg = f"SFT adapter checkpoint at step {step_num}" if step_num else "SFT adapter"
        api.upload_folder(
            folder_path=local_path,
            repo_id=HF_REPO,
            path_in_repo=hf_path,
            repo_type="model",
            commit_message=commit_msg
        )
        print(f"[huggingface] Uploaded to {HF_REPO}/{hf_path}")
        return True
    except Exception as e:
        print(f"[huggingface] Upload failed: {e}")
        return False


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    assert torch.cuda.is_available(), "CUDA not available."
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    # Login to HuggingFace
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print(f"[huggingface] Logged in, will upload to {HF_REPO}")
    except Exception as e:
        print(f"[huggingface] Login failed: {e}")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = MODEL_NAME,
        dtype           = None,
        max_seq_length  = MAX_TOKENS,
        load_in_4bit    = True,
        full_finetuning = False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load existing adapter to continue
    model = PeftModel.from_pretrained(model, LATEST_ADAPTER, is_trainable=True)

    # (Recommended for GC) disable KV cache during training
    try:
        model.config.use_cache = False
    except Exception:
        pass

    data = read_data()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Cosine schedule after warmup
    total_steps = max(1, (len(data) // GRADIENT_ACC_STEP))
    # we’ll step the scheduler only post-warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - WARMUP_STEPS))

    # Short run: adjust total_steps to NUM_STEPS
    total_steps = NUM_STEPS
    for i in tqdm(range(total_steps)):
        batch_start = i * GRADIENT_ACC_STEP
        batch_end = batch_start + GRADIENT_ACC_STEP
        batch = [
            {
                "input_ids": tokenizer.apply_chat_template(
                    item["messages"][:-1],
                    add_generation_prompt=True
                ),
                "answer_ids": tokenizer(item["messages"][-1]["content"])["input_ids"],
            }
            for item in data[batch_start:batch_end] if batch_start < len(data)
        ]
        if not batch: break

        loss = train_step(model, tokenizer, optimizer, batch, i, scheduler)
        print(f"\rStep {i}/{total_steps}, lr={optimizer.param_groups[0]['lr']:.6g}, loss: {loss:.4f}")

        if i % 5 == 0:  # Save more frequently for short run
            save_path = f"sft-adapter-step-{i}"
            model.save_pretrained(save_path)
            print(f"[checkpoint] Saved to {save_path}")
            
            # Upload to HuggingFace
            hf_path = f"sft/step-{i}"
            upload_to_huggingface(save_path, hf_path, step_num=i)
