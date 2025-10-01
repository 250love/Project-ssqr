#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add [CODE0..CODE(M-1)] tokens into a HF tokenizer and expand model embeddings.

Usage:
  python scripts/add_code_tokens.py \
    --base_model meta-llama/Llama-3.1-8B \
    --out_dir ./checkpoints/llama3.1_with_codes \
    --M 2048 \
    --init mean  # ['mean', 'random']

Notes:
- M 是码本大小（论文 LLM 微调阶段推荐 M=2048）。你可改成 4096 等。
- 改了 M 后需要重新运行本脚本导出新 tokenizer+model。
"""
import argparse, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default='/root/autodl-tmp/SSQR_Stage_2/llama2-7b')
    ap.add_argument("--out_dir", type=str, default='model2code')
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--init", type=str, default="mean", choices=["mean","random"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Loading base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )

    # Construct code tokens
    code_tokens = [f"[CODE{i}]" for i in range(args.M)]
    to_add = [t for t in code_tokens if t not in tok.get_vocab()]
    added = tok.add_tokens(to_add, special_tokens=False)
    print(f"[INFO] Requested M={args.M}, actually added {added} new tokens.")

    if added > 0:
        model.resize_token_embeddings(len(tok))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            old_vocab_size = emb.shape[0] - added
            if args.init == "mean":
                mean_vec = emb[:old_vocab_size].mean(dim=0, keepdim=True)
                emb[old_vocab_size:] = mean_vec + 0.01 * torch.randn_like(emb[old_vocab_size:])
            else:
                torch.nn.init.normal_(emb[old_vocab_size:], mean=0.0, std=0.02)

    print(f"[INFO] Saving extended tokenizer & model to: {args.out_dir}")
    tok.save_pretrained(args.out_dir)
    model.save_pretrained(args.out_dir)
    print("[OK] Done.")

if __name__ == "__main__":
    main()
