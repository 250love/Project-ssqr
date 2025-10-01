#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert SSQR JSON files to Alpaca format with answer-first reordering.

Rules:
1. Keep only the first 3 candidates from `completion`.
2. If `answer` is not in those candidates, insert it at the first position.
3. If `answer` is present but not first, move it to first, preserving others' order.

Outputs Alpaca format:
{
  "instruction": "<prompt>",
  "input": "",
  "output": "<reordered completion text>"
}
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

def reorder_completion(completion: str, answer: str) -> str:
    """Reorder completion so that answer is first, keep only 3 items."""
    lines = [l.strip() for l in completion.splitlines() if l.strip()]
    # 保留前三行
    top3 = lines[:3]
    # 查找是否已有 answer
    idx = next((i for i,l in enumerate(top3) if answer in l), None)
    if idx is None:
        # 插入到第一位
        top3 = [f"1, {answer}"] + [f"{i+2}, {l.split(',',1)[-1].strip()}" for i,l in enumerate(top3)]
    elif idx != 0:
        # 把 answer 那行移到最前
        ans_line = top3.pop(idx)
        # 重新编号
        top3 = [ans_line] + top3
        top3 = [f"{i+1}, {l.split(',',1)[-1].strip()}" for i,l in enumerate(top3)]
    else:
        # already first -> 重新编号保证1,2,3
        top3 = [f"{i+1}, {l.split(',',1)[-1].strip()}" for i,l in enumerate(top3)]
    return "\n".join(top3)

def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        content = f.read().strip()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass
    # JSONL
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                items.append(json.loads(line))
    return items

def to_alpaca(items: List[Dict[str, Any]]) -> List[Dict[str,str]]:
    out=[]
    for it in items:
        prompt = it.get('prompt','')
        comp = it.get('completion','')
        ans  = it.get('answer','')
        if comp and ans:
            comp = reorder_completion(comp, ans)
        elif ans:
            comp = ans
        out.append({
            "instruction": prompt,
            "input": "",
            "output": comp
        })
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="/root/autodl-tmp/SSQR-main/output_json/")
    parser.add_argument("--out_dir", default="/root/autodl-tmp/SSQR_Stage_2/data")
    args = parser.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    for name in ["train","valid","test"]:
        src = in_dir / f"{name}.json"
        if not src.exists():
            alt = in_dir / name
            if alt.exists(): src = alt
            else:
                print(f"Skip {name}, not found.")
                continue
        items = load_items(src)
        alpaca = to_alpaca(items)
        out_file = out_dir / f"{name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open('w',encoding='utf-8') as f:
            json.dump(alpaca, f, ensure_ascii=False, indent=2)
        print(f"Wrote {out_file} ({len(alpaca)} samples)")

if __name__ == "__main__":
    main()
