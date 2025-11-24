"""
JudgeBench 평가 스크립트: 두 응답 리스트(pkl)를 비교하고 요약 통계를 출력/저장합니다.
사용 전 OPENAI_API_KEY를 환경변수로 설정하세요.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
from typing import Any, Dict, List, Optional

from datasets import load_dataset  # ScalerLab/JudgeBench
from openai import OpenAI


def load_list(path: str) -> List[str]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _norm(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip().upper()
    if x in ("A", "B", "C"):
        return x
    return None


async def judge_pair(
    client: OpenAI,
    model: str,
    question: str,
    a: str,
    b: str,
) -> Optional[str]:
    prompt = f"""You are an expert judge evaluating two responses to the same question.

Question:
{question}

Response A:
{a}

Response B:
{b}

Which response is better?
- Answer "A" if Response A is better
- Answer "B" if Response B is better
- Answer "C" if they are equally good

Answer(answer ONLY):"""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return _norm(r.choices[0].message.content)
    except Exception:
        return None


def summarize(temp: List[Optional[str]]) -> Dict[str, Any]:
    total = len(temp)
    valid = [t for t in temp if t in ("A", "B")]
    a_w = sum(1 for t in valid if t == "A")
    b_w = sum(1 for t in valid if t == "B")
    ties = sum(1 for t in temp if t == "C")
    invalid = total - len(valid) - ties
    return {
        "pairs_total": total,
        "valid_pairs": len(valid),
        "invalid": invalid,
        "A_wins": a_w,
        "B_wins": b_w,
        "Ties": ties,
        "A_win_rate_%": round(100.0 * a_w / len(valid), 2) if valid else 0.0,
        "B_win_rate_%": round(100.0 * b_w / len(valid), 2) if valid else 0.0,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-a", required=True, help="pickle path for response list A")
    ap.add_argument("--path-b", required=True, help="pickle path for response list B")
    ap.add_argument("--judge-model", required=True, help="OpenAI model id (e.g., gpt-4o-mini)")
    ap.add_argument("--reverse-order", action="store_true", help="also judge (B,A) to reduce order bias")
    ap.add_argument("--out", help="optional JSONL output path for raw judgments")
    args = ap.parse_args()

    # Load data
    resp_a = load_list(args.path_a)
    resp_b = load_list(args.path_b)
    ds = load_dataset("ScalerLab/JudgeBench")["claude"]
    questions = ds["question"]
    if len(questions) != len(resp_a) or len(resp_a) != len(resp_b):
        raise ValueError("Length mismatch among questions/A/B")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if client.api_key is None:
        raise EnvironmentError("Set OPENAI_API_KEY")

    temp: List[Optional[str]] = []
    temp_rev: List[Optional[str]] = []
    raw_records: List[Dict[str, Any]] = []

    for i, q in enumerate(questions):
        r = await judge_pair(client, args.judge_model, q, resp_a[i], resp_b[i])
        temp.append(r)
        raw_records.append({"pair_id": i, "order": "A_first", "judgment": r})
        if args.reverse_order:
            r2 = await judge_pair(client, args.judge_model, q, resp_b[i], resp_a[i])
            temp_rev.append(r2)
            raw_records.append({"pair_id": i, "order": "B_first", "judgment": r2})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for rec in raw_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("=== A first ===")
    print(json.dumps(summarize(temp), ensure_ascii=False, indent=2))
    if args.reverse_order:
        print("\n=== B first ===")
        print(json.dumps(summarize(temp_rev), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
