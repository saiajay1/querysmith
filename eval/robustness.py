#!/usr/bin/env python3
"""Phrasing-robustness eval — the Querysmith thesis, measured.

For each dev question we create several surface paraphrases (different ways a user
might phrase the *same* request via an API), generate SQL for each, and check
execution correctness. A phrasing-robust model answers all paraphrases correctly;
a brittle one only handles the wording it expects.

Metrics per model:
  consistency : mean fraction of paraphrases answered correctly (per question)
  robust_acc  : % of questions correct on ALL paraphrases
  best_acc    : % of questions correct on AT LEAST ONE paraphrase

Usage:  python eval/robustness.py [--per 20]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
E = importlib.import_module("eval")          # reuse extract_sql/run_sql/user_msg/load_dev/MODEL
from mlx_lm import load, generate

# Surface paraphrases: same intent, different phrasing/format (mimics the API gap).
PARAPHRASES = [
    lambda q: q,
    lambda q: f"Could you tell me {q[0].lower() + q[1:]}",
    lambda q: f"{q.rstrip('?. ')}? Please return the SQL.",
    lambda q: f"I want to know {q[0].lower() + q[1:]}",
    lambda q: f"Question: {q}\nAnswer with a SQL query only.",
]


def gen(model, tok, schema, question, evidence=""):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": E.SYSTEM_PROMPT},
         {"role": "user", "content": E.user_msg(schema, question, evidence)}],
        add_generation_prompt=True, tokenize=False)
    return E.extract_sql(generate(model, tok, prompt=prompt, max_tokens=192, verbose=False))


def eval_model(adapter, label, per):
    print(f"\n==> [{label}]" + (f" adapter={adapter}" if adapter else ""))
    model, tok = load(E.MODEL, adapter_path=adapter)
    rows = []
    for tag in ("spider", "bird"):
        rows += E.load_dev(tag)[:per]
    tot_correct = tot = robust = best = nq = 0
    for i, r in enumerate(rows, 1):
        gold = E.run_sql(r["db"], r["query"])
        if gold is None:
            continue
        nq += 1
        c = 0
        for pf in PARAPHRASES:
            ps = E.run_sql(r["db"], gen(model, tok, r["schema"], pf(r["question"]), r.get("evidence", "")))
            if ps is not None and ps == gold:
                c += 1
        tot_correct += c
        tot += len(PARAPHRASES)
        robust += (c == len(PARAPHRASES))
        best += (c > 0)
        if i % 10 == 0:
            print(f"   [{i}/{len(rows)}] running consistency={100*tot_correct/tot:.1f}%")
    return {"label": label, "questions": nq,
            "consistency": round(100*tot_correct/tot, 1),
            "robust_acc": round(100*robust/nq, 1),
            "best_acc": round(100*best/nq, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20, help="questions per dataset")
    ap.add_argument("--adapter", default="adapters")
    args = ap.parse_args()

    out = [eval_model(None, "base", args.per), eval_model(args.adapter, "fine-tuned", args.per)]
    print("\n" + "=" * 60)
    print(f"{'model':12s}{'consistency':>14s}{'robust_acc':>13s}{'best_acc':>11s}")
    print("-" * 60)
    for r in out:
        print(f"{r['label']:12s}{r['consistency']:>13.1f}%{r['robust_acc']:>12.1f}%{r['best_acc']:>10.1f}%")
    print("=" * 60)
    print(f"{len(PARAPHRASES)} paraphrases/question. Consistency gap = the phrasing-robustness win.")
    (ROOT / "eval" / "robustness.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote eval/robustness.json")


if __name__ == "__main__":
    main()
