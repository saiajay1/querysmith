#!/usr/bin/env python3
"""Execution-accuracy eval for Querysmith on Spider + BIRD dev sets.

For each dev question we generate SQL, run BOTH the prediction and the gold query
against the real SQLite database, and compare result sets (order-insensitive). This
is the field-standard 'execution accuracy' metric.

Metrics per dataset:
  exec_acc   : % where predicted result set == gold result set
  valid_sql  : % where the predicted SQL runs without error

Usage:
  python eval/eval.py                 # base vs fine-tuned on Spider + BIRD dev
  python eval/eval.py --limit 40      # quick smoke test
  python eval/eval.py --datasets spider
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

from mlx_lm import load, generate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a text-to-SQL generator. Given a database schema and a question, "
    "reply with a single valid SQLite query that answers it. Output only the SQL "
    "on one line, with no explanation and no markdown."
)


def user_msg(schema, question, evidence=""):
    hint = f"\n\nHint: {evidence.strip()}" if evidence.strip() else ""
    return f"Schema:\n{schema}{hint}\n\nQuestion: {question.strip()}"


def extract_sql(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:sql)?\s*(.+?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    m = re.search(r"(?is)\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b", t)
    if m:
        t = t[m.start():]
    t = t.split(";")[0]
    return " ".join(t.split())


def run_sql(db_path, sql, max_ops=3_000_000):
    """Return frozenset of result rows, or None if invalid/timed out."""
    if not sql:
        return None
    con = sqlite3.connect(db_path)
    counter = {"n": 0}
    def progress():
        counter["n"] += 1
        return 1 if counter["n"] > max_ops else 0   # nonzero aborts long queries
    con.set_progress_handler(progress, 10000)
    try:
        rows = con.execute(sql).fetchall()
        return frozenset(tuple(r) for r in rows)
    except Exception:
        return None
    finally:
        con.close()


def load_dev(tag):
    rows = [json.loads(l) for l in (RAW / f"{tag}_dev.jsonl").open()]
    schemas = json.loads((RAW / "schemas.json").read_text())
    out = []
    for r in rows:
        schema = schemas.get(f"{tag}/{r['db_id']}")
        db = RAW / "db" / tag / f"{r['db_id']}.sqlite"
        if schema and db.exists():
            out.append({**r, "schema": schema, "db": str(db), "tag": tag})
    return out


def predict(model, tok, row):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_msg(row["schema"], row["question"], row.get("evidence", ""))}],
        add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, prompt=prompt, max_tokens=192, verbose=False)
    return extract_sql(out)


def evaluate(adapter, datasets, label, limit):
    print(f"\n==> Loading [{label}]" + (f" adapter={adapter}" if adapter else ""))
    model, tok = load(MODEL, adapter_path=adapter)
    results = {}
    for tag in datasets:
        rows = load_dev(tag)
        if limit:
            rows = rows[:limit]
        ok = valid = 0
        for i, row in enumerate(rows, 1):
            pred = predict(model, tok, row)
            pset = run_sql(row["db"], pred)
            gset = run_sql(row["db"], row["query"])
            if pset is not None:
                valid += 1
            if pset is not None and gset is not None and pset == gset:
                ok += 1
            if i % 25 == 0:
                print(f"   {tag} [{i}/{len(rows)}] exec_acc={100*ok/i:.1f}%")
        n = len(rows) or 1
        results[tag] = {"n": len(rows), "exec_acc": round(100*ok/n, 1),
                        "valid_sql": round(100*valid/n, 1)}
        print(f"   {tag}: exec_acc={results[tag]['exec_acc']}%  "
              f"valid_sql={results[tag]['valid_sql']}%  (n={len(rows)})")
    return {"label": label, **results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapters")
    ap.add_argument("--datasets", nargs="+", default=["spider", "bird"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--ft-only", action="store_true")
    args = ap.parse_args()

    out = []
    if not args.ft_only:
        out.append(evaluate(None, args.datasets, "base", args.limit))
    if not args.base_only:
        out.append(evaluate(args.adapter, args.datasets, "fine-tuned", args.limit))

    print("\n" + "=" * 60)
    print(f"{'model':12s}" + "".join(f"{d+' exec%':>14s}" for d in args.datasets))
    print("-" * 60)
    for r in out:
        print(f"{r['label']:12s}" + "".join(f"{r[d]['exec_acc']:>14.1f}" for d in args.datasets))
    print("=" * 60)
    (ROOT / "eval" / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote eval/results.json")


if __name__ == "__main__":
    main()
