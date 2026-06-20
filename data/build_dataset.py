#!/usr/bin/env python3
"""Build schema-grounded text-to-SQL training data from the downloaded Spider+BIRD
slice (run data/download_data.py first).

Each record (mlx-lm chat format):
  system: the text-to-SQL instruction
  user:   schema (CREATE TABLEs) [+ BIRD hint] + the question
  assistant: the gold SQL

Writes data/train.jsonl + data/valid.jsonl.  Run:  python data/build_dataset.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

SYSTEM_PROMPT = (
    "You are a text-to-SQL generator. Given a database schema and a question, "
    "reply with a single valid SQLite query that answers it. Output only the SQL "
    "on one line, with no explanation and no markdown."
)

MAX_SCHEMA_CHARS = 3500   # drop examples whose schema is too long to train efficiently


def user_msg(schema: str, question: str, evidence: str = "") -> str:
    hint = f"\n\nHint: {evidence.strip()}" if evidence.strip() else ""
    return f"Schema:\n{schema}{hint}\n\nQuestion: {question.strip()}"


def to_record(schema, question, query, evidence=""):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg(schema, question, evidence)},
        {"role": "assistant", "content": " ".join(query.strip().split())},
    ]}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    schemas = json.loads((RAW / "schemas.json").read_text())
    records, dropped_noschema, dropped_long = [], 0, 0

    sources = [("spider", "spider_train.jsonl"), ("bird", "bird_train.jsonl")]
    for tag, fname in sources:
        rows = load_jsonl(RAW / fname)
        kept = 0
        for r in rows:
            schema = schemas.get(f"{tag}/{r['db_id']}")
            if not schema:
                dropped_noschema += 1
                continue
            if len(schema) > MAX_SCHEMA_CHARS:
                dropped_long += 1
                continue
            if not r.get("query"):
                continue
            records.append(to_record(schema, r["question"], r["query"], r.get("evidence", "")))
            kept += 1
        print(f"  {tag:6s}: kept {kept}/{len(rows)}")

    rng = random.Random(1707)
    rng.shuffle(records)
    n_valid = 300
    valid, train = records[:n_valid], records[n_valid:]

    for name, rows in [("train", train), ("valid", valid)]:
        with (HERE / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(rows):5d} -> {name}.jsonl")
    print(f"\ntotal {len(records)} records "
          f"(dropped: {dropped_noschema} no-schema, {dropped_long} schema-too-long)")


if __name__ == "__main__":
    main()
