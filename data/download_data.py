#!/usr/bin/env python3
"""Download a lean Spider + BIRD slice for schema-grounded text-to-SQL.

Pulls only what we need:
  - Q/SQL pairs (Spider parquet via xlangai/spider; BIRD json via premai-io/birdbench)
  - schemas as text (Spider per-db schema.sql; BIRD rendered from train_tables.json)
  - dev-only SQLite databases (for execution-accuracy eval)

Writes into data/raw/.  Run:  python data/download_data.py
"""
import json
import random
import sqlite3
from pathlib import Path

RNG = random.Random(7)


def read_ddl(path, cap=262144):
    """Read only the CREATE TABLE DDL from a .sql file (drop INSERT data dumps).
    Bounded read so we never load a multi-hundred-MB data dump into memory."""
    with open(path, "r", errors="ignore") as f:
        head = f.read(cap)
    idx = head.upper().find("INSERT INTO")
    if idx != -1:
        head = head[:idx]
    return head.strip()

from datasets import load_dataset
from huggingface_hub import hf_hub_download

RAW = Path(__file__).resolve().parent / "raw"


def dl(repo, path, tries=3):
    """hf_hub_download with a few retries for transient network errors; None if absent."""
    from huggingface_hub.utils import EntryNotFoundError
    for attempt in range(tries):
        try:
            return hf_hub_download(repo, path, repo_type="dataset")
        except EntryNotFoundError:
            return None
        except Exception:
            if attempt == tries - 1:
                return None
    return None
DBDIR = RAW / "db"
(RAW).mkdir(parents=True, exist_ok=True)
(DBDIR / "spider").mkdir(parents=True, exist_ok=True)
(DBDIR / "bird").mkdir(parents=True, exist_ok=True)

# how many dev examples to keep for execution eval (kept tractable for a 7B on a Mac)
DEV_SAMPLE = 200


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows):5d} -> {path.name}")


def render_schema_from_tables(entry) -> str:
    """Render CREATE TABLE text from a Spider/BIRD tables.json entry."""
    tbls = entry["table_names_original"]
    cols = entry["column_names_original"]      # [[table_idx, col_name], ...] ([-1,'*'] first)
    types = entry.get("column_types", [])
    pks = set()
    for pk in (entry.get("primary_keys") or []):
        pks.update(pk if isinstance(pk, list) else [pk])
    lines = []
    for ti, tname in enumerate(tbls):
        coldefs = []
        for ci, (tidx, cname) in enumerate(cols):
            if tidx != ti:
                continue
            ctype = types[ci] if ci < len(types) else "text"
            pk = " PRIMARY KEY" if ci in pks else ""
            coldefs.append(f"  {cname} {ctype}{pk}")
        lines.append(f"CREATE TABLE {tname} (\n" + ",\n".join(coldefs) + "\n);")
    # foreign keys as a comment hint
    fks = entry.get("foreign_keys", [])
    if fks:
        names = {i: (cols[i][1], tbls[cols[i][0]]) for i in range(len(cols)) if cols[i][0] >= 0}
        fk_txt = "; ".join(f"{names[a][1]}.{names[a][0]} -> {names[b][1]}.{names[b][0]}"
                           for a, b in fks if a in names and b in names)
        if fk_txt:
            lines.append(f"-- foreign keys: {fk_txt}")
    return "\n".join(lines)


def schema_from_sqlite(path) -> str:
    con = sqlite3.connect(path)
    rows = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    con.close()
    return "\n".join(r[0].strip() + ";" for r in rows)


def main():
    schemas = {}   # "spider/<db>" or "bird/<db>" -> schema text

    # ---------------- SPIDER ----------------
    print("== Spider: pairs ==")
    sp = load_dataset("xlangai/spider")
    sp_train = [{"db_id": r["db_id"], "question": r["question"], "query": r["query"]}
                for r in sp["train"]]
    sp_dev = [{"db_id": r["db_id"], "question": r["question"], "query": r["query"]}
              for r in sp["validation"]]
    write_jsonl(RAW / "spider_train.jsonl", sp_train)

    # Spider schemas (all dbs that appear) from premai-io/spider schema.sql
    print("== Spider: schemas (schema.sql per db) ==")
    sp_dbs = sorted({r["db_id"] for r in sp_train + sp_dev})
    for db in sp_dbs:
        p = (dl("premai-io/spider", f"database/{db}/schema.sql")
             or dl("premai-io/spider", f"database/{db}/{db}.sql"))
        if p:
            schemas[f"spider/{db}"] = read_ddl(p)
        else:
            print(f"   ! no schema for {db}")
    print(f"   got {sum(1 for k in schemas if k.startswith('spider/'))} spider schemas")

    # Spider dev DBs (sample ACROSS databases for a representative eval)
    RNG.shuffle(sp_dev)
    sp_dev = sp_dev[:DEV_SAMPLE]
    write_jsonl(RAW / "spider_dev.jsonl", sp_dev)
    print("== Spider: dev sqlite DBs ==")
    for db in sorted({r["db_id"] for r in sp_dev}):
        p = dl("premai-io/spider", f"database/{db}/{db}.sqlite")
        if p:
            (DBDIR / "spider" / f"{db}.sqlite").write_bytes(Path(p).read_bytes())
        else:
            print(f"   ! no sqlite for {db}")

    # ---------------- BIRD ----------------
    print("== BIRD: pairs + train schemas ==")
    bt = hf_hub_download("premai-io/birdbench", "train/train.json", repo_type="dataset")
    bv = hf_hub_download("premai-io/birdbench", "validation/validation.json", repo_type="dataset")
    btab = hf_hub_download("premai-io/birdbench", "train/train_databases/train_tables.json",
                           repo_type="dataset")
    bird_train_raw = json.loads(Path(bt).read_text())
    bird_dev_raw = json.loads(Path(bv).read_text())
    for entry in json.loads(Path(btab).read_text()):
        try:
            schemas[f"bird/{entry['db_id']}"] = render_schema_from_tables(entry)
        except Exception as e:
            print(f"   ! schema render failed for {entry.get('db_id')}: {type(e).__name__}")
    print(f"   got {sum(1 for k in schemas if k.startswith('bird/'))} bird train schemas")

    def bird_row(r):
        return {"db_id": r["db_id"], "question": r["question"],
                "query": r.get("SQL") or r.get("query"), "evidence": r.get("evidence", "")}
    write_jsonl(RAW / "bird_train.jsonl", [bird_row(r) for r in bird_train_raw])

    bird_dev_all = [bird_row(r) for r in bird_dev_raw]
    RNG.shuffle(bird_dev_all)
    bird_dev = bird_dev_all[:DEV_SAMPLE]
    write_jsonl(RAW / "bird_dev.jsonl", bird_dev)
    print("== BIRD: dev sqlite DBs + dev schemas ==")
    for db in sorted({r["db_id"] for r in bird_dev}):
        p = dl("premai-io/birdbench", f"validation/dev_databases/{db}/{db}.sqlite")
        if p:
            dest = DBDIR / "bird" / f"{db}.sqlite"
            dest.write_bytes(Path(p).read_bytes())
            schemas[f"bird/{db}"] = schema_from_sqlite(dest)   # authoritative dev schema
        else:
            print(f"   ! no dev sqlite for {db}")

    (RAW / "schemas.json").write_text(json.dumps(schemas, indent=0))
    print(f"\nDONE. {len(schemas)} schemas, DBs in data/raw/db/, pairs in data/raw/*.jsonl")


if __name__ == "__main__":
    main()
