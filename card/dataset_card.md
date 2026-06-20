---
license: cc-by-sa-4.0
task_categories:
  - text-generation
language:
  - en
tags:
  - text-to-sql
  - sql
  - spider
  - bird
size_categories:
  - 10K<n<100K
---

# querysmith-spider-bird

Schema-grounded **text-to-SQL** training data used to fine-tune
[`ajayk007/Qwen2.5-Coder-7B-Querysmith`](https://huggingface.co/ajayk007/Qwen2.5-Coder-7B-Querysmith).
~13.7k examples derived from [Spider](https://yale-lily.github.io/spider) and
[BIRD](https://bird-bench.github.io/).

## Format
mlx-lm chat format, one example per line:
```json
{"messages": [
  {"role": "system", "content": "You are a text-to-SQL generator ..."},
  {"role": "user", "content": "Schema:\nCREATE TABLE ...\n\nQuestion: ..."},
  {"role": "assistant", "content": "SELECT ..."}
]}
```
The user turn contains the database's `CREATE TABLE` DDL (and, for BIRD examples, a
`Hint:` line with external-knowledge evidence) followed by the question. Schemas longer
than 3,500 chars are dropped to keep prompts trainable.

| File | Rows |
| --- | --- |
| `train.jsonl` | 13,685 |
| `valid.jsonl` | 300 |

Reproduce with `data/download_data.py` + `data/build_dataset.py` in the
[GitHub repo](https://github.com/saiajay1/querysmith).

## Attribution & license
This is a derivative of **Spider** (Yu et al., 2018) and **BIRD** (Li et al., 2023),
both licensed **CC BY-SA 4.0**. This dataset is released under the same
**CC BY-SA 4.0** license. Please cite the original Spider and BIRD papers if you use it.
