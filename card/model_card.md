---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
  - mlx
  - lora
  - text-generation
  - text-to-sql
  - sql
language:
  - en
pipeline_tag: text-generation
library_name: mlx
---

# Qwen2.5-Coder-7B-Querysmith

A schema-grounded **text-to-SQL** model: given a database schema and a plain-English
question, it returns a single SQLite query. LoRA fine-tune of
[`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct),
trained and quantized on Apple Silicon with [MLX](https://github.com/ml-explore/mlx),
on a mix of [Spider](https://yale-lily.github.io/spider) and
[BIRD](https://bird-bench.github.io/).

This model is the artifact of an **honest investigation**, not a cherry-picked demo —
see Results and "What I learned".

## Results (execution accuracy)

Evaluated by **execution accuracy**: run the predicted and gold SQL against the real
SQLite database and compare result sets. Dev splits the model never saw in training.

| Metric | Base (7B) | Querysmith | Δ |
| --- | :---: | :---: | :---: |
| Spider exec-acc (n=180) | 61.7% | **67.2%** | **+5.5** |
| BIRD exec-acc (n=200) | 55.0% | 53.0% | −2.0 |
| Phrasing consistency (n=40) | 62.0% | 63.5% | +1.5 |

The fine-tune gives a solid **+5.5** on Spider, is flat/slightly down on the much
harder BIRD, and shows no meaningful phrasing-robustness gain.

## What I learned (the honest part)

This project started from a real observation: **Databricks Genie** (NL→SQL) works well
in its UI but is brittle to question phrasing when driven via API. The hypothesis was
that a fine-tuned, phrasing-robust model would close that gap.

The robustness eval did **not** support that hypothesis — and the *reason why* is the
real finding: **Genie's brittleness is a property of its retrieval / example-grounding
layer, not of the underlying LLM's phrasing sensitivity.** A strong base model
(Qwen2.5-Coder-7B) is already robust to surface paraphrases, so fine-tuning had little
robustness headroom to capture. The right lever for the Genie-API problem is better
example-grounding / retrieval — not fine-tuning the model.

The model itself is still a useful, honestly-benchmarked schema-grounded text-to-SQL
fine-tune with a real Spider gain.

## Usage

```python
from mlx_lm import load, generate
model, tok = load("ajayk007/Qwen2.5-Coder-7B-Querysmith")
SYSTEM = ("You are a text-to-SQL generator. Given a database schema and a question, "
          "reply with a single valid SQLite query. Output only the SQL on one line.")
schema = "CREATE TABLE employees(id, name, dept_id, salary);\nCREATE TABLE departments(id, name);"
question = "average salary per department name"
user = f"Schema:\n{schema}\n\nQuestion: {question}"
prompt = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    add_generation_prompt=True, tokenize=False)
print(generate(model, tok, prompt=prompt, max_tokens=192, verbose=False))
```

For BIRD-style questions you can add a `\n\nHint: <external knowledge>` line before
the question, matching the training format.

## Training
- **Method:** LoRA (rank 16, 8 layers), MLX, 1200 iters, batch 2, seq 1536. Val loss 1.52 → 0.137.
- **Data:** ~13.7k schema-grounded (schema + question → SQL) pairs from Spider + BIRD.
  See [`ajayk007/querysmith-spider-bird`](https://huggingface.co/datasets/ajayk007/querysmith-spider-bird).
- **Eval:** execution accuracy on Spider + BIRD dev; methodology in the
  [GitHub repo](https://github.com/saiajay1/querysmith).

## Limitations & safety
- SQLite dialect; schema must be provided in the prompt (no built-in retrieval).
- BIRD-hard questions needing external knowledge remain weak.
- **Review generated SQL before running it** against any real database.

## Related
Part of a series of focused "English → developer DSL" fine-tunes:
- [Qwen2.5-Coder-1.5B-Shellsmith](https://huggingface.co/ajayk007/Qwen2.5-Coder-1.5B-Shellsmith) — English → shell command.

## License
Apache-2.0 (model weights, inheriting from the base). Training data derives from
Spider and BIRD (CC BY-SA 4.0) — see the dataset card.
