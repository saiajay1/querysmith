#!/usr/bin/env python3
"""Live dashboard for eval/eval.py — tails its log and shows running
execution accuracy per (model x dataset).

Usage:
  python eval/watch_eval.py --log <eval-output-file>
"""
import argparse
import re
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

LOAD_RE = re.compile(r"==> Loading \[(base|fine-tuned)\]")
PROG_RE = re.compile(r"(spider|bird)\s+\[(\d+)/(\d+)\]\s+exec_acc=([\d.]+)%")
DONE_RE = re.compile(r"(spider|bird):\s+exec_acc=([\d.]+)%\s+valid_sql=([\d.]+)%\s+\(n=(\d+)\)")

BAR_W = 24


def bar(done, total):
    if not total:
        return "·" * BAR_W
    f = int(BAR_W * done / total)
    return "█" * f + "·" * (BAR_W - f)


def render(state):
    t = Table(box=None, expand=True)
    for c in ("model", "dataset", "progress", "exec-acc", "valid"):
        t.add_column(c, style="cyan" if c in ("model", "dataset") else "white")
    for model in ("base", "fine-tuned"):
        for ds in ("spider", "bird"):
            s = state.get((model, ds), {})
            done, total = s.get("i", 0), s.get("n", 0)
            acc = s.get("acc")
            valid = s.get("valid")
            mark = "[green]✓[/]" if s.get("final") else ""
            color = "green" if model == "fine-tuned" else "white"
            t.add_row(f"[{color}]{model}[/]", ds,
                      f"{bar(done, total)} {done}/{total or '?'}",
                      f"[bold]{acc:.1f}%[/]" if acc is not None else "—",
                      f"{valid:.1f}% {mark}" if valid is not None else "")
    return Panel(t, title="[bold]🐦 Querysmith — execution-accuracy eval[/]",
                 border_style="blue", padding=(1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    path = Path(args.log)
    state, current = {}, "base"
    console = Console()
    with Live(render(state), console=console, refresh_per_second=4) as live:
        with path.open() as f:
            while True:
                line = f.readline()
                if not line:
                    live.update(render(state))
                    time.sleep(0.4)
                    if "Done" in "".join(state.get("_tail", [])):
                        break
                    continue
                m = LOAD_RE.search(line)
                if m:
                    current = m.group(1)
                m = PROG_RE.search(line)
                if m:
                    ds, i, n, acc = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
                    state[(current, ds)] = {**state.get((current, ds), {}), "i": i, "n": n, "acc": acc}
                m = DONE_RE.search(line)
                if m:
                    ds, acc, valid, n = m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))
                    state[(current, ds)] = {"i": n, "n": n, "acc": acc, "valid": valid, "final": True}
                if "wrote eval/results.json" in line:
                    break
                live.update(render(state))
    console.print("[green]eval finished[/] — see eval/results.json")


if __name__ == "__main__":
    main()
