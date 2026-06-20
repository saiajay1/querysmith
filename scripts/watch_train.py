#!/usr/bin/env python3
"""Live training dashboard: launches mlx_lm.lora and renders progress, train/val
loss, throughput, ETA, and a loss sparkline.  Requires: rich

Usage:
  python scripts/watch_train.py                 # run training + live dashboard
  python scripts/watch_train.py --log file.log  # tail an existing log instead
"""
import argparse
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
TITLE = "🗄️ querysmith — LoRA fine-tune"
SPARK = "▁▂▃▄▅▆▇█"
TRAIN_RE = re.compile(
    r"Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.eE+-]+), "
    r"It/sec ([\d.]+), Tokens/sec ([\d.]+), Trained Tokens (\d+), Peak mem ([\d.]+) GB")
VAL_RE = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
SAVE_RE = re.compile(r"Saved adapter weights")


def sparkline(values, width=48):
    if not values:
        return ""
    vals = list(values)[-width:]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(SPARK[min(7, int((v - lo) / rng * 7))] for v in vals)


class State:
    def __init__(self, total):
        self.total = total
        self.iter = 0
        self.train_loss = self.val_loss = self.it_sec = self.tok_sec = self.lr = None
        self.trained_tokens = self.saves = 0
        self.peak_mem = None
        self.losses = deque(maxlen=200)
        self.val_losses = deque(maxlen=64)
        self.start = time.monotonic()
        self.status = "starting"
        self.first_loss = None

    def feed(self, line):
        m = TRAIN_RE.search(line)
        if m:
            self.iter = int(m.group(1)); self.train_loss = float(m.group(2))
            self.lr = float(m.group(3)); self.it_sec = float(m.group(4))
            self.tok_sec = float(m.group(5)); self.trained_tokens = int(m.group(6))
            self.peak_mem = float(m.group(7)); self.losses.append(self.train_loss)
            if self.first_loss is None:
                self.first_loss = self.train_loss
            self.status = "training"; return
        m = VAL_RE.search(line)
        if m:
            self.iter = max(self.iter, int(m.group(1)))
            self.val_loss = float(m.group(2)); self.val_losses.append(self.val_loss); return
        if SAVE_RE.search(line):
            self.saves += 1


def render(st, progress, task_id):
    progress.update(task_id, completed=min(st.iter, st.total))
    elapsed = time.monotonic() - st.start
    table = Table(box=None, expand=True, pad_edge=False, show_header=False)
    for _ in range(4):
        table.add_column()

    def fmt(v, suf="", nd=3):
        return f"{v:.{nd}f}{suf}" if isinstance(v, float) else (str(v) if v is not None else "—")

    drop = f"  ([green]↓{st.first_loss - st.train_loss:.2f}[/])" if st.first_loss and st.train_loss else ""
    table.add_row("train loss", fmt(st.train_loss) + drop, "val loss", f"[magenta]{fmt(st.val_loss)}[/]")
    table.add_row("it/sec", fmt(st.it_sec, nd=2), "tokens/sec", fmt(st.tok_sec, "", 0))
    table.add_row("trained tokens", f"{st.trained_tokens:,}", "peak mem", fmt(st.peak_mem, " GB", 2))
    table.add_row("learning rate", f"{st.lr:.1e}" if st.lr else "—", "checkpoints", str(st.saves))
    table.add_row("elapsed", f"{elapsed:5.1f}s", "status", f"[yellow]{st.status}[/]")
    spark = (f"[green]{sparkline(st.losses)}[/]\n[dim]loss min {min(st.losses):.3f} → now {st.train_loss:.3f}[/]"
             if st.losses else "[dim]waiting for first loss…[/]")
    return Panel(Group(progress, table, Panel(spark, title="loss curve", border_style="green", padding=(0, 1))),
                 title=f"[bold]{TITLE}[/]  [dim]{st.iter}/{st.total} iters[/]", border_style="blue", padding=(1, 2))


def line_source(args):
    if args.log:
        path = Path(args.log)
        while not path.exists():
            time.sleep(0.2)
        with path.open() as f:
            while True:
                line = f.readline()
                yield line if line else None
                if not line:
                    time.sleep(0.2)
    else:
        proc = subprocess.Popen([sys.executable, "-m", "mlx_lm", "lora", "--config", args.config],
                                cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            yield line
        proc.wait(); yield "__DONE__"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/lora_config.yaml")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    cfg = (ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    total = int(yaml.safe_load(cfg.read_text()).get("iters", 400))

    console = Console()
    st = State(total)
    progress = Progress(TextColumn("[bold blue]progress"), BarColumn(bar_width=None),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeRemainingColumn(), expand=True)
    task_id = progress.add_task("train", total=total)
    done = False
    with Live(render(st, progress, task_id), console=console, refresh_per_second=8) as live:
        for line in line_source(args):
            if line == "__DONE__":
                st.status = "done ✓"; done = True
            elif line:
                st.feed(line)
            live.update(render(st, progress, task_id))
            if done:
                break
    if st.first_loss and st.train_loss:
        best = min(st.val_losses) if st.val_losses else float("nan")
        console.print(f"\n[bold green]Finished.[/] train loss {st.first_loss:.2f} → "
                      f"{st.train_loss:.3f} | best val {best:.3f} | adapters in [cyan]adapters/[/]")


if __name__ == "__main__":
    main()
