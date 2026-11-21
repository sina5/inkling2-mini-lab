"""Measure prompt and generation speed for several expert offload levels with llama-bench."""

import argparse
import json
import subprocess

from inklab.constants import BENCH_EXE, MAIN_MODEL, MICRO_BATCH, RUNS_DIR


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cpu-moe", default="12,14,16,18,20,24", help="comma-separated offload levels")
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run llama-bench and save its JSON output."""
    args = parse_args()
    command = [str(BENCH_EXE), "-m", str(MAIN_MODEL), "-ngl", "99", "-ncmoe", args.n_cpu_moe,
               "-ub", str(MICRO_BATCH), "-p", "512", "-n", "128", "-r", str(args.repetitions), "-o", "json"]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = json.loads(completed.stdout)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "bench_speed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        kind = "prompt" if row["n_prompt"] else "generate"
        print(f"n_cpu_moe={row['n_cpu_moe']:>2} {kind:8} {row['avg_ts']:7.1f} tokens/s")


if __name__ == "__main__":
    main()
