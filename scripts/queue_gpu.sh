#!/usr/bin/env bash
# Run every GPU experiment one after another. Each step resumes from its JSON Lines file.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() {
    for attempt in 1 2 3; do
        echo "=== $(date +%H:%M) attempt $attempt: $*"
        uv run python "$@" && return 0
        echo "FAILED attempt $attempt: $*"
        taskkill //F //IM llama-server.exe > /dev/null 2>&1
        sleep 10
    done
}

run scripts/run_cifar.py --model main --size 240 --per-class 100
run scripts/run_cifar.py --model main --size 32 --per-class 50
for p in 1 2 3; do run scripts/run_cifar.py --model varloop --passes "$p" --size 240 --per-class 50; done
run scripts/run_gsm8k.py --model main --effort none low medium high --n 50
for p in 1 2 3; do run scripts/run_gsm8k.py --model varloop --passes "$p" --effort low --n 50; done
run scripts/bench_speed.py
run scripts/run_cifar.py --model main --size 240 --per-class 1000
echo "=== $(date +%H:%M) queue done"
