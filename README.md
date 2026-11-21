# inkling2-mini-lab

This repository holds small experiments with [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on one 12 GB GPU (RTX 3080 Ti) with llama.cpp.

The experiments are:

1. Zero-shot CIFAR-10: the model names the class of each test image with no CIFAR-10 training.
2. Loop depth: the variable-loop research checkpoint runs its loop core 1, 2, or 3 times, and I measure accuracy and speed.
3. Loop gates: a probe reads the gates that control how much each pass takes from the previous pass.
4. Expert routing: the same probe records which of the 128 experts each token uses in each of the 48 layer visits.
5. Speed and reasoning effort: prompt and generation speed for each expert offload level, and GSM8K accuracy against the number of reasoning tokens.

## Requirements

- Windows with an NVIDIA GPU that has 12 GB of memory or more, and 32 GB of RAM.
- [uv](https://docs.astral.sh/uv/) for Python 3.12 or later.
- About 40 GB of free disk space for the two model files.
- Visual Studio 2026 with the C++ workload, only for the probe tool.

## Setup

1. Install the Python packages:

   ```console
   $ uv sync
   ```

2. Download llama.cpp build b12600 or later (`llama-*-bin-win-cuda-12.4-x64.zip` and the matching `cudart` file) and extract both into `bin/`.
3. Download the models into `models/`:

   ```console
   $ uvx --from huggingface_hub hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf mmproj-Inkling-2-Mini-BF16.gguf --local-dir models/inkling-2-mini
   $ uvx --from huggingface_hub hf download ggml-org/TML-d24-varloop-research-GGUF TML-d24-varloop-research-IQ4_NL.gguf mmproj-TML-d24-varloop-research-BF16.gguf --local-dir models/varloop
   ```

4. Datasets live in the shared folder `../data` next to this repository. Put the CIFAR-10 Python files in `../data/cifar-10-batches-py/`. The GSM8K test file downloads to `../data/gsm8k/` when you run this command:

   ```console
   $ uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('openai/gsm8k', 'main/test-00000-of-00001.parquet', repo_type='dataset', local_dir='../data/gsm8k')"
   ```

## Run the experiments

1. Run all GPU experiments in sequence. Each step continues from its result file if you stop it:

   ```console
   $ bash scripts/queue_gpu.sh
   ```

2. Build the probe tool (CPU only). Clone llama.cpp at commit `29b52d398` into `third_party/llama.cpp`, apply `probe/gate-names.patch`, copy `probe/` to `tools/inkling-probe/`, add `add_subdirectory(inkling-probe)` to `tools/CMakeLists.txt`, and build the `llama-inkling-probe` target.
3. Run the probes:

   ```console
   $ uv run python scripts/run_probes.py
   ```

4. Make the charts and `runs/summary.json`:

   ```console
   $ uv run python scripts/make_figures.py
   $ uv run python scripts/draw_architecture.py
   ```

## Known llama.cpp problem

With llama.cpp b12600, the CUDA build returns NaN logits for this model when one micro-batch holds 32 tokens or more. Short prompts work, so the problem is easy to miss. A CPU build from the same commit gives correct output. The scripts set `-ub 16` to avoid the problem. Prompt processing is slower with this setting.

## Layout

- `inklab/` holds the Python package: server control, the chat client, datasets, parsers, the probe reader, and the analysis.
- `scripts/` holds the experiment, probe, and chart scripts.
- `probe/` holds the C++ probe tool and the llama.cpp patch that names the loop gate tensors.
- `tests/` holds the unit tests. Run them with `uv run pytest`.
- `runs/`, `logs/`, `models/`, `bin/`, `data/`, and `third_party/` are local only and Git ignores them.

## License

The code is under the Apache 2.0 license. Inkling 2 Mini is also under the Apache 2.0 license.
