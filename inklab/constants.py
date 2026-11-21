"""Central constants: paths, model files, server defaults, and dataset labels."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = REPO_ROOT / "bin"
MODELS_DIR = REPO_ROOT / "models"
# Datasets live in a data folder next to the repositories, shared by every project in the parent folder.
DATA_DIR = REPO_ROOT.parent / "data"
RUNS_DIR = REPO_ROOT / "runs"
LOGS_DIR = REPO_ROOT / "logs"
FIGURES_DIR = REPO_ROOT / "figures"

SERVER_EXE = BIN_DIR / "llama-server.exe"
BENCH_EXE = BIN_DIR / "llama-bench.exe"

MAIN_MODEL = MODELS_DIR / "inkling-2-mini" / "Inkling-2-Mini-IQ4_NL.gguf"
MAIN_MMPROJ = MODELS_DIR / "inkling-2-mini" / "mmproj-Inkling-2-Mini-BF16.gguf"
VARLOOP_MODEL = MODELS_DIR / "varloop" / "TML-d24-varloop-research-IQ4_NL.gguf"
VARLOOP_MMPROJ = MODELS_DIR / "varloop" / "mmproj-TML-d24-varloop-research-BF16.gguf"

LOOP_COUNT_KEY = "inkling2_loop.loop.count"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8091
SERVER_READY_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 900.0
DEFAULT_N_CPU_MOE = 14
DEFAULT_PARALLEL = 4
CONTEXT_PER_SLOT = 4096
# llama.cpp b12600 returns NaN logits for this model on CUDA when a micro-batch holds 32 or more tokens.
MICRO_BATCH = 16
LOGICAL_BATCH = 512

CIFAR_DIR = DATA_DIR / "cifar-10-batches-py"
CIFAR_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CIFAR_NATIVE_SIZE = 32
# The vision encoder uses 20 px patches merged 2x2, so a 40 px square is one image token.
VISION_TOKEN_EDGE = 40

GSM8K_REPO = "openai/gsm8k"
GSM8K_TEST_FILE = "main/test-00000-of-00001.parquet"

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
