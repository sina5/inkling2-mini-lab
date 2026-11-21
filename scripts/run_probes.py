"""Run llama-inkling-probe on every probe prompt (CPU build) and save one dump per prompt."""

import argparse
import json
import os
import subprocess

from inklab.cifar import CifarTestSet
from inklab.constants import CIFAR_DIR, LOGS_DIR, MAIN_MMPROJ, MAIN_MODEL, REPO_ROOT, RUNS_DIR
from inklab.experiments import balanced_subset
from inklab.probe_prompts import ChatFormatter, default_prompts

PROBE_EXE = REPO_ROOT / "third_party" / "llama.cpp" / "build-cpu" / "bin" / "Release" / "llama-inkling-probe.exe"
PROBE_DIR = RUNS_DIR / "probe"
IMAGE_SIZE = 240


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-per-class", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Write the images, run the probe for each prompt, and save an index file."""
    args = parse_args()
    dataset = CifarTestSet(CIFAR_DIR).load()
    indices = balanced_subset(dataset.labels, args.images_per_class)
    formatter = ChatFormatter()
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for prompt in default_prompts(indices):
        out = PROBE_DIR / f"{prompt.name}.bin"
        prompt_path = PROBE_DIR / f"{prompt.name}.txt"
        prompt_path.write_text(formatter.render(prompt), encoding="utf-8")
        command = [str(PROBE_EXE), "-m", str(MAIN_MODEL), "--mmproj", str(MAIN_MMPROJ),
                   "-f", str(prompt_path), "-t", str(args.threads), "-c", "2048"]
        if prompt.cifar_index is not None:
            image_path = PROBE_DIR / f"{prompt.name}.png"
            image_path.write_bytes(dataset.png(prompt.cifar_index, IMAGE_SIZE))
            command += ["--image", str(image_path)]
        if not out.exists():
            with (LOGS_DIR / "probe_last.log").open("wb") as log:
                subprocess.run(command, env={**os.environ, "INKLING_PROBE_OUT": str(out)},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"probed {prompt.name}", flush=True)
        label = int(dataset.labels[prompt.cifar_index]) if prompt.cifar_index is not None else None
        index.append({"name": prompt.name, "category": prompt.category.value, "label": label})
    (PROBE_DIR / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
