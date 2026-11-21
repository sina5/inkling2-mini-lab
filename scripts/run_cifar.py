"""Zero-shot CIFAR-10 classification with Inkling 2 Mini through llama-server."""

import argparse

from inklab.cifar import CifarTestSet
from inklab.client import ChatClient
from inklab.constants import CIFAR_DIR, DEFAULT_N_CPU_MOE, DEFAULT_PARALLEL
from inklab.experiments import ModelChoice, RunSpec, balanced_subset
from inklab.runner import JsonlRunner
from inklab.server import LlamaServer
from inklab.zero_shot import ZeroShotClassifier


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[m.value for m in ModelChoice], default="main")
    parser.add_argument("--passes", type=int, default=None, help="loop passes; default keeps the file value")
    parser.add_argument("--size", type=int, default=240, help="image edge in pixels after resizing")
    parser.add_argument("--per-class", type=int, default=200, help="images per class; 1000 is the full test set")
    parser.add_argument("--effort", default="none")
    parser.add_argument("--budget", type=int, default=-1, help="reasoning token budget; -1 is no limit")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--n-cpu-moe", type=int, default=DEFAULT_N_CPU_MOE)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    return parser.parse_args()


def main() -> None:
    """Start the server, classify the subset, and append results."""
    args = parse_args()
    spec = RunSpec("cifar", ModelChoice(args.model), args.passes, args.effort,
                   f"s{args.size}" + (f"_b{args.budget}" if args.budget >= 0 else ""))
    dataset = CifarTestSet(CIFAR_DIR).load()
    indices = balanced_subset(dataset.labels, args.per_class)
    runner = JsonlRunner(spec.output, workers=args.parallel)
    with LlamaServer(spec.server_config(True, args.n_cpu_moe, args.parallel), spec.server_log) as server:
        classifier = ZeroShotClassifier(ChatClient(server.base_url), dataset.class_names, args.effort, args.max_tokens,
                                        args.budget)

        def job(index: int) -> dict:
            p = classifier.classify(index, int(dataset.labels[index]), dataset.png(index, args.size))
            r = p.result
            return {"index": p.index, "label": p.label, "predicted": p.predicted, "answer": p.answer,
                    "reasoning": r.reasoning, "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens, "seconds": r.seconds, "finish": r.finish_reason}

        added = runner.run(indices, job)
    print(f"{spec.name}: added {added} records to {spec.output.name}")


if __name__ == "__main__":
    main()
