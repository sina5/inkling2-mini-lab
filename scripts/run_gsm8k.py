"""GSM8K accuracy and reasoning length for one model, loop depth, and reasoning effort."""

import argparse

from inklab.client import ChatClient
from inklab.constants import DATA_DIR, DEFAULT_N_CPU_MOE, DEFAULT_PARALLEL, GSM8K_TEST_FILE
from inklab.experiments import ModelChoice, RunSpec
from inklab.gsm8k import FinalNumberGrader, Gsm8kTestSet
from inklab.runner import JsonlRunner
from inklab.server import LlamaServer


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[m.value for m in ModelChoice], default="main")
    parser.add_argument("--passes", type=int, default=None)
    parser.add_argument("--effort", nargs="+", default=["low"], help="one or more reasoning efforts")
    parser.add_argument("--n", type=int, default=50, help="first n test questions")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--n-cpu-moe", type=int, default=DEFAULT_N_CPU_MOE)
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    return parser.parse_args()


def main() -> None:
    """Start one server and run every requested effort on it."""
    args = parse_args()
    questions = Gsm8kTestSet(DATA_DIR / "gsm8k" / GSM8K_TEST_FILE).load()[: args.n]
    grader = FinalNumberGrader()
    first = RunSpec("gsm8k", ModelChoice(args.model), args.passes, args.effort[0], f"n{args.n}")
    with LlamaServer(first.server_config(False, args.n_cpu_moe, args.parallel), first.server_log) as server:
        client = ChatClient(server.base_url)
        for effort in args.effort:
            spec = RunSpec("gsm8k", ModelChoice(args.model), args.passes, effort, f"n{args.n}")

            def job(i: int, effort: str = effort) -> dict:
                q = questions[i]
                r = client.chat(grader.prompt(q), effort=effort, max_tokens=args.max_tokens)
                return {"index": i, "reference": q.answer, "extracted": grader.extract(r.content),
                        "correct": grader.correct(r.content, q), "answer": r.content, "reasoning": r.reasoning,
                        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                        "seconds": r.seconds, "finish": r.finish_reason}

            added = JsonlRunner(spec.output, workers=args.parallel).run(range(len(questions)), job, progress_every=10)
            print(f"{spec.name}: added {added} records", flush=True)


if __name__ == "__main__":
    main()
