"""Run many requests in parallel and append each record to a JSON Lines file."""

import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


class JsonlRunner:
    """Run one job per key with a thread pool, skipping keys that the output file already has.

    The output file is the checkpoint: an interrupted run continues where it stopped.
    """

    def __init__(self, output: Path, workers: int, key_field: str = "key") -> None:
        self.output = output
        self.workers = workers
        self.key_field = key_field
        self._lock = threading.Lock()

    def done_keys(self) -> set:
        """Return the keys that the output file already has."""
        if not self.output.exists():
            return set()
        with self.output.open(encoding="utf-8") as handle:
            return {json.loads(line)[self.key_field] for line in handle if line.strip()}

    def run(self, keys: Iterable, job: Callable[[object], dict], progress_every: int = 50) -> int:
        """Run job(key) for each new key and append its record. Return the number of new records."""
        todo = [key for key in keys if key not in self.done_keys()]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool, self.output.open("a", encoding="utf-8") as handle:
            futures = [pool.submit(self._keyed, job, key) for key in todo]
            for future in as_completed(futures):
                record = future.result()
                with self._lock:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                count += 1
                if count % progress_every == 0:
                    print(f"{self.output.name}: {count}/{len(todo)}", flush=True)
        return count

    def _keyed(self, job: Callable[[object], dict], key: object) -> dict:
        record = job(key)
        record[self.key_field] = key
        return record


def read_jsonl(path: Path) -> list[dict]:
    """Read every record of a JSON Lines file."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
