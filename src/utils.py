import csv
import gzip
import json
import time
from typing import Any, Dict, List, Union


class Timer:
    """
    Context manager to measure and report execution time of code blocks.
    """

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"Elapsed time: {self.interval:.4f} seconds")


def load_json(filepath: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Loads and parses a standard JSON file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidates(filepath: str) -> List[Dict[str, Any]]:
    """
    Loads candidates from a JSONL or gzipped JSONL (.gz) file.
    """
    candidates = []
    if filepath.endswith(".gz"):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    return candidates


def save_csv(data: List[Dict[str, Any]], filepath: str, headers: List[str]) -> None:
    """
    Saves a list of dictionaries to a CSV file.
    """
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in data:
            # Only output specified headers
            writer.writerow({k: row.get(k, "") for k in headers})
