"""GSM8K grade-school math questions and a final-number answer checker."""

import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from inklab.errors import DatasetError

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
UNICODE_MINUS = "−"


@dataclass(frozen=True)
class MathQuestion:
    """One GSM8K test question with its reference number."""

    index: int
    question: str
    answer: float


def to_number(text: str) -> float | None:
    """Parse a number such as '1,234' or '-3.5'; return None if it does not parse."""
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


class Gsm8kTestSet:
    """The 1,319 GSM8K test questions from the Hugging Face parquet file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[MathQuestion]:
        """Read all questions.

        Raises:
            DatasetError: If the file is missing or a reference answer has no '####' number.
        """
        if not self.path.exists():
            raise DatasetError(f"GSM8K file not found: {self.path}")
        rows = pq.read_table(self.path).to_pylist()
        return [MathQuestion(i, row["question"], self._reference(row["answer"])) for i, row in enumerate(rows)]

    @staticmethod
    def _reference(answer: str) -> float:
        value = to_number(answer.split("####")[-1].strip())
        if value is None:
            raise DatasetError(f"reference answer has no number: {answer[-40:]!r}")
        return value


class FinalNumberGrader:
    """Grade a reply by the last number in it, so the prompt asks for the number at the end."""

    PROMPT_SUFFIX = "\n\nSolve the problem. End your reply with the final answer as a plain number on its own line."

    def prompt(self, question: MathQuestion) -> str:
        """Return the full prompt for one question."""
        return question.question + self.PROMPT_SUFFIX

    def extract(self, reply: str) -> float | None:
        """Return the last number in the reply, or None."""
        matches = NUMBER.findall(reply.replace(UNICODE_MINUS, "-"))
        return to_number(matches[-1]) if matches else None

    def correct(self, reply: str, question: MathQuestion) -> bool:
        """Return True if the last number in the reply equals the reference."""
        value = self.extract(reply)
        return value is not None and abs(value - question.answer) < 1e-6
