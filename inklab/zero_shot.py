"""Zero-shot image classification with a chat prompt and a strict answer parser."""

import re
from dataclasses import dataclass

from inklab.client import ChatClient, ChatResult, png_data_url

UNPARSED = -1

SYNONYMS = {
    "aeroplane": "airplane", "plane": "airplane", "jet": "airplane",
    "car": "automobile", "auto": "automobile",
    "lorry": "truck",
    "boat": "ship",
    "kitten": "cat", "puppy": "dog", "toad": "frog", "pony": "horse",
}


@dataclass(frozen=True)
class Prediction:
    """The parsed answer for one image.

    Attributes:
        index: Image index in the dataset.
        label: True class index.
        predicted: Predicted class index, or UNPARSED if the answer names no class.
        answer: Raw answer text.
        result: The full chat result.
    """

    index: int
    label: int
    predicted: int
    answer: str
    result: ChatResult


class AnswerParser:
    """Map a free-text answer to one class index."""

    def __init__(self, class_names: tuple[str, ...]) -> None:
        self.class_names = class_names
        self._lookup = {name: i for i, name in enumerate(class_names)}
        for word, name in SYNONYMS.items():
            if name in self._lookup:
                self._lookup[word] = self._lookup[name]

    def parse(self, answer: str) -> int:
        """Return the class index of the first class word in the answer, or UNPARSED."""
        for word in re.findall(r"[a-z]+", answer.lower()):
            if word in self._lookup:
                return self._lookup[word]
            if word.endswith("s") and word[:-1] in self._lookup:
                return self._lookup[word[:-1]]
        return UNPARSED


class ZeroShotClassifier:
    """Ask the model which class an image shows."""

    def __init__(self, client: ChatClient, class_names: tuple[str, ...], effort: str, max_tokens: int,
                 reasoning_budget: int = -1) -> None:
        self.client = client
        self.class_names = class_names
        self.effort = effort
        self.max_tokens = max_tokens
        self.reasoning_budget = reasoning_budget
        self.parser = AnswerParser(class_names)

    def prompt(self) -> str:
        """Return the text part of the prompt."""
        names = ", ".join(self.class_names)
        return f"This is a small photo. Which one of these classes does it show: {names}? Answer with the class name only."

    def classify(self, index: int, label: int, png: bytes) -> Prediction:
        """Classify one image and parse the answer."""
        content = [
            {"type": "image_url", "image_url": {"url": png_data_url(png)}},
            {"type": "text", "text": self.prompt()},
        ]
        result = self.client.chat(content, effort=self.effort, max_tokens=self.max_tokens,
                                  reasoning_budget=self.reasoning_budget)
        return Prediction(index, label, self.parser.parse(result.content), result.content, result)
