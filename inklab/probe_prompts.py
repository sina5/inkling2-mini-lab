"""Probe inputs for the routing and loop-gate analysis.

The text prompts are short passages that I wrote for this probe. They are inputs to measure
routing on, not data that the analysis grades.
"""

from dataclasses import dataclass
from enum import Enum

MEDIA_MARKER = "<__media__>"


class Category(Enum):
    """Kinds of input that the probe compares."""

    IMAGE = "image"
    ENGLISH = "english"
    CODE = "code"
    MATH = "math"
    OTHER_LANGUAGE = "other_language"


@dataclass(frozen=True)
class ProbePrompt:
    """One probe input.

    Attributes:
        name: File-safe identifier.
        category: Kind of input.
        text: User text; for image prompts, the question after the image.
        cifar_index: CIFAR-10 test index for image prompts, else None.
    """

    name: str
    category: Category
    text: str
    cifar_index: int | None = None


class ChatFormatter:
    """Build the raw prompt string that llama-server renders for one user turn."""

    def __init__(self, effort_value: str = "0") -> None:
        self.effort_value = effort_value

    def render(self, prompt: ProbePrompt) -> str:
        """Return the raw prompt, with the media marker for image prompts."""
        parts = [f"<|message_system|><|content_text|>Thinking effort level: {self.effort_value}<|end_message|>"]
        if prompt.cifar_index is not None:
            parts.append(f"<|message_user|>{MEDIA_MARKER}<|end_message|>")
        parts.append(f"<|message_user|><|content_text|>{prompt.text}<|end_message|>")
        return "".join(parts)


IMAGE_QUESTION = (
    "This is a small photo. Which one of these classes does it show: airplane, automobile, bird, cat, deer, "
    "dog, frog, horse, ship, truck? Answer with the class name only."
)

ENGLISH = [
    "The library opens at nine, but most people arrive after lunch when the reading room gets its best light.",
    "My neighbor grows tomatoes on her balcony and gives the extra ones to anyone who walks past in August.",
    "The train was late again, so I read two chapters of my book while I waited on the cold platform.",
    "We painted the kitchen a pale green last spring, and now the morning sun makes it look like a garden.",
    "He fixed the old bicycle with parts from three other bikes and rode it to work the next morning.",
    "The museum moved the dinosaur skeleton to a bigger hall so that school groups could walk all around it.",
    "After the storm, the beach was covered in seaweed, and the dogs ran back and forth for hours.",
    "She keeps a notebook of every recipe she has tried, with a small star next to the ones worth repeating.",
]

CODE = [
    "def mean(values):\n    total = sum(values)\n    return total / len(values) if values else 0.0",
    "for i in range(10):\n    if i % 2 == 0:\n        print(i, 'is even')\n    else:\n        print(i, 'is odd')",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)",
    "import json\nwith open('config.json') as f:\n    config = json.load(f)\nprint(config.get('port', 8080))",
    "SELECT name, COUNT(*) AS orders FROM customers JOIN orders USING (id) GROUP BY name ORDER BY orders DESC;",
    "const total = items.filter(x => x.active).map(x => x.price).reduce((a, b) => a + b, 0);",
    "#include <stdio.h>\nint main(void) {\n    for (int i = 0; i < 3; i++) printf(\"%d\\n\", i);\n    return 0;\n}",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
]

MATH = [
    "Solve for x: 3x + 7 = 22. Then compute 2x squared minus 5.",
    "A train travels 180 km in 2.5 hours. What is its average speed in km per hour?",
    "Find the derivative of f(x) = x^3 - 4x^2 + 6x - 1 and evaluate it at x = 2.",
    "If a rectangle has a perimeter of 36 cm and a width of 7 cm, what is its area?",
    "Compute the sum of the integers from 1 to 100, then divide the result by 25.",
    "What is 15% of 240, plus one third of 81?",
    "The probability of rain is 0.3 each day. What is the probability of no rain on two days in a row?",
    "Simplify (2a + 3b)^2 - (2a - 3b)^2.",
]

OTHER_LANGUAGE = [
    "Die Bibliothek öffnet um neun Uhr, aber die meisten Leute kommen erst nach dem Mittagessen.",
    "Mi vecina cultiva tomates en su balcón y regala los que sobran a quien pasa por la calle.",
    "Le train était encore en retard, alors j'ai lu deux chapitres de mon livre sur le quai froid.",
    "我们去年春天把厨房刷成了浅绿色，现在早上的阳光让它看起来像一个花园。",
    "او دوچرخه قدیمی را با قطعات سه دوچرخه دیگر تعمیر کرد و صبح روز بعد با آن سر کار رفت.",
    "Il museo ha spostato lo scheletro del dinosauro in una sala più grande per i gruppi scolastici.",
    "嵐の後、浜辺は海藻で覆われていて、犬たちは何時間も走り回っていた。",
    "Она ведёт тетрадь со всеми рецептами, которые попробовала, и ставит звёздочку рядом с лучшими.",
]


def default_prompts(image_indices: list[int]) -> list[ProbePrompt]:
    """Return the full probe set: one image prompt per CIFAR index and eight prompts per text category."""
    prompts = [ProbePrompt(f"image_{i:05d}", Category.IMAGE, IMAGE_QUESTION, i) for i in image_indices]
    for category, texts in ((Category.ENGLISH, ENGLISH), (Category.CODE, CODE), (Category.MATH, MATH),
                            (Category.OTHER_LANGUAGE, OTHER_LANGUAGE)):
        prompts += [ProbePrompt(f"{category.value}_{j}", category, text) for j, text in enumerate(texts)]
    return prompts
