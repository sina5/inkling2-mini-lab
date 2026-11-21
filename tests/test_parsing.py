"""Tests for answer parsing, grading, subsets, and slot mapping."""

import numpy as np
import pytest

from inklab.constants import CIFAR_CLASSES
from inklab.experiments import balanced_subset
from inklab.gsm8k import FinalNumberGrader, MathQuestion, to_number
from inklab.probe_data import layer_pass_to_slot, slot_to_layer_pass
from inklab.zero_shot import UNPARSED, AnswerParser


@pytest.fixture
def parser() -> AnswerParser:
    return AnswerParser(CIFAR_CLASSES)


@pytest.mark.parametrize("answer, expected", [
    ("cat", 3), ("Cat.", 3), ("  truck\n", 9), ("It is a ship", 8), ("car", 1), ("aeroplane", 0), ("horses", 7),
])
def test_parse_known_answers(parser: AnswerParser, answer: str, expected: int) -> None:
    assert parser.parse(answer) == expected


def test_parse_returns_first_class_word(parser: AnswerParser) -> None:
    assert parser.parse("dog, not cat") == 5


@pytest.mark.parametrize("answer", ["", "I am not sure", "????"])
def test_parse_unknown_answers(parser: AnswerParser, answer: str) -> None:
    assert parser.parse(answer) == UNPARSED


@pytest.mark.parametrize("text, expected", [("1,234", 1234.0), ("-3.5", -3.5), ("abc", None)])
def test_to_number(text: str, expected: float | None) -> None:
    assert to_number(text) == expected


def test_grader_uses_last_number() -> None:
    grader = FinalNumberGrader()
    question = MathQuestion(0, "q", 18.0)
    assert grader.correct("9 * 2 = 18\n\n18", question)
    assert not grader.correct("18 then 19", question)


def test_grader_reads_unicode_minus() -> None:
    assert FinalNumberGrader().extract("answer: −4") == -4.0


def test_grader_without_number() -> None:
    assert FinalNumberGrader().extract("no idea") is None


def test_balanced_subset_takes_first_per_class() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1])
    assert balanced_subset(labels, 2) == [0, 1, 2, 3]


def test_balanced_subset_rejects_small_class() -> None:
    with pytest.raises(ValueError):
        balanced_subset(np.array([0, 0, 1]), 2)


@pytest.mark.parametrize("slot, layer, pass_number", [(0, 0, 1), (17, 17, 1), (23, 23, 1), (24, 6, 2), (35, 17, 2),
                                                      (36, 6, 3), (47, 17, 3)])
def test_slot_mapping_round_trip(slot: int, layer: int, pass_number: int) -> None:
    assert slot_to_layer_pass(slot) == (layer, pass_number)
    assert layer_pass_to_slot(layer, pass_number) == slot


def test_slot_mapping_rejects_layer_outside_core() -> None:
    with pytest.raises(ValueError):
        layer_pass_to_slot(3, 2)
