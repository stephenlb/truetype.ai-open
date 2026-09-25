"""Fast, deterministic contracts for parsing, rendering, and scoring.

These 50 tests deliberately avoid loading the model.  They cover the pure-Python
parts of the public request/response path, which makes failures local and keeps
the full suite useful on machines that do not have the model weights available.
"""

from __future__ import annotations

import math

import pytest

from src.truetype.letters import distribution_from_letter_logits
from src.truetype.questions import (
    _predicate_forms,
    build_question,
    confidence_from_probabilities,
    score_question,
    score_question_tensor,
)
from src.truetype.render import _interleaved_order, render_example_prefix, render_question, render_target_block


# 9 tests
@pytest.mark.parametrize(
    ("instruction", "affirmative", "negative"),
    [
        ("Does the text request a refund?", "Yes, the text request a refund.", "No, the text does not request a refund."),
        ("Does the text express urgency?", "Yes, the text express urgency.", "No, the text does not express urgency."),
        ("Is the text urgent?", "Yes, the text urgent.", "No, the text does not urgent."),
        ("Is this a bug?", "Yes, this a bug.", "No, it is not the case that this a bug."),
        ("Does this need help?", "Yes, this need help.", "No, it is not the case that this need help."),
        ("Does the order ship?", "Yes, the order ship.", "No, the order does not ship."),
        ("Is the order ready?", "Yes, the order ready.", "No, the order does not ready."),
        ("Did the order arrive?", "Yes, the order arrive.", "No, the order does not arrive."),
        ("Can the order ship?", "Yes, the order ship.", "No, the order does not ship."),
    ],
)
def test_predicate_forms_are_stable(instruction, affirmative, negative):
    assert _predicate_forms(instruction) == (affirmative, negative)


# 10 tests
@pytest.mark.parametrize(
    ("size", "expected"),
    [(0, []), (1, [0]), (2, [0, 1]), (3, [0, 2, 1]), (4, [0, 3, 1, 2]),
     (5, [0, 4, 1, 2, 3]), (6, [0, 5, 1, 2, 3, 4]),
     (7, [0, 6, 1, 2, 3, 4, 5]), (8, [0, 7, 1, 2, 3, 4, 5, 6]),
     (9, [0, 8, 1, 2, 3, 4, 5, 6, 7])],
)
def test_interleaved_order_visits_each_option_once(size, expected):
    assert _interleaved_order(size) == expected


# 7 tests
@pytest.mark.parametrize(
    ("logits", "top_k", "temperature", "letters"),
    [
        ({"A": 2.0, "B": 1.0}, 2, 1.0, ("A", "B")),
        ({"A": 1.0, "B": 2.0}, 2, 1.0, ("B", "A")),
        ({"A": 3.0, "B": 2.0, "C": 1.0}, 2, 1.0, ("A", "B")),
        ({"A": 3.0, "B": 2.0, "C": 1.0}, 99, 1.0, ("A", "B", "C")),
        ({"A": 3.0, "B": 2.0}, 0, 1.0, ("A",)),
        ({"A": 3.0, "B": 2.0}, 2, 0.5, ("A", "B")),
        ({"Z": 1.0}, 5, 1.0, ("Z",)),
    ],
)
def test_letter_distribution_is_normalized_and_ranked(logits, top_k, temperature, letters):
    result = distribution_from_letter_logits(logits, top_k=top_k, temperature=temperature)
    assert tuple(letter for letter, _ in result.ranked) == letters
    assert result.top_k == len(letters)
    assert math.isclose(sum(result.probabilities.values()), 1.0)
    assert result.top_letter == letters[0]


# 7 tests
@pytest.mark.parametrize(
    ("spec", "kind", "labels", "letters"),
    [
        ({"type": "noul", "instructions": "Does the text request a refund?"}, "noul", ["yes", "no"], ["Y", "N"]),
        ({"type": "noul", "instructions": "Does the text request a refund?", "criteria": {"true": "refund", "false": "other"}}, "noul", ["yes", "no"], ["Y", "N"]),
        ({"type": "choice", "instructions": "Route it", "criteria": {"billing": "charges", "shipping": "delivery"}}, "choice", ["billing", "shipping"], ["A", "B"]),
        ({"type": "choice", "instructions": "Route it", "criteria": {"x": 1}}, "choice", ["x"], ["A"]),
        ({"type": "score", "instructions": "Severity", "criteria": ["low", "high"]}, "score", ["level 0", "level 1"], ["A", "B"]),
        ({"type": "score", "instructions": "Severity", "criteria": ["low", "medium", "high"]}, "score", ["level 0", "level 1", "level 2"], ["A", "B", "C"]),
        ({"type": "choice", "instructions": "Alphabet", "criteria": {chr(97 + i): str(i) for i in range(26)}}, "choice", [chr(97 + i) for i in range(26)], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    ],
)
def test_build_question_assigns_ordered_letters(spec, kind, labels, letters):
    question = build_question("question-id", spec)
    assert question.id == "question-id"
    assert question.type == kind
    assert [option.label for option in question.options] == labels
    assert [option.letter for option in question.options] == letters


# 7 tests
@pytest.mark.parametrize(
    "spec",
    [
        {"type": "other", "instructions": "x"},
        {"type": "noul"},
        {"type": "noul", "instructions": "   "},
        {"type": "noul", "instructions": 1},
        {"type": "choice", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": {}},
        {"type": "score", "instructions": "x", "criteria": ["only level"]},
    ],
)
def test_build_question_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        build_question("bad", spec)


# 5 tests
@pytest.mark.parametrize(
    ("spec", "logits", "expected_type", "expected_value"),
    [
        ({"type": "noul", "instructions": "Does it work?"}, {"Y": 2.0, "N": 0.0}, "noul", 0.880797),
        ({"type": "choice", "instructions": "Pick", "criteria": {"a": "first", "b": "second"}}, {"A": 0.0, "B": 3.0}, "choice", "b"),
        ({"type": "choice", "instructions": "Pick", "criteria": {"a": "first", "b": "second"}}, {"A": 3.0, "B": 0.0}, "choice", "a"),
        ({"type": "score", "instructions": "Rate", "criteria": ["low", "high"]}, {"A": 0.0, "B": 0.0}, "score", 0.5),
        ({"type": "score", "instructions": "Rate", "criteria": ["low", "middle", "high"]}, {"A": 0.0, "B": 0.0, "C": 10.0}, "score", 1.999864),
    ],
)
def test_score_question_returns_typed_answers(spec, logits, expected_type, expected_value):
    answer = score_question(build_question("q", spec), logits)
    assert answer.type == expected_type
    if expected_type == "noul":
        assert answer.noul == pytest.approx(expected_value, abs=1e-6)
    elif expected_type == "choice":
        assert answer.choice == expected_value
        assert math.isclose(sum(answer.probabilities.values()), 1.0)
    else:
        assert answer.score == pytest.approx(expected_value, abs=1e-6)
        assert math.isclose(sum(answer.probabilities.values()), 1.0)


@pytest.mark.parametrize(
    ("spec", "values"),
    [
        ({"type": "noul", "instructions": "Does it work?"}, {"Y": 2.0, "N": 0.0}),
        ({"type": "choice", "instructions": "Pick", "criteria": {"a": "first", "b": "second"}}, {"A": 0.0, "B": 3.0}),
        ({"type": "score", "instructions": "Rate", "criteria": ["low", "middle", "high"]}, {"A": 0.0, "B": 0.0, "C": 10.0}),
    ],
)
def test_tensor_question_scoring_matches_dict_scoring(spec, values):
    torch = pytest.importorskip("torch")
    question = build_question("q", spec)
    full = {letter: -100.0 for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    full.update(values)
    tensor = torch.tensor([full[letter] for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    actual = score_question_tensor(question, tensor)
    expected = score_question(question, full)
    assert actual.type == expected.type
    assert actual.choice == expected.choice
    if expected.noul is None:
        assert actual.noul is None
    else:
        assert actual.noul == pytest.approx(expected.noul)
    if expected.score is None:
        assert actual.score is None
    else:
        assert actual.score == pytest.approx(expected.score)
    if expected.confidence is None:
        assert actual.confidence is None
    else:
        assert actual.confidence == pytest.approx(expected.confidence)
    assert actual.legend == expected.legend
    assert actual.probabilities == pytest.approx(expected.probabilities)


# 3 tests
@pytest.mark.parametrize(
    ("spec", "state", "required"),
    [
        ({"type": "noul", "instructions": "Does the text request a refund?"}, "refund please", "Answer: Y"),
        ({"type": "choice", "instructions": "Route", "criteria": {"billing": "charges", "shipping": "delivery", "returns": "refunds"}}, "charged twice", "Answer: C"),
        ({"type": "score", "instructions": "Severity", "criteria": ["low", "high", "critical"]}, "service down", "Answer: C"),
    ],
)
def test_rendered_prompt_has_examples_and_an_unstranded_target_slot(spec, state, required):
    question = build_question("q", spec)
    prompt = render_question(question, state).text
    assert required in render_example_prefix(question)
    assert prompt.endswith(render_target_block(question, state))
    assert prompt.endswith("Answer:")
    assert not prompt.endswith("Answer: ")


# 2 tests
@pytest.mark.parametrize(("probabilities", "expected"), [([1.0, 0.0], 1.0), ([0.5, 0.5], 0.0)])
def test_confidence_has_expected_endpoints(probabilities, expected):
    assert confidence_from_probabilities(probabilities) == pytest.approx(expected)
