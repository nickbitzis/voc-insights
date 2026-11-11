import pytest

from vocinsights.data import Feedback
from vocinsights.sentiment import AspectSentiment, clause_score, evaluate_aspects


@pytest.fixture(scope="module")
def model():
    m = AspectSentiment()
    m.prior = {"missing_export": "negative", "love_the_design": "positive"}
    return m


@pytest.mark.parametrize(
    "text, sign",
    [
        ("Support was really helpful", 1),
        ("The app crashes all the time", -1),
        ("No crashes in two years", 1),
        ("Not easy to set up", -1),
        ("Great, another notification at 3am, love that", -1),
        ("Support never replied", -1),
    ],
)
def test_clause_scores(text, sign):
    s = clause_score(text)
    assert (s > 0) - (s < 0) == sign


def test_aspects_split_by_clause(model):
    out = model.predict("Love the design, but sync is slow.", ["love_the_design", "slow_sync"])
    assert out == {"love_the_design": "positive", "slow_sync": "negative"}


def test_sarcasm_is_negative(model):
    out = model.predict("Love paying more every year for the same features.", ["pricing_too_high"])
    assert out["pricing_too_high"] == "negative"


def test_request_is_neutral(model):
    assert model.predict("Please add a dark mode.", ["dark_mode_request"]) == {"dark_mode_request": "neutral"}


def test_prior_when_no_sentiment_words(model):
    assert model.predict("There is no export button", ["missing_export"])["missing_export"] == "negative"


def test_overall_mixed(model):
    assert model.overall("Love the design. The app crashes.", ["love_the_design", "app_stability"]) == "mixed"


def test_fit_learns_prior():
    f = Feedback("F1", "nps", "2025-01-01", "2025Q1", "2024-12-30", "x", ("a",), {"a": "positive"}, "positive", 9)
    assert AspectSentiment().fit([f, f]).prior == {"a": "positive"}


def test_evaluate_aspects_math():
    f = Feedback(
        "F1",
        "nps",
        "2025-01-01",
        "2025Q1",
        "2024-12-30",
        "x",
        ("a", "b"),
        {"a": "positive", "b": "negative"},
        "mixed",
        5,
    )
    m = evaluate_aspects([f], [{"a": "positive", "b": "positive", "c": "negative"}])
    assert m["aspect_pairs"] == 2 and m["aspect_accuracy"] == 0.5 and m["overall_accuracy"] == 1.0  # c makes it mixed


def test_aspect_accuracy_on_bench(bench):
    s = bench.evaluation["rows"]["cascade"]["sentiment"]
    assert s["aspect_accuracy"] > 0.85 and s["overall_accuracy"] > 0.75
