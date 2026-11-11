import json

import pytest

from vocinsights import data
from vocinsights.llm import (
    OUTPUT_SCHEMA,
    ClaudeTagger,
    SimulatedTagger,
    StructuredOutputError,
    best_span,
    validate_output,
)
from vocinsights.pricing import call_cost

TEXT = "The app crashes every morning, but support was really helpful."


def reply(themes, overall="mixed"):
    return {"themes": themes, "overall": overall}


def test_valid_reply_passes():
    out = validate_output(
        reply(
            [
                {"theme": "app_stability", "sentiment": "negative", "evidence": "The app crashes every morning"},
                {"theme": "great_support", "sentiment": "positive", "evidence": "support was really helpful"},
            ]
        ),
        TEXT,
    )
    assert [t["theme"] for t in out["themes"]] == ["app_stability", "great_support"]


def test_evidence_must_be_verbatim():
    with pytest.raises(StructuredOutputError, match="verbatim"):
        validate_output(
            reply([{"theme": "app_stability", "sentiment": "negative", "evidence": "the app crashed on launch"}]), TEXT
        )


def test_evidence_match_ignores_case_and_spacing():
    validate_output(
        reply([{"theme": "app_stability", "sentiment": "negative", "evidence": "the  APP crashes"}], "negative"), TEXT
    )


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({"themes": []}, "missing"),
        ({"themes": [], "overall": "mixed", "x": 1}, "unexpected"),
        ({"themes": [], "overall": "angry"}, "overall"),
        (reply([{"theme": "teleport", "sentiment": "negative", "evidence": "The app crashes"}]), "taxonomy"),
        (reply([{"theme": "app_stability", "sentiment": "furious", "evidence": "The app crashes"}]), "sentiment"),
        (reply([{"theme": "app_stability", "sentiment": "negative"}]), "exactly"),
        (reply([{"theme": "app_stability", "sentiment": "negative", "evidence": "The app crashes"}] * 2), "duplicate"),
        (reply([{"theme": "app_stability", "sentiment": "negative", "evidence": "x"}]), "at least 3"),
        ([1, 2], "JSON object"),
    ],
)
def test_invalid_replies(bad, msg):
    with pytest.raises(StructuredOutputError, match=msg):
        validate_output(bad, TEXT)


def test_schema_lists_other_theme():
    enum = OUTPUT_SCHEMA["properties"]["themes"]["items"]["properties"]["theme"]["enum"]
    assert "other" in enum and "app_stability" in enum and "widget_crash_v6" not in enum


def test_claude_tagger_request_and_parse(fake_client):
    body = json.dumps(
        reply([{"theme": "app_stability", "sentiment": "negative", "evidence": "app crashes"}], "negative")
    )
    client = fake_client(body)
    t = ClaudeTagger(client=client)
    res = t.tag(TEXT)
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "<feedback>" in call["messages"][0]["content"]
    assert res.labels == ["app_stability"] and not res.simulated
    assert res.cost == pytest.approx(call_cost("claude-haiku-5-5", 520, 90))


def test_claude_tagger_rejects_bad_json(fake_client):
    with pytest.raises(StructuredOutputError, match="not valid JSON"):
        ClaudeTagger(client=fake_client("not json")).tag(TEXT)


def test_claude_tagger_refusal(fake_client):
    with pytest.raises(StructuredOutputError, match="declined"):
        ClaudeTagger(client=fake_client(("{}", "refusal"))).tag(TEXT)


def test_claude_tagger_rejects_invented_evidence(fake_client):
    body = json.dumps(reply([{"theme": "app_stability", "sentiment": "negative", "evidence": "deleted my files"}]))
    with pytest.raises(StructuredOutputError):
        ClaudeTagger(client=fake_client(body)).tag(TEXT)


def test_pricing():
    assert call_cost("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)
    assert call_cost("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.0)


def test_simulated_tagger_is_deterministic_and_verbatim():
    items = data.generate(n=400, seed=3)
    sim = SimulatedTagger()
    for f in items[:100]:
        a, b = sim.tag(f.text, gold=f), sim.tag(f.text, gold=f)
        assert a.themes == b.themes and a.simulated
        validate_output({"themes": a.themes, "overall": a.overall}, f.text)


def test_simulated_tagger_maps_new_issue_to_other():
    items = data.generate(n=25000, seed=7)
    bug = next(f for f in items if "widget_crash_v6" in f.themes)
    res = SimulatedTagger().tag(bug.text, gold=bug)
    assert any(t["theme"] == "other" for t in res.themes)
    assert "widget_crash_v6" not in res.labels


def test_best_span_is_a_substring():
    text = "Sync is slow. The widget crashes on launch."
    assert best_span(text, "widget crashes") == "The widget crashes on launch."
