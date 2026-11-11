import pytest

from vocinsights.report import ClaudeWriter, TemplateWriter, build_report, facts, markdown_to_html, validate


@pytest.fixture(scope="module")
def f(bench):
    return facts(bench)


def corpus(bench):
    return [x.text for x in bench.items]


def test_facts_shape(f):
    assert f["quarter"] == "2025Q4" and f["previous_quarter"] == "2025Q3"
    assert f["themes"][0]["count"] >= f["themes"][-1]["count"]
    assert len(f["movers"]) == 4 and f["drivers_negative"]


def test_quotes_are_verbatim(bench, f):
    texts = set(corpus(bench))
    for qs in f["quotes"].values():
        assert qs and all(q in texts for q in qs)


def test_template_summary_passes_validator(bench, f):
    assert validate(TemplateWriter().write(f), f, corpus(bench)) == []


def test_validator_rejects_invented_number(bench, f):
    probs = validate("- NPS rose to 87.3 this quarter.", f, corpus(bench))
    assert probs and "87.3" in probs[0]


def test_validator_rejects_invented_quote(bench, f):
    probs = validate('- One customer said "this app changed my life completely".', f, corpus(bench))
    assert any("does not appear" in p for p in probs)


def test_validator_accepts_real_quote_and_numbers(bench, f):
    q = next(iter(f["quotes"].values()))[0]
    text = f'- {f["volume"]:,} comments. NPS {f["nps"]:.1f}. A customer wrote "{q}"'
    assert validate(text, f, corpus(bench)) == []


def test_validator_rejects_unknown_date(bench, f):
    assert validate("- Flagged on 2025-02-30.", f, corpus(bench))


def test_claude_writer_good_summary_is_used(bench, f, fake_client):
    good = TemplateWriter().write(f)
    client = fake_client(good)
    r = build_report(bench, writer=ClaudeWriter(client=client))
    assert r["writer"] == "claude" and r["problems"] == []
    assert "<facts>" in client.messages.calls[0]["messages"][0]["content"]


def test_claude_writer_invented_number_falls_back(bench, fake_client):
    r = build_report(bench, writer=ClaudeWriter(client=fake_client("- NPS jumped 42.42 points.")))
    assert r["writer"] == "template (fallback)" and r["problems"]
    assert "rejected by the validator" in r["markdown"]


def test_claude_writer_error_falls_back(bench, fake_client):
    r = build_report(bench, writer=ClaudeWriter(client=fake_client(RuntimeError("boom"))))
    assert r["writer"] == "template (fallback)"


def test_markdown_and_html(bench):
    r = build_report(bench, quarter="2025Q3", fmt="html")
    md, page = r["markdown"], r["html"]
    for section in ("## Summary", "## Top themes", "## Emerging issues", "## What customers said", "## NPS drivers"):
        assert section in md
    assert page.startswith("<!doctype html>") and "<table>" in page and "prefers-color-scheme" in page


def test_html_escapes():
    assert "&lt;script&gt;" in markdown_to_html("> <script>x</script>", "t")
