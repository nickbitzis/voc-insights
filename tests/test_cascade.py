import pytest

from vocinsights.cascade import Cascade, Policy
from vocinsights.llm import ClaudeTagger, SimulatedTagger


class CountingTagger(SimulatedTagger):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def tag(self, text, gold=None, hint=None):
        self.calls += 1
        return super().tag(text, gold=gold, hint=hint)


def cascade_with(bench, policy, tagger=None):
    return Cascade(bench.cascade.clf, bench.cascade.sentiment, tagger or CountingTagger(), policy)


def test_rules_tier_takes_clean_matches(bench):
    c = cascade_with(bench, Policy(t_rules=0.9, t_clf=0.5))
    d = c.run(["I was charged twice this month."])[0]
    assert d.tier == "rules" and d.themes == ["billing_errors"] and c.llm.calls == 0


def test_rules_off_goes_to_classifier(bench):
    c = cascade_with(bench, Policy(t_rules=None, t_clf=0.0))
    d = c.run(["I was charged twice this month."])[0]
    assert d.tier == "classifier"


def test_low_confidence_goes_to_llm(bench):
    c = cascade_with(bench, Policy(t_rules=None, t_clf=1.01))
    d = c.run(["hmm the thing with the stuff"])[0]
    assert d.tier == "llm" and c.llm.calls == 1 and d.simulated


def test_threshold_controls_llm_share(bench):
    texts = [f.text for f in bench.test[:300]]
    low = cascade_with(bench, Policy(t_rules=None, t_clf=0.3))
    high = cascade_with(bench, Policy(t_rules=None, t_clf=0.9))
    low.run(texts)
    high.run(texts)
    assert 0 < low.llm.calls < high.llm.calls


def test_llm_off_never_calls(bench):
    c = cascade_with(bench, Policy(t_rules=None, t_clf=1.01, use_llm=False))
    c.run([f.text for f in bench.test[:50]])
    assert c.llm.calls == 0


def test_llm_cost_is_added(bench):
    c = cascade_with(bench, Policy(t_rules=None, t_clf=1.01))
    d = c.run(["The app crashes every morning."])[0]
    assert d.cost > 1e-6


def test_bad_llm_output_falls_back_to_classifier(bench, fake_client):
    tagger = ClaudeTagger(client=fake_client("oops"))
    c = cascade_with(bench, Policy(t_rules=None, t_clf=1.01), tagger)
    d = c.run(["The app crashes every morning."])[0]
    assert d.tier == "classifier" and d.llm_error and "app_stability" in d.themes


def test_llm_other_is_kept_as_new_issue(bench, fake_client):
    import json

    body = json.dumps(
        {"themes": [{"theme": "other", "sentiment": "negative", "evidence": "widget is blank"}], "overall": "negative"}
    )
    c = cascade_with(bench, Policy(t_rules=None, t_clf=1.01), ClaudeTagger(client=fake_client(body)))
    d = c.run(["After the update the widget is blank."])[0]
    assert d.tier == "llm" and d.themes == [] and d.new_issue == "widget is blank"


def test_evidence_comes_from_the_text(bench):
    texts = [f.text for f in bench.test[:200]]
    for t, d in zip(texts, bench.cascade.run(texts, golds=bench.test[:200]), strict=True):
        for span in d.evidence.values():
            assert span.lower() in t.lower() or span == t[:80]


@pytest.mark.parametrize("t_rules", [None, 0.9])
def test_policy_label(t_rules):
    assert "classifier 0.50" in Policy(t_rules=t_rules, t_clf=0.5).label()
