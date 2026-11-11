import json

import pytest
from fastapi.testclient import TestClient

from vocinsights.app import Settings, create_app
from vocinsights.llm import ClaudeTagger


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(Settings(texts=4000, seed=7)))


def test_health_and_metrics(client):
    assert client.get("/healthz").json()["status"] == "ok"
    m = client.get("/metrics").text
    assert "voc_eval_micro_f1" in m and "voc_http_requests_total" in m


def test_index_and_static(client):
    assert "VoC Insights" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/secret.txt").status_code == 404


def test_summary_and_themes(client):
    s = client.get("/api/summary").json()
    assert (
        s["texts"] == 4000
        and s["simulated_llm"]
        and set(s["nps_by_quarter"]) == {"2025Q1", "2025Q2", "2025Q3", "2025Q4"}
    )
    t = client.get("/api/themes").json()
    assert len(t["themes"]) == 20 and t["y_max"] >= max(q["total"] for r in t["themes"] for q in r["quarters"])


def test_theme_detail(client):
    d = client.get("/api/themes/app_stability?quotes=5").json()
    assert d["label"] == "Crashes and freezes" and len(d["quotes"]) == 5 and len(d["weeks"]) == len(d["count"])
    assert client.get("/api/themes/nope").status_code == 404


def test_emerging_drivers_discovery_eval(client):
    e = client.get("/api/emerging").json()
    assert e["detection"]["caught"] and e["events"]
    assert client.get("/api/drivers").json()["drivers"]
    assert "nmf_k20" in client.get("/api/discovery").json()
    ev = client.get("/api/eval?per_theme=false").json()
    assert "per_theme" not in ev["rows"]["cascade"]


def test_classify(client):
    r = client.post("/api/classify", json={"text": "I was charged twice this month. Love the design."})
    d = r.json()
    assert r.status_code == 200 and {"billing_errors", "love_the_design"} <= set(d["themes"])
    assert d["sentiments"]["billing_errors"] == "negative"
    assert client.post("/api/classify", json={"text": ""}).status_code == 422


def test_ingest_and_list(client):
    r = client.post(
        "/api/feedback",
        json={
            "items": [
                {"text": "Sync takes forever", "source": "review", "score": 2},
                {"text": "Support was really helpful", "source": "nps", "score": 10, "date": "2025-12-01"},
            ]
        },
    )
    assert r.status_code == 200 and r.json()["ingested"] == 2
    listed = client.get("/api/feedback").json()
    assert listed["total"] >= 2 and listed["items"][0]["decision"]["themes"] == ["great_support"]
    bad = client.post("/api/feedback", json={"items": [{"text": "ok then", "source": "review", "score": 9}]})
    assert bad.status_code == 422
    assert client.post("/api/feedback", json={"items": []}).status_code == 422


def test_report_formats(client):
    md = client.get("/api/report")
    assert md.status_code == 200 and md.text.startswith("# Fernhollow Planner insights report, 2025Q4")
    assert client.get("/api/report?format=html").text.startswith("<!doctype html>")
    j = client.get("/api/report?format=json&quarter=2025Q2").json()
    assert j["quarter"] == "2025Q2" and j["problems"] == []
    assert client.get("/api/report?quarter=2030Q1").status_code == 404
    assert "voc_reports_total" in client.get("/metrics").text


def test_service_with_fake_claude_tagger(fake_client):
    body = json.dumps(
        {
            "themes": [{"theme": "app_stability", "sentiment": "negative", "evidence": "keeps quitting"}],
            "overall": "negative",
        }
    )
    tagger = ClaudeTagger(client=fake_client(body))
    c = TestClient(create_app(Settings(texts=2500, seed=3), tagger=tagger))
    # Batch analysis stays simulated; live requests below the classifier threshold go to the fake Claude.
    c.app.state.s.bench.cascade.policy = type(c.app.state.s.bench.cascade.policy)(t_rules=None, t_clf=1.01)
    d = c.post("/api/classify", json={"text": "the app keeps quitting on me"}).json()
    assert d["tier"] == "llm" and not d["simulated"] and d["themes"] == ["app_stability"]
