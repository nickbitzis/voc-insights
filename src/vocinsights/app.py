"""FastAPI service: ingest and classify feedback, themes over time, emerging issues, NPS drivers, the
classification evaluation, theme discovery, and the quarterly report."""

from __future__ import annotations

import datetime as dt
import os
import random
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__, data
from .llm import ClaudeTagger
from .logs import get_logger, log_event
from .metrics import Metrics
from .pipeline import Bench, build
from .report import ClaudeWriter, TemplateWriter, build_report
from .taxonomy import LABELS, SOURCES, THEME_NAMES

STATIC = Path(__file__).parent / "static"


@dataclass
class Settings:
    texts: int = int(os.environ.get("VOC_TEXTS", "25000"))
    seed: int = int(os.environ.get("VOC_SEED", "7"))
    llm: str = os.environ.get("VOC_LLM", "sim")  # sim | claude
    writer: str = os.environ.get("VOC_WRITER", "template")  # template | claude
    max_ingest: int = 10_000


class ClassifyIn(BaseModel):
    text: str = Field(min_length=2, max_length=4000)


class FeedbackItem(BaseModel):
    text: str = Field(min_length=2, max_length=4000)
    source: Literal["nps", "review", "ticket"] = "nps"
    date: dt.date | None = None
    score: int | None = Field(default=None, ge=0, le=10)


class IngestIn(BaseModel):
    items: list[FeedbackItem] = Field(min_length=1, max_length=500)


class State:
    def __init__(self, settings: Settings, tagger=None, writer=None):
        self.settings = settings
        if tagger is None and settings.llm == "claude":
            tagger = ClaudeTagger()
        self.bench: Bench = build(n=settings.texts, seed=settings.seed, llm=tagger)
        if writer is None:
            writer = ClaudeWriter() if settings.writer == "claude" else TemplateWriter()
        self.writer = writer
        self.metrics = Metrics()
        self.lock = threading.Lock()
        self.ingested: list[dict] = []
        self.themes = self._themes_by_quarter()
        self.by_theme = self._index()

    def _themes_by_quarter(self) -> dict:
        b = self.bench
        rows = {t: {q: Counter() for q in data.QUARTERS} for t in THEME_NAMES}
        for f, d in zip(b.items, b.decisions, strict=True):
            for t in d.themes:
                if t in rows:
                    rows[t][f.quarter][d.sentiments.get(t, "neutral")] += 1
        out = []
        for t in THEME_NAMES:
            qs = [
                {
                    "quarter": q,
                    "negative": c["negative"],
                    "neutral": c["neutral"],
                    "positive": c["positive"],
                    "total": sum(c.values()),
                }
                for q, c in rows[t].items()
            ]
            out.append({"theme": t, "label": LABELS[t], "quarters": qs, "total": sum(q["total"] for q in qs)})
        out.sort(key=lambda r: -r["total"])
        return {
            "quarters": list(data.QUARTERS),
            "themes": out,
            "y_max": max(q["total"] for r in out for q in r["quarters"]),
        }

    def _index(self) -> dict[str, list[int]]:
        idx: dict[str, list[int]] = defaultdict(list)
        for k, d in enumerate(self.bench.decisions):
            for t in d.themes:
                idx[t].append(k)
        return idx

    def classify(self, text: str) -> dict:
        d = self.bench.cascade.run([text])[0]
        self.metrics.observe_decision(d.tier, d.cost)
        return d.to_dict()


def create_app(settings: Settings | None = None, tagger=None, writer=None) -> FastAPI:
    settings = settings or Settings()
    log = get_logger()
    t0 = time.perf_counter()
    state = State(settings, tagger=tagger, writer=writer)
    b = state.bench
    build_s = round(time.perf_counter() - t0, 1)
    state.metrics.set_gauge(
        "voc_build_seconds", "Seconds to generate, train, evaluate, and analyze at startup.", build_s
    )
    state.metrics.set_gauge(
        "voc_eval_micro_f1", "Cascade micro F1 on the labeled test set.", b.evaluation["rows"]["cascade"]["micro_f1"]
    )
    state.metrics.set_gauge("voc_emerging_alerts", "Theme alerts raised over the year.", len(b.alerts))
    state.metrics.set_gauge("voc_corpus_texts", "Texts in the analyzed corpus.", len(b.items))
    log_event(
        log,
        "ready",
        seconds=build_s,
        texts=settings.texts,
        policy=b.cascade.policy.label(),
        llm=settings.llm,
        micro_f1=b.evaluation["rows"]["cascade"]["micro_f1"],
        timings=b.timings,
    )
    app = FastAPI(title="VoC Insights", version=__version__)
    app.state.s = state

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = route.path if route else request.url.path
        state.metrics.observe_http(path, response.status_code)
        if path not in ("/healthz", "/metrics"):
            log_event(
                log,
                "request",
                method=request.method,
                path=path,
                status=response.status_code,
                ms=round((time.perf_counter() - start) * 1000, 2),
            )
        return response

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "version": __version__,
            "texts": len(b.items),
            "llm": settings.llm,
            "writer": state.writer.name,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return state.metrics.render()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/static/{name}")
    def static(name: str):
        path = STATIC / name
        if name not in {"app.js", "styles.css"} or not path.exists():
            raise HTTPException(404, "not found")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.get("/api/summary")
    def summary():
        rows = b.evaluation["rows"]
        nps_by_q = {}
        for q in data.QUARTERS:
            scores = [f.score for f in b.items if f.quarter == q and f.source == "nps"]
            pro = sum(s >= 9 for s in scores)
            det = sum(s <= 6 for s in scores)
            nps_by_q[q] = round(100 * (pro - det) / len(scores), 1) if scores else None
        return {
            "company": "Fernhollow",
            "product": "Fernhollow Planner",
            "texts": len(b.items),
            "sources": dict(Counter(f.source for f in b.items)),
            "labeled": len(b.train) + len(b.test),
            "nps_by_quarter": nps_by_q,
            "eval": {k: {m: rows[k][m] for m in ("micro_f1", "macro_f1", "cost_per_1k", "llm_share")} for k in rows},
            "policy": b.evaluation["operating_point"],
            "detection": {k: v for k, v in b.detection.items() if k not in ("cluster", "other_alerts")},
            "false_alarms": len(b.detection["other_alerts"]),
            "simulated_llm": settings.llm != "claude",
            "timings": b.timings,
        }

    @app.get("/api/themes")
    def themes():
        return state.themes

    @app.get("/api/themes/{name}")
    def theme(name: str, quotes: int = 6):
        if name not in THEME_NAMES:
            raise HTTPException(404, f"unknown theme {name}")
        idx = state.by_theme.get(name, [])
        weeks = sorted({f.week for f in b.items})
        wi = {w: i for i, w in enumerate(weeks)}
        count, neg, total = [0] * len(weeks), [0] * len(weeks), [0] * len(weeks)
        for f in b.items:
            total[wi[f.week]] += 1
        by_source: dict[str, Counter] = {s: Counter() for s in SOURCES}
        for k in idx:
            f, d = b.items[k], b.decisions[k]
            count[wi[f.week]] += 1
            s = d.sentiments.get(name, "neutral")
            neg[wi[f.week]] += int(s == "negative")
            by_source[f.source][s] += 1
        rng = random.Random(f"{settings.seed}|{name}")
        picks = rng.sample(idx, min(max(1, min(quotes, 20)), len(idx))) if idx else []
        drv = next((r for r in b.drivers["drivers"] if r["theme"] == name), None)
        return {
            "theme": name,
            "label": LABELS[name],
            "mentions": len(idx),
            "weeks": weeks,
            "count": count,
            "negative": neg,
            "total": total,
            "by_source": {s: dict(c) for s, c in by_source.items()},
            "quotes": [
                {
                    "id": b.items[k].id,
                    "text": b.items[k].text,
                    "source": b.items[k].source,
                    "date": b.items[k].date,
                    "sentiment": b.decisions[k].sentiments.get(name),
                    "evidence": b.decisions[k].evidence.get(name),
                    "tier": b.decisions[k].tier,
                }
                for k in sorted(picks, key=lambda k: b.items[k].date)
            ],
            "alerts": [a for a in b.alerts if a["theme"] == name],
            "driver": drv,
            "eval": b.evaluation["rows"]["cascade"]["per_theme"].get(name),
        }

    @app.get("/api/emerging")
    def emerging_view():
        weeks = sorted({f.week for f in b.items})
        flagged = sorted({a["theme"] for a in b.alerts})
        wi = {w: i for i, w in enumerate(weeks)}
        series = {t: [0] * len(weeks) for t in flagged}
        total = [0] * len(weeks)
        for f, d in zip(b.items, b.decisions, strict=True):
            total[wi[f.week]] += 1
            for t in d.themes:
                if t in series:
                    series[t][wi[f.week]] += 1
        cluster_series = {}
        for c in b.clusters:
            s = [0] * len(weeks)
            ids = set(c["doc_ids"])
            for f in b.items:
                if f.id in ids:
                    s[wi[f.week]] += 1
            cluster_series[c["name"]] = s
        return {
            "alerts": b.alerts,
            "clusters": [{k: v for k, v in c.items() if k != "doc_ids"} for c in b.clusters],
            "detection": {k: v for k, v in b.detection.items() if k != "cluster"},
            "weeks": weeks,
            "total": total,
            "series": series,
            "cluster_series": cluster_series,
            "events": [
                {"date": data.PRICE_CHANGE.isoformat(), "label": "price increase"},
                {"date": data.RELEASE.isoformat(), "label": "release 6.0"},
                {"date": data.HOTFIX.isoformat(), "label": "hotfix 6.0.2"},
            ],
        }

    @app.get("/api/drivers")
    def drivers_view():
        return {**b.drivers, "nps_vs_sentiment": b.nps_sentiment}

    @app.get("/api/discovery")
    def discovery_view():
        return b.discovery

    @app.get("/api/eval")
    def eval_view(per_theme: bool = True):
        rows = b.evaluation["rows"]
        if not per_theme:
            rows = {k: {m: v for m, v in r.items() if m != "per_theme"} for k, r in rows.items()}
        return {**b.evaluation, "rows": rows, "n_train": len(b.train), "simulated_llm": True}

    @app.post("/api/classify")
    def classify(body: ClassifyIn):
        d = state.classify(body.text.strip())
        log_event(log, "classified", tier=d["tier"], themes=d["themes"], cost=d["cost"])
        return d

    @app.post("/api/feedback")
    def ingest(body: IngestIn):
        texts = [i.text.strip() for i in body.items]
        decisions = b.cascade.run(texts)
        out = []
        with state.lock:
            if len(state.ingested) + len(texts) > settings.max_ingest:
                raise HTTPException(413, f"ingest store holds at most {settings.max_ingest} items")
            for item, d in zip(body.items, decisions, strict=True):
                if item.source == "review" and item.score is not None and not 1 <= item.score <= 5:
                    raise HTTPException(422, "review score must be 1 to 5 stars")
                rec = {
                    "id": f"N{len(state.ingested) + 1:06d}",
                    "text": item.text.strip(),
                    "source": item.source,
                    "date": (item.date or dt.date.today()).isoformat(),
                    "score": item.score,
                    "decision": d.to_dict(),
                }
                state.ingested.append(rec)
                state.metrics.observe_decision(d.tier, d.cost)
                out.append(rec)
        log_event(log, "ingested", count=len(out), tiers=dict(Counter(r["decision"]["tier"] for r in out)))
        return {"ingested": len(out), "items": out}

    @app.get("/api/feedback")
    def list_ingested(limit: int = 50):
        return {"total": len(state.ingested), "items": state.ingested[-max(1, min(limit, 500)) :][::-1]}

    @app.get("/api/report")
    def report_view(quarter: str = data.QUARTERS[-1], format: Literal["md", "html", "json"] = "md"):
        if quarter not in data.QUARTERS:
            raise HTTPException(404, f"quarter must be one of {list(data.QUARTERS)}")
        r = build_report(b, quarter=quarter, writer=state.writer, fmt="html" if format == "html" else "md")
        state.metrics.observe_report(r["writer"], bool(r["problems"]))
        log_event(log, "report", quarter=quarter, writer=r["writer"], problems=len(r["problems"]))
        if format == "html":
            return HTMLResponse(r["html"])
        if format == "json":
            return {k: v for k, v in r.items() if k != "html"}
        return PlainTextResponse(r["markdown"], media_type="text/markdown")

    return app
