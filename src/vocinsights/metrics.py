"""Prometheus text exposition without extra dependencies."""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.classified = defaultdict(int)  # tier
        self.cost = 0.0
        self.http = defaultdict(int)  # (path, status)
        self.reports = defaultdict(int)  # writer
        self.rejections = 0
        self.gauges: dict[str, tuple[str, float]] = {}

    def observe_decision(self, tier: str, cost: float) -> None:
        with self.lock:
            self.classified[tier] += 1
            self.cost += cost

    def observe_http(self, path: str, status: int) -> None:
        with self.lock:
            self.http[(path, str(status))] += 1

    def observe_report(self, writer: str, rejected: bool) -> None:
        with self.lock:
            self.reports[writer] += 1
            self.rejections += int(rejected)

    def set_gauge(self, name: str, help_text: str, value: float) -> None:
        with self.lock:
            self.gauges[name] = (help_text, value)

    def render(self) -> str:
        out: list[str] = []

        def family(name, kind, help_text, samples):
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} {kind}")
            for labels, value in samples:
                lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
                out.append(f"{name}{{{lab}}} {value}" if lab else f"{name} {value}")

        with self.lock:
            family(
                "voc_texts_classified_total",
                "counter",
                "Texts classified through the API, by deciding tier.",
                [({"tier": t}, v) for t, v in sorted(self.classified.items())],
            )
            family(
                "voc_llm_cost_usd_total",
                "counter",
                "LLM spend in USD for API classifications.",
                [({}, f"{self.cost:.8f}")],
            )
            family(
                "voc_reports_total",
                "counter",
                "Reports generated, by summary writer.",
                [({"writer": w}, v) for w, v in sorted(self.reports.items())],
            )
            family(
                "voc_report_rejections_total",
                "counter",
                "Writer summaries rejected by the validator.",
                [({}, self.rejections)],
            )
            family(
                "voc_http_requests_total",
                "counter",
                "HTTP requests by path and status.",
                [({"path": p, "status": s}, v) for (p, s), v in sorted(self.http.items())],
            )
            for name, (help_text, value) in sorted(self.gauges.items()):
                family(name, "gauge", help_text, [({}, value)])
        return "\n".join(out) + "\n"
