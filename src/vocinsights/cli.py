"""Command line: serve the API, run the evaluation (with the CI gate), and write the quarterly report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .evaluate import DEFAULT_F1_FLOOR, gate, table


def _bench(args):
    from .pipeline import build

    return build(n=args.texts, seed=args.seed)


def cmd_serve(args) -> None:
    import uvicorn

    from .app import Settings, create_app

    settings = Settings(texts=args.texts, seed=args.seed)
    uvicorn.run(
        create_app(settings),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8873")),
        log_level="warning",
    )


def cmd_eval(args) -> None:
    b = _bench(args)
    ev = b.evaluation
    det = b.detection
    lines = [
        f"Labeled test set: {ev['n_test']:,} texts (classifier trained on {len(b.train):,}).",
        f"Operating point: {ev['operating_point']['label']}.",
        "",
        table(ev),
        "",
        f"Quarter 4 release bug ({det['release']}): theme alert on {det['theme_alert_day']} "
        f"({det['theme_alert_lag_days']} days), new-cluster alert on {det['cluster_alert_day']} "
        f"({det['cluster_alert_lag_days']} days). Alerts not tied to a planted event: {len(det['other_alerts'])}.",
    ]
    for name, d in b.discovery.items():
        lines.append(
            f"Discovery {name}: purity {d['purity']:.3f}, NMI {d['nmi']:.3f}, themes recovered {d['themes_recovered']}."
        )
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rows = {k: {m: v for m, v in r.items() if m != "per_theme"} for k, r in ev["rows"].items()}
        payload = {
            "evaluation": {**ev, "rows": rows},
            "per_theme": ev["rows"]["cascade"]["per_theme"],
            "detection": {k: v for k, v in det.items() if k != "cluster"},
            "discovery": {k: {m: v for m, v in d.items() if m != "clusters"} for k, d in b.discovery.items()},
            "drivers": b.drivers,
            "timings": b.timings,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write("## VoC evaluation\n\n" + text + "\n")
    if args.gate:
        problems = gate(ev, args.floor)
        if not det["caught"]:
            problems.append("the planted quarter 4 release bug was not flagged")
        if problems:
            print("GATE FAILED: " + "; ".join(problems), file=sys.stderr)
            sys.exit(1)
        print(
            f"Gate passed: micro F1 {ev['rows']['cascade']['micro_f1']:.3f} >= {args.floor:.3f} and the quarter 4 "
            "bug was flagged."
        )


def cmd_report(args) -> None:
    from .report import build_report

    b = _bench(args)
    r = build_report(b, quarter=args.quarter, fmt=args.format)
    body = r["html"] if args.format == "html" else r["markdown"]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body)
        print(f"wrote {args.out}")
    else:
        print(body)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vocinsights")
    ap.add_argument("--texts", type=int, default=int(os.environ.get("VOC_TEXTS", "25000")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("VOC_SEED", "7")))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve").set_defaults(fn=cmd_serve)
    e = sub.add_parser("eval")
    e.add_argument("--gate", action="store_true")
    e.add_argument("--floor", type=float, default=DEFAULT_F1_FLOOR)
    e.add_argument("--out")
    e.add_argument("--summary")
    e.set_defaults(fn=cmd_eval)
    r = sub.add_parser("report")
    r.add_argument("--quarter", default=None)
    r.add_argument("--format", choices=("md", "html"), default="md")
    r.add_argument("--out")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)
