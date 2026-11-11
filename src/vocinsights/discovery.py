"""Unsupervised theme discovery: TF-IDF with NMF or k-means, named from top terms, scored against the
hidden taxonomy with purity and normalized mutual information (NMI).

This is what a team would run before it has a taxonomy, or to check whether the taxonomy still fits.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

STOP = sorted(
    ENGLISH_STOP_WORDS
    | {
        "app",
        "just",
        "really",
        "customer",
        "user",
        "says",
        "reports",
        "summary",
        "chat",
        "email",
        "ticket",
        "writes",
        "caller",
        "said",
        "update",
        "honest",
        "review",
        "thanks",
        "otherwise",
        "fine",
        "used",
        "year",
        "ok",
        "using",
        "start",
        "cents",
        "look",
        "please",
        "edit",
        "way",
        "day",
        "days",
        "lot",
        "thing",
        "tool",
        "planner",
        "application",
        "stuff",
        "seriously",
        "truly",
        "rly",
        "time",
    }
)


def purity(clusters, labels) -> float:
    """Share of items whose cluster's majority label matches their own label."""
    by: dict = {}
    for c, y in zip(clusters, labels, strict=True):
        by.setdefault(c, Counter())[y] += 1
    return sum(cnt.most_common(1)[0][1] for cnt in by.values()) / len(labels) if labels else 0.0


def nmi(clusters, labels) -> float:
    """Normalized mutual information with arithmetic-mean normalization (the scikit-learn default)."""
    n = len(labels)
    if n == 0:
        return 0.0
    cc, lc = Counter(clusters), Counter(labels)
    joint = Counter(zip(clusters, labels, strict=True))
    mi = sum(v / n * math.log((v / n) / ((cc[c] / n) * (lc[y] / n))) for (c, y), v in joint.items())
    hc = -sum(v / n * math.log(v / n) for v in cc.values())
    hl = -sum(v / n * math.log(v / n) for v in lc.values())
    if hc == 0 and hl == 0:
        return 1.0
    denom = (hc + hl) / 2
    return max(0.0, mi / denom) if denom else 0.0


def primary_theme(f) -> str | None:
    return f.clauses[0][0] if f.clauses else None


def discover(items, k: int = 20, method: str = "nmf", sample: int = 8000, seed: int = 7, top_terms: int = 4) -> dict:
    pool = [f for f in items if primary_theme(f)]
    rng = random.Random(seed)
    pool = rng.sample(pool, min(sample, len(pool)))
    texts = [f.text for f in pool]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.4, stop_words=STOP, sublinear_tf=True)
    x = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    if method == "nmf":
        model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
        w = model.fit_transform(x)
        assign = w.argmax(axis=1)
        comps = model.components_
    else:
        model = KMeans(n_clusters=k, n_init=3, random_state=seed)
        assign = model.fit_predict(x)
        comps = model.cluster_centers_
    truth = [primary_theme(f) for f in pool]
    clusters = []
    for c in range(k):
        members = [truth[i] for i in np.flatnonzero(assign == c)]
        if not members:
            continue
        top = [str(t) for t in terms[np.argsort(comps[c])[::-1][:top_terms]]]
        majority, cnt = Counter(members).most_common(1)[0]
        clusters.append(
            {
                "cluster": int(c),
                "name": " / ".join(top[:3]),
                "top_terms": top,
                "size": len(members),
                "majority_theme": majority,
                "majority_share": round(cnt / len(members), 3),
            }
        )
    clusters.sort(key=lambda r: -r["size"])
    found = {c["majority_theme"] for c in clusters if c["majority_share"] >= 0.5}
    return {
        "method": method,
        "k": k,
        "n": len(pool),
        "purity": round(purity(list(assign), truth), 4),
        "nmi": round(nmi(list(assign), truth), 4),
        "themes_recovered": len(found),
        "clusters": clusters,
    }
