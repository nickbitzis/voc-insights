"""Tier 2: TF-IDF features and one-vs-rest logistic regression, multi-label.

Word unigrams and bigrams carry the meaning; character n-grams within words make the model robust to
typos ("synk", "pasword"). Confidence for a text is the smallest distance of any theme probability
from the 0.5 decision line, scaled to 0 to 1: a text is only as certain as its least certain theme.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .taxonomy import THEME_NAMES


def to_matrix(label_sets, names=THEME_NAMES) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    y = np.zeros((len(label_sets), len(names)), dtype=np.int8)
    for r, labels in enumerate(label_sets):
        for name in labels:
            if name in idx:
                y[r, idx[name]] = 1
    return y


class ThemeClassifier:
    def __init__(self, c: float = 6.0, seed: int = 7):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=40000
        )
        self.model = OneVsRestClassifier(LogisticRegression(C=c, max_iter=2000, random_state=seed))
        self.names = THEME_NAMES

    def _x(self, texts, fit: bool = False):
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts: list[str], label_sets) -> ThemeClassifier:
        x = self._x(texts, fit=True)
        self.model.fit(x, to_matrix(label_sets, self.names))
        return self

    def proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(self._x(texts))

    @staticmethod
    def decide(p: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Binary labels and a per-text confidence. A text may have no theme ("Love it!")."""
        y = (p >= threshold).astype(np.int8)
        conf = (np.abs(p - threshold) / np.maximum(threshold, 1 - threshold)).min(axis=1)
        return y, conf
