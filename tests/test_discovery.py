import pytest
from sklearn.metrics import normalized_mutual_info_score

from vocinsights.discovery import nmi, purity


def test_perfect_clustering():
    assert purity([0, 0, 1, 1], ["a", "a", "b", "b"]) == 1.0
    assert nmi([5, 5, 7, 7], ["a", "a", "b", "b"]) == pytest.approx(1.0)


def test_known_purity():
    assert purity([0, 0, 0, 1, 1, 1], ["a", "a", "b", "b", "b", "b"]) == pytest.approx(5 / 6)


def test_independent_clusters_have_zero_nmi():
    assert nmi([0, 1, 0, 1], ["a", "a", "b", "b"]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "c, y",
    [
        ([0, 0, 0, 1, 1, 1], ["a", "a", "b", "b", "b", "b"]),
        ([0, 1, 2, 0, 1, 2, 2], list("xxyyzzz")),
        ([1, 1, 1, 1], list("abab")),
    ],
)
def test_nmi_matches_scikit_learn(c, y):
    assert nmi(c, y) == pytest.approx(normalized_mutual_info_score(y, c))


def test_discovery_on_bench(bench):
    d = bench.discovery
    assert set(d) == {"nmf_k20", "kmeans_k20", "nmf_k40"}
    for r in d.values():
        assert 0 < r["purity"] <= 1 and 0 < r["nmi"] <= 1
        assert all(" / " in c["name"] for c in r["clusters"])
    # more clusters gives purer groups
    assert d["nmf_k40"]["purity"] > d["nmf_k20"]["purity"]
