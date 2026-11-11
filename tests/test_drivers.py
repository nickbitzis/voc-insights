import numpy as np
import pytest

from vocinsights.drivers import nps, ols, planted_effect


def test_ols_recovers_planted_effect():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, size=(4000, 3)).astype(float)
    y = 7.5 - 2.0 * x[:, 0] + 0.8 * x[:, 1] + rng.normal(0, 1.0, 4000)
    fit = ols(x, y)
    assert fit["beta"][0] == pytest.approx(7.5, abs=0.1)
    assert fit["beta"][1] == pytest.approx(-2.0, abs=0.1)
    assert fit["beta"][2] == pytest.approx(0.8, abs=0.1)
    assert abs(fit["beta"][3]) < 0.1 and fit["p"][3] > 0.01 and fit["p"][1] < 1e-10


def test_nps_formula():
    assert nps([10, 9, 8, 7, 6, 0]) == pytest.approx(0.0)
    assert nps([10, 10, 3]) == pytest.approx(33.3)
    assert nps([]) == 0.0


def test_planted_effect_mixes_sentiment():
    assert planted_effect("data_loss") == pytest.approx(-3.2)
    assert -1.0 < planted_effect("collaboration") < 1.0


def test_drivers_recover_generator_effects(bench):
    drv = {d["theme"]: d for d in bench.drivers["drivers"]}
    assert bench.drivers["drivers"][0]["coef"] < -2  # strongest negative driver is large
    for name in ("data_loss", "billing_errors", "app_stability", "great_support", "easy_to_use"):
        assert abs(drv[name]["coef"] - drv[name]["planted"]) < 0.6
    assert drv["data_loss"]["coef"] < drv["notifications_spam"]["coef"]


def test_nps_falls_with_negative_aspects(bench):
    rows = bench.nps_sentiment["by_negative_aspects"]
    means = [r["mean_score"] for r in rows]
    assert means == sorted(means, reverse=True)
    assert bench.nps_sentiment["corr_score_net_sentiment"] > 0.3
