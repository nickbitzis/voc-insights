from vocinsights import rules


def test_split_clauses_on_sentences_and_but():
    assert rules.split_clauses("Sync is slow. Love the design!") == ["Sync is slow.", "Love the design!"]
    assert rules.split_clauses("Looks great, but it crashes") == ["Looks great", "it crashes"]


def test_single_clause_match_is_confident():
    r = rules.match("The app crashes every morning.")
    assert r.themes == ["app_stability"] and r.confidence >= 0.9


def test_two_clauses_two_themes():
    r = rules.match("I was charged twice this month. Way too many notifications.")
    assert r.themes == ["billing_errors", "notifications_spam"] and r.confidence >= 0.9


def test_unexplained_clause_lowers_confidence():
    r = rules.match("The app crashes all the time. Where did all my stuff go.")
    assert "app_stability" in r.themes and r.confidence < 0.9


def test_no_match():
    r = rules.match("Love it!")
    assert r.themes == [] and r.confidence == 0.0


def test_ambiguous_clause_is_not_confident():
    # "sync" and "notes disappeared" fire two themes in one clause
    r = rules.match("My notes disappeared after a sync")
    assert set(r.themes) == {"slow_sync", "data_loss"} and r.confidence < 0.9
