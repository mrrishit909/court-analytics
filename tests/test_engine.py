"""Unit tests for the statistics. No database."""
import numpy as np
import pytest

from ca import engine, train, world


def test_documented_is_read_from_the_facts_or_left_unknown():
    assert engine.documented("Plaintiff attaches correspondence supporting each allegation.") is True
    assert engine.documented("The complaint identifies no written record of the key events.") is False
    assert engine.documented("Plaintiff seeks damages.") is None


def test_keyword_rules_trip_where_expected():
    assert engine.keyword_treatment("The court does not follow 20-cv-01234 (D. X), which involved a different statute.") == "followed"       # really: distinguished
    assert engine.keyword_treatment("20-cv-01234 was abrogated and cannot be followed.") == "overruled"


def test_shrinkage_pulls_small_samples_towards_the_typical_judge():
    prior = engine.beta_prior([(40, 100), (55, 100), (50, 100), (45, 100), (60, 100)])
    assert prior[0] / sum(prior) == pytest.approx(0.5, abs=0.01)
    few, many, none = engine.posterior(3, 3, prior), engine.posterior(90, 100, prior), engine.posterior(0, 0, prior)
    assert few["raw_rate"] == 1.0 and 0.5 < few["rate"] < 0.7 and few["thin_record"] and (few["high"] - few["low"]) > (many["high"] - many["low"])
    assert 0.7 < many["rate"] < 0.9 and not many["thin_record"]            # even 100 rulings are pulled in a little when judges rarely differ this much
    assert none["rate"] == pytest.approx(0.5, abs=0.01) and none["raw_rate"] is None and none["low"] < none["rate"] < none["high"]
    assert engine.beta_prior([(1, 2)]) == (1.0, 1.0)                              # too few judges to learn a prior from: flat


def test_auc_and_calibration():
    assert engine.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0 and engine.auc([0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2]) == 0.0
    ece, table = engine.calibration([1] * 7 + [0] * 3, [0.7] * 10)
    assert ece == 0.0 and table == [{"range": "60% to 80%", "n": 10, "predicted": 0.7, "observed": 0.7}]
    assert engine.calibration([0] * 10, [0.9] * 10)[0] == pytest.approx(0.9)


def test_kaplan_meier_counts_open_cases_for_as_long_as_they_were_open():
    #                       closed at 10, 20, 30; two still open at 25 and 40
    curve = engine.kaplan_meier([10, 20, 30, 25, 40], [True, True, True, False, False])
    assert curve == [(10, pytest.approx(0.8)), (20, pytest.approx(0.6)), (30, pytest.approx(0.3))]
    assert engine.quantile(curve, 0.5) == 30 and engine.quantile(curve, 0.9) is None      # never observed 90% closing
    assert float(np.median([10, 20, 30])) == 20                                            # what ignoring open cases would say


def test_the_estimate_widens_when_the_judge_is_unknown_and_explains_itself():
    rng = np.random.default_rng(0)
    rows, y = [], []
    for _ in range(1500):
        jr, pro = rng.uniform(0.2, 0.8), rng.random() < 0.3
        rows.append(engine.row("dismiss", "contract", jr, 0.5, pro, 0, None, False, 3))
        y.append(rng.random() < 1 / (1 + np.exp(-(engine.logit(jr) + 0.8 * pro))))
    model = engine.fit(rows, y)
    known = engine.estimate(model, engine.row("dismiss", "contract", 0.75, 0.5, True, 0, None, False, 3), judge_sd=0.02)
    unknown = engine.estimate(model, engine.row("dismiss", "contract", 0.75, 0.5, True, 0, None, False, 3), judge_sd=0.25)
    assert known["low"] <= known["probability"] <= known["high"] and (unknown["high"] - unknown["low"]) > 2 * (known["high"] - known["low"])
    assert known["probability"] > 0.75 and {d["factor"] for d in known["drivers"][:2]} == {"this judge's record on this motion", "plaintiff has no lawyer"}
    assert all(d["direction"] == "towards granted" for d in known["drivers"][:2])


def test_training_rows_do_not_see_their_own_outcome():
    w = world.build(41, 400)
    t = train.tables(w, world.SPLIT)
    m = next(m for m in w["motions"] if m["decided"] < world.SPLIT)
    case = w["cases"][m["case"]]
    nb = np.array([], dtype=int)
    j = len(engine.MOTIONS) + len(engine.SUITS)
    flipped = {**m, "granted": not m["granted"]}
    g, n = t[0][case["judge"], m["type"]]
    t2 = ({**t[0], (case["judge"], m["type"]): (g - m["granted"] + flipped["granted"], n)}, *t[1:])
    assert train.feature_row(m, case, m["case"], t, nb, leave_out=True)[j] == pytest.approx(train.feature_row(flipped, case, m["case"], t2, nb, leave_out=True)[j])
    assert train.feature_row(m, case, m["case"], t, nb)[j] != pytest.approx(train.feature_row(flipped, case, m["case"], t2, nb)[j])


def test_similar_cases_share_the_hidden_fact_pattern_and_exclude_the_case_itself():
    w = world.build(41, 400)
    index = engine.Similar([c["facts"] for c in w["cases"]])
    got = index.query(w["cases"][7]["facts"], 5, exclude=7)
    assert 7 not in [i for i, _ in got] and all(w["cases"][i]["pattern"] == w["cases"][7]["pattern"] for i, _ in got)
    nb = index.neighbours(k=5)
    assert nb.shape == (400, 5) and all(k not in nb[k] for k in range(400))
    assert index.query("zzzz qqqq", 5) == []
