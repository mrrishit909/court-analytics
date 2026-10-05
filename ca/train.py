"""Build the judge statistics and the outcome model from a world's motions. Shared by the ingest job and the offline
evaluation so both use exactly the same procedure. Only motions decided before the split date are learnt from."""
from . import engine, world


def judge_tables(motions, cases, cut=None):
    """-> ({(judge, motion type): (granted, rulings)}, {motion type: beta prior})"""
    counts = {}
    for m in motions:
        if cut is None or m["decided"] < cut:
            g, n = counts.get((cases[m["case"]]["judge"], m["type"]), (0, 0))
            counts[cases[m["case"]]["judge"], m["type"]] = (g + m["granted"], n + 1)
    priors = {t: engine.beta_prior([v for (_, mt), v in counts.items() if mt == t]) for t in world.MOTIONS}
    return counts, priors


def similar_rate(neighbours, by_case, motion_type, base, skip=None):
    """Grant rate of this motion type in the similar cases, blended with the overall rate so a handful of cases cannot dominate."""
    hits = [g for k in neighbours if k != skip for g in by_case.get((int(k), motion_type), ())]
    return (sum(hits) + engine.NEIGHBOUR_PRIOR * base) / (len(hits) + engine.NEIGHBOUR_PRIOR), len(hits)


def tables(w, cut):
    """Everything the model's features are computed from, using only motions decided before `cut`."""
    counts, priors = judge_tables(w["motions"], w["cases"], cut)
    past = [m for m in w["motions"] if m["decided"] < cut]
    base = {t: sum(m["granted"] for m in past if m["type"] == t) / max(1, sum(m["type"] == t for m in past)) for t in world.MOTIONS}
    by_case = {}
    for m in past:
        by_case.setdefault((m["case"], m["type"]), []).append(m["granted"])
    return counts, priors, base, by_case


def feature_row(m, case, k, t, neighbours, leave_out=False):
    counts, priors, base, by_case = t
    g, n = counts.get((case["judge"], m["type"]), (0, 0))
    if leave_out:                                              # a training row must not see its own outcome in the judge's record
        g, n = g - m["granted"], n - 1
    rate = engine.posterior(g, n, priors[m["type"]])["rate"]
    sim, _ = similar_rate(neighbours, by_case, m["type"], base[m["type"]], skip=k)
    return engine.row(m["type"], case["suit"], rate, sim, case["pro_se"], case["amended"], engine.documented(case["facts"]), case["large_firm_defence"], case["claims"])


def train(w, cut=world.SPLIT):
    """-> (model, tables, similarity index, neighbours, holdout rows: [(motion index, features, granted)])"""
    t = tables(w, cut)
    index = engine.Similar([c["facts"] for c in w["cases"]])
    nb = index.neighbours()
    tr = [(feature_row(m, w["cases"][m["case"]], m["case"], t, nb[m["case"]], leave_out=True), m["granted"]) for m in w["motions"] if m["decided"] < cut]
    hold = [(i, feature_row(m, w["cases"][m["case"]], m["case"], t, nb[m["case"]]), m["granted"]) for i, m in enumerate(w["motions"]) if m["decided"] >= cut]
    return engine.fit([x for x, _ in tr], [y for _, y in tr]), t, index, nb, hold
