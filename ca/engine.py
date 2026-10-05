"""Case similarity, citation treatment, judge statistics with shrinkage, a calibrated motion-outcome model with
uncertainty and attribution, and Kaplan-Meier time to resolution. Every number is a base rate with an interval:
none of it predicts what a court will do in a particular case."""
import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import world

VERSION = "ca-1.0"
MOTIONS, SUITS = list(world.MOTIONS), list(world.PATTERNS)
FEATURES = [f"motion: {m}" for m in MOTIONS] + [f"suit: {s}" for s in SUITS] + ["this judge's record on this motion", "how this motion fared in similar cases", "plaintiff has no lawyer", "times the complaint was amended", "key events documented in writing", "large defence firm", "number of claims (log)"]
NEIGHBOURS, NEIGHBOUR_PRIOR = 40, 10     # similar cases consulted, and how many imaginary average cases they are blended with
THIN_RECORD = 10                # fewer rulings than this on a motion type and the judge-specific figure is flagged
BOOT = 40


# ---------------------------------------------------------------- text

def documented(facts):
    """A structured feature read from the facts: do they say the key events are in writing? None if the facts do not say."""
    low = facts.lower()
    if re.search(r"no written record|recollection alone", low):
        return False
    return True if re.search(r"documented in contemporaneous|attaches correspondence", low) else None


def train_treatment(pairs):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True), LogisticRegression(max_iter=1000, C=10)).fit([t for t, _ in pairs], [k for _, k in pairs])


def keyword_treatment(text):
    low = text.lower()
    for label, pat in (("overruled", r"overruled|abrogated"), ("followed", r"follow|controls|adopts|consistent|as in"), ("criticized", r"disagree|questioned|unpersuasive|doubt"), ("distinguished", r"distinguish|unlike|misplaced")):
        if re.search(pat, low):
            return label
    return "followed"


class Similar:
    """TF-IDF over the facts of every case; cosine similarity."""

    def __init__(self, facts):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english")
        self.x = self.vec.fit_transform(facts)

    def query(self, text, k=10, mask=None, exclude=None):
        s = (self.x @ self.vec.transform([text]).T).toarray().ravel()
        if mask is not None:
            s = np.where(mask, s, -1)
        if exclude is not None:
            s[exclude] = -1
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    def neighbours(self, k=NEIGHBOURS, chunk=500):
        """The k most similar other cases for every case. -> array [case, k]"""
        out = np.empty((self.x.shape[0], k), dtype=np.int64)
        for a in range(0, self.x.shape[0], chunk):
            s = (self.x[a:a + chunk] @ self.x.T).toarray()
            s[np.arange(s.shape[0]), a + np.arange(s.shape[0])] = -1
            out[a:a + chunk] = np.argpartition(-s, k, axis=1)[:, :k]
        return out


# ---------------------------------------------------------------- judge statistics

def beta_prior(rates_n):
    """Method-of-moments Beta prior from each judge's (granted, rulings): what a typical judge looks like on this motion."""
    ps = [g / n for g, n in rates_n if n >= 20]
    if len(ps) < 3:
        return 1.0, 1.0
    m, v = float(np.mean(ps)), float(np.var(ps))
    k = max(2.0, m * (1 - m) / max(v, 1e-6) - 1)
    return m * k, (1 - m) * k


def posterior(granted, n, prior):
    """Shrunk grant rate with a 90% credible interval (normal approximation to the Beta posterior)."""
    a, b = prior[0] + granted, prior[1] + n - granted
    mean = a / (a + b)
    sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    return {"rulings": n, "granted": granted, "raw_rate": round(granted / n, 4) if n else None, "rate": round(mean, 4), "low": round(max(0.0, mean - 1.645 * sd), 4), "high": round(min(1.0, mean + 1.645 * sd), 4),
            "sd": sd, "thin_record": n < THIN_RECORD}


def logit(p):
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p / (1 - p))


# ---------------------------------------------------------------- outcome model

def row(motion_type, suit, judge_rate, similar_rate, pro_se, amended, is_documented, large_firm, claims):
    return ([float(motion_type == m) for m in MOTIONS] + [float(suit == s) for s in SUITS]
            + [logit(judge_rate), logit(similar_rate), float(pro_se), float(amended), 0.0 if is_documented is None else (1.0 if is_documented else -1.0), float(large_firm), math.log(max(1, claims))])


def fit(rows, y, seed=0):
    """Logistic regression plus BOOT bootstrap refits, which give the spread of the estimate."""
    x, y = np.array(rows), np.array(y, dtype=int)
    rng = np.random.default_rng(seed)
    main = LogisticRegression(C=1.0, max_iter=2000).fit(x, y)
    boots = []
    for _ in range(BOOT):
        idx = rng.integers(0, len(y), len(y))
        m = LogisticRegression(C=1.0, max_iter=2000).fit(x[idx], y[idx])
        boots.append(np.append(m.coef_[0], m.intercept_[0]))
    return {"coef": main.coef_[0], "intercept": float(main.intercept_[0]), "boots": np.array(boots), "mean": x.mean(axis=0), "n": len(y), "base_rate": float(y.mean())}


def predict(model, x):
    return 1 / (1 + np.exp(-(np.asarray(x) @ model["coef"] + model["intercept"])))


def estimate(model, x, judge_sd=0.0, seed=0):
    """-> probability, 80% interval and what moved it. The interval combines model uncertainty (bootstrap) with how
    little may be known about the judge (judge_sd: posterior sd of that judge's grant rate)."""
    x = np.array(x, dtype=float)
    p = float(predict(model, x))
    rng = np.random.default_rng(seed)
    j = len(MOTIONS) + len(SUITS)
    rate = 1 / (1 + math.exp(-x[j]))
    draws = []
    for b in model["boots"]:
        xb = x.copy()
        xb[j] = logit(float(np.clip(rng.normal(rate, judge_sd), 0.01, 0.99)))
        draws.append(1 / (1 + math.exp(-(xb @ b[:-1] + b[-1]))))
    lo, hi = np.percentile(draws, [10, 90])
    contrib = model["coef"] * (x - model["mean"])
    a, b = len(MOTIONS), len(MOTIONS) + len(SUITS)                 # the one-hot blocks are each reported as one factor
    parts = [("the type of motion", contrib[:a].sum()), ("the nature of suit", contrib[a:b].sum())] + list(zip(FEATURES[b:], contrib[b:]))
    parts = sorted(parts, key=lambda kv: -abs(kv[1]))[:5]
    return {"probability": round(p, 3), "low": round(float(min(lo, p)), 3), "high": round(float(max(hi, p)), 3),
            "drivers": [{"factor": f, "effect_log_odds": round(float(v), 3), "direction": "towards granted" if v > 0 else "towards denied"} for f, v in parts if abs(v) > 0.02]}


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y.sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def calibration(y, p, bins=5):
    """-> (expected calibration error, [{range, n, predicted, observed}])"""
    y, p = np.asarray(y, dtype=float), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges, edges[1:]):
        k = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if k.sum():
            table.append({"range": f"{lo:.0%} to {hi:.0%}", "n": int(k.sum()), "predicted": round(float(p[k].mean()), 3), "observed": round(float(y[k].mean()), 3)})
            ece += k.sum() / len(y) * abs(p[k].mean() - y[k].mean())
    return round(float(ece), 4), table


# ---------------------------------------------------------------- time to resolution

def kaplan_meier(durations, closed):
    """Survival curve for "still open after t days", using open cases for as long as they have been observed.
    -> [(day, share still open)]"""
    t, e = np.asarray(durations), np.asarray(closed, dtype=bool)
    s, out = 1.0, []
    for day in np.unique(t[e]):
        at_risk = (t >= day).sum()
        s *= 1 - ((t == day) & e).sum() / at_risk
        out.append((int(day), float(s)))
    return out


def quantile(curve, q=0.5):
    """Days by which a share q of cases has closed; None if the curve never gets there."""
    return next((d for d, s in curve if s <= 1 - q), None)
