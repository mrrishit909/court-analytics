"""Public API (modular monolith). Blueprint services map to: docket-ingestion + document-parser + citation-service (the ingest
job), precedent-graph, judge-analytics, similarity-service, prediction-service (motion estimates) and workspace-service (saved
brief analyses, audit).

    uvicorn ca.api:app          python -m core.jobs ca.api      # the worker
"""
import datetime
import hashlib
import json
import pickle
import uuid
from typing import Literal

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine, train, world
from .seed import TENANTS

READ = {"records:read", "jobs:read"}
WORK = READ | {"briefs:analyze"}
PERMISSIONS = {"associate": WORK, "partner": WORK | {"audit:read"}, "admin": WORK | {"records:ingest", "audit:read"}}
app = create_app("court-analytics", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
Motion = Literal["dismiss", "summary_judgment", "compel", "preliminary_injunction"]
Suit = Literal["employment", "contract", "civil_rights", "patent", "injury", "securities"]
CAVEAT = "This is a statistical summary of past rulings, not legal advice and not a prediction of what the court will do in this case."
_cache = {}


def clf():
    if "clf" not in _cache:
        with db.tx() as c:
            _cache["clf"] = pickle.loads(c.execute("SELECT artifact FROM model_registry WHERE name = 'citation-treatment' AND state = 'production'").fetchone()["artifact"])   # our own training output
    return _cache["clf"]


def index(c, t):
    """This tenant's fitted artefacts: outcome model, similarity index, neighbours and the tables behind the features."""
    row = c.execute("SELECT fitted_at FROM indexes").fetchone()
    if not row:
        raise Problem(409, "nothing_ingested", "ingest dockets first: POST /v1/dockets/ingest")
    if _cache.get(t, (None,))[0] != row["fitted_at"]:
        _cache[t] = (row["fitted_at"], pickle.loads(c.execute("SELECT artifact FROM indexes").fetchone()["artifact"]))        # written by this service's own ingest job
    return _cache[t][1]


# ---------------------------------------------------------------- docket-ingestion

class IngestIn(BaseModel):
    source: str = Field("demo-dockets", pattern="^demo-dockets$", description="The one source in this build: a deterministic synthetic court system")


@app.post("/v1/dockets/ingest", status_code=202, tags=["dockets"], summary="(+) Ingest dockets, classify every citation, and fit this tenant's similarity index and outcome model (job)")
def ingest(body: IngestIn, ctx: Ctx = Depends(auth("records:ingest")), idem: str | None = IdemKey):
    return run(ctx, idem, body, lambda c: (202, jobs.enqueue(c, ctx, "dockets.ingest", body.model_dump())))


@jobs.handler("dockets.ingest")
def ingest_job(c, job):
    t = job["tenant_id"]
    if c.execute("SELECT 1 FROM cases LIMIT 1").fetchone():
        return {"cases": 0, "note": "these dockets are already loaded; nothing added"}
    name = c.execute("SELECT name FROM tenants WHERE id = %s", [t]).fetchone()["name"]
    w = world.build(*next(v[1:] for v in TENANTS.values() if v[0] == name))
    jid = [uuid.uuid5(t, f"judge-{i}") for i in range(len(w["judges"]))]
    cid = [uuid.uuid5(t, f"case-{i}") for i in range(len(w["cases"]))]
    mid = [uuid.uuid5(t, f"motion-{i}") for i in range(len(w["motions"]))]
    oid = [uuid.uuid5(t, f"order-{i}") for i in range(len(w["motions"]))]
    at = lambda d: datetime.datetime.combine(d, datetime.time(12), datetime.UTC)      # noqa: E731
    db.load(c, "judges", ["id", "tenant_id", "name", "court", "appointed_at", "ordinal"], [(jid[i], t, j["name"], j["court"], j["appointed"], i) for i, j in enumerate(w["judges"])])
    db.load(c, "cases", ["id", "tenant_id", "court", "docket_number", "filed_at", "closed_at", "nature_of_suit", "judge_id", "status", "facts", "features", "ordinal"],
            [(cid[i], t, x["court"], x["docket"], x["filed"], x["closed"], x["suit"], jid[x["judge"]], "closed" if x["closed"] else "open", x["facts"],
              json.dumps({k: x[k] for k in ("pro_se", "amended", "claims", "large_firm_defence")}), i) for i, x in enumerate(w["cases"])])
    db.load(c, "filings", ["id", "tenant_id", "case_id", "filing_type", "filed_at", "document_uri", "text_hash"],
            [(uuid.uuid5(mid[i], "filing"), t, cid[m["case"]], "motion", at(m["filed"]), f"synthetic://motion/{i}", hashlib.sha256(f"m{i}".encode()).hexdigest()) for i, m in enumerate(w["motions"])]
            + [(oid[i], t, cid[m["case"]], "order", at(m["decided"]), f"synthetic://order/{i}", hashlib.sha256(f"o{i}".encode()).hexdigest()) for i, m in enumerate(w["motions"])])
    db.load(c, "motions", ["id", "tenant_id", "case_id", "motion_type", "filed_at", "outcome", "decided_at", "order_filing"],
            [(mid[i], t, cid[m["case"]], m["type"], m["filed"], "granted" if m["granted"] else "denied", m["decided"], oid[i]) for i, m in enumerate(w["motions"])])
    model = clf()
    proba = model.predict_proba([x["context"] for x in w["cites"]])
    seen, rows, counts = set(), [], {}
    for x, pr in zip(w["cites"], proba):
        label = str(model.classes_[pr.argmax()])                  # the generator's own label (x["treatment"]) is never read here
        if (x["motion"], x["cited"]) not in seen:
            seen.add((x["motion"], x["cited"]))
            counts[label] = counts.get(label, 0) + 1
            rows.append((oid[x["motion"]], cid[x["cited"]], x["context"], label, round(float(pr.max()), 4), t))
    db.load(c, "citations", ["source_filing", "cited_case", "context", "treatment", "confidence", "tenant_id"], rows)
    fitted, tables, sim, nb, hold = train.train(w)
    ys, ps = np.array([int(y) for _, _, y in hold]), np.array([float(engine.predict(fitted, x)) for _, x, _ in hold])
    judge_only = np.array([engine.posterior(*tables[0].get((w["cases"][w["motions"][i]["case"]]["judge"], w["motions"][i]["type"]), (0, 0)), tables[1][w["motions"][i]["type"]])["rate"] for i, _, _ in hold])
    base = np.array([tables[2][w["motions"][i]["type"]] for i, _, _ in hold])
    ece, table = engine.calibration(ys, ps)
    brier = lambda q: round(float(((q - ys) ** 2).mean()), 4)      # noqa: E731
    metrics = {"trained_on": int(fitted["n"]), "trained_through": str(world.SPLIT - datetime.timedelta(days=1)), "held_out": len(ys), "held_out_granted": round(float(ys.mean()), 4),
               "model": {"auc": round(engine.auc(ys, ps), 4), "brier": brier(ps), "calibration_error": ece}, "judge_rate_only": {"auc": round(engine.auc(ys, judge_only), 4), "brier": brier(judge_only)},
               "motion_base_rate": {"auc": round(engine.auc(ys, base), 4), "brier": brier(base)}, "calibration": table, "version": engine.VERSION,
               "coefficients": [{"factor": f, "log_odds": round(float(v), 3)} for f, v in zip(engine.FEATURES, fitted["coef"])]}
    c.execute("INSERT INTO indexes (tenant_id, artifact, metrics, version) VALUES (%s,%s,%s,%s)", [t, pickle.dumps({"model": fitted, "tables": tables, "sim": sim, "nb": nb}), Jsonb(metrics), engine.VERSION])
    audit.record(c, Ctx(t, uuid.UUID(job["payload"]["actor_id"]), "worker", "system"), "dockets.ingested", "tenant", t, {"cases": len(cid), "motions": len(mid), "model": engine.VERSION})
    return {"cases": len(cid), "courts": len(world.COURTS), "judges": len(jid), "motions": len(mid), "citations": len(rows), "citation_treatment": counts, "open_cases": sum(x["closed"] is None for x in w["cases"]),
            "filed": [str(w["cases"][0]["filed"]), str(w["cases"][-1]["filed"])], "outcome_model": {k: metrics[k] for k in ("trained_on", "trained_through", "held_out", "model", "judge_rate_only", "motion_base_rate")}}


# ---------------------------------------------------------------- search + similarity

@app.get("/v1/cases/search", tags=["cases"], summary="Search case facts (full text) with court, nature-of-suit and judge filters; newest first, cursor-paginated")
def search(q: str = "", court: str | None = None, nature_of_suit: str | None = None, judge_id: uuid.UUID | None = None, limit: int = 10, before: uuid.UUID | None = None, ctx: Ctx = Depends(auth("records:read"))):
    limit = max(1, min(limit, 100))
    where = """(%(q)s = '' OR k.tsv @@ websearch_to_tsquery('english', %(q)s)) AND (%(court)s::text IS NULL OR k.court = %(court)s) AND (%(suit)s::text IS NULL OR k.nature_of_suit = %(suit)s)
               AND (%(judge)s::uuid IS NULL OR k.judge_id = %(judge)s)"""
    p = {"q": q, "court": court, "suit": nature_of_suit, "judge": judge_id, "before": before, "limit": limit + 1}
    with db.tx(ctx.tenant_id) as c:
        rows = c.execute(f"""SELECT k.id, k.docket_number, k.court, k.filed_at, k.closed_at, k.nature_of_suit, k.status, j.name AS judge, k.facts FROM cases k JOIN judges j ON j.id = k.judge_id
                              WHERE {where} AND (%(before)s::uuid IS NULL OR (k.filed_at, k.id) < (SELECT filed_at, id FROM cases WHERE id = %(before)s)) ORDER BY k.filed_at DESC, k.id DESC LIMIT %(limit)s""", p).fetchall()
        total = c.execute(f"SELECT count(*) AS n FROM cases k WHERE {where}", p).fetchone()["n"]
    return jsonable_encoder({"matches": total, "items": rows[:limit], "next_before": rows[limit - 1]["id"] if len(rows) > limit else None})


def describe(c, ordinals, scores, motion_type=None):
    rows = {r["ordinal"]: r for r in c.execute("""SELECT k.id, k.ordinal, k.docket_number, k.court, k.filed_at, k.nature_of_suit, k.facts, j.name AS judge,
                                                         (SELECT jsonb_agg(jsonb_build_object('motion_type', m.motion_type, 'outcome', m.outcome, 'decided_at', m.decided_at)) FROM motions m WHERE m.case_id = k.id) AS motions
                                                    FROM cases k JOIN judges j ON j.id = k.judge_id WHERE k.ordinal = ANY(%s)""", [list(ordinals)])}
    out = []
    for o, s in zip(ordinals, scores):
        r = dict(rows[o])
        r["similarity"] = round(s, 3)
        r["this_motion"] = next((m["outcome"] for m in r["motions"] or [] if m["motion_type"] == motion_type), None) if motion_type else None
        out.append(r)
    return out


@app.get("/v1/cases/{case_id}/similar", tags=["cases"], summary="Cases with the most similar facts")
def similar(case_id: uuid.UUID, limit: int = 8, ctx: Ctx = Depends(auth("records:read"))):
    with db.tx(ctx.tenant_id) as c:
        k = c.execute("SELECT ordinal, facts, docket_number FROM cases WHERE id = %s", [case_id]).fetchone()
        if not k:
            raise Problem(404, "case_not_found")
        got = index(c, ctx.tenant_id)["sim"].query(k["facts"], max(1, min(limit, 30)), exclude=k["ordinal"])
        return jsonable_encoder({"case": k["docket_number"], "method": "TF-IDF cosine on the facts", "items": describe(c, [i for i, _ in got], [s for _, s in got])})


# ---------------------------------------------------------------- judge-analytics

@app.get("/v1/judges", tags=["judges"], summary="(+) Judges, with how many rulings each has on record")
def judges(court: str | None = None, ctx: Ctx = Depends(auth("records:read"))):
    with db.tx(ctx.tenant_id) as c:
        return jsonable_encoder({"items": c.execute("""SELECT j.id, j.name, j.court, j.appointed_at, (SELECT count(*) FROM motions m JOIN cases k ON k.id = m.case_id WHERE k.judge_id = j.id) AS rulings
                                                         FROM judges j WHERE %s::text IS NULL OR j.court = %s ORDER BY rulings DESC, j.name""", [court, court]).fetchall()})


def judge_record(c, t, judge):
    """Shrunk grant rates per motion type from every ruling on record (the outcome model's features use the pre-split subset)."""
    counts = {r["motion_type"]: (r["g"], r["n"]) for r in c.execute("""SELECT m.motion_type, count(*) FILTER (WHERE m.outcome = 'granted') AS g, count(*) AS n FROM motions m JOIN cases k ON k.id = m.case_id
                                                                         WHERE k.judge_id = %s GROUP BY 1""", [judge["id"]])}
    priors = index(c, t)["tables"][1]
    return {m: {**engine.posterior(*counts.get(m, (0, 0)), priors[m]), "typical_judge": round(priors[m][0] / sum(priors[m]), 4)} for m in world.MOTIONS}


@app.get("/v1/judges/{judge_id}/analytics", tags=["judges"], summary="A judge's grant rates by motion type with credible intervals, time to ruling and time to resolution against the court")
def judge_analytics(judge_id: uuid.UUID, ctx: Ctx = Depends(auth("records:read"))):
    with db.tx(ctx.tenant_id) as c:
        j = c.execute("SELECT id, name, court, appointed_at FROM judges WHERE id = %s", [judge_id]).fetchone()
        if not j:
            raise Problem(404, "judge_not_found")
        record = judge_record(c, ctx.tenant_id, j)
        ruling = c.execute("""SELECT m.motion_type, percentile_cont(0.5) WITHIN GROUP (ORDER BY m.decided_at - m.filed_at) AS median_days FROM motions m JOIN cases k ON k.id = m.case_id WHERE k.judge_id = %s GROUP BY 1""", [judge_id]).fetchall()
        cut = c.execute("SELECT max(decided_at) AS d FROM motions").fetchone()["d"]
        cs = c.execute("SELECT judge_id = %s AS mine, (coalesce(closed_at, %s) - filed_at) AS days, closed_at IS NOT NULL AS closed, nature_of_suit FROM cases WHERE court = %s", [judge_id, cut, j["court"]]).fetchall()
    mine = [x for x in cs if x["mine"]]

    def resolution(rows):
        curve = engine.kaplan_meier([x["days"] for x in rows], [x["closed"] for x in rows]) if rows else []
        return {"cases": len(rows), "still_open": sum(not x["closed"] for x in rows), "median_days": engine.quantile(curve), "quarter_closed_by": engine.quantile(curve, 0.25), "three_quarters_closed_by": engine.quantile(curve, 0.75)}
    suits = {}
    for x in mine:
        suits[x["nature_of_suit"]] = suits.get(x["nature_of_suit"], 0) + 1
    for r in ruling:
        record[r["motion_type"]]["median_days_to_ruling"] = float(r["median_days"])
    return jsonable_encoder({"judge": j, "motions": [{"motion_type": m, "name": world.MOTION_NAME[m], **{k: v for k, v in s.items() if k != "sd"}} for m, s in record.items()],
                             "time_to_resolution": {"this_judge": resolution(mine), "court": resolution(cs), "method": "Kaplan-Meier: open cases count for as long as they have been open"},
                             "caseload": sorted(suits.items(), key=lambda kv: -kv[1]), "warnings": [f"{world.MOTION_NAME[m]}: only {s['rulings']} rulings, so the figure is mostly the court average" for m, s in record.items() if s["thin_record"]],
                             "note": "Rates are shrunk towards a typical judge of these courts in proportion to how few rulings there are. Intervals are 90% credible intervals."})


# ---------------------------------------------------------------- prediction-service

class EstimateIn(BaseModel):
    motion_type: Motion
    judge_id: uuid.UUID
    nature_of_suit: Suit
    facts: str = Field(min_length=40, max_length=5000, description="The fact section. Confidential: stored only in this tenant's workspace, never logged")
    pro_se: bool = False
    amended: int = Field(0, ge=0, le=10)
    claims: int = Field(1, ge=1, le=50)
    large_firm_defence: bool = False


def estimate_for(c, t, body):
    ix = index(c, t)
    j = c.execute("SELECT id, name, court, ordinal FROM judges WHERE id = %s", [body.judge_id]).fetchone()
    if not j:
        raise Problem(404, "judge_not_found")
    counts, priors, base, by_case = ix["tables"]
    post = engine.posterior(*counts.get((j["ordinal"], body.motion_type), (0, 0)), priors[body.motion_type])
    near = ix["sim"].query(body.facts, engine.NEIGHBOURS)
    sim_rate, sim_n = train.similar_rate([i for i, _ in near], by_case, body.motion_type, base[body.motion_type])
    x = engine.row(body.motion_type, body.nature_of_suit, post["rate"], sim_rate, body.pro_se, body.amended, engine.documented(body.facts), body.large_firm_defence, body.claims)
    est = engine.estimate(ix["model"], x, post["sd"], seed=j["ordinal"])
    flags = ([f"this judge had only {post['rulings']} rulings on this kind of motion when the model was fitted: the estimate leans on the court average and the range is wide"] if post["thin_record"] else []) \
        + (["the facts resemble no case on record closely: the similar-case figure is the overall base rate"] if not near else []) \
        + (["the facts do not say whether the key events are documented: that factor was left neutral"] if engine.documented(body.facts) is None else [])
    return j, near, {**est, "outcome": "granted", "base_rate_for_this_motion": round(base[body.motion_type], 3), "similar_cases_rate": round(sim_rate, 3), "similar_rulings_used": sim_n, "flags": flags,
                     "interval": "80% range from refitting the model on resampled data and from uncertainty in the judge's record. It does not cover what the model cannot see.", "model_version": engine.VERSION, "caveat": CAVEAT}


@app.post("/v1/motions/estimate", tags=["estimates"], summary="Estimated chance a motion is granted, with a range, the factors behind it and warnings. Nothing is stored")
def estimate(body: EstimateIn, ctx: Ctx = Depends(auth("briefs:analyze"))):
    with db.tx(ctx.tenant_id) as c:
        _, _, est = estimate_for(c, ctx.tenant_id, body)
    return est


@app.get("/v1/models/motion-outcome", tags=["estimates"], summary="(+) How the outcome model did on rulings it was not trained on: discrimination, calibration, and simpler baselines")
def model_card(ctx: Ctx = Depends(auth("records:read"))):
    with db.tx(ctx.tenant_id) as c:
        row = c.execute("SELECT metrics, fitted_at FROM indexes").fetchone()
    if not row:
        raise Problem(409, "nothing_ingested", "ingest dockets first: POST /v1/dockets/ingest")
    return jsonable_encoder({**row["metrics"], "fitted_at": row["fitted_at"], "caveat": CAVEAT})


# ---------------------------------------------------------------- briefs

class BriefIn(EstimateIn):
    cited_dockets: list[str] = Field(default_factory=list, max_length=20)


def treatment_of(c, case_ids):
    rows = c.execute("""SELECT x.cited_case, x.treatment, count(*) AS n, (array_agg(x.source_filing ORDER BY f.filed_at DESC))[1:3] AS filings, (array_agg(x.context ORDER BY f.filed_at DESC))[1] AS latest
                          FROM citations x JOIN filings f ON f.id = x.source_filing WHERE x.cited_case = ANY(%s) GROUP BY 1, 2""", [case_ids]).fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["cited_case"], {})[r["treatment"]] = r
    return out


@app.post("/v1/briefs/analyze", status_code=201, tags=["briefs"], summary="Analyse a motion: similar cases, how its cited cases have been treated since, the judge's record, an estimate, and a memo in which every sentence cites its sources")
def analyze(body: BriefIn, ctx: Ctx = Depends(auth("briefs:analyze")), idem: str | None = IdemKey):
    def work(c):
        j, near, est = estimate_for(c, ctx.tenant_id, body)
        sims = describe(c, [i for i, _ in near[:8]], [s for _, s in near[:8]], body.motion_type)
        ruled = describe(c, [i for i, _ in near], [s for _, s in near], body.motion_type)
        ruled = [r for r in ruled if r["this_motion"]]
        cited = {r["docket_number"]: r for r in c.execute("SELECT id, docket_number, court, filed_at FROM cases WHERE docket_number = ANY(%s)", [body.cited_dockets])}
        treat = treatment_of(c, [r["id"] for r in cited.values()])
        checks = []
        for d in body.cited_dockets:
            if d not in cited:
                checks.append({"docket": d, "status": "not found", "detail": "no case with this docket number is on record"})
                continue
            tr = treat.get(cited[d]["id"], {})
            n = {k: tr[k]["n"] if k in tr else 0 for k in world.TREATMENT}
            status = "overruled" if n["overruled"] else "caution" if n["criticized"] >= 2 or n["distinguished"] > n["followed"] else "good" if n["followed"] else "never cited"
            worst = tr.get("overruled") or (tr.get("followed") if status == "good" else None) or tr.get("criticized") or tr.get("distinguished") or tr.get("followed")
            checks.append({"docket": d, "case_id": cited[d]["id"], "court": cited[d]["court"], "status": status, **n, "latest": worst["latest"] if worst else None, "sources": worst["filings"] if worst else []})
        rec = judge_record(c, ctx.tenant_id, j)[body.motion_type]
        name = world.MOTION_NAME[body.motion_type].lower()
        memo = [{"text": f"This is a {name} in {'an' if body.nature_of_suit[0] in 'aeiou' else 'a'} {body.nature_of_suit.replace('_', ' ')} case before {j['name']} ({j['court']}).", "sources": [{"type": "judge", "id": j["id"]}]}]
        memo.append({"text": (f"{j['name']} has ruled on this kind of motion {rec['rulings']} times and granted {rec['granted']} ({rec['raw_rate']:.0%}). Allowing for the sample size the rate is put at {rec['rate']:.0%} "
                              f"(90% range {rec['low']:.0%} to {rec['high']:.0%}); a typical judge of these courts grants {rec['typical_judge']:.0%}.") if rec["rulings"]
                     else f"{j['name']} has no rulings on this kind of motion on record; the court-wide rate of {rec['typical_judge']:.0%} is the only guide.", "sources": [{"type": "judge", "id": j["id"]}]})
        if ruled:
            g = sum(r["this_motion"] == "granted" for r in ruled)
            memo.append({"text": f"Among the {len(near)} cases with the most similar facts, {len(ruled)} had a ruling on a {name}: {g} granted, {len(ruled) - g} denied.", "sources": [{"type": "case", "id": r["id"]} for r in ruled]})
        for k in checks:
            if k["status"] == "overruled":
                memo.append({"text": f"{k['docket']}, cited in the brief, has been overruled according to {k['overruled']} later order{'s' if k['overruled'] > 1 else ''}. Do not rely on it without reading them.", "sources": [{"type": "filing", "id": f} for f in k["sources"]]})
            elif k["status"] == "caution":
                memo.append({"text": f"{k['docket']} has been distinguished {k['distinguished']} times and criticised {k['criticized']} times against {k['followed']} follows: check that it supports the point it is cited for.", "sources": [{"type": "filing", "id": f} for f in k["sources"]]})
            elif k["status"] == "good":
                memo.append({"text": f"{k['docket']} has been followed {k['followed']} times and never overruled.", "sources": [{"type": "filing", "id": f} for f in k["sources"]]})
            elif k["status"] == "not found":
                memo.append({"text": f"{k['docket']} could not be found on record, so its later treatment was not checked.", "sources": []})
        drivers = "; ".join(f"{d['factor']} ({d['direction']})" for d in est["drivers"][:3])
        memo.append({"text": f"Estimated chance the motion is granted: {est['probability']:.0%} (range {est['low']:.0%} to {est['high']:.0%}). Largest factors: {drivers}.", "sources": [{"type": "model", "id": engine.VERSION}, {"type": "judge", "id": j["id"]}]})
        memo.append({"text": CAVEAT, "sources": [{"type": "model", "id": engine.VERSION}]})
        kept = [s for s in memo if s["sources"]]                    # a sentence that can point at nothing is not shown as analysis
        aid = uuid.uuid7()
        result = {"analysis_id": aid, "motion": world.MOTION_NAME[body.motion_type], "judge": {"id": j["id"], "name": j["name"], "court": j["court"], "record": {k: v for k, v in rec.items() if k != "sd"}},
                  "similar_cases": sims, "similar_rulings": {"considered": len(near), "with_this_motion": len(ruled), "granted": sum(r["this_motion"] == "granted" for r in ruled)},
                  "cited_cases": checks, "estimate": est, "memo": kept, "unsourced_statements": [s["text"] for s in memo if not s["sources"]]}
        result = jsonable_encoder(result)
        c.execute("INSERT INTO analyses (id, tenant_id, created_by, request, result) VALUES (%s,%s,%s,%s,%s)", [aid, ctx.tenant_id, ctx.actor_id, Jsonb(jsonable_encoder(body)), Jsonb(result)])
        audit.record(c, ctx, "brief.analyzed", "analysis", aid, {"motion_type": body.motion_type, "judge_id": str(j["id"]), "cited": len(body.cited_dockets), "model": engine.VERSION})      # no facts in the audit trail
        return 201, result
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- precedent-graph

@app.get("/v1/precedent/graph", tags=["precedent"], summary="Who a case cites and who has cited it since, with how each citation treated it")
def precedent(case_id: uuid.UUID, limit: int = 40, ctx: Ctx = Depends(auth("records:read"))):
    limit = max(1, min(limit, 200))
    with db.tx(ctx.tenant_id) as c:
        k = c.execute("SELECT id, docket_number, court, filed_at, nature_of_suit FROM cases WHERE id = %s", [case_id]).fetchone()
        if not k:
            raise Problem(404, "case_not_found")
        cited_by = c.execute("""SELECT s.id, s.docket_number, s.court, s.filed_at, x.treatment, x.context, f.filed_at AS at FROM citations x JOIN filings f ON f.id = x.source_filing JOIN cases s ON s.id = f.case_id
                                 WHERE x.cited_case = %s ORDER BY f.filed_at DESC LIMIT %s""", [case_id, limit]).fetchall()
        cites = c.execute("""SELECT d.id, d.docket_number, d.court, d.filed_at, x.treatment, x.context, f.filed_at AS at FROM citations x JOIN filings f ON f.id = x.source_filing JOIN cases d ON d.id = x.cited_case
                              WHERE f.case_id = %s ORDER BY f.filed_at DESC LIMIT %s""", [case_id, limit]).fetchall()
        totals = c.execute("SELECT treatment, count(*) AS n FROM citations WHERE cited_case = %s GROUP BY 1", [case_id]).fetchall()
    n = {r["treatment"]: r["n"] for r in totals}
    return jsonable_encoder({"case": k, "treated_since": {t: n.get(t, 0) for t in world.TREATMENT}, "status": "overruled" if n.get("overruled") else "good law on this record",
                             "cited_by": cited_by, "cites": cites})
