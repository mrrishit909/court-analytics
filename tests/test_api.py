"""Integration, end-to-end and security tests against a real Postgres."""
import pytest

from core import db, jobs, scenario
from ca import world
from ca.seed import TENANTS

from .conftest import bearer

ADMIN, PARTNER, ASSOC, OTHER = bearer("firm", "admin"), bearer("firm", "partner"), bearer("firm", "associate"), bearer("other", "admin")
ZERO = "00000000-0000-0000-0000-000000000000"
FACTS = "Plaintiff alleges the delivered machinery failed to meet the agreed specification. The complaint describes repeated breakdowns within the warranty period."


def test_citation_classifier_beats_keyword_rules(seeded):
    m = seeded["metrics"]
    assert m["model"]["accuracy"] > m["baseline"]["accuracy"] + 0.1


def test_nothing_works_before_ingest(client):
    assert client.get("/v1/models/motion-outcome", headers=ASSOC).json()["code"] == "nothing_ingested"


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


@pytest.fixture(scope="module")
def truth():
    return world.build(*TENANTS["firm"][1:])


def test_demo_journey_matches_the_generated_courts(demo, truth):
    ing = demo["ing"]["result"]
    assert ing["cases"] == len(truth["cases"]) and ing["motions"] == len(truth["motions"]) and ing["open_cases"] == sum(c["closed"] is None for c in truth["cases"])
    om = ing["outcome_model"]
    assert om["trained_on"] == sum(m["decided"] < world.SPLIT for m in truth["motions"]) and om["held_out"] == len(truth["motions"]) - om["trained_on"]
    assert om["model"]["auc"] > om["judge_rate_only"]["auc"] > om["motion_base_rate"]["auc"] and om["model"]["brier"] < om["motion_base_rate"]["brier"] and om["model"]["calibration_error"] < 0.04
    assert demo["search"]["matches"] > 20 and all(r["nature_of_suit"] == "employment" for r in demo["search"]["items"])
    b = demo["brief"]
    by_docket = {c["docket"]: i for i, c in enumerate(truth["cases"])}
    assert all(truth["cases"][by_docket[s["docket_number"]]]["pattern"] == "retaliation" for s in b["similar_cases"]) and len(b["similar_cases"]) == 8
    assert [k["status"] for k in b["cited_cases"]] == ["good", "caution", "overruled", "not found"]
    overruled = sum(x["treatment"] == "overruled" and x["cited"] == by_docket[b["cited_cases"][2]["docket"]] for x in truth["cites"])
    assert 1 <= b["cited_cases"][2]["overruled"] <= overruled                    # repeated citations by one order are stored once
    est = b["estimate"]
    assert est["low"] <= est["probability"] <= est["high"] and est["drivers"] and not est["flags"] and "not legal advice" in est["caveat"]
    assert all(s["sources"] for s in b["memo"]) and len(b["unsourced_statements"]) == 1 and "99-cv-00000" in b["unsourced_statements"][0]
    assert any("overruled" in s["text"] for s in b["memo"]) and b["memo"][-1]["text"].startswith("This is a statistical summary")
    rec = next(m for m in demo["judge"]["motions"] if m["motion_type"] == "dismiss")
    assert rec == {**b["judge"]["record"], "motion_type": "dismiss", "name": "Motion to dismiss", "median_days_to_ruling": rec["median_days_to_ruling"]}
    assert rec["low"] < rec["rate"] < rec["high"] and abs(rec["rate"] - rec["typical_judge"]) < abs(rec["raw_rate"] - rec["typical_judge"])
    tr = demo["judge"]["time_to_resolution"]
    assert tr["this_judge"]["still_open"] > 0 and tr["this_judge"]["quarter_closed_by"] < tr["this_judge"]["median_days"] < tr["this_judge"]["three_quarters_closed_by"]
    assert demo["graph"]["status"] == "overruled" and any(x["treatment"] == "overruled" for x in demo["graph"]["cited_by"])
    thin = demo["thin"]
    assert thin["flags"] and (thin["high"] - thin["low"]) > 2 * (est["high"] - est["low"]) and len(demo["newjudge"]["warnings"]) == 4
    assert demo["card"]["calibration"] and demo["card"]["model"] == om["model"]
    event = next(e for e in demo["audit"]["events"] if e["action"] == "brief.analyzed")
    assert demo["audit"]["chain_valid"] and "facts" not in event["detail"] and "regulator" not in str(event["detail"])


def test_the_brief_is_stored_only_in_the_firms_workspace_and_a_retry_stores_it_once(client, demo):
    body = {"motion_type": "summary_judgment", "judge_id": demo["brief"]["judge"]["id"], "nature_of_suit": "contract", "facts": FACTS, "cited_dockets": []}
    key = {**ASSOC, "Idempotency-Key": "brief-retry"}
    first, again = client.post("/v1/briefs/analyze", headers=key, json=body), client.post("/v1/briefs/analyze", headers=key, json=body)
    assert first.status_code == 201 and again.json() == first.json() and again.headers["Idempotent-Replay"] == "true"
    assert "the facts do not say whether the key events are documented" in " ".join(first.json()["estimate"]["flags"])
    with db.tx() as c:
        assert c.execute("SELECT count(*) AS n FROM analyses WHERE request->>'facts' = %s", [FACTS]).fetchone()["n"] == 1
        assert c.execute("SELECT count(*) AS n FROM audit_events WHERE detail::text LIKE '%%machinery%%'").fetchone()["n"] == 0
    estimate = client.post("/v1/motions/estimate", headers=ASSOC, json={k: v for k, v in body.items() if k != "cited_dockets"})
    assert estimate.status_code == 200 and estimate.json()["probability"] == first.json()["estimate"]["probability"]


def test_search_pages_without_repeats_and_similar_excludes_the_case(client, demo):
    seen, before = [], None
    while True:
        page = client.get("/v1/cases/search", headers=ASSOC, params={"q": "overtime", "limit": 100, **({"before": before} if before else {})}).json()
        seen += [r["id"] for r in page["items"]]
        before = page["next_before"]
        if not before:
            break
    assert len(seen) == len(set(seen)) == page["matches"] > 100
    sim = client.get(f"/v1/cases/{seen[0]}/similar", headers=ASSOC).json()["items"]
    assert len(sim) == 8 and seen[0] not in [s["id"] for s in sim] and all("overtime" in s["facts"] or "hours" in s["facts"] or "meal" in s["facts"] or "hourly" in s["facts"] for s in sim)


def test_ingesting_again_adds_nothing(client, demo):
    job = client.post("/v1/dockets/ingest", headers=ADMIN, json={}).json()
    jobs.drain()
    assert client.get(job["status_url"], headers=ADMIN).json()["result"]["cases"] == 0


@pytest.mark.parametrize("headers,method,path,body,expected", [
    ({}, "GET", "/v1/judges", None, 401),
    (ASSOC, "GET", "/v1/judges", None, 200),
    (ASSOC, "POST", "/v1/dockets/ingest", {}, 403),
    (ASSOC, "GET", "/v1/audit", None, 403),
    (PARTNER, "GET", "/v1/audit", None, 200),
    (PARTNER, "POST", "/v1/dockets/ingest", {}, 403),
    (ASSOC, "GET", f"/v1/judges/{ZERO}/analytics", None, 404),
    (ASSOC, "GET", f"/v1/cases/{ZERO}/similar", None, 404),
    (ASSOC, "GET", f"/v1/precedent/graph?case_id={ZERO}", None, 404),
    (ASSOC, "GET", "/v1/cases/search?q=%27%29%3B+DROP+TABLE+cases%3B--", None, 200),
    (ASSOC, "POST", "/v1/motions/estimate", {"motion_type": "dismiss", "judge_id": ZERO, "nature_of_suit": "contract", "facts": FACTS}, 404),
    (ASSOC, "POST", "/v1/motions/estimate", {"motion_type": "dismiss", "judge_id": ZERO, "nature_of_suit": "contract", "facts": "too short"}, 422),
    (ASSOC, "POST", "/v1/motions/estimate", {"motion_type": "strike", "judge_id": ZERO, "nature_of_suit": "contract", "facts": FACTS}, 422),
    (ASSOC, "POST", "/v1/briefs/analyze", {"motion_type": "dismiss", "judge_id": ZERO, "nature_of_suit": "contract", "facts": FACTS, "cited_dockets": ["x"] * 21}, 422),
])
def test_authorization_and_validation(client, demo, headers, method, path, body, expected):
    assert client.request(method, path, headers=headers, json=body).status_code == expected


def test_one_firm_cannot_see_anothers_records_or_briefs(client, demo):
    judge, case = demo["brief"]["judge"]["id"], demo["brief"]["cited_cases"][2]["case_id"]
    assert client.get("/v1/judges", headers=OTHER).json()["items"] == [] and client.get("/v1/cases/search", headers=OTHER).json()["matches"] == 0
    assert client.get(f"/v1/judges/{judge}/analytics", headers=OTHER).status_code == 404 and client.get(f"/v1/precedent/graph?case_id={case}", headers=OTHER).status_code == 404
    assert client.post("/v1/motions/estimate", headers=OTHER, json={"motion_type": "dismiss", "judge_id": judge, "nature_of_suit": "contract", "facts": FACTS}).status_code == 409
    job = client.post("/v1/dockets/ingest", headers=OTHER, json={}).json()
    jobs.drain()
    assert client.get(job["status_url"], headers=OTHER).json()["result"]["cases"] == TENANTS["other"][2]
    assert client.post("/v1/motions/estimate", headers=OTHER, json={"motion_type": "dismiss", "judge_id": judge, "nature_of_suit": "contract", "facts": FACTS}).status_code == 404      # not their judge
    theirs = client.get("/v1/judges", headers=OTHER).json()["items"]
    assert theirs and not {j["id"] for j in theirs} & {j["id"] for j in client.get("/v1/judges", headers=ASSOC).json()["items"]}
    with db.tx() as c:
        firm = c.execute("SELECT id FROM tenants WHERE name LIKE 'Second%'").fetchone()["id"]
    with db.tx(firm) as c:
        assert c.execute("SELECT count(*) AS n FROM analyses").fetchone()["n"] == 0                # the other firm's workspace holds none of this firm's briefs
    assert client.get("/v1/models/motion-outcome", headers=ASSOC).json()["model"] == demo["card"]["model"]
