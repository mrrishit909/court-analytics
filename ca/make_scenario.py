"""Write web/scenario.json. Judges and docket numbers in the demo come from the deterministic world, so they are looked up here.

    python -m ca.make_scenario
"""
import collections
import json
import uuid

from core import db

from . import world
from .seed import TENANTS, tenant_id


def main():
    w = world.build(*TENANTS["firm"][1:])
    t = tenant_id("firm")
    by = collections.Counter((w["cases"][m["case"]]["judge"], m["type"]) for m in w["motions"])
    judge = max(range(len(w["judges"]) - 4), key=lambda j: by[j, "dismiss"])               # the judge with the most motion-to-dismiss rulings
    new = len(w["judges"]) - 1                                                             # a 2024 appointee with almost no record
    tr = collections.defaultdict(collections.Counter)
    for x in w["cites"]:
        tr[x["cited"]][x["treatment"]] += 1
    retaliation = [k for k, c in enumerate(w["cases"]) if c["pattern"] == "retaliation"]
    good = max(retaliation, key=lambda k: tr[k]["followed"] - 50 * tr[k]["overruled"] - 5 * tr[k]["criticized"])
    bad = max(retaliation, key=lambda k: (tr[k]["overruled"], tr[k]["followed"]))
    mixed = max((k for k in retaliation if not tr[k]["overruled"] and k != good), key=lambda k: tr[k]["distinguished"] - tr[k]["followed"])
    jid = lambda j: str(uuid.uuid5(t, f"judge-{j}"))      # noqa: E731
    brief = {"motion_type": "dismiss", "judge_id": jid(judge), "nature_of_suit": "employment", "pro_se": False, "amended": 1, "claims": 3, "large_firm_defence": True,
             "facts": "Plaintiff alleges she was dismissed three weeks after reporting unsafe equipment to a regulator. She reported the hazard in writing. Supervisors allegedly referred to the safety report when announcing the termination. "
                      "Plaintiff attaches correspondence supporting each allegation. Plaintiff seeks back pay and reinstatement.",
             "cited_dockets": [w["cases"][good]["docket"], w["cases"][mixed]["docket"], w["cases"][bad]["docket"], "99-cv-00000"]}
    C = lambda key, who, method, path, body=None: {"key": key, "as": who, "method": method, "path": path, **({"body": body} if body is not None else {})}      # noqa: E731
    s = {"service": "court-analytics", "run": "demo-1", "tokens": {"admin": "firm-admin-demo", "partner": "firm-partner-demo", "associate": "firm-associate-demo"},
         "steps": [
             {"title": "Ingest ten years of dockets", "say": "An invented court system: four districts, 36 judges, 9,000 civil cases with their motions, rulings and the citations in each order. The outcome model learns from rulings before 2023 and is scored on the two years after.",
              "calls": [C("ing", "admin", "POST", "/v1/dockets/ingest", {"source": "demo-dockets"})]},
             {"title": "Search the record", "say": "Full-text search over the facts of every case, with filters. Here: employment cases about retaliation after a safety report.",
              "calls": [C("search", "associate", "GET", "/v1/cases/search?q=reporting+safety+violations+terminated&nature_of_suit=employment&limit=5")]},
             {"title": "Upload a motion and its facts", "say": "A motion to dismiss in a retaliation case. The system finds the cases with the most similar facts, checks how the four cases the brief cites have been treated since, pulls the judge's record, and writes a memo in which every sentence points at its sources.",
              "calls": [C("brief", "associate", "POST", "/v1/briefs/analyze", brief)]},
             {"title": "The judge's record, with honest error bars", "say": "Grant rates are pulled towards a typical judge in proportion to how few rulings there are, and shown with a range. Time to resolution counts cases still open, which a simple median of closed cases gets wrong.",
              "calls": [C("judge", "associate", "GET", f"/v1/judges/{jid(judge)}/analytics")]},
             {"title": "The cited case that was overruled", "say": "One of the brief's authorities has been overruled. The precedent graph shows which later orders said so, in their words.",
              "calls": [C("graph", "associate", "GET", "/v1/precedent/graph?case_id={{brief.cited_cases.2.case_id}}&limit=12")]},
             {"title": "A judge with almost no record", "say": "The same motion before a judge appointed last year. The system does not pretend to know: the estimate falls back towards the court average, the range widens, and it says so.",
              "calls": [C("thin", "associate", "POST", "/v1/motions/estimate", {**{k: v for k, v in brief.items() if k != "cited_dockets"}, "judge_id": jid(new)}), C("newjudge", "associate", "GET", f"/v1/judges/{jid(new)}/analytics")]},
             {"title": "How good is the model, really?", "say": "Its results on two years of rulings it never saw, beside two simpler rules, and whether a stated 70% actually came true 70% of the time. Then the audit trail, which records that a brief was analysed but never what it said.",
              "calls": [C("card", "partner", "GET", "/v1/models/motion-outcome"), C("audit", "partner", "GET", "/v1/audit?limit=8")]}]}
    (db.ROOT / "web" / "scenario.json").write_text(json.dumps(s, indent=1))
    print("judge:", w["judges"][judge]["name"], by[judge, "dismiss"], "| new:", w["judges"][new]["name"], by[new, "dismiss"], "| cited:", [(w["cases"][k]["docket"], dict(tr[k])) for k in (good, mixed, bad)])


if __name__ == "__main__":
    main()
