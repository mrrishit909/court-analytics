"""Tenants, demo tokens and the citation-treatment classifier (fitted on one set of synthetic sentences, scored on another,
then registered). Dockets arrive through POST /v1/dockets/ingest, which also fits the outcome model on that tenant's data.

    python -m ca.seed [--if-empty]
"""
import hashlib
import pickle
import sys
import uuid

from psycopg.types.json import Jsonb

from core import db

from . import engine, world

TENANTS = {"firm": ("Harlow & Pike LLP", 9, 9000), "other": ("Second Firm LLP", 21, 1500)}       # name, world seed, cases
ROLES = ["associate", "partner", "admin"]


def token(tenant, role):
    """Demo credentials for local use only."""
    return f"{tenant}-{role}-demo"


def tenant_id(key):
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{key}.court-analytics.example")


def train():
    clf = engine.train_treatment(world.contexts(101))
    test = world.contexts(202)
    acc = sum(p == k for p, (_, k) in zip(clf.predict([t for t, _ in test]), test)) / len(test)
    base = sum(engine.keyword_treatment(t) == k for t, k in test) / len(test)
    return clf, {"model": {"accuracy": round(acc, 4), "n_test": len(test)}, "baseline": {"name": "keyword rules", "accuracy": round(base, 4)}}


def main(if_empty=False):
    db.migrate()
    with db.tx() as c:
        if if_empty and c.execute("SELECT 1 FROM tenants LIMIT 1").fetchone():
            return print("already seeded")
    clf, metrics = train()
    out = {"metrics": metrics}
    with db.tx() as c:
        c.execute("INSERT INTO model_registry (name, version, data_snapshot, metrics, baseline, artifact) VALUES ('citation-treatment',%s,%s,%s,%s,%s)",
                  [engine.VERSION, "synthetic citation sentences, seeds 101 (fit) and 202 (score)", Jsonb(metrics["model"]), Jsonb(metrics["baseline"]), pickle.dumps(clf)])
        for key, (name, _, _) in TENANTS.items():
            t = tenant_id(key)
            c.execute("INSERT INTO tenants (id, name) VALUES (%s,%s)", [t, name])
            for role in ROLES:
                c.execute("INSERT INTO api_tokens VALUES (%s,%s,%s,%s,%s)", [hashlib.sha256(token(key, role).encode()).hexdigest(), t, uuid.uuid5(t, role), f"{role} ({name})", role])
            out[key] = t
    print("classifier registered:", metrics)
    print("demo tokens:", ", ".join(token(t, r) for t in TENANTS for r in ROLES))
    return out


if __name__ == "__main__":
    main("--if-empty" in sys.argv)
