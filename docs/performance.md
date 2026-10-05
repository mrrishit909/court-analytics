# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 9,000 cases, 12,555 motions, 25,110 filings,
42,921 citations. Reproduce with `make demo && make load-test`.

## Reads: full-text case search (20 results), the judge list, the model card (in rotation), closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 599 | 59.2 req/s | 18.9 ms | 22.9 ms | 119.2 ms | 0.00% |
| 8 | 1408 | 139.2 req/s | 19.3 ms | 143.6 ms | 160.5 ms | 0.00% |
| 32 | 1283 | 126.0 req/s | 168.1 ms | 309.7 ms | 345.6 ms | 0.00% |

Target from the blueprint: p95 under 400 ms for reads: met, with less room than the other endpoints suggest. The slow one is the
judge list (about 120 ms): it counts each judge's rulings with a correlated subquery, 36 scans of the motions-to-cases join. A single
grouped join would remove that.

## Compute

| Work | Time |
|---|---|
| Seed: fit and score the citation classifier | under 1 s |
| Ingest job: 9,000 cases loaded, 42,921 citation sentences classified, TF-IDF index and 40 nearest neighbours for every case, model + 40 bootstrap refits | about 3.5 s |
| Brief analysis: similarity query, cited-case treatment, judge record, estimate, memo, stored | about 30 ms |

Each API process unpickles its tenant's fitted artefacts (model, TF-IDF matrix, neighbour table: a few megabytes) once and keeps
them in memory until the tenant is refitted.

Not measured: real dockets. Millions of cases would need an approximate-nearest-neighbour index instead of a dense similarity pass,
and incremental refits instead of one fit at ingest.
