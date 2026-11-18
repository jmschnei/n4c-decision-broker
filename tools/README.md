# tools

`normalize_datasets.py` turns public text-to-SPARQL datasets into one JSONL schema for
evaluating the broker's model recommendations.

```bash
# raw data: one directory per dataset under datasets/raw/{spider4sparql,golem,qald10,qald9plus}
python3 -I tools/normalize_datasets.py datasets/raw datasets/normalized
```

Sources and licenses: Spider4SPARQL dev (Apache-2.0), QALD-10 (MIT), QALD-9-plus (CC-BY-4.0),
GOLEM (no license file found — internal use only). `datasets/` is git-ignored on purpose:
do not redistribute third-party data from this repository.

Splits (`calibration` / `test`) are deterministic and made per `group` (KG for Spider4SPARQL,
SPARQL skeleton for GOLEM, question id for QALD) so templates and translations never leak
across splits. `features` and `complexity` are heuristic (regex-based, no SPARQL parser).
