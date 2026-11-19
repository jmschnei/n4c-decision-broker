# n4c-decision-broker

**Confidence-governed model broker** for *System One* decision models served **locally by
[Ollama](https://docs.ollama.com/capabilities/decision)** (default: `nimble`, 9B). The model
evaluates typed questions (Choice / Score / Noul) against a *state* and returns **typed
answers with probability distributions and a confidence value**. The API
(`POST /v1/systemone`) is Jev-compatible, so hosted Jev or Ollama cloud models also work by
changing the base URL.

`n4c-decision-broker` wraps the model in a governance layer for production use inside NFDI4Culture
pipelines:

```
        ┌────────────┐  state + questions  ┌────────────┐
 app ──▶│  broker    │ ───────────────────▶ │   Ollama  │
        │ (this pkg) │ ◀─────────────────── │ ollama local│
        └─────┬──────┘   answers+confidence └────────────┘
              │ confidence routing per question
   ┌──────────┼──────────────────────────────┐
   ▼          ▼                              ▼
 decided    review (flagged)            escalated
  auto       human-confirm queue      fallback LLM or human hold
   │          │                              │
   └──────────┴──────────────┬───────────────┘
                             ▼
                  cache · audit log · metrics
```

## Why a broker, not a direct API call?

| Concern | How the broker handles it |
|---|---|
| **Calibrated uncertainty** | Every answer is routed by confidence: `decided` (act), `review` (flag for confirmation), `escalated` (model is unsure) |
| **Fallback** | Escalated questions can be re-asked to an OpenAI-compatible LLM (non-calibrated, clearly marked) or held for a human |
| **Noul semantics** | Noul values are probabilities: ≥ 0.7 → yes, ≤ 0.3 → no, in between → review band (per TypeSafe docs) |
| **Cost** | One Ollama call evaluates *all* questions for a state in parallel; identical states are served from cache |
| **Traceability (NFDI)** | JSONL audit log of every decision: status, probabilities, confidence, thresholds, route, latency |
| **Batch enrichment** | `broker.batch()` for enriching whole collections (e.g. museum catalog records) |

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[server,dev]"
ollama pull nimble            # needs Ollama >= 0.35; no API key required
# optional: OLLAMA_HOST / N4C_BROKER_BASE_URL, N4C_BROKER_MODEL (see .env.example)
```

## Python API

```python
from n4c_decision_broker import ModelBroker, QuestionSpec, QuestionPolicy, JevClient

broker = ModelBroker(
    client=JevClient(model="nimble"),   # local Ollama (alias: OllamaClient)
    policy=QuestionPolicy(auto_threshold=0.85, review_threshold=0.5),
)

resp = broker.decide(
    state="Record #42: 'Kupferstich, Ansicht von Rom, 17. Jh.'",
    questions=[
        QuestionSpec(id="period", type="choice",
                     instructions="Historical period of the object",
                     criteria={"early_modern": "1500-1800",
                               "modern": "1800-1945",
                               "contemporary": "1945-present"}),
        QuestionSpec(id="open_access", type="noul",
                     instructions="The record contains no personal data and no rights issues"),
    ],
)

for a in resp.answers:
    print(a.question_id, a.status, a.choice or a.noul_label, a.confidence, a.source)
```

Batch over a collection:

```python
records = ["Kupferstich, Ansicht von Rom, 17. Jh.", "Photographie, Bauhaus, 1927"]
results = broker.batch(records, questions=questions)
```

## CLI

```bash
# Single decision (local Nimble call)
n4c-decision-broker decide \
  --state "Kupferstich, Ansicht von Rom, 17. Jh." \
  --choice period=Historical period of the object|early_modern:1400-1800|modern:1800-1945|contemporary:1945-present \
  --noul open_access=The record contains no personal data and no rights issues

# Questions from JSON instead of flags
n4c-decision-broker decide --state "..." --questions-file examples/questions.example.json

# Offline demo (no API key needed) — shows decided / review / escalated paths
n4c-decision-broker demo

# Batch enrich records from JSON, write results to JSONL
n4c-decision-broker batch --input examples/records.example.json \
                 --questions-file examples/questions.example.json --out audit/batch.jsonl

# Introspection
n4c-decision-broker metrics
n4c-decision-broker audit --limit 10
```

## REST server

```bash
n4c-decision-broker serve --host 127.0.0.1 --port 8787
```

**Authentication.** Set `N4C_BROKER_API_KEYS` (comma-separated, several keys allow rotation)
and send `Authorization: Bearer <key>` or `X-API-Key: <key>`. Everything except
`/v1/broker/health` requires a key. `serve` refuses to bind to a non-loopback host
without keys; on `127.0.0.1` it runs open with a warning.

```bash
export N4C_BROKER_API_KEYS=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
n4c-decision-broker serve --host 0.0.0.0
```

| Endpoint | Description |
|---|---|
| `POST /v1/broker/decide` | Single decision (body: `state`, `questions`, optional `policy`) |
| `POST /v1/broker/batch` | Batch decisions (body: `records`, `questions`) |
| `GET  /v1/broker/health` | Liveness |
| `GET  /v1/broker/metrics` | Calls, errors, cache stats, policy |
| `GET  /v1/broker/audit?limit=20` | Recent audit entries |

## Docker

```bash
export N4C_BROKER_API_KEYS=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")
docker compose up --build        # broker on :8787, Ollama expected on the host
```

The image runs as a non-root user, persists the audit log/cache in the `broker-audit`
volume and reaches Ollama through `OLLAMA_HOST` (default `http://host.docker.internal:11434`).
Without `N4C_BROKER_API_KEYS` the container refuses to start.
Plain Docker: `docker run -p 8787:8787 -e N4C_BROKER_API_KEYS=... n4c-decision-broker`.

## License

MIT — see [LICENSE](LICENSE).

## Model evaluation on a Slurm cluster

Compares candidate Ollama models on text-to-SPARQL (Spider4SPARQL dev, GOLEM, QALD-10/9-plus;
700 sampled questions) to build the per-query model-quality table the broker's recommendations
are evaluated against. Everything runs in user space; no root needed.

```bash
git clone https://github.com/jmschnei/n4c-decision-broker && cd n4c-decision-broker
export N4C_WORK=/path/to/shared/scratch/n4c      # ~100 GB free, visible from all nodes

bash slurm/setup.sh                              # once, WITH internet (login node):
                                                 # venv, Ollama, 7 models (~80 GB), datasets
bash slurm/submit.sh --partition gpu --time 03:00:00 --max-parallel 8
```

`submit.sh` launches an array job (models x shards, one GPU each) and a scoring job that
starts when the array ends. Useful options: `--shards N`, `--models "m1 m2"`, `--run-name`,
`--account`, `--gres gpu:a100:1`, `--limit 20` (smoke test), `--dry-run`, `--execute`
(also run queries on public endpoints; needs internet, see `tools/eval/score.py`).
Re-running the same command resumes unfinished work. Results: `$N4C_WORK/results/<run>/scores/`
(`summary.md`, `summary.json`, `item_scores.jsonl`).

### Pyxis/Enroot clusters: build the image once

Ollama is installed **into a container image** instead of the setup step:

```bash
BASE=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh
bash slurm/build_image.sh --base $BASE --out /netscratch/$USER/n4c-eval.sqsh --partition <partition>
```

This starts the base image with Pyxis (`--container-save`, `--container-remap-root`), runs
`docker/install_in_image.sh` (Ollama into `/usr/local` with checksum check + `httpx`, `rdflib`)
and saves the `.sqsh`. Without Pyxis' save option, build it with Docker elsewhere
(`docker build -f docker/Dockerfile.eval -t n4c-eval .`, then `enroot import -o ... dockerd://n4c-eval`).
Models (~80 GB) are not baked in; they go to `$N4C_WORK/ollama_models`:

```bash
IMG=/netscratch/$USER/n4c-eval.sqsh
C="--container-image=$IMG --container-workdir=`pwd` --container-mounts=/netscratch/$USER:/netscratch/$USER,`pwd`:`pwd`"
srun -p <partition> --cpus-per-task=4 --mem=16G --time=06:00:00 $C --export=ALL bash slurm/setup.sh  # models + data
bash slurm/submit.sh --partition <partition> --container-image $IMG [--container-mounts "/ds:/ds:ro"]
```

`setup.sh` detects the Ollama and Python libraries of the image and skips installing them.

If the login node forbids heavy work: `sbatch --partition=<internet-partition> slurm/setup.sbatch`.
Needs NVIDIA GPUs with >= 24 GB for the 30B models (`OLLAMA_FLAVOR=rocm` for AMD).

## Robustness

- **Retries:** the client retries connection errors and HTTP 408/425/429/5xx with exponential
  backoff + jitter (`max_retries=3`, honours `Retry-After`); 4xx errors are not retried.
- **Parallel batch:** `broker.batch(records, questions, max_workers=4)` /
  `batch --workers N`; order is preserved and one failing record becomes an `error`
  response instead of aborting the batch.
- **Fallback:** errors inside an escalation backend never break the flow
  (counted in `metrics()["fallback_errors"]`).

## Threshold calibration

```bash
n4c-decision-broker calibrate --labeled examples/labeled.example.json \
    --questions-file examples/questions.example.json     # live Nimble; saves raw observations
n4c-decision-broker calibrate --from-observations audit/calibration_observations.json \
    --target-auto 0.9 --target-review 0.75               # recompute offline, no model calls
```

Reports per-tier accuracy, ECE and a recommended `QuestionPolicy`. Thresholds are the
lowest confidence whose answers meet the target accuracy (`--target-auto`,
`--target-review`, backed by `--min-support` answers). `examples/labeled.example.json`
is a small illustrative set (38 records) — **use your own labeled NFDI4Culture records**
before adopting the thresholds.

## Differences vs. hosted Jev

- **No `box` questions** (rejected with a clear error); only `choice`, `score`, `noul`.
- **`confidence` is entropy-based** (`1 - H(p)/ln N`) and measures how concentrated the
  distribution is, **not calibrated correctness**. Re-tune `auto_threshold` /
  `review_threshold` on labeled NFDI4Culture records before trusting the defaults.
- Local inference: no per-token cost (`cost_estimate_usd` is 0 unless you set `cost_usd_per_btok`).

## Decision policy

`QuestionPolicy` (defaults are conservative and configurable per request):

- **Choice / Score**: `confidence >= auto_threshold` (0.85) → `decided`;
  `>= review_threshold` (0.5) → `review`; below → `escalated`.
- **Noul**: `p >= noul_yes` (0.7) → `yes`; `p <= noul_no` (0.3) → `no`;
  in between → `review` (or `escalated` if closer to 0.5).

`escalated` questions go to the configured `EscalationBackend`:

- `HoldBackend` (default): no automated action — answer is `None`, status `escalated`,
  your app routes it to a human.
- `LLMFallbackBackend`: asks an OpenAI-compatible chat endpoint for a second opinion;
  the result is marked `calibrated: false` and `source: "fallback_llm"`, and the
  original original answer is preserved under `primary`.

## Tests

```bash
.venv/bin/pytest -q
```

Tests run fully offline (`FakeJev` deterministic stand-in + `httpx.MockTransport`).
