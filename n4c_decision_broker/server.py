"""FastAPI app for the Jev model broker (optional dependency)."""
from __future__ import annotations

import hmac
import os
from typing import Any, Optional, Sequence, Union

from pydantic import BaseModel, Field

from n4c_decision_broker import __version__
from n4c_decision_broker.broker import ModelBroker
from n4c_decision_broker.models import BrokerResponse, QuestionPolicy, QuestionSpec

__all__ = ["BrokerRequestBody", "BatchRequestBody", "create_app", "load_api_keys"]


class BrokerRequestBody(BaseModel):
    state: Union[str, dict, list]
    questions: Union[dict[str, QuestionSpec], list[QuestionSpec]] = Field(..., min_length=1)
    policy: Union[QuestionPolicy, None] = None
    request_id: Union[str, None] = None


class BatchRequestBody(BaseModel):
    records: list[Union[str, dict, list]] = Field(..., min_length=1)
    questions: Union[dict[str, QuestionSpec], list[QuestionSpec]] = Field(..., min_length=1)
    policy: Union[QuestionPolicy, None] = None


def load_api_keys(env: Optional[dict] = None) -> list[str]:
    """API keys from N4C_BROKER_API_KEYS (comma-separated, allows key rotation)."""
    raw = (env if env is not None else os.environ).get("N4C_BROKER_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def create_app(broker: ModelBroker, api_keys: Optional[Sequence[str]] = None):
    """Build the FastAPI app.

    ``api_keys``: accepted keys, sent as ``Authorization: Bearer <key>`` or
    ``X-API-Key: <key>``. ``None``/empty disables authentication (local use only).
    ``/v1/broker/health`` is always open so probes keep working.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI is not installed. Install with: pip install 'n4c-decision-broker[server]'"
        ) from exc

    keys = [k for k in (api_keys or []) if k]

    def require_key(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
    ) -> None:
        if not keys:
            return
        supplied = x_api_key
        if supplied is None and authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        # compare against every key (constant time per comparison, no early exit)
        ok = False
        for k in keys:
            ok |= hmac.compare_digest((supplied or "").encode(), k.encode())
        if not ok:
            raise HTTPException(status_code=401, detail="invalid or missing API key",
                                headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="N4C Decision Broker", version=__version__)
    auth = [Depends(require_key)]

    @app.get("/v1/broker/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "n4c-decision-broker", "version": __version__}

    @app.post("/v1/broker/decide", response_model=BrokerResponse, dependencies=auth)
    def decide(body: BrokerRequestBody) -> BrokerResponse:
        try:
            return broker.decide(
                state=body.state,
                questions=body.questions,
                policy=body.policy,
                request_id=body.request_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/broker/batch", dependencies=auth)
    def batch(body: BatchRequestBody) -> dict[str, Any]:
        try:
            responses = broker.batch(records=body.records, questions=body.questions, policy=body.policy)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "count": len(responses),
            "results": [r.model_dump(mode="json") for r in responses],
        }

    @app.get("/v1/broker/metrics", dependencies=auth)
    def metrics() -> dict[str, Any]:
        return broker.metrics()

    @app.get("/v1/broker/audit", dependencies=auth)
    def audit(limit: int = 20) -> list[dict[str, Any]]:
        if broker.audit is None:
            raise HTTPException(status_code=404, detail="no audit log configured")
        return broker.audit.recent(limit=limit)

    return app
