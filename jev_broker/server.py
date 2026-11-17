"""FastAPI app for the Jev model broker (optional dependency)."""
from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel, Field

from jev_broker import __version__
from jev_broker.broker import ModelBroker
from jev_broker.models import BrokerResponse, QuestionPolicy, QuestionSpec

__all__ = ["BrokerRequestBody", "BatchRequestBody", "create_app"]


class BrokerRequestBody(BaseModel):
    state: Union[str, dict, list]
    questions: Union[dict[str, QuestionSpec], list[QuestionSpec]] = Field(..., min_length=1)
    policy: Union[QuestionPolicy, None] = None
    request_id: Union[str, None] = None


class BatchRequestBody(BaseModel):
    records: list[Union[str, dict, list]] = Field(..., min_length=1)
    questions: Union[dict[str, QuestionSpec], list[QuestionSpec]] = Field(..., min_length=1)
    policy: Union[QuestionPolicy, None] = None


def create_app(broker: ModelBroker):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FastAPI is not installed. Install with: pip install 'jev-broker[server]'"
        ) from exc

    app = FastAPI(title="Jev Model Broker", version=__version__)

    @app.get("/v1/broker/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "jev-broker", "version": __version__}

    @app.post("/v1/broker/decide", response_model=BrokerResponse)
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

    @app.post("/v1/broker/batch")
    def batch(body: BatchRequestBody) -> dict[str, Any]:
        try:
            responses = broker.batch(records=body.records, questions=body.questions, policy=body.policy)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "count": len(responses),
            "results": [r.model_dump(mode="json") for r in responses],
        }

    @app.get("/v1/broker/metrics")
    def metrics() -> dict[str, Any]:
        return broker.metrics()

    @app.get("/v1/broker/audit")
    def audit(limit: int = 20) -> list[dict[str, Any]]:
        if broker.audit is None:
            raise HTTPException(status_code=404, detail="no audit log configured")
        return broker.audit.recent(limit=limit)

    return app
