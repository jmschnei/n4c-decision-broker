"""jev-broker: confidence-governed decision broker over TypeSafe's Jev model."""

from jev_broker.models import (
    AnswerResult,
    BrokerRequest,
    BrokerResponse,
    DecisionStatus,
    QuestionPolicy,
    QuestionSpec,
    QuestionType,
    state_sha256,
)
from jev_broker.decision import (
    Routing,
    choice_confidence,
    route_answer,
    score_confidence,
)
from jev_broker.client import JevAPIError, JevClient, OllamaAPIError, OllamaClient
from jev_broker.fallback import EscalationBackend, HoldBackend, LLMFallbackBackend
from jev_broker.cache import DecisionCache
from jev_broker.audit import AuditLog
from jev_broker.broker import ModelBroker

__version__ = "0.1.0"

__all__ = [
    "AnswerResult",
    "AuditLog",
    "BrokerRequest",
    "BrokerResponse",
    "DecisionCache",
    "DecisionStatus",
    "EscalationBackend",
    "HoldBackend",
    "JevAPIError",
    "JevClient",
    "LLMFallbackBackend",
    "ModelBroker",
    "OllamaAPIError",
    "OllamaClient",
    "QuestionPolicy",
    "QuestionSpec",
    "QuestionType",
    "Routing",
    "choice_confidence",
    "route_answer",
    "score_confidence",
    "state_sha256",
    "__version__",
]
