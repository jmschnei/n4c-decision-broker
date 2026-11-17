"""n4c-decision-broker: confidence-governed decision broker over TypeSafe's Jev model."""

from n4c_decision_broker.models import (
    AnswerResult,
    BrokerRequest,
    BrokerResponse,
    DecisionStatus,
    QuestionPolicy,
    QuestionSpec,
    QuestionType,
    state_sha256,
)
from n4c_decision_broker.decision import (
    Routing,
    choice_confidence,
    route_answer,
    score_confidence,
)
from n4c_decision_broker.client import JevAPIError, JevClient, OllamaAPIError, OllamaClient
from n4c_decision_broker.fallback import EscalationBackend, HoldBackend, LLMFallbackBackend
from n4c_decision_broker.cache import DecisionCache
from n4c_decision_broker.audit import AuditLog
from n4c_decision_broker.broker import ModelBroker

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
