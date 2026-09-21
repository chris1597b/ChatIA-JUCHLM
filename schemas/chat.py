"""Schemas API (§24-25). Re-exportan dominio para FastAPI/OpenAPI."""
from core.models import ActionButton, AssistantResponse, ConversationRequest, ConversationState, PredioData

__all__ = ["ActionButton", "AssistantResponse", "ConversationRequest", "ConversationState", "PredioData"]
