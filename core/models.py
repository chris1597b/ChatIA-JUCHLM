"""Modelos de dominio. Contrato único backend <-> frontend (§25)."""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ConversationState(str, Enum):
    START = "START"
    MAIN_MENU = "MAIN_MENU"
    JUNTA_MENU = "JUNTA_MENU"
    RAG_QUERY = "RAG_QUERY"
    RAG_RESPONSE = "RAG_RESPONSE"
    PREDIO_ELEGIR_METODO = "PREDIO_ELEGIR_METODO"
    PREDIO_REQUEST_NAME = "PREDIO_REQUEST_NAME"
    PREDIO_SQL_QUERY = "PREDIO_SQL_QUERY"
    PREDIO_RESULT = "PREDIO_RESULT"
    PREDIO_DISAMBIGUATE = "PREDIO_DISAMBIGUATE"
    ASK_TRAMITE = "ASK_TRAMITE"
    TRAMITE_MENU = "TRAMITE_MENU"
    CONSTANCIA_USUARIO = "CONSTANCIA_USUARIO"
    END = "END"


class ActionButton(BaseModel):
    id: str = Field(..., max_length=64)
    label: str = Field(..., max_length=120)
    type: str = Field(default="action", max_length=32)  # action | menu | rag | workflow | tramite | download


class ConversationRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=64)
    message: str = Field(default="", max_length=2000)
    action_id: Optional[str] = Field(default=None, max_length=64)


class AssistantResponse(BaseModel):
    message: str
    message_type: str = "assistant"
    state: ConversationState = ConversationState.MAIN_MENU
    actions: list[ActionButton] = Field(default_factory=list)
    data: Optional[Any] = None
    source: str = "orchestrator"  # orchestrator | rag | sql | tramite | menu | security


class PredioData(BaseModel):
    nombre: str = ""
    codigo_predio: str = ""
    sector: str = ""
    area: Optional[float] = None
    estado: str = ""
    raw: Optional[dict] = None
