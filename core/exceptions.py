"""Excepciones de dominio. Nunca exponer tracebacks al usuario (§27)."""


class ChatJuchlmError(Exception):
    user_message = "No fue posible completar la consulta en este momento."


class InvalidStateError(ChatJuchlmError):
    pass


class UnknownCapabilityError(ChatJuchlmError):
    user_message = "Esa función aún no está disponible."


class DomainValidationError(ChatJuchlmError):
    pass


class DatabaseUnavailableError(ChatJuchlmError):
    user_message = "No fue posible completar la consulta en este momento."


class TramiteNotFoundError(ChatJuchlmError):
    user_message = "El documento solicitado no se encuentra disponible."
