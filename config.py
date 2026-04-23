"""Configuración central de MediSource AI.

Carga variables de entorno (también desde `st.secrets` si estamos en Streamlit)
y expone constantes de modelos, rutas y umbrales usados por el resto del sistema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_COLLECTION = "medisource_devices"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_CHAT_MODEL_PREMIUM = "gpt-4o"

DEFAULT_EMBED_BATCH = 96
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_FLOOR = 0.55  # Por debajo, alternativa "dudosa".
DEFAULT_PRICE_MIN = 40.0
DEFAULT_PRICE_MAX = 1800.0


def _get_secret(key: str) -> Optional[str]:
    """Lee variable de entorno y, si existe, también `st.secrets`."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st  # type: ignore

        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return None


@dataclass
class Settings:
    """Ajustes de ejecución de MediSource AI."""

    openai_api_key: Optional[str] = field(default_factory=lambda: _get_secret("OPENAI_API_KEY"))
    db_path: str = field(default_factory=lambda: _get_secret("MEDISOURCE_DB_PATH") or DEFAULT_DB_PATH)
    collection: str = field(
        default_factory=lambda: _get_secret("MEDISOURCE_COLLECTION") or DEFAULT_COLLECTION
    )
    embed_model: str = field(
        default_factory=lambda: _get_secret("MEDISOURCE_EMBED_MODEL") or DEFAULT_EMBED_MODEL
    )
    chat_model: str = field(
        default_factory=lambda: _get_secret("MEDISOURCE_CHAT_MODEL") or DEFAULT_CHAT_MODEL_PREMIUM
    )
    embed_batch: int = DEFAULT_EMBED_BATCH
    top_k: int = DEFAULT_TOP_K
    similarity_floor: float = DEFAULT_SIMILARITY_FLOOR
    price_min: float = DEFAULT_PRICE_MIN
    price_max: float = DEFAULT_PRICE_MAX

    def has_api_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())


_settings: Optional[Settings] = None


def get_settings(refresh: bool = False) -> Settings:
    """Devuelve la configuración cargada (singleton liviano)."""
    global _settings
    if refresh or _settings is None:
        _settings = Settings()
    return _settings
