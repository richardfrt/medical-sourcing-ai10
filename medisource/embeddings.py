"""Servicio de embeddings sobre OpenAI.

Responsabilidades:
- Instanciar el cliente con la API key activa.
- Generar embeddings en lotes (chunking) respetando límites de la API.
- Reintentar con backoff exponencial ante errores transitorios.
- Exponer un `EmbeddingError` amigable para la UI.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, List, Optional, Sequence

from medisource.config import DEFAULT_EMBED_BATCH, DEFAULT_EMBED_MODEL, get_settings

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Fallo al generar embeddings (propagado a la UI de forma controlada)."""


def _sanitize(text: str) -> str:
    text = (text or "").strip()
    return text.replace("\x00", " ")


def _make_client(api_key: Optional[str] = None):
    """Crea un cliente OpenAI usando la API key efectiva.

    Se importa `openai` de forma perezosa para no penalizar el arranque de la UI.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingError(
            "El paquete 'openai' no está instalado. Ejecuta: pip install -r requirements.txt"
        ) from exc

    key = api_key or get_settings().openai_api_key
    if not key:
        raise EmbeddingError(
            "Falta la API key de OpenAI. Configura OPENAI_API_KEY o introdúcela en la barra lateral."
        )
    return OpenAI(api_key=key)


class OpenAIEmbedder:
    """Generador de embeddings con batching y reintentos."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_EMBED_MODEL,
        batch_size: int = DEFAULT_EMBED_BATCH,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(1, int(max_retries))
        self._client = _make_client(api_key)

    def embed_one(self, text: str) -> List[float]:
        """Embebe un único texto."""
        return self.embed_many([text])[0]

    def embed_many(
        self,
        texts: Sequence[str],
        *,
        progress_cb: Optional[callable] = None,
    ) -> List[List[float]]:
        """Embebe una lista de textos en lotes, con callback opcional de progreso."""
        cleaned = [_sanitize(t) for t in texts]
        vectors: List[List[float]] = [None] * len(cleaned)  # type: ignore[list-item]

        # Filtramos vacíos: devolverán vector vacío [] y los saltamos en quien use esto.
        idx_valid = [i for i, t in enumerate(cleaned) if t]
        idx_empty = [i for i, t in enumerate(cleaned) if not t]
        for i in idx_empty:
            vectors[i] = []

        total = len(idx_valid)
        processed = 0
        for start in range(0, total, self.batch_size):
            batch_indexes = idx_valid[start : start + self.batch_size]
            batch_texts = [cleaned[i] for i in batch_indexes]
            batch_vectors = self._embed_batch_with_retry(batch_texts)
            for i, vec in zip(batch_indexes, batch_vectors):
                vectors[i] = vec
            processed += len(batch_indexes)
            if progress_cb:
                try:
                    progress_cb(processed, total)
                except Exception:
                    log.debug("progress_cb falló (ignorado)", exc_info=True)
        return vectors  # type: ignore[return-value]

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.embeddings.create(model=self.model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception as exc:  # noqa: BLE001  (OpenAI expone varias clases)
                last_exc = exc
                wait = min(30.0, 1.5 ** attempt)
                log.warning(
                    "Embedding batch failed (attempt %d/%d): %s -> retry in %.1fs",
                    attempt, self.max_retries, exc, wait,
                )
                time.sleep(wait)
        raise EmbeddingError(f"La API de OpenAI falló tras {self.max_retries} intentos: {last_exc}")


def iter_chunks(items: Iterable, size: int) -> Iterable[list]:
    """Agrupa un iterable en listas de tamaño `size` (útil para CLIs)."""
    batch: list = []
    for it in items:
        batch.append(it)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
