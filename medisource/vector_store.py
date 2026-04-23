"""Wrapper sobre ChromaDB con persistencia en SQLite.

Centraliza:
- Creación / apertura de la colección.
- Upsert de dispositivos (id estable por hash del UDI-DI).
- Búsqueda semántica con filtro opcional por metadatos (p.ej. GMDN).
- Recuperación de un dispositivo concreto por id o por UDI-DI.

Se importa DESPUÉS del patch SQLite (`medisource._bootstrap`).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from medisource import _bootstrap as _bootstrap  # noqa: F401
from medisource.config import get_settings
from medisource.schemas import MedicalDevice

log = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Fallo genérico al operar con la base vectorial."""


def stable_id(device_identifier: str) -> str:
    """ID determinista a partir del UDI-DI (evita duplicados en reindexación)."""
    h = hashlib.sha1(device_identifier.strip().encode("utf-8")).hexdigest()
    return f"dev_{h[:16]}"


class ChromaStore:
    """Cliente persistente de ChromaDB orientado a nuestro dominio."""

    def __init__(
        self,
        *,
        path: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.path = path or settings.db_path
        self.collection_name = collection or settings.collection

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError(
                "El paquete 'chromadb' no está instalado. Ejecuta: pip install -r requirements.txt"
            ) from exc

        try:
            self._client = chromadb.PersistentClient(
                path=self.path,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"No se pudo abrir ChromaDB: {exc}") from exc

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def upsert_devices(
        self,
        devices: Sequence[MedicalDevice],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Inserta/actualiza dispositivos con sus vectores. Devuelve nº persistidos."""
        if len(devices) != len(embeddings):
            raise VectorStoreError("devices y embeddings deben tener la misma longitud")

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict] = []
        embs: List[List[float]] = []

        for d, emb in zip(devices, embeddings):
            if not emb:
                continue
            ids.append(stable_id(d.deviceIdentifier))
            docs.append(d.deviceDescription)
            metas.append(d.to_metadata())
            embs.append(list(emb))

        if not ids:
            return 0

        try:
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Fallo al persistir en Chroma: {exc}") from exc
        return len(ids)

    def semantic_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: Optional[Dict] = None,
        exclude_ids: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, MedicalDevice, float]]:
        """Devuelve lista de (id, device, similarity 0-1) ordenada desc por similitud."""
        if not query_embedding:
            return []

        # Pedimos más resultados de los necesarios por si hay que filtrar exclusiones.
        n_query = max(top_k * 2, top_k + len(list(exclude_ids or [])))
        try:
            res = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=max(1, n_query),
                where=where or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Error consultando Chroma: {exc}") from exc

        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        excluded = set(exclude_ids or [])

        out: List[Tuple[str, MedicalDevice, float]] = []
        for _id, meta, dist in zip(ids, metas, dists):
            if _id in excluded:
                continue
            try:
                device = MedicalDevice(**meta)
            except Exception:
                log.warning("Metadatos inválidos para id=%s, se ignora", _id)
                continue
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))  # distancia coseno -> similitud
            out.append((_id, device, similarity))
            if len(out) >= top_k:
                break
        return out

    def get_by_id(self, _id: str) -> Optional[MedicalDevice]:
        try:
            res = self._collection.get(ids=[_id], include=["metadatas"])
        except Exception:
            return None
        metas = res.get("metadatas") or []
        if not metas:
            return None
        try:
            return MedicalDevice(**metas[0])
        except Exception:
            return None

    def list_gmdn_terms(self, limit: int = 2000) -> List[str]:
        """Devuelve una lista única de nombres GMDN presentes en la colección."""
        try:
            res = self._collection.get(limit=limit, include=["metadatas"])
        except Exception:
            return []
        metas = res.get("metadatas") or []
        return sorted({(m.get("gmdnPTName") or "").strip() for m in metas if m.get("gmdnPTName")})
