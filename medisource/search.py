"""Pipeline de búsqueda multicapa (PRD §4).

Capa 1 (Hard Filter): restringe por código o nombre GMDN.
Capa 2 (Semantic Search): similitud del coseno vía ChromaDB.
Capa 3 (Generative Validation): se delega en `medisource.agent` y se invoca
sólo bajo demanda del usuario en la UI (evita consumo innecesario de GPT-4o).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from medisource.embeddings import EmbeddingError, OpenAIEmbedder
from medisource.ingest import build_embedding_text
from medisource.pricing import deterministic_price
from medisource.schemas import MedicalDevice, SearchHit
from medisource.vector_store import ChromaStore, VectorStoreError, stable_id

log = logging.getLogger(__name__)


class SearchError(RuntimeError):
    """Fallo en el pipeline de búsqueda (propagado a UI con `st.error`)."""


def _build_where(
    device: Optional[MedicalDevice],
    *,
    use_gmdn_filter: bool,
) -> Optional[dict]:
    if not use_gmdn_filter or device is None:
        return None
    if device.gmdnCode:
        return {"gmdnCode": device.gmdnCode}
    if device.gmdnPTName:
        return {"gmdnPTName": device.gmdnPTName}
    return None


def _price_delta(reference: float, candidate: float) -> Tuple[float, float]:
    delta = round(reference - candidate, 2)
    pct = round((delta / reference) * 100.0, 2) if reference > 0 else 0.0
    return delta, pct


def text_prefilter(
    store: ChromaStore,
    query: str,
    *,
    limit: int = 25,
) -> List[Tuple[str, MedicalDevice]]:
    """Búsqueda textual simple sobre la metadata para seleccionar un producto origen.

    No es un full-text search potente pero basta como "caja de búsqueda"
    para que el usuario localice un producto del inventario indexado.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    try:
        res = store._collection.get(limit=5000, include=["metadatas"])
    except Exception as exc:  # noqa: BLE001
        raise SearchError(f"No se pudo consultar el índice: {exc}") from exc

    ids = res.get("ids") or []
    metas = res.get("metadatas") or []

    scored: List[Tuple[int, str, MedicalDevice]] = []
    tokens = [t for t in q.split() if t]
    for _id, meta in zip(ids, metas):
        haystack = " ".join(
            str(meta.get(k, "") or "")
            for k in ("brandName", "companyName", "gmdnPTName", "deviceDescription", "versionModelNumber", "deviceIdentifier")
        ).lower()
        if not haystack:
            continue
        if q in haystack:
            score = 100
        else:
            score = sum(1 for t in tokens if t in haystack)
        if score <= 0:
            continue
        try:
            device = MedicalDevice(**meta)
        except Exception:
            continue
        scored.append((score, _id, device))

    scored.sort(key=lambda x: (-x[0], x[2].brandName))
    return [(i, d) for _, i, d in scored[:limit]]


def find_similar(
    store: ChromaStore,
    reference: MedicalDevice,
    *,
    embedder: OpenAIEmbedder,
    top_k: int = 5,
    use_gmdn_filter: bool = True,
    similarity_floor: float = 0.0,
) -> List[SearchHit]:
    """Ejecuta las capas 1 (GMDN) + 2 (coseno) del pipeline."""
    try:
        vector = embedder.embed_one(build_embedding_text(reference))
    except EmbeddingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SearchError(f"No se pudo generar el embedding de la consulta: {exc}") from exc

    where = _build_where(reference, use_gmdn_filter=use_gmdn_filter)

    try:
        results = store.semantic_search(
            vector,
            top_k=top_k,
            where=where,
            exclude_ids=[stable_id(reference.deviceIdentifier)],
        )
    except VectorStoreError:
        # Fallback: reintento sin filtro GMDN (puede que la colección no tenga esa clave).
        if where is not None:
            log.warning("Fallo filtrando por GMDN, reintentando sin filtro.")
            results = store.semantic_search(
                vector,
                top_k=top_k,
                where=None,
                exclude_ids=[stable_id(reference.deviceIdentifier)],
            )
        else:
            raise

    hits: List[SearchHit] = []
    ref_price = reference.estimated_price or deterministic_price(reference.deviceIdentifier)
    for _id, device, similarity in results:
        if similarity < similarity_floor:
            continue
        cand_price = device.estimated_price or deterministic_price(device.deviceIdentifier)
        delta, delta_pct = _price_delta(ref_price, cand_price)
        hits.append(
            SearchHit(
                device=device,
                similarity=similarity,
                price_delta_unit=delta,
                price_delta_unit_pct=delta_pct,
            )
        )
    return hits
