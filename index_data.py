"""CLI de ingesta e indexación de MediSource AI.

Ejecución local (equivale al "Sprint 1" del PRD):

    python scripts/index_data.py --csv gudid_filtrado.csv

Genera embeddings con OpenAI en lotes y persiste en ChromaDB (`./chroma_db`).
Usa `OPENAI_API_KEY` del entorno. Soporta reindexación idempotente (los ids
se derivan del UDI-DI vía hash).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medisource.config import get_settings  # noqa: E402
from medisource.embeddings import EmbeddingError, OpenAIEmbedder  # noqa: E402
from medisource.ingest import build_embedding_text, read_devices_from_csv  # noqa: E402
from medisource.vector_store import ChromaStore, VectorStoreError  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("medisource.ingest")


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Ingestar CSV GUDID y generar embeddings en ChromaDB.")
    p.add_argument("--csv", required=True, help="Ruta al CSV de entrada.")
    p.add_argument("--max-rows", type=int, default=0, help="Límite de filas (0 = todas).")
    p.add_argument("--batch-size", type=int, default=settings.embed_batch,
                   help="Tamaño de lote de embeddings.")
    p.add_argument("--embed-model", default=settings.embed_model, help="Modelo de embeddings.")
    p.add_argument("--db-path", default=settings.db_path, help="Directorio de ChromaDB.")
    p.add_argument("--collection", default=settings.collection, help="Nombre de la colección.")
    p.add_argument("--dry-run", action="store_true",
                   help="Sólo valida el CSV sin llamar a OpenAI ni escribir en Chroma.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        log.error("No existe el archivo CSV: %s", csv_path)
        return 2

    log.info("Leyendo y validando %s …", csv_path)
    devices = read_devices_from_csv(csv_path, max_rows=args.max_rows or None)
    log.info("Dispositivos válidos tras Pydantic: %d", len(devices))
    if not devices:
        log.error("No se pudo construir ningún dispositivo válido.")
        return 1

    if args.dry_run:
        for d in devices[:5]:
            log.info("• %s | %s | %s", d.deviceIdentifier, d.brandName, d.gmdnPTName)
        log.info("Dry-run OK (%d devices). Saliendo sin tocar ChromaDB.", len(devices))
        return 0

    try:
        embedder = OpenAIEmbedder(
            model=args.embed_model,
            batch_size=args.batch_size,
        )
    except EmbeddingError as exc:
        log.error("%s", exc)
        return 3

    texts = [build_embedding_text(d) for d in devices]
    log.info("Generando embeddings con %s (batch=%d) …", args.embed_model, args.batch_size)
    t0 = time.time()

    def _cb(done: int, total: int) -> None:
        if done == total or done % (args.batch_size * 5) == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            log.info("  progreso: %d/%d (%.1f/s)", done, total, rate)

    try:
        vectors = embedder.embed_many(texts, progress_cb=_cb)
    except EmbeddingError as exc:
        log.error("Falló la generación de embeddings: %s", exc)
        return 4

    try:
        store = ChromaStore(path=args.db_path, collection=args.collection)
        persisted = store.upsert_devices(devices, vectors)
    except VectorStoreError as exc:
        log.error("Falló el guardado en ChromaDB: %s", exc)
        return 5

    log.info("Indexación completada: %d dispositivos persistidos en '%s'.", persisted, args.collection)
    log.info("Colección '%s' ahora contiene %d documentos.", args.collection, store.count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
