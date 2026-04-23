from __future__ import annotations

try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass

import logging
import os
from pathlib import Path
from typing import List, Optional

import streamlit as st

from medisource.agent import AgentError, ClinicalJustificationAgent
from medisource.config import get_settings
from medisource.embeddings import EmbeddingError, OpenAIEmbedder
from medisource.ingest import build_embedding_text, read_devices_from_csv
from medisource.pricing import estimate_savings, format_eur
from medisource.schemas import EquivalenceAnalysis, MedicalDevice, SearchHit
from medisource.search import SearchError, find_similar, text_prefilter
from medisource.ui import (
    apply_theme,
    build_alternatives_dataframe,
    kpi_row,
    render_alternatives_table,
    render_device_card,
    render_empty_state,
    render_equivalence_report,
    render_hero,
)
from medisource.vector_store import ChromaStore, VectorStoreError, stable_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medisource.app")


st.set_page_config(
    page_title="MediSource AI · Clinical Spend Intelligence",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ---------------------------------------------------------------------------
# Recursos cacheados (store y embedder se reutilizan entre reruns).
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_store(db_path: str, collection: str) -> ChromaStore:
    return ChromaStore(path=db_path, collection=collection)


@st.cache_resource(show_spinner=False)
def _get_embedder(api_key: str, model: str) -> OpenAIEmbedder:
    return OpenAIEmbedder(api_key=api_key, model=model)


@st.cache_resource(show_spinner=False)
def _get_agent(api_key: str, model: str) -> ClinicalJustificationAgent:
    return ClinicalJustificationAgent(api_key=api_key, model=model)


@st.cache_data(ttl=300, show_spinner=False)
def _search_reference(query: str, db_path: str, collection: str) -> List[tuple]:
    store = _get_store(db_path, collection)
    results = text_prefilter(store, query, limit=25)
    return [(rid, d.model_dump()) for rid, d in results]


# ---------------------------------------------------------------------------
# Sidebar: configuración y operaciones de mantenimiento (ingesta).
# ---------------------------------------------------------------------------

def render_sidebar(settings) -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")

        api_key_env = settings.openai_api_key or ""
        api_key = st.text_input(
            "OpenAI API Key",
            value=api_key_env,
            type="password",
            help="Si la configuras como OPENAI_API_KEY (o en .streamlit/secrets.toml) se precarga.",
            placeholder="sk-...",
        )
        if api_key and api_key != api_key_env:
            os.environ["OPENAI_API_KEY"] = api_key

        chat_model = st.selectbox(
            "Modelo de razonamiento",
            ["gpt-4o", "gpt-4o-mini"],
            index=0 if settings.chat_model == "gpt-4o" else 1,
            help="GPT-4o da mejor calidad clínica; mini reduce coste.",
        )

        st.markdown("---")
        st.markdown("### 🏥 Hospital")
        hospital = st.text_input("Nombre del centro", value="Hospital Universitario")
        annual_volume = st.number_input(
            "Volumen anual (uds)", min_value=1, max_value=1_000_000, value=1000, step=50,
            help="Consumo anual del producto actual — usado para calcular ahorros.",
        )

        st.markdown("---")
        st.markdown("### 🔎 Búsqueda")
        top_k = st.slider("Nº de alternativas", min_value=3, max_value=10, value=5)
        use_gmdn_filter = st.checkbox(
            "Filtrar por código GMDN (Capa 1)",
            value=True,
            help="Restringe candidatos a la misma categoría clínica GMDN.",
        )
        similarity_floor = st.slider(
            "Similitud mínima (%)", min_value=0, max_value=95, value=50, step=5,
        ) / 100.0

        st.markdown("---")
        st.markdown("### 🗄️ Base vectorial")
        try:
            store = _get_store(settings.db_path, settings.collection)
            count = store.count()
        except VectorStoreError as exc:
            st.error(f"No se pudo abrir ChromaDB: {exc}")
            count = 0
        st.metric("Dispositivos indexados", f"{count:,}")

        with st.expander("📥 Ingestar CSV GUDID", expanded=(count == 0)):
            st.caption(
                "Carga un CSV de GUDID (columnas en inglés o el formato traducido "
                "de `gudid_filter.py`). Se generarán embeddings y se guardarán en ChromaDB."
            )
            default_path = "gudid_filtrado.csv"
            csv_path_input = st.text_input("Ruta del CSV", value=default_path)
            max_rows = st.number_input(
                "Límite de filas (0 = todas)", min_value=0, max_value=200_000, value=0, step=100,
            )
            uploaded = st.file_uploader("... o sube un CSV", type=["csv"], accept_multiple_files=False)

            if st.button("🚀 Ejecutar ingesta", type="primary", use_container_width=True):
                _run_ingest_ui(
                    csv_path=csv_path_input,
                    uploaded=uploaded,
                    max_rows=int(max_rows) or None,
                    settings=settings,
                    api_key=api_key,
                )

    return {
        "api_key": api_key,
        "chat_model": chat_model,
        "hospital": hospital,
        "annual_volume": int(annual_volume),
        "top_k": int(top_k),
        "use_gmdn_filter": bool(use_gmdn_filter),
        "similarity_floor": float(similarity_floor),
    }


def _run_ingest_ui(
    *,
    csv_path: str,
    uploaded,
    max_rows: Optional[int],
    settings,
    api_key: str,
) -> None:
    if not api_key:
        st.error("Necesitas una API key de OpenAI para generar embeddings.")
        return

    target_path: Optional[Path] = None
    if uploaded is not None:
        target_path = Path("gudid_uploaded.csv")
        target_path.write_bytes(uploaded.getvalue())
    else:
        p = Path(csv_path).expanduser()
        if not p.exists():
            st.error(f"No existe el archivo: {p}")
            return
        target_path = p

    try:
        with st.spinner(f"Leyendo y validando {target_path.name}…"):
            devices = read_devices_from_csv(target_path, max_rows=max_rows)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error leyendo el CSV: {exc}")
        return

    if not devices:
        st.error("El CSV no produjo dispositivos válidos tras la validación.")
        return

    st.info(f"{len(devices):,} dispositivos válidos. Generando embeddings…")

    try:
        embedder = _get_embedder(api_key, settings.embed_model)
    except EmbeddingError as exc:
        st.error(str(exc))
        return

    progress = st.progress(0.0, text="Embeddings 0/0")

    def _cb(done: int, total: int) -> None:
        progress.progress(done / max(1, total), text=f"Embeddings {done:,}/{total:,}")

    try:
        texts = [build_embedding_text(d) for d in devices]
        vectors = embedder.embed_many(texts, progress_cb=_cb)
    except EmbeddingError as exc:
        st.error(f"Falló la generación de embeddings: {exc}")
        return

    try:
        store = _get_store(settings.db_path, settings.collection)
        persisted = store.upsert_devices(devices, vectors)
    except VectorStoreError as exc:
        st.error(f"Falló el guardado en ChromaDB: {exc}")
        return

    progress.empty()
    st.success(f"Indexación completada: {persisted:,} dispositivos persistidos en `{settings.collection}`.")
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Vistas principales.
# ---------------------------------------------------------------------------

def _ensure_session_state() -> None:
    st.session_state.setdefault("selected_device_id", None)
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("alternatives", [])
    st.session_state.setdefault("analyses", {})  # {udi_di_b: EquivalenceAnalysis}


def _render_reference_picker(settings) -> Optional[MedicalDevice]:
    st.markdown("#### 1 · Selecciona el producto actual")
    query = st.text_input(
        "Buscar por marca, fabricante, UDI-DI o descripción",
        value=st.session_state["last_query"],
        placeholder="Ej: catéter central 7 french",
        label_visibility="collapsed",
    )
    st.session_state["last_query"] = query

    if not query.strip():
        render_empty_state(
            "Empieza buscando un producto",
            "Escribe una marca, un fabricante o parte de la descripción técnica.",
        )
        return None

    try:
        raw = _search_reference(query.strip(), settings.db_path, settings.collection)
    except SearchError as exc:
        st.error(str(exc))
        return None

    if not raw:
        render_empty_state("Sin coincidencias", "Prueba con otros términos o amplía la indexación.")
        return None

    candidates = [(rid, MedicalDevice(**data)) for rid, data in raw]

    def _label(item):
        _, d = item
        suffix = f" · {d.gmdnPTName}" if d.gmdnPTName else ""
        return f"{d.brandName} — {d.companyName}{suffix}  [{d.deviceIdentifier}]"

    default_idx = 0
    if st.session_state["selected_device_id"]:
        for i, (rid, _d) in enumerate(candidates):
            if rid == st.session_state["selected_device_id"]:
                default_idx = i
                break

    selected = st.selectbox(
        "Coincidencias",
        options=candidates,
        format_func=_label,
        index=default_idx,
        label_visibility="collapsed",
    )
    st.session_state["selected_device_id"] = selected[0]
    return selected[1]


def _render_alternatives(
    reference: MedicalDevice,
    settings,
    cfg: dict,
) -> None:
    st.markdown("#### 2 · Alternativas clínicamente equivalentes")

    if not cfg["api_key"]:
        st.warning("Introduce tu API key de OpenAI en la barra lateral para buscar alternativas.")
        return

    cache_key = (
        reference.deviceIdentifier,
        cfg["top_k"],
        cfg["use_gmdn_filter"],
        round(cfg["similarity_floor"], 3),
    )
    cache_store = st.session_state.setdefault("alt_cache", {})

    if cache_key not in cache_store:
        try:
            embedder = _get_embedder(cfg["api_key"], settings.embed_model)
        except EmbeddingError as exc:
            st.error(str(exc))
            return

        try:
            store = _get_store(settings.db_path, settings.collection)
            with st.spinner("Calculando similitud semántica…"):
                hits = find_similar(
                    store,
                    reference,
                    embedder=embedder,
                    top_k=cfg["top_k"],
                    use_gmdn_filter=cfg["use_gmdn_filter"],
                    similarity_floor=cfg["similarity_floor"],
                )
        except (EmbeddingError, SearchError, VectorStoreError) as exc:
            st.error(f"No se pudieron calcular alternativas: {exc}")
            return
        cache_store[cache_key] = hits

    hits: List[SearchHit] = cache_store[cache_key]
    if not hits:
        render_empty_state(
            "Sin alternativas bajo los filtros actuales",
            "Prueba a desactivar el filtro GMDN o a reducir la similitud mínima.",
        )
        return

    df = build_alternatives_dataframe(hits, annual_volume=cfg["annual_volume"])
    render_alternatives_table(df)

    st.markdown("#### 3 · Análisis clínico asistido (Auditor IA)")
    option_labels = {
        i: f"#{i+1} · {h.device.brandName} — {h.device.companyName} (sim {h.similarity*100:.1f}%)"
        for i, h in enumerate(hits)
    }
    picked = st.radio(
        "Selecciona la alternativa para evaluar",
        options=list(option_labels.keys()),
        format_func=lambda i: option_labels[i],
        horizontal=False,
    )
    candidate = hits[picked]

    savings = estimate_savings(
        price_a=reference.estimated_price,
        price_b=candidate.device.estimated_price,
        annual_volume=cfg["annual_volume"],
    )
    kpi_row(savings)

    analyses = st.session_state["analyses"]
    analysis_key = f"{reference.deviceIdentifier}->{candidate.device.deviceIdentifier}"

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run = st.button(
            "🧠 Analizar equivalencia",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            f"Sustitución: **{reference.brandName}** → **{candidate.device.brandName}** · "
            f"Ahorro anual: **{format_eur(savings.annual_savings)}** "
            f"({savings.annual_savings_pct:+.1f}%)"
        )

    if run:
        try:
            agent = _get_agent(cfg["api_key"], cfg["chat_model"])
            with st.spinner(f"Consultando {cfg['chat_model']}…"):
                analyses[analysis_key] = agent.analyze_equivalence(reference, candidate.device)
        except AgentError as exc:
            st.error(str(exc))

    analysis: Optional[EquivalenceAnalysis] = analyses.get(analysis_key)
    if analysis:
        render_equivalence_report(analysis, reference, candidate.device)
        _render_report_download(analysis, reference, candidate.device, savings, cfg)


def _render_report_download(
    analysis: EquivalenceAnalysis,
    device_a: MedicalDevice,
    device_b: MedicalDevice,
    savings,
    cfg: dict,
) -> None:
    md = [
        f"# Informe de Equivalencia Clínica",
        f"**Centro:** {cfg['hospital']}",
        "",
        f"## Sustitución propuesta",
        f"- Producto actual: **{device_a.brandName}** ({device_a.companyName}) — UDI-DI {device_a.deviceIdentifier}",
        f"- Alternativa: **{device_b.brandName}** ({device_b.companyName}) — UDI-DI {device_b.deviceIdentifier}",
        "",
        f"## Ahorro estimado",
        f"- Precio unidad actual: {format_eur(savings.unit_price_a)}",
        f"- Precio alternativa: {format_eur(savings.unit_price_b)}",
        f"- Ahorro por unidad: {format_eur(savings.unit_savings)} ({savings.unit_savings_pct:+.1f}%)",
        f"- Volumen anual: {savings.annual_volume:,} uds",
        f"- **Ahorro anual: {format_eur(savings.annual_savings)} ({savings.annual_savings_pct:+.1f}%)**",
        "",
        f"## Veredicto",
        f"- Compatibilidad: **{analysis.compatibility_score}/100**",
        f"- Veredicto: **{analysis.verdict_es}**",
        f"- Resumen: {analysis.executive_summary}",
        "",
        "## Similitudes críticas",
        *(f"- {s}" for s in analysis.similarities),
        "",
        "## Diferencias clínicas",
        *(f"- {s}" for s in analysis.differences),
        "",
    ]
    if analysis.missing_data:
        md += ["## Datos a verificar", *(f"- {s}" for s in analysis.missing_data), ""]
    if analysis.clinical_recommendation:
        md += ["## Recomendación clínica", analysis.clinical_recommendation, ""]
    md += [
        "---",
        "Firma del Jefe de Servicio Médico: ______________________________",
    ]

    payload = "\n".join(md).encode("utf-8")
    st.download_button(
        "⬇️ Descargar informe (Markdown)",
        data=payload,
        file_name=f"equivalencia_{device_a.deviceIdentifier}_{device_b.deviceIdentifier}.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Flujo principal.
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_session_state()
    settings = get_settings(refresh=True)
    cfg = render_sidebar(settings)

    try:
        store = _get_store(settings.db_path, settings.collection)
        db_count = store.count()
    except VectorStoreError as exc:
        st.error(f"No se pudo abrir ChromaDB: {exc}")
        db_count = 0

    render_hero(db_count=db_count, has_api_key=bool(cfg["api_key"]))

    if db_count == 0:
        render_empty_state(
            "Aún no hay dispositivos indexados",
            "Abre la sección \"Ingestar CSV GUDID\" de la barra lateral para cargar los datos.",
        )
        return

    col_left, col_right = st.columns([1, 1.35], gap="large")
    with col_left:
        reference = _render_reference_picker(settings)
        if reference:
            render_device_card(reference, title="Producto actual")

    with col_right:
        if reference:
            _render_alternatives(reference, settings, cfg)
        else:
            render_empty_state(
                "Selecciona un producto para ver alternativas",
                "Las sugerencias se calcularán al elegir una coincidencia.",
            )


if __name__ == "__main__":
    main()
