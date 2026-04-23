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
    render_how_it_works,
    render_onboarding_no_data,
    render_savings_banner,
)
from medisource.vector_store import ChromaStore, VectorStoreError, stable_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medisource.app")

DEMO_CATALOG_PATH = Path(__file__).parent / "data" / "demo_catalog.csv"


st.set_page_config(
    page_title="MediSource AI · Clinical Spend Intelligence",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ---------------------------------------------------------------------------
# Recursos cacheados (store, embedder y agente se reutilizan entre reruns).
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
# Sidebar simplificada para el usuario final.
# Todo lo técnico (API key, ingesta, ajustes avanzados) queda tras un expander.
# ---------------------------------------------------------------------------

def render_sidebar(settings, db_count: int) -> dict:
    with st.sidebar:
        st.markdown("### 🏥 Tu hospital")
        hospital = st.text_input(
            "Nombre del centro",
            value=st.session_state.get("hospital", "Hospital Universitario"),
            label_visibility="collapsed",
            placeholder="Nombre del hospital",
        )
        st.session_state["hospital"] = hospital

        annual_volume = st.number_input(
            "Unidades que usas al año (para calcular ahorro)",
            min_value=1,
            max_value=1_000_000,
            value=int(st.session_state.get("annual_volume", 1000)),
            step=50,
            help="Consumo anual estimado del producto actual. Se usa para calcular el ahorro.",
        )
        st.session_state["annual_volume"] = int(annual_volume)

        st.markdown("---")

        # Estado de la base de datos, visible pero compacto
        if db_count > 0:
            st.success(f"✓ {db_count:,} productos disponibles")
        else:
            st.warning("Aún no hay catálogo cargado")

        # Estado de la API key (sin exponerla)
        has_secret_key = bool(settings.openai_api_key)
        manual_key = st.session_state.get("manual_api_key", "")
        effective_key = manual_key or settings.openai_api_key or ""

        if has_secret_key:
            st.caption("✓ Conexión con IA configurada")
        elif manual_key:
            st.caption("✓ Conexión con IA configurada (sesión)")
        else:
            st.error("⚠ Falta la conexión con IA. Abre Administración.")

        # --- Panel de administración (colapsado por defecto) -----------------
        with st.expander("🔧 Administración", expanded=(db_count == 0 or not effective_key)):
            st.caption(
                "Panel para el equipo técnico. Configura la conexión con IA y carga "
                "el catálogo de productos la primera vez."
            )

            if not has_secret_key:
                st.markdown("**Conexión con OpenAI**")
                new_key = st.text_input(
                    "API Key",
                    value=manual_key,
                    type="password",
                    placeholder="sk-...",
                    help="Pégala solo si tu administrador no la ha configurado en secrets.",
                    label_visibility="collapsed",
                )
                if new_key != manual_key:
                    st.session_state["manual_api_key"] = new_key
                    os.environ["OPENAI_API_KEY"] = new_key
                    effective_key = new_key
                st.caption(
                    "Mejor práctica: el administrador la configura en `secrets.toml` y "
                    "este campo desaparece."
                )
                st.markdown("---")

            st.markdown("**Cargar catálogo**")

            if DEMO_CATALOG_PATH.exists() and st.button(
                "🚀 Cargar catálogo demo (43 productos)",
                use_container_width=True,
                type="primary",
                key="sidebar_demo_btn",
            ):
                _run_ingest_ui(
                    csv_path=str(DEMO_CATALOG_PATH),
                    uploaded=None,
                    max_rows=None,
                    settings=settings,
                    api_key=effective_key,
                )

            st.caption("— o sube tu propio CSV —")
            uploaded = st.file_uploader(
                "Sube el CSV de productos",
                type=["csv"],
                label_visibility="collapsed",
            )
            csv_path_input = st.text_input(
                "…o indica una ruta local",
                value="gudid_filtrado.csv",
                help="Ruta al CSV GUDID si no lo subes directamente.",
            )
            max_rows = st.number_input(
                "Limitar filas (0 = todas, útil para pruebas)",
                min_value=0, max_value=200_000, value=0, step=100,
            )
            if st.button("📥 Cargar mi catálogo", use_container_width=True):
                _run_ingest_ui(
                    csv_path=csv_path_input,
                    uploaded=uploaded,
                    max_rows=int(max_rows) or None,
                    settings=settings,
                    api_key=effective_key,
                )

            st.markdown("---")
            st.markdown("**Ajustes avanzados**")
            top_k = st.slider("Nº de alternativas a mostrar", 3, 10,
                              int(st.session_state.get("top_k", 5)))
            use_gmdn_filter = st.checkbox(
                "Restringir a misma categoría clínica (GMDN)",
                value=st.session_state.get("use_gmdn_filter", True),
                help="Recomendado. Descarta alternativas de otras categorías médicas.",
            )
            similarity_floor = st.slider(
                "Similitud mínima aceptada (%)",
                0, 95,
                int(st.session_state.get("similarity_floor_pct", 50)),
                step=5,
            )
            chat_model = st.selectbox(
                "Modelo del auditor IA",
                ["gpt-4o", "gpt-4o-mini"],
                index=0 if st.session_state.get("chat_model", settings.chat_model) == "gpt-4o" else 1,
                help="GPT-4o ofrece análisis clínico más riguroso; mini reduce coste.",
            )
            st.session_state["top_k"] = int(top_k)
            st.session_state["use_gmdn_filter"] = bool(use_gmdn_filter)
            st.session_state["similarity_floor_pct"] = int(similarity_floor)
            st.session_state["chat_model"] = chat_model

    return {
        "api_key": effective_key,
        "chat_model": st.session_state.get("chat_model", settings.chat_model),
        "hospital": hospital,
        "annual_volume": int(annual_volume),
        "top_k": int(st.session_state.get("top_k", 5)),
        "use_gmdn_filter": bool(st.session_state.get("use_gmdn_filter", True)),
        "similarity_floor": float(st.session_state.get("similarity_floor_pct", 50)) / 100.0,
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
        st.error("Antes de cargar el catálogo necesitas configurar la conexión con OpenAI.")
        return

    target_path: Optional[Path] = None
    if uploaded is not None:
        target_path = Path("gudid_uploaded.csv")
        target_path.write_bytes(uploaded.getvalue())
    else:
        p = Path(csv_path).expanduser()
        if not p.exists():
            st.error(f"No se encuentra el archivo: {p}")
            return
        target_path = p

    try:
        with st.spinner(f"Leyendo y validando {target_path.name}…"):
            devices = read_devices_from_csv(target_path, max_rows=max_rows)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error leyendo el CSV: {exc}")
        return

    if not devices:
        st.error("El CSV no produjo productos válidos tras la validación.")
        return

    st.info(f"{len(devices):,} productos válidos. Generando índice semántico…")

    try:
        embedder = _get_embedder(api_key, settings.embed_model)
    except EmbeddingError as exc:
        st.error(str(exc))
        return

    progress = st.progress(0.0, text="Procesando 0/0")

    def _cb(done: int, total: int) -> None:
        progress.progress(done / max(1, total), text=f"Procesando {done:,}/{total:,}")

    try:
        texts = [build_embedding_text(d) for d in devices]
        vectors = embedder.embed_many(texts, progress_cb=_cb)
    except EmbeddingError as exc:
        st.error(f"Falló la generación del índice: {exc}")
        return

    try:
        store = _get_store(settings.db_path, settings.collection)
        persisted = store.upsert_devices(devices, vectors)
    except VectorStoreError as exc:
        st.error(f"Falló el guardado del catálogo: {exc}")
        return

    progress.empty()
    st.success(f"✓ Catálogo cargado. {persisted:,} productos listos para analizar.")
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Flujo principal: 3 pasos guiados.
# ---------------------------------------------------------------------------

def _ensure_session_state() -> None:
    st.session_state.setdefault("selected_device_id", None)
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("analyses", {})
    st.session_state.setdefault("alt_cache", {})


def _render_step_header(n: int, title: str, subtitle: str = "") -> None:
    sub = f'<div class="ms-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin: 18px 0 8px 0;">
            <div style="
                width:32px; height:32px; border-radius:50%;
                background: linear-gradient(135deg, #22d3ee, #34d399);
                color:#0b1020; font-weight:800; display:flex;
                align-items:center; justify-content:center;">
                {n}
            </div>
            <div>
                <div style="font-size:1.1rem; font-weight:600;">{title}</div>
                {sub}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_step1_search(settings) -> Optional[MedicalDevice]:
    _render_step_header(
        1,
        "¿Qué producto compras hoy a precio premium?",
        "Marca, fabricante o tipo de producto. Buscaremos el equivalente clínico más barato del catálogo FDA.",
    )

    query = st.text_input(
        "buscar",
        value=st.session_state["last_query"],
        placeholder="Ej: bisturí nº 11  ·  catéter venoso central 7 french  ·  Medtronic",
        label_visibility="collapsed",
    )
    st.session_state["last_query"] = query

    if not query.strip():
        st.caption("💡 Pista: prueba con una marca (Medtronic, BD) o un término clínico.")
        return None

    try:
        raw = _search_reference(query.strip(), settings.db_path, settings.collection)
    except SearchError as exc:
        st.error(str(exc))
        return None

    if not raw:
        st.warning("No encontramos coincidencias. Prueba otros términos o revisa el catálogo cargado.")
        return None

    candidates = [(rid, MedicalDevice(**data)) for rid, data in raw]

    def _label(item):
        _, d = item
        suffix = f" · {d.gmdnPTName}" if d.gmdnPTName else ""
        return f"{d.brandName} — {d.companyName}{suffix}"

    default_idx = 0
    if st.session_state["selected_device_id"]:
        for i, (rid, _d) in enumerate(candidates):
            if rid == st.session_state["selected_device_id"]:
                default_idx = i
                break

    selected = st.selectbox(
        f"Selecciona uno ({len(candidates)} coincidencias)",
        options=candidates,
        format_func=_label,
        index=default_idx,
    )
    st.session_state["selected_device_id"] = selected[0]
    return selected[1]


def _render_step2_alternatives(
    reference: MedicalDevice,
    settings,
    cfg: dict,
) -> Optional[SearchHit]:
    _render_step_header(
        2,
        "Tus alternativas más baratas",
        "Productos del catálogo FDA equivalentes al actual, ordenados por similitud clínica. "
        "Cuanto mayor el porcentaje, más parecidos son técnicamente.",
    )

    if not cfg["api_key"]:
        st.error("⚠ Configura la conexión con IA en el panel Administración de la izquierda.")
        return None

    cache_key = (
        reference.deviceIdentifier,
        cfg["top_k"],
        cfg["use_gmdn_filter"],
        round(cfg["similarity_floor"], 3),
    )
    cache_store = st.session_state["alt_cache"]

    if cache_key not in cache_store:
        try:
            embedder = _get_embedder(cfg["api_key"], settings.embed_model)
        except EmbeddingError as exc:
            st.error(str(exc))
            return None

        try:
            store = _get_store(settings.db_path, settings.collection)
            with st.spinner("Buscando alternativas…"):
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
            return None
        cache_store[cache_key] = hits

    hits: List[SearchHit] = cache_store[cache_key]
    if not hits:
        st.info(
            "No hemos encontrado alternativas con el mínimo de similitud actual. "
            "Prueba a desactivar el filtro de categoría clínica o a bajar la similitud mínima "
            "en **Ajustes avanzados**."
        )
        return None

    # Banner de ahorro potencial: la alternativa más rentable.
    best = max(hits, key=lambda h: h.price_delta_unit)
    if best.price_delta_unit > 0:
        render_savings_banner(
            best_unit_savings=best.price_delta_unit,
            best_savings_pct=best.price_delta_unit_pct,
            best_brand=f"{best.device.brandName} ({best.device.companyName})",
            annual_savings_top=best.price_delta_unit * cfg["annual_volume"],
            annual_volume=cfg["annual_volume"],
        )

    df = build_alternatives_dataframe(hits, annual_volume=cfg["annual_volume"])
    render_alternatives_table(df)

    option_labels = {
        i: f"#{i+1} · {h.device.brandName} — {h.device.companyName} · similitud {h.similarity*100:.0f}%"
        for i, h in enumerate(hits)
    }
    st.markdown("**Elige la alternativa que quieres evaluar:**")
    picked = st.radio(
        "opciones",
        options=list(option_labels.keys()),
        format_func=lambda i: option_labels[i],
        label_visibility="collapsed",
    )
    return hits[picked]


def _render_step3_analysis(
    reference: MedicalDevice,
    candidate: SearchHit,
    cfg: dict,
) -> None:
    _render_step_header(
        3,
        "Verificación clínica e informe firmable",
        "Pedimos al auditor IA que confirme que la sustitución es segura y emitimos el "
        "informe que tu Jefe de Servicio Médico necesita para aprobar el cambio.",
    )

    savings = estimate_savings(
        price_a=reference.estimated_price,
        price_b=candidate.device.estimated_price,
        annual_volume=cfg["annual_volume"],
    )
    kpi_row(savings)

    analyses = st.session_state["analyses"]
    analysis_key = f"{reference.deviceIdentifier}->{candidate.device.deviceIdentifier}"

    col_btn, col_info = st.columns([1, 2.2])
    with col_btn:
        run = st.button(
            "🧠 Verificar con auditor IA",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            f"Si sustituyes **{reference.brandName}** por **{candidate.device.brandName}** "
            f"en tu inventario anual, tu hospital ahorraría "
            f"**{format_eur(savings.annual_savings)}** ({savings.annual_savings_pct:+.1f}%)."
        )

    if run:
        try:
            agent = _get_agent(cfg["api_key"], cfg["chat_model"])
            with st.spinner(f"Consultando auditor IA ({cfg['chat_model']})…"):
                analyses[analysis_key] = agent.analyze_equivalence(reference, candidate.device)
        except AgentError as exc:
            st.error(str(exc))

    analysis: Optional[EquivalenceAnalysis] = analyses.get(analysis_key)
    if analysis:
        render_equivalence_report(analysis, reference, candidate.device)
        _render_report_download(analysis, reference, candidate.device, savings, cfg)
    else:
        st.caption(
            "Pulsa **Analizar equivalencia clínica** para obtener el informe detallado "
            "que tu Jefe de Servicio Médico necesita para aprobar la sustitución."
        )


def _render_report_download(
    analysis: EquivalenceAnalysis,
    device_a: MedicalDevice,
    device_b: MedicalDevice,
    savings,
    cfg: dict,
) -> None:
    md = [
        "# Informe de Equivalencia Clínica",
        f"**Centro:** {cfg['hospital']}",
        "",
        "## Sustitución propuesta",
        f"- Producto actual: **{device_a.brandName}** ({device_a.companyName}) — UDI-DI {device_a.deviceIdentifier}",
        f"- Alternativa: **{device_b.brandName}** ({device_b.companyName}) — UDI-DI {device_b.deviceIdentifier}",
        "",
        "## Ahorro estimado",
        f"- Precio unidad actual: {format_eur(savings.unit_price_a)}",
        f"- Precio alternativa: {format_eur(savings.unit_price_b)}",
        f"- Ahorro por unidad: {format_eur(savings.unit_savings)} ({savings.unit_savings_pct:+.1f}%)",
        f"- Volumen anual: {savings.annual_volume:,} uds",
        f"- **Ahorro anual: {format_eur(savings.annual_savings)} ({savings.annual_savings_pct:+.1f}%)**",
        "",
        "## Veredicto",
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
        "⬇️ Descargar informe para aprobación clínica",
        data=payload,
        file_name=f"equivalencia_{device_a.deviceIdentifier}_{device_b.deviceIdentifier}.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Acciones rápidas para el primer uso (catálogo vacío).
# ---------------------------------------------------------------------------

def _render_quick_actions(settings, cfg: dict) -> None:
    """Botonera directamente accionable cuando no hay catálogo indexado."""
    demo_exists = DEMO_CATALOG_PATH.exists()
    has_key = bool(cfg["api_key"])

    st.markdown("### Empieza en 1 clic")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="ms-card" style="min-height:180px;">
                <h3 style="margin-top:0;">🚀 Probar con catálogo demo</h3>
                <p class="ms-subtitle">
                    Carga un catálogo de ejemplo con 43 productos reales (bisturís, catéteres,
                    jeringas, sondas, mascarillas…) y empieza a buscar al instante.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        disabled = not (demo_exists and has_key)
        if st.button(
            "Cargar catálogo demo",
            type="primary",
            use_container_width=True,
            disabled=disabled,
            key="btn_demo",
        ):
            _run_ingest_ui(
                csv_path=str(DEMO_CATALOG_PATH),
                uploaded=None,
                max_rows=None,
                settings=settings,
                api_key=cfg["api_key"],
            )
        if not demo_exists:
            st.caption("⚠ No se encuentra `data/demo_catalog.csv` en el despliegue.")
        elif not has_key:
            st.caption("⚠ Necesitas configurar la conexión con IA (panel Administración).")
        else:
            st.caption("Tiempo estimado: ~20 segundos · coste OpenAI: menos de 0,01 €.")

    with col2:
        st.markdown(
            """
            <div class="ms-card" style="min-height:180px;">
                <h3 style="margin-top:0;">📁 Subir mi propio catálogo</h3>
                <p class="ms-subtitle">
                    Ya tienes un CSV con tus productos (formato GUDID o traducido al español).
                    Subirlo desde el panel <b>Administración</b> en la barra lateral.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("👉 Abre **🔧 Administración** en la barra lateral → **Cargar catálogo**.")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_session_state()
    settings = get_settings(refresh=True)

    try:
        store = _get_store(settings.db_path, settings.collection)
        db_count = store.count()
    except VectorStoreError as exc:
        st.error(f"Error abriendo la base de datos: {exc}")
        db_count = 0

    cfg = render_sidebar(settings, db_count)

    render_hero(db_count=db_count, has_api_key=bool(cfg["api_key"]))

    # Caso 1: catálogo vacío → onboarding + acción directa de carga.
    if db_count == 0:
        render_how_it_works()
        render_onboarding_no_data()
        _render_quick_actions(settings, cfg)
        return

    # Caso 2: catálogo cargado pero aún sin API key.
    if not cfg["api_key"]:
        render_how_it_works()
        render_empty_state(
            "Falta configurar la conexión con IA",
            "Abre el panel Administración en la barra lateral y añade tu API key.",
        )
        return

    # Caso 3: flujo normal en 3 pasos.
    reference = _render_step1_search(settings)
    if not reference:
        return

    st.markdown("---")
    col_info, _ = st.columns([1, 0.0001])
    with col_info:
        render_device_card(reference, title="Producto actual seleccionado")

    candidate = _render_step2_alternatives(reference, settings, cfg)
    if not candidate:
        return

    st.markdown("---")
    _render_step3_analysis(reference, candidate, cfg)


if __name__ == "__main__":
    main()
