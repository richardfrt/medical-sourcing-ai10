"""Componentes visuales reutilizables del dashboard MediSource AI.

El diseño busca un acabado de SaaS B2B de analítica financiera:
- Paleta sobria (fondo casi negro, acentos cian/verde).
- Tarjetas con glassmorphism ligero.
- Métricas KPI con cifras grandes y deltas de color.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import streamlit as st

from medisource.pricing import SavingsEstimate, format_eur
from medisource.schemas import EquivalenceAnalysis, MedicalDevice, SearchHit


CUSTOM_CSS = """
<style>
    :root {
        --ms-bg: #0b1020;
        --ms-panel: rgba(255,255,255,0.04);
        --ms-border: rgba(255,255,255,0.08);
        --ms-text: #e6edf3;
        --ms-muted: #8b95a7;
        --ms-accent: #22d3ee;
        --ms-positive: #34d399;
        --ms-negative: #f87171;
    }

    section.main > div.block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Cabecera branding */
    .ms-hero {
        background: radial-gradient(120% 80% at 0% 0%, rgba(34,211,238,0.18), transparent 60%),
                    radial-gradient(120% 80% at 100% 0%, rgba(52,211,153,0.12), transparent 60%),
                    var(--ms-panel);
        border: 1px solid var(--ms-border);
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 18px;
    }
    .ms-hero h1 {
        font-size: 1.6rem; margin: 0 0 4px 0; letter-spacing: -0.01em;
    }
    .ms-hero p { color: var(--ms-muted); margin: 0; }

    /* Tarjeta genérica */
    .ms-card {
        background: var(--ms-panel);
        border: 1px solid var(--ms-border);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .ms-card h3 {
        margin: 0 0 6px 0; font-size: 1.05rem;
    }
    .ms-card .ms-subtitle {
        color: var(--ms-muted); font-size: 0.85rem; margin-bottom: 10px;
    }

    /* Chips */
    .ms-chip {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; background: rgba(34,211,238,0.12);
        color: var(--ms-accent); margin-right: 6px; margin-bottom: 4px;
        border: 1px solid rgba(34,211,238,0.25);
    }
    .ms-chip.neutral { background: rgba(255,255,255,0.06); color: var(--ms-text); border-color: var(--ms-border); }
    .ms-chip.positive { background: rgba(52,211,153,0.12); color: var(--ms-positive); border-color: rgba(52,211,153,0.3); }
    .ms-chip.negative { background: rgba(248,113,113,0.12); color: var(--ms-negative); border-color: rgba(248,113,113,0.3); }

    /* Tabla de alternativas */
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* Verdict badge */
    .ms-verdict {
        display: inline-block; padding: 6px 12px; border-radius: 10px;
        font-weight: 600; letter-spacing: 0.02em;
    }
    .ms-verdict.ok { background: rgba(52,211,153,0.15); color: var(--ms-positive); }
    .ms-verdict.warn { background: rgba(250,204,21,0.15); color: #facc15; }
    .ms-verdict.ko { background: rgba(248,113,113,0.15); color: var(--ms-negative); }

    /* Listas clínicas */
    .ms-list { margin: 0; padding-left: 1.1rem; }
    .ms-list li { margin-bottom: 4px; }

    /* Botones primarios */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #22d3ee 0%, #34d399 100%);
        color: #0b1020; border: 0; font-weight: 600;
    }
    .stButton>button[kind="primary"]:hover { filter: brightness(1.05); }
</style>
"""


def apply_theme() -> None:
    """Inyecta el CSS custom. Llamar una única vez al arrancar la app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero(db_count: int, has_api_key: bool) -> None:
    status = (
        f"{db_count:,} productos en catálogo"
        if db_count
        else "Catálogo no cargado"
    )
    key_status = "Auditor IA conectado" if has_api_key else "Auditor IA desconectado"
    st.markdown(
        f"""
        <div class="ms-hero">
            <h1>MediSource AI · El comparador clínico de tu hospital</h1>
            <p>
                El "buscador de equivalencias" para material sanitario. Compras una marca premium
                a 100&nbsp;€, te enseñamos la genérica equivalente a 70&nbsp;€ y la IA garantiza
                que es clínicamente segura. <b>Ahorro medio detectado: 12-18% del gasto MedSurg.</b>
            </p>
            <div style="margin-top:12px;">
                <span class="ms-chip">{status}</span>
                <span class="ms-chip {'positive' if has_api_key else 'negative'}">{key_status}</span>
                <span class="ms-chip neutral">Datos: FDA GUDID</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_device_card(device: MedicalDevice, *, title: str = "Producto actual") -> None:
    st.markdown(
        f"""
        <div class="ms-card">
            <h3>{title}</h3>
            <div class="ms-subtitle">{device.brandName} · {device.companyName}</div>
            <div>
                <span class="ms-chip neutral">UDI-DI: {device.deviceIdentifier}</span>
                {f'<span class="ms-chip neutral">Ref: {device.versionModelNumber}</span>' if device.versionModelNumber else ''}
                {f'<span class="ms-chip">{device.gmdnPTName}</span>' if device.gmdnPTName else ''}
            </div>
            <p style="margin-top:12px; color: var(--ms-muted); font-size: 0.92rem; line-height: 1.45;">
                {device.deviceDescription}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(savings: SavingsEstimate) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Precio unidad actual",
        format_eur(savings.unit_price_a),
    )
    c2.metric(
        "Precio alternativa",
        format_eur(savings.unit_price_b),
        delta=f"{savings.unit_savings_pct:+.1f}%",
        delta_color="inverse",
    )
    c3.metric(
        "Ahorro unitario",
        format_eur(savings.unit_savings),
        delta=f"{savings.unit_savings_pct:+.1f}%",
    )
    c4.metric(
        f"Ahorro anual ({savings.annual_volume:,} uds)",
        format_eur(savings.annual_savings),
        delta=f"{savings.annual_savings_pct:+.1f}%",
    )


def build_alternatives_dataframe(
    hits: List[SearchHit],
    *,
    annual_volume: int,
) -> pd.DataFrame:
    rows = []
    for idx, hit in enumerate(hits, start=1):
        rows.append(
            {
                "#": idx,
                "Marca": hit.device.brandName,
                "Fabricante": hit.device.companyName,
                "Categoría GMDN": hit.device.gmdnPTName or "(no informado)",
                "UDI-DI": hit.device.deviceIdentifier,
                "Similitud semántica": round(hit.similarity * 100.0, 1),
                "Precio unidad": round(hit.device.estimated_price, 2),
                "Δ precio unidad": hit.price_delta_unit,
                "Δ precio %": hit.price_delta_unit_pct,
                "Ahorro anual estimado": round(hit.price_delta_unit * annual_volume, 2),
            }
        )
    return pd.DataFrame(rows)


def render_alternatives_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No se han encontrado alternativas con los filtros actuales.")
        return

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Similitud semántica": st.column_config.ProgressColumn(
                "Similitud semántica", format="%.1f %%", min_value=0, max_value=100,
            ),
            "Precio unidad": st.column_config.NumberColumn("Precio unidad", format="%.2f €"),
            "Δ precio unidad": st.column_config.NumberColumn("Δ precio unidad", format="%.2f €"),
            "Δ precio %": st.column_config.NumberColumn("Δ precio %", format="%.1f %%"),
            "Ahorro anual estimado": st.column_config.NumberColumn(
                "Ahorro anual estimado", format="%.0f €",
            ),
        },
    )


def _verdict_class(verdict: str) -> str:
    return {"EQUIVALENT": "ok", "CONDITIONAL": "warn", "NOT_EQUIVALENT": "ko"}.get(verdict, "warn")


def render_equivalence_report(
    analysis: EquivalenceAnalysis,
    device_a: MedicalDevice,
    device_b: MedicalDevice,
) -> None:
    klass = _verdict_class(analysis.verdict)
    st.markdown(
        f"""
        <div class="ms-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                <div>
                    <h3 style="margin:0;">Informe de Equivalencia Clínica</h3>
                    <div class="ms-subtitle">
                        {device_a.brandName} <span style="color:var(--ms-muted)">→</span> {device_b.brandName}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="ms-verdict {klass}">{analysis.verdict_es}</div>
                    <div style="font-size:2rem; font-weight:700; margin-top:6px;">
                        {analysis.compatibility_score}<span style="color:var(--ms-muted); font-size:1rem;">/100</span>
                    </div>
                </div>
            </div>
            <p style="margin-top:14px;">{analysis.executive_summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Similitudes críticas**")
        if analysis.similarities:
            st.markdown("<ul class='ms-list'>" + "".join(f"<li>{s}</li>" for s in analysis.similarities) + "</ul>", unsafe_allow_html=True)
        else:
            st.caption("No se han identificado similitudes relevantes.")

    with col_b:
        st.markdown("**Diferencias clínicas**")
        if analysis.differences:
            st.markdown("<ul class='ms-list'>" + "".join(f"<li>{s}</li>" for s in analysis.differences) + "</ul>", unsafe_allow_html=True)
        else:
            st.caption("No se han identificado diferencias relevantes.")

    if analysis.missing_data:
        st.warning(
            "⚠ Datos no informados en GUDID que conviene revisar con el fabricante:\n- "
            + "\n- ".join(analysis.missing_data)
        )

    if analysis.clinical_recommendation:
        st.markdown(
            f"""
            <div class="ms-card" style="border-color: rgba(34,211,238,0.3);">
                <h3>Recomendación para el jefe de servicio médico</h3>
                <p>{analysis.clinical_recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_empty_state(message: str, hint: Optional[str] = None) -> None:
    st.markdown(
        f"""
        <div class="ms-card" style="text-align:center; padding:40px 20px;">
            <h3 style="margin-top:0;">{message}</h3>
            {f'<p class="ms-subtitle">{hint}</p>' if hint else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    """Panel de 'cómo funciona' alineado con el pitch de negocio."""
    st.markdown(
        """
        <div class="ms-card">
            <h3 style="margin-top:0;">Cómo funciona en 3 pasos</h3>
            <p class="ms-subtitle" style="margin: 0 0 14px 0;">
                Sin entrenar modelos médicos caros. Usamos la base oficial de la FDA + IA generativa.
            </p>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:18px;">
                <div>
                    <div style="font-size:1.4rem;">🧭</div>
                    <div style="font-weight:600; margin: 6px 0;">1 · Coordenadas inteligentes</div>
                    <div class="ms-subtitle">
                        OpenAI lee la descripción técnica de cada producto del catálogo FDA y la
                        convierte en una huella matemática. Productos parecidos quedan
                        <b>cerca, como vecinos en un mapa</b>.
                    </div>
                </div>
                <div>
                    <div style="font-size:1.4rem;">🔍</div>
                    <div style="font-weight:600; margin: 6px 0;">2 · Buscamos los vecinos baratos</div>
                    <div class="ms-subtitle">
                        Cuando buscas un producto premium, la base vectorial encuentra al instante
                        los más similares — incluyendo los <b>genéricos equivalentes</b> que
                        nadie había detectado.
                    </div>
                </div>
                <div>
                    <div style="font-size:1.4rem;">👨‍⚕️</div>
                    <div style="font-weight:600; margin: 6px 0;">3 · Auditor clínico GPT-4o</div>
                    <div class="ms-subtitle">
                        Antes de proponer la sustitución, GPT-4o revisa material, dimensiones y
                        esterilización, y emite un <b>informe firmable</b> por tu Jefe de Servicio
                        Médico.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_onboarding_no_data() -> None:
    """Pantalla de bienvenida cuando la base vectorial está vacía.

    Incluye un ejemplo realista del valor (bisturí 100€ vs 70€) para
    enganchar al CFO antes incluso de cargar datos.
    """
    st.markdown(
        """
        <div class="ms-card" style="padding:36px 28px;">
            <div style="text-align:center;">
                <div style="font-size:2.2rem;">👋</div>
                <h3 style="margin:6px 0;">Bienvenido a MediSource AI</h3>
                <p class="ms-subtitle" style="max-width:620px; margin:0 auto;">
                    Antes de empezar, mira un caso real de lo que esta herramienta detecta:
                </p>
            </div>

            <div style="
                display:grid; grid-template-columns: 1fr auto 1fr; gap:18px;
                align-items:center; margin: 24px auto; max-width: 760px;">
                <div style="
                    border:1px solid var(--ms-border); border-radius:14px;
                    padding:18px; text-align:center; background: rgba(248,113,113,0.06);">
                    <div class="ms-subtitle" style="font-size:0.75rem; letter-spacing:0.1em;">PRODUCTO PREMIUM</div>
                    <div style="font-weight:700; margin: 6px 0;">Bisturí marca X</div>
                    <div style="font-size:1.6rem; font-weight:700; color: var(--ms-negative);">100&nbsp;€</div>
                    <div class="ms-subtitle" style="font-size:0.8rem;">precio actual / unidad</div>
                </div>
                <div style="font-size:1.6rem; color: var(--ms-accent); text-align:center;">→</div>
                <div style="
                    border:1px solid rgba(52,211,153,0.4); border-radius:14px;
                    padding:18px; text-align:center; background: rgba(52,211,153,0.08);">
                    <div class="ms-subtitle" style="font-size:0.75rem; letter-spacing:0.1em;">GENÉRICO EQUIVALENTE</div>
                    <div style="font-weight:700; margin: 6px 0;">Bisturí marca Y</div>
                    <div style="font-size:1.6rem; font-weight:700; color: var(--ms-positive);">70&nbsp;€</div>
                    <div class="ms-subtitle" style="font-size:0.8rem;">mismo material · misma certificación</div>
                </div>
            </div>

            <div style="text-align:center; margin-bottom: 18px;">
                <span class="ms-chip positive" style="font-size:0.95rem; padding:6px 14px;">
                    Ahorras 30&nbsp;€ por unidad · auditor IA aprueba la sustitución
                </span>
            </div>

            <hr style="border-color: var(--ms-border); margin: 20px 0;">

            <div style="text-align:center;">
                <p class="ms-subtitle" style="max-width:620px; margin:0 auto 10px auto;">
                    Para encontrar oportunidades como esta en tu hospital, carga tu catálogo
                    de productos (CSV). Sólo hace falta hacerlo una vez.
                </p>
                <p class="ms-subtitle" style="margin: 6px 0 0 0;">
                    👉 Abre el panel <b>🔧 Administración</b> en la barra lateral y pulsa
                    <b>«Cargar catálogo»</b>.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_savings_banner(
    *,
    best_unit_savings: float,
    best_savings_pct: float,
    best_brand: str,
    annual_savings_top: float,
    annual_volume: int,
) -> None:
    """Banner con el ahorro potencial detectado para enganchar al CFO."""
    if best_unit_savings <= 0:
        return

    color = "var(--ms-positive)" if best_savings_pct >= 10 else "var(--ms-accent)"
    st.markdown(
        f"""
        <div class="ms-card" style="
            background: linear-gradient(135deg, rgba(52,211,153,0.10), rgba(34,211,238,0.06));
            border-color: rgba(52,211,153,0.35);">
            <div style="display:flex; flex-wrap:wrap; gap:18px; align-items:center; justify-content:space-between;">
                <div>
                    <div class="ms-subtitle" style="font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase;">
                        💰 Mejor oportunidad detectada
                    </div>
                    <div style="font-weight:700; font-size:1.05rem; margin-top:4px;">
                        Sustituyendo por <span style="color:{color};">{best_brand}</span>
                    </div>
                    <div class="ms-subtitle" style="margin-top:2px;">
                        Ahorro de <b>{best_savings_pct:.0f}%</b> por unidad ·
                        volumen anual estimado <b>{annual_volume:,} uds</b>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="ms-subtitle" style="font-size:0.8rem;">AHORRO ANUAL POTENCIAL</div>
                    <div style="font-size:2.1rem; font-weight:800; color:{color}; line-height:1;">
                        {_format_eur_inline(annual_savings_top)}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_eur_inline(value: float) -> str:
    """Formato es-ES sin importar pricing.format_eur (evita import circular)."""
    try:
        s = f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)
    return s.replace(",", ".") + " €"
