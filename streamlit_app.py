"""
Comparador de precios simple y funcional.

El usuario escribe un producto y la app consulta en tiempo real la API pública
de DummyJSON (https://dummyjson.com/products/search) para traer productos
reales con precios variados, calcular el más barato, el más caro, descuentos,
y mostrarlos ordenados para facilitar la comparación.

DummyJSON es una API pública gratuita, sin API key, con más de 200 productos
reales en categorías como smartphones, laptops, fragancias, ropa, muebles,
electrodomésticos, cuidado personal, etc.
"""

from __future__ import annotations

from statistics import median
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

API_URL = "https://dummyjson.com/products/search"

st.set_page_config(
    page_title="Comparador de Precios",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def search_products(query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Consulta la API pública y normaliza la respuesta."""
    resp = requests.get(API_URL, params={"q": query, "limit": limit}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("products", []) or []

    normalized: list[dict[str, Any]] = []
    for p in results:
        price = p.get("price")
        if price is None:
            continue
        discount_pct = float(p.get("discountPercentage") or 0)
        price = float(price)
        original_price = round(price / (1 - discount_pct / 100), 2) if discount_pct else price
        normalized.append({
            "id": p.get("id"),
            "title": p.get("title") or "(sin título)",
            "description": p.get("description") or "",
            "price": price,
            "original_price": original_price,
            "discount_pct": discount_pct,
            "rating": float(p.get("rating") or 0),
            "stock": int(p.get("stock") or 0),
            "brand": p.get("brand") or "—",
            "category": p.get("category") or "—",
            "thumbnail": p.get("thumbnail") or "",
            "availability_status": p.get("availabilityStatus") or "",
        })
    return normalized


def format_price(amount: float) -> str:
    return f"${amount:,.2f}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding: 6px 0 18px 0;">
            <h1 style="margin-bottom:4px;">🔍 Comparador de Precios</h1>
            <p style="color:#6b7280; margin-top:0;">
                Escribe el producto que buscas y compara precios al instante.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### 🔎 Filtros")
        category_filter = st.text_input(
            "Categoría (opcional)",
            value="",
            placeholder="Ej: smartphones",
            help="Filtra los resultados por categoría (smartphones, laptops, fragrances…).",
        )
        min_rating = st.slider("Puntuación mínima", 0.0, 5.0, 0.0, step=0.5)
        only_with_stock = st.checkbox("Solo con stock disponible", value=True)
        only_discount = st.checkbox("Solo productos con descuento", value=False)
        st.markdown("---")
        st.markdown("### ↕️ Orden")
        sort = st.selectbox(
            "Ordenar por",
            [
                "Precio: menor a mayor",
                "Precio: mayor a menor",
                "Mayor descuento",
                "Mejor puntuación",
            ],
            index=0,
        )
        st.markdown("---")
        limit = st.slider("Máximo de resultados a traer", 10, 50, 30, step=5)
        st.markdown("---")
        st.caption(
            "Datos en vivo desde la API pública de DummyJSON. "
            "El catálogo incluye smartphones, laptops, ropa, muebles, fragancias, "
            "electrodomésticos y más."
        )
    return {
        "category_filter": category_filter.strip().lower(),
        "min_rating": min_rating,
        "only_with_stock": only_with_stock,
        "only_discount": only_discount,
        "sort": sort,
        "limit": limit,
    }


def render_stats(items: list[dict[str, Any]]) -> None:
    prices = [i["price"] for i in items]
    if not prices:
        return
    cheapest = min(items, key=lambda x: x["price"])
    most_expensive = max(items, key=lambda x: x["price"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Productos", f"{len(items)}")
    c2.metric("Precio mínimo", format_price(cheapest["price"]))
    c3.metric("Mediana", format_price(median(prices)))
    c4.metric("Precio máximo", format_price(most_expensive["price"]))

    spread = most_expensive["price"] - cheapest["price"]
    if cheapest["price"] > 0:
        pct = spread / cheapest["price"] * 100
        st.success(
            f"💡 Entre el más barato y el más caro hay **{format_price(spread)}** "
            f"de diferencia ({pct:.0f}%). El más barato es **{cheapest['title']}** "
            f"({cheapest['brand']}) a **{format_price(cheapest['price'])}**."
        )


def render_product_card(item: dict[str, Any], rank: int, cheapest_price: float) -> None:
    is_cheapest = item["price"] == cheapest_price
    border = "2px solid #10b981" if is_cheapest else "1px solid #e5e7eb"
    badge_cheap = (
        '<span style="background:#10b981;color:white;padding:3px 10px;border-radius:10px;'
        'font-size:0.75rem;font-weight:700;">⭐ MÁS BARATO</span>'
        if is_cheapest else ""
    )

    discount_html = ""
    if item["discount_pct"] >= 1:
        discount_html = (
            f'<span style="background:#ef4444;color:white;padding:3px 8px;border-radius:6px;'
            f'font-size:0.8rem;font-weight:700;margin-left:8px;">'
            f'-{item["discount_pct"]:.0f}%</span>'
        )

    original_html = ""
    if item["discount_pct"] >= 1 and item["original_price"] > item["price"]:
        original_html = (
            f'<span style="text-decoration:line-through;color:#9ca3af;font-size:0.95rem;'
            f'margin-left:10px;">{format_price(item["original_price"])}</span>'
        )

    stars = "★" * int(round(item["rating"])) + "☆" * (5 - int(round(item["rating"])))
    stock_color = "#10b981" if item["stock"] > 0 else "#ef4444"
    stock_text = f"Stock: {item['stock']}" if item["stock"] > 0 else "Sin stock"

    st.markdown(
        f"""
        <div style="border:{border}; border-radius:12px; padding:14px; margin-bottom:10px; background:white;">
            <div style="display:flex; gap:14px; align-items:flex-start;">
                <img src="{item['thumbnail']}" style="width:120px;height:120px;object-fit:contain;border-radius:8px;background:#f9fafb;padding:4px;" />
                <div style="flex:1;">
                    <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">
                        <div style="font-weight:600;color:#6b7280;font-size:0.85rem;">
                            #{rank} · {item['category']} · <b>{item['brand']}</b>
                        </div>
                        <div>{badge_cheap}</div>
                    </div>
                    <div style="font-size:1.05rem;font-weight:600;margin:4px 0;color:#111827;">
                        {item['title']}
                    </div>
                    <div style="color:#6b7280;font-size:0.88rem;margin-bottom:6px;">
                        {item['description'][:140]}{'…' if len(item['description']) > 140 else ''}
                    </div>
                    <div style="display:flex;align-items:baseline;gap:4px;">
                        <span style="font-size:1.6rem;font-weight:700;color:#111827;">
                            {format_price(item['price'])}
                        </span>
                        {discount_html}
                        {original_html}
                    </div>
                    <div style="color:#6b7280;font-size:0.85rem;margin-top:6px;">
                        <span style="color:#f59e0b;">{stars}</span>
                        <span style="margin-left:4px;">{item['rating']:.1f}/5</span>
                        · <span style="color:{stock_color};font-weight:600;">{stock_text}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(items: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    result = list(items)
    if cfg["category_filter"]:
        result = [i for i in result if cfg["category_filter"] in i["category"].lower()]
    if cfg["min_rating"] > 0:
        result = [i for i in result if i["rating"] >= cfg["min_rating"]]
    if cfg["only_with_stock"]:
        result = [i for i in result if i["stock"] > 0]
    if cfg["only_discount"]:
        result = [i for i in result if i["discount_pct"] >= 1]

    if cfg["sort"] == "Precio: menor a mayor":
        result.sort(key=lambda x: x["price"])
    elif cfg["sort"] == "Precio: mayor a menor":
        result.sort(key=lambda x: x["price"], reverse=True)
    elif cfg["sort"] == "Mayor descuento":
        result.sort(key=lambda x: x["discount_pct"], reverse=True)
    elif cfg["sort"] == "Mejor puntuación":
        result.sort(key=lambda x: x["rating"], reverse=True)
    return result


def render_intro() -> None:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            ### 1. Escribe
            Pon el producto que quieres comparar.
            Ejemplos: `iphone`, `laptop`, `perfume`, `sunglasses`, `watch`.
            """
        )
    with c2:
        st.markdown(
            """
            ### 2. Compara
            La app trae productos similares y los ordena
            de **más barato a más caro**.
            """
        )
    with c3:
        st.markdown(
            """
            ### 3. Ahorra
            Destacamos en verde el **precio más bajo** y
            calculamos cuánto ahorras frente al más caro.
            """
        )
    st.markdown("---")
    st.markdown("#### Pruebas rápidas")
    quick = ["phone", "laptop", "watch", "perfume", "sunglasses", "shoes"]
    cols = st.columns(len(quick))
    for col, term in zip(cols, quick):
        with col:
            if st.button(term.capitalize(), use_container_width=True, key=f"quick_{term}"):
                st.session_state["last_query"] = term
                st.session_state["trigger_search"] = True
                st.rerun()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    render_header()
    cfg = render_sidebar()

    query = st.text_input(
        "¿Qué producto quieres comparar?",
        value=st.session_state.get("last_query", ""),
        placeholder="Ej: iphone · laptop · perfume · running shoes · sunglasses",
        label_visibility="collapsed",
        key="query_input",
    )
    st.session_state["last_query"] = query

    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        do_search = st.button("🔎 Buscar", type="primary", use_container_width=True)
    with col_info:
        st.caption(
            "🌐 Datos en vivo · sin registros ni API keys · "
            "más de 200 productos en múltiples categorías."
        )

    trigger = do_search or st.session_state.pop("trigger_search", False)

    if not query.strip():
        render_intro()
        return

    if not trigger and query == st.session_state.get("searched_query"):
        items = st.session_state.get("last_items")
        if items is None:
            render_intro()
            return
    else:
        try:
            with st.spinner(f"Buscando «{query}»…"):
                items = search_products(query.strip(), limit=cfg["limit"])
        except requests.RequestException as exc:
            st.error(f"No se pudo conectar con el servicio de búsqueda: {exc}")
            return
        st.session_state["searched_query"] = query
        st.session_state["last_items"] = items

    if not items:
        st.warning(
            f"No encontramos resultados para «{query}». "
            "Prueba con otro término (en inglés funciona mejor: phone, laptop, watch, shoes…)."
        )
        return

    filtered = apply_filters(items, cfg)
    if not filtered:
        st.warning(
            "Los filtros activos dejan la lista vacía. Ajusta los filtros en la barra lateral."
        )
        return

    st.markdown("---")
    render_stats(filtered)
    st.markdown(f"### 📋 Resultados ({len(filtered)})")

    cheapest_price = min(i["price"] for i in filtered)
    for idx, item in enumerate(filtered, start=1):
        render_product_card(item, idx, cheapest_price)

    with st.expander("📊 Ver como tabla / descargar CSV"):
        df = pd.DataFrame(filtered)[
            ["title", "brand", "category", "price", "original_price",
             "discount_pct", "rating", "stock"]
        ]
        df.columns = [
            "Producto", "Marca", "Categoría", "Precio",
            "Precio original", "Descuento %", "Puntuación", "Stock",
        ]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Descargar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"comparador_{quote_plus(query)}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
