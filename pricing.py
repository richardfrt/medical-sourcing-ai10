"""Simulación determinista de precios unitarios y cálculo de ahorros.

El dataset GUDID no contiene precios; para el MVP derivamos un precio unitario
reproducible a partir del `deviceIdentifier`. Cuando el cliente nos aporte
su Item Master real, este módulo se sustituye por una lectura a su ERP.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from medisource.config import DEFAULT_PRICE_MAX, DEFAULT_PRICE_MIN


def deterministic_price(
    device_identifier: str,
    *,
    minimo: float = DEFAULT_PRICE_MIN,
    maximo: float = DEFAULT_PRICE_MAX,
) -> float:
    """Precio unitario reproducible a partir del UDI-DI."""
    if not device_identifier:
        return 0.0
    h = hashlib.sha256(device_identifier.encode("utf-8")).digest()
    n = int.from_bytes(h[:4], "big") / 2**32
    return round(minimo + n * (maximo - minimo), 2)


@dataclass(frozen=True)
class SavingsEstimate:
    """Resumen económico de una sustitución A -> B."""

    unit_price_a: float
    unit_price_b: float
    unit_savings: float
    unit_savings_pct: float
    annual_volume: int
    annual_cost_a: float
    annual_cost_b: float
    annual_savings: float
    annual_savings_pct: float


def estimate_savings(price_a: float, price_b: float, annual_volume: int) -> SavingsEstimate:
    """Calcula el ahorro por unidad y anualizado de sustituir A por B."""
    price_a = max(0.0, float(price_a))
    price_b = max(0.0, float(price_b))
    annual_volume = max(0, int(annual_volume))

    unit_savings = round(price_a - price_b, 2)
    unit_savings_pct = round((unit_savings / price_a) * 100.0, 2) if price_a > 0 else 0.0

    annual_cost_a = round(price_a * annual_volume, 2)
    annual_cost_b = round(price_b * annual_volume, 2)
    annual_savings = round(annual_cost_a - annual_cost_b, 2)
    annual_savings_pct = (
        round((annual_savings / annual_cost_a) * 100.0, 2) if annual_cost_a > 0 else 0.0
    )

    return SavingsEstimate(
        unit_price_a=price_a,
        unit_price_b=price_b,
        unit_savings=unit_savings,
        unit_savings_pct=unit_savings_pct,
        annual_volume=annual_volume,
        annual_cost_a=annual_cost_a,
        annual_cost_b=annual_cost_b,
        annual_savings=annual_savings,
        annual_savings_pct=annual_savings_pct,
    )


def format_eur(value: float) -> str:
    """Formatea en euros (es-ES): 12.345,67 €."""
    try:
        s = f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)
    return s.replace(",", "X").replace(".", ",").replace("X", ".") + " €"
