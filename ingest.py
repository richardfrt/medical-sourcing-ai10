"""Ingesta y normalización del CSV GUDID.

Acepta tanto el formato original de la FDA (columnas en inglés: `primary_di`,
`brand_name`, ...) como el CSV traducido que produce `gudid_filter.py`
(columnas en español: "Nombre Comercial", "Descripción Técnica", ...).

Devuelve objetos `MedicalDevice` ya validados por Pydantic.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from medisource.pricing import deterministic_price
from medisource.schemas import MedicalDevice

log = logging.getLogger(__name__)


COLUMN_ALIASES = {
    "deviceIdentifier": [
        "deviceIdentifier", "deviceidentifier", "primary_di", "primarydi",
        "udi-di", "udidi", "di", "id",
    ],
    "companyName": [
        "companyName", "companyname", "company_name", "fabricante", "manufacturer",
    ],
    "brandName": [
        "brandName", "brandname", "brand_name", "nombre comercial", "nombre_comercial",
        "brand",
    ],
    "versionModelNumber": [
        "versionModelNumber", "versionmodelnumber", "version_model_number",
        "modelo", "model", "referencia", "ref",
    ],
    "gmdnPTName": [
        "gmdnPTName", "gmdnptname", "gmdn_pt_name", "gmdn", "gmdn term",
        "código gmdn", "codigo gmdn", "código gmdn (categoría global)",
        "gmdn_name", "gmdn_term_name",
    ],
    "gmdnCode": [
        "gmdnCode", "gmdncode", "gmdn_code", "código gmdn numérico",
    ],
    "deviceDescription": [
        "deviceDescription", "devicedescription", "device_description",
        "descripción técnica", "descripcion tecnica", "descripción",
        "descripcion", "description",
    ],
}


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _build_column_map(df_columns: Iterable[str]) -> dict[str, str]:
    """Para cada campo canónico, busca la primera columna real que haga match."""
    normalized = {col: _norm(col) for col in df_columns}
    reverse: dict[str, str] = {v: k for k, v in normalized.items()}

    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _norm(alias)
            if key in reverse:
                mapping[canonical] = reverse[key]
                break
    return mapping


def _maybe_split_gmdn(value: str) -> tuple[str, str]:
    """El CSV de `gudid_filter.py` guarda 'Código GMDN' como 'nombre | codigo'.
    Separamos nombre y código numérico si viene mezclado.
    """
    if not value:
        return "", ""
    if "|" in value:
        parts = [p.strip() for p in value.split("|", 1)]
        name, code = parts[0], parts[1] if len(parts) > 1 else ""
        return name, code
    # "1234 - Catheter" o "Catheter (1234)"
    m = re.match(r"^\s*(\d{3,})\s*[-–—:]\s*(.+)$", value)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    m = re.match(r"^(.*?)\((\d{3,})\)\s*$", value)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return value.strip(), ""


def read_devices_from_csv(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> List[MedicalDevice]:
    """Lee un CSV GUDID y devuelve la lista de MedicalDevice validados."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo CSV: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=max_rows)
    mapping = _build_column_map(df.columns)

    required = {"deviceIdentifier", "brandName", "companyName", "deviceDescription"}
    missing = required - mapping.keys()
    if missing:
        raise ValueError(
            f"El CSV no contiene columnas requeridas: {sorted(missing)}. "
            f"Columnas detectadas: {list(df.columns)}"
        )

    devices: List[MedicalDevice] = []
    seen_ids: set[str] = set()

    for row in df.itertuples(index=False):
        data = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))

        def get(key: str, default: str = "") -> str:
            col = mapping.get(key)
            if not col:
                return default
            return str(data.get(col, default) or "").strip()

        device_id = get("deviceIdentifier")
        if not device_id or device_id in seen_ids:
            continue

        gmdn_pt = get("gmdnPTName")
        gmdn_code = get("gmdnCode")
        if not gmdn_code and gmdn_pt:
            gmdn_pt, gmdn_code = _maybe_split_gmdn(gmdn_pt)

        description = get("deviceDescription") or get("brandName")

        try:
            device = MedicalDevice(
                deviceIdentifier=device_id,
                companyName=get("companyName") or "Desconocido",
                brandName=get("brandName") or "Desconocido",
                versionModelNumber=get("versionModelNumber"),
                gmdnPTName=gmdn_pt,
                gmdnCode=gmdn_code,
                deviceDescription=description,
                estimated_price=deterministic_price(device_id),
            )
        except Exception as exc:  # Pydantic ValidationError u otros
            log.debug("Fila descartada (%s): %s", device_id, exc)
            continue

        seen_ids.add(device_id)
        devices.append(device)

    return devices


def build_embedding_text(device: MedicalDevice) -> str:
    """Construye el texto enriquecido que realmente vectorizamos.

    Incluir marca, fabricante y GMDN en el texto mejora la recuperación
    semántica cuando el cliente busca con términos comerciales.
    """
    partes = [
        device.deviceDescription,
        f"Categoría clínica: {device.gmdnPTName}" if device.gmdnPTName else "",
        f"Fabricante: {device.companyName}" if device.companyName else "",
        f"Marca: {device.brandName}" if device.brandName else "",
        f"Modelo: {device.versionModelNumber}" if device.versionModelNumber else "",
    ]
    return " | ".join(p for p in partes if p)
