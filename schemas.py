"""Modelos Pydantic del dominio clínico.

Usados para:
- Validar filas del CSV GUDID (evita que datos sucios lleguen a la IA).
- Tipar las respuestas del LLM (el "Auditor Clínico" de Equivalencia).
- Homogeneizar los resultados de búsqueda mostrados en la UI.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MedicalDevice(BaseModel):
    """Representación canónica de un dispositivo médico GUDID."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    deviceIdentifier: str = Field(..., description="UDI-DI / Primary DI del producto.")
    companyName: str = Field(..., description="Fabricante / Company Name.")
    brandName: str = Field(..., description="Nombre comercial.")
    versionModelNumber: str = Field("", description="Referencia o número de modelo.")
    gmdnPTName: str = Field("", description="Nombre del término GMDN (categoría clínica).")
    gmdnCode: str = Field("", description="Código GMDN numérico (si está disponible).")
    deviceDescription: str = Field(..., description="Descripción técnica libre (se vectoriza).")
    estimated_price: float = Field(0.0, ge=0.0, description="Precio simulado (determinista).")

    @field_validator("deviceIdentifier", "brandName", "companyName", "deviceDescription")
    @classmethod
    def _no_vacios(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("campo obligatorio vacío")
        return v

    def to_metadata(self) -> dict:
        """Metadatos que se guardan junto al vector en ChromaDB."""
        return {
            "deviceIdentifier": self.deviceIdentifier,
            "companyName": self.companyName,
            "brandName": self.brandName,
            "versionModelNumber": self.versionModelNumber,
            "gmdnPTName": self.gmdnPTName,
            "gmdnCode": self.gmdnCode,
            "deviceDescription": self.deviceDescription,
            "estimated_price": float(self.estimated_price),
        }


class SearchHit(BaseModel):
    """Un candidato devuelto por el pipeline de búsqueda."""

    device: MedicalDevice
    similarity: float = Field(..., ge=0.0, le=1.0)
    price_delta_unit: float = 0.0
    price_delta_unit_pct: float = 0.0


EquivalenceVerdict = Literal["EQUIVALENT", "CONDITIONAL", "NOT_EQUIVALENT"]


class EquivalenceAnalysis(BaseModel):
    """Informe estructurado del Clinical Justification Agent (GPT-4o)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    compatibility_score: int = Field(..., ge=0, le=100, description="Porcentaje de compatibilidad 0-100.")
    verdict: EquivalenceVerdict = Field(..., description="Veredicto de intercambiabilidad.")
    executive_summary: str = Field(..., description="Resumen ejecutivo en 1-2 frases.")
    similarities: List[str] = Field(default_factory=list, description="Similitudes críticas.")
    differences: List[str] = Field(default_factory=list, description="Diferencias clínicas relevantes.")
    missing_data: List[str] = Field(default_factory=list, description="Datos no informados en el GUDID.")
    clinical_recommendation: str = Field("", description="Recomendación final al jefe de servicio médico.")

    @property
    def verdict_es(self) -> str:
        return {
            "EQUIVALENT": "Equivalente",
            "CONDITIONAL": "Equivalente con reservas",
            "NOT_EQUIVALENT": "No equivalente",
        }[self.verdict]


class AppError(BaseModel):
    """Mensaje estructurado para mostrar errores en UI sin crashear."""

    code: str
    message: str
    details: Optional[str] = None
