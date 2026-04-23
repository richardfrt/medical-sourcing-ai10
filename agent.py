"""Clinical Justification Agent (PRD §6, Sprint 3).

Usa GPT-4o (o el modelo configurado) como "Auditor Clínico" para decidir si
dos dispositivos médicos son intercambiables, devolviendo un informe
estructurado (Pydantic) listo para mostrar en la UI o exportar a PDF.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from medisource.config import DEFAULT_CHAT_MODEL_PREMIUM, get_settings
from medisource.schemas import EquivalenceAnalysis, MedicalDevice

log = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Fallo al invocar el agente clínico."""


SYSTEM_PROMPT = (
    "Eres el Auditor Clínico Senior de un hospital universitario. Tu tarea es "
    "determinar si dos dispositivos médicos extraídos de la base GUDID de la FDA "
    "son intercambiables en la práctica clínica.\n\n"
    "Criterios a evaluar SIEMPRE:\n"
    "1. Material de construcción y biocompatibilidad.\n"
    "2. Dimensiones críticas (French/Fr, gauge, longitud, diámetro, volumen).\n"
    "3. Mecanismo de esterilización.\n"
    "4. Indicación / uso clínico previsto.\n"
    "5. Contraindicaciones o diferencias que afecten a la seguridad del paciente.\n\n"
    "Reglas estrictas:\n"
    "- No inventes parámetros que no aparezcan en los datos aportados: si faltan, "
    "  decláralo en 'missing_data'.\n"
    "- Responde SIEMPRE en español clínico profesional.\n"
    "- Devuelve EXCLUSIVAMENTE un JSON válido conforme al esquema indicado."
)

JSON_SCHEMA_HINT = {
    "compatibility_score": "entero 0-100",
    "verdict": "uno de: EQUIVALENT | CONDITIONAL | NOT_EQUIVALENT",
    "executive_summary": "1-2 frases orientadas al CFO.",
    "similarities": ["lista de similitudes críticas para el clínico"],
    "differences": ["lista de diferencias clínicas relevantes"],
    "missing_data": ["campos no informados en el GUDID que faltaría revisar"],
    "clinical_recommendation": "veredicto final para el jefe de servicio médico.",
}


def _device_block(label: str, d: MedicalDevice) -> str:
    return (
        f"[{label}]\n"
        f"- UDI-DI: {d.deviceIdentifier}\n"
        f"- Marca: {d.brandName}\n"
        f"- Fabricante: {d.companyName}\n"
        f"- Referencia: {d.versionModelNumber or '(no informado)'}\n"
        f"- GMDN: {d.gmdnPTName or '(no informado)'} ({d.gmdnCode or 's/código'})\n"
        f"- Descripción técnica: {d.deviceDescription}\n"
    )


def _build_user_prompt(a: MedicalDevice, b: MedicalDevice) -> str:
    return (
        "Evalúa la intercambiabilidad clínica entre estos dos dispositivos.\n\n"
        f"{_device_block('DISPOSITIVO A - Producto actual', a)}\n"
        f"{_device_block('DISPOSITIVO B - Alternativa propuesta', b)}\n\n"
        "Devuelve únicamente un objeto JSON con exactamente estas claves:\n"
        f"{json.dumps(JSON_SCHEMA_HINT, ensure_ascii=False, indent=2)}"
    )


class ClinicalJustificationAgent:
    """Orquestador del razonamiento clínico para el RAG (Capa 3)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise AgentError(
                "Falta la API key de OpenAI para el agente clínico. "
                "Configúrala en OPENAI_API_KEY o en la barra lateral."
            )
        self.model = model or settings.chat_model or DEFAULT_CHAT_MODEL_PREMIUM
        self.temperature = float(temperature)

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise AgentError("El paquete 'openai' no está instalado.") from exc
        self._client = OpenAI(api_key=self.api_key)

    def analyze_equivalence(
        self,
        device_a: MedicalDevice,
        device_b: MedicalDevice,
    ) -> EquivalenceAnalysis:
        """Pide a GPT-4o un veredicto de equivalencia y lo valida con Pydantic."""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(device_a, device_b)},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise AgentError(f"Error llamando al modelo {self.model}: {exc}") from exc

        content = (resp.choices[0].message.content or "").strip()
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AgentError(f"El modelo devolvió un JSON inválido: {exc}\nContenido: {content[:400]}")

        try:
            return EquivalenceAnalysis(**payload)
        except Exception as exc:  # ValidationError
            raise AgentError(
                "La respuesta del modelo no cumple el esquema esperado.\n"
                f"Detalle: {exc}\nPayload: {json.dumps(payload, ensure_ascii=False)[:600]}"
            ) from exc
