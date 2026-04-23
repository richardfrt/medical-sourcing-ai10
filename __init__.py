"""MediSource AI - Clinical Spend Intelligence.

Paquete core de la aplicación (ingesta GUDID, embeddings, vector store,
pipeline de equivalencia clínica y UI de Streamlit).
"""

from medisource import _bootstrap as _bootstrap  # noqa: F401  (aplica patch SQLite al importar)

__version__ = "0.1.0"
