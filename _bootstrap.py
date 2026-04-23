"""Parche de compatibilidad SQLite para Streamlit Cloud (ChromaDB).

Debe importarse ANTES que `chromadb` en cualquier entrada del programa.
En entornos donde `pysqlite3-binary` no esté instalado (ej. Windows local)
se ignora silenciosamente y se usa el sqlite3 de la biblioteca estándar.
"""

from __future__ import annotations

import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass
