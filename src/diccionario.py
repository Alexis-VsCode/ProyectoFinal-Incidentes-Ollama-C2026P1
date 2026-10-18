"""Contrato compartido de variables: nombres, etiquetas en español, tipos analíticos y constantes.

No depende de Streamlit, de Ollama ni de los demás módulos de src; lo consumen
data_loader, eda, visualizations y la interfaz.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

DURACION: Final = "tiempo_resolucion_horas"
SIN_DATO: Final = "(sin dato)"
OTROS: Final = "(otras categorías)"
MIN_MUESTRA: Final = 30
PISO_LOG_HORAS: Final = 1 / 60
TIPOS_ANALITICOS: Final = (
    "identificador", "categórica ordinal", "categórica nominal",
    "fecha y hora", "conteo discreto", "continua derivada",
)
VARIABLES_SEGMENTO: Final = ("category", "assignment_group")
VARIABLES_RELACION: Final = ("sys_mod_count", "reassignment_count", "reopen_count")
VARIABLES_EXPLORAR: Final = (
    DURACION, "reassignment_count", "reopen_count", "sys_mod_count",
    "priority", "category", "assignment_group", "incident_state",
)
COLUMNAS_DICCIONARIO: Final = ("variable", "etiqueta", "tipo_analitico", "descripcion", "origen")


@dataclass(frozen=True)
class Variable:
    """Descripción inmutable de una columna analítica de la tabla de incidentes."""
    nombre: str
    etiqueta: str
    tipo: str
    descripcion: str
    origen: str


DICCIONARIO: Final[tuple[Variable, ...]] = (
    Variable("number", "Identificador del incidente", "identificador",
             "Código anonimizado que agrupa los eventos de un mismo incidente.", "CSV"),
    Variable("priority", "Prioridad", "categórica ordinal",
             "Prioridad calculada por la plataforma a partir de impacto y urgencia (1 - Critical a 4 - Low).", "CSV"),
    Variable("incident_state", "Estado del incidente", "categórica nominal",
             "Estado del proceso en la última actualización disponible.", "CSV"),
    Variable("assignment_group", "Grupo de soporte", "categórica nominal",
             "Identificador anonimizado del grupo de soporte a cargo.", "CSV"),
    Variable("category", "Categoría del servicio", "categórica nominal",
             "Descripción anonimizada de primer nivel del servicio afectado.", "CSV"),
    Variable("opened_at", "Fecha de apertura", "fecha y hora",
             "Apertura registrada (día/mes/año hora:minuto, sin zona horaria).", "CSV"),
    Variable("resolved_at", "Fecha de resolución", "fecha y hora", "Resolución registrada.", "CSV"),
    Variable("closed_at", "Fecha de cierre", "fecha y hora", "Cierre registrado.", "CSV"),
    Variable("sys_updated_at", "Última actualización", "fecha y hora",
             "Actualización que generó el evento; define la última observación del incidente.", "CSV"),
    Variable("reassignment_count", "Reasignaciones", "conteo discreto",
             "Veces que cambió el grupo o el analista asignado.", "CSV"),
    Variable("reopen_count", "Reaperturas", "conteo discreto",
             "Veces que el solicitante rechazó la resolución.", "CSV"),
    Variable("sys_mod_count", "Modificaciones del registro", "conteo discreto",
             "Número de actualizaciones del incidente hasta la última observación.", "CSV"),
    Variable(DURACION, "Horas hasta resolución", "continua derivada",
             "resolved_at − opened_at en horas; negativas y faltantes quedan sin dato. "
             "Describe el histórico: no es SLO, SLA ni MTTR, ni equivale a indisponibilidad.", "Derivada"),
)

ETIQUETAS: Final[dict[str, str]] = {v.nombre: v.etiqueta for v in DICCIONARIO}
TIPOS: Final[dict[str, str]] = {v.nombre: v.tipo for v in DICCIONARIO}


def etiqueta(variable: str) -> str:
    """Etiqueta en español de la variable; si no está en el diccionario, devuelve el nombre recibido."""
    return ETIQUETAS.get(variable, variable)


def tabla_diccionario() -> pd.DataFrame:
    """Tabla nueva en cada llamada, en el orden de DICCIONARIO, con las columnas COLUMNAS_DICCIONARIO."""
    return pd.DataFrame(
        [(v.nombre, v.etiqueta, v.tipo, v.descripcion, v.origen) for v in DICCIONARIO],
        columns=list(COLUMNAS_DICCIONARIO),
    )
