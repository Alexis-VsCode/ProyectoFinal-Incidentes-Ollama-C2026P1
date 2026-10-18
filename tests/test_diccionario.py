"""Pruebas del contrato compartido de variables (src/diccionario.py).

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src import diccionario
from src.data_loader import COLUMNAS, DatosIncidentes
from src.diccionario import (
    DICCIONARIO, DURACION, ETIQUETAS, MIN_MUESTRA, OTROS, PISO_LOG_HORAS, SIN_DATO,
    TIPOS, TIPOS_ANALITICOS, VARIABLES_EXPLORAR, VARIABLES_RELACION, VARIABLES_SEGMENTO,
    Variable, etiqueta, tabla_diccionario,
)


def test_diccionario_cubre_las_13_columnas_analiticas() -> None:
    nombres = tuple(v.nombre for v in DICCIONARIO)
    assert nombres == COLUMNAS + (DURACION,)
    assert len(nombres) == 13
    assert len(set(nombres)) == len(nombres)


def test_etiquetas_y_descripciones_en_espanol_no_vacias() -> None:
    for variable in DICCIONARIO:
        assert variable.etiqueta.strip()
        assert variable.descripcion.strip()
        assert variable.origen in {"CSV", "Derivada"}
    assert ETIQUETAS[DURACION] == "Horas hasta resolución"
    assert ETIQUETAS["priority"] == "Prioridad"
    assert "MTTR" in next(v for v in DICCIONARIO if v.nombre == DURACION).descripcion


def test_descripcion_de_duracion_niega_slo_sla_y_mttr() -> None:
    descripcion = next(v for v in DICCIONARIO if v.nombre == DURACION).descripcion
    assert "no es SLO, SLA ni MTTR" in descripcion
    assert tabla_diccionario().set_index("variable").loc[DURACION, "descripcion"] == descripcion


def test_tipos_analiticos_pertenecen_al_catalogo() -> None:
    assert len(TIPOS_ANALITICOS) == 6
    assert {v.tipo for v in DICCIONARIO} <= set(TIPOS_ANALITICOS)
    assert TIPOS["priority"] == "categórica ordinal"
    assert TIPOS[DURACION] == "continua derivada"
    assert TIPOS["sys_mod_count"] == "conteo discreto"


def test_tabla_diccionario_columnas_y_orden() -> None:
    tabla = tabla_diccionario()
    assert list(tabla.columns) == ["variable", "etiqueta", "tipo_analitico", "descripcion", "origen"]
    assert len(tabla) == 13
    assert tuple(tabla["variable"]) == COLUMNAS + (DURACION,)
    assert tabla.loc[tabla["variable"] == DURACION, "etiqueta"].item() == "Horas hasta resolución"


def test_variables_de_segmento_relacion_y_explorar_existen_en_diccionario() -> None:
    nombres = {v.nombre for v in DICCIONARIO}
    assert set(VARIABLES_SEGMENTO) <= nombres
    assert set(VARIABLES_RELACION) <= nombres
    assert set(VARIABLES_EXPLORAR) <= nombres
    assert VARIABLES_SEGMENTO == ("category", "assignment_group")
    assert VARIABLES_RELACION == ("sys_mod_count", "reassignment_count", "reopen_count")
    assert VARIABLES_EXPLORAR[0] == DURACION


def test_csv_sintetico_consolida_9_eventos_en_8_incidentes(datos_sinteticos: DatosIncidentes) -> None:
    assert len(datos_sinteticos.eventos) == 9
    assert len(datos_sinteticos.incidentes) == 8
    assert datos_sinteticos.control["incidentes_sin_duracion"] == 1
    inc01 = datos_sinteticos.incidentes.set_index("number").loc["INC01"]
    assert inc01["incident_state"] == "Closed"
    assert inc01[DURACION] == 2.0


def test_constantes_textuales_y_umbrales_del_contrato() -> None:
    assert SIN_DATO == "(sin dato)"
    assert OTROS == "(otras categorías)"
    assert MIN_MUESTRA == 30
    assert PISO_LOG_HORAS == pytest.approx(1 / 60)


def test_etiqueta_devuelve_el_nombre_si_la_variable_no_existe() -> None:
    assert etiqueta("reopen_count") == "Reaperturas"
    assert etiqueta("columna_inexistente") == "columna_inexistente"


def test_variable_es_inmutable() -> None:
    variable = DICCIONARIO[0]
    assert isinstance(variable, Variable)
    with pytest.raises(FrozenInstanceError):
        variable.etiqueta = "otra"  # type: ignore[misc]


def test_tabla_diccionario_devuelve_copia_independiente() -> None:
    tabla = tabla_diccionario()
    tabla.loc[0, "etiqueta"] = "modificada"
    assert tabla_diccionario().loc[0, "etiqueta"] == "Identificador del incidente"
    assert diccionario.ETIQUETAS["number"] == "Identificador del incidente"
