"""Contrato entre módulos: lo que la interfaz consume de eda, visualizaciones, agente y observabilidad.

Comprueba con datos sintéticos que las salidas de cada módulo alimentan al siguiente sin
adaptadores, que los ayudantes de app.py entienden los registros reales del agente y que los
símbolos citados en la pestaña Arquitectura existen. La prueba con el CSV real lleva el
marcador `csv_real`.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

import app
from src.agent import AgenteIncidentes
from src.data_loader import DatosIncidentes, cargar_ruta
from src.diccionario import VARIABLES_EXPLORAR, VARIABLES_SEGMENTO
from src.eda import AnalisisIncidentes
from src.observabilidad import (
    COLUMNAS_TABLA_SLOS, COLUMNAS_ULTIMAS_CONSULTAS, MuestraConsulta, evaluar_slos, tabla_slos, ultimas_consultas,
)
from src.visualizations import (
    figura_a_png, figura_boxplot_prioridad, figura_mapa_spearman, grafico_duracion, grafico_prioridades,
    grafico_segmentos, grafico_variable,
)
from tests.dobles_agente import ClienteFalso, analisis_agente, respuesta_con_llamadas

RUTA_CSV_REAL = Path(__file__).resolve().parents[1] / "data" / "incident_event_log.csv"


def _resolver(simbolo: str) -> object:
    """Objeto nombrado por "modulo.atributo[.atributo]", probando el prefijo de módulo más largo."""
    partes = simbolo.split(".")
    for corte in range(len(partes) - 1, 0, -1):
        try:
            objeto: object = importlib.import_module(".".join(partes[:corte]))
        except ModuleNotFoundError:
            continue
        for nombre in partes[corte:]:
            objeto = getattr(objeto, nombre)
        return objeto
    raise LookupError(simbolo)


def test_salidas_de_eda_alimentan_visualizaciones(datos_sinteticos: DatosIncidentes) -> None:
    analisis = AnalisisIncidentes(datos_sinteticos.incidentes, datos_sinteticos.control,
                                  ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"])
    figuras_plotly = [grafico_prioridades(analisis.prioridades_resumen()), grafico_duracion(analisis.datos)]
    figuras_plotly += [grafico_variable(analisis.datos, v) for v in VARIABLES_EXPLORAR]
    figuras_plotly += [grafico_segmentos(analisis.segmentar(v, 2), v) for v in VARIABLES_SEGMENTO]
    assert all(isinstance(f, go.Figure) for f in figuras_plotly)
    figuras = [figura_boxplot_prioridad(analisis.datos), figura_mapa_spearman(analisis.matriz_relaciones())]
    assert all(isinstance(f, Figure) for f in figuras)
    assert all(figura_a_png(f).startswith(b"\x89PNG") for f in figuras)


@pytest.mark.csv_real
def test_figuras_con_csv_real() -> None:
    datos = cargar_ruta(RUTA_CSV_REAL)
    analisis = AnalisisIncidentes(datos.incidentes, datos.control, sorted(datos.incidentes["priority"].unique()))
    boxplot = figura_boxplot_prioridad(analisis.datos)
    assert len(boxplot.axes[0].get_xticklabels()) == 4
    mapa = figura_mapa_spearman(analisis.matriz_relaciones())
    assert len(mapa.axes[0].texts) == 10


def test_registro_del_agente_alimenta_traza_insignias_y_descarga_de_la_interfaz() -> None:
    muestras: list[MuestraConsulta] = []
    cliente = ClienteFalso([respuesta_con_llamadas(("segmentar", {"variable": "assignment_group", "top": 3}))])
    registro = AgenteIncidentes(analisis_agente(), cliente, al_finalizar=muestras.append).responder(
        "¿Qué tres grupos concentran más incidentes?")
    filas = app.filas_traza(registro.traza)
    assert any(f["Herramienta"] == "segmentar" and "assignment_group" in f["Argumentos"] for f in filas)
    assert all(set(f) == set(app.COLUMNAS_TRAZA) for f in filas)
    color, texto = app.insignia_cifras(registro.verificacion_cifras)
    assert color in {"green", "orange", "gray"} and texto
    assert app.ORIGENES[registro.origen_seleccion]
    assert json.loads(registro.a_json())["trace_id"] == registro.trace_id
    assert [m.trace_id for m in muestras] == [registro.trace_id]


def test_insignia_de_cifras_verde_naranja_y_gris() -> None:
    assert app.insignia_cifras({"evaluadas": 2, "verificadas": 2}) == ("green", "Cifras verificadas: 2/2")
    assert app.insignia_cifras({"evaluadas": 3, "verificadas": 1}) == ("orange", "Cifras verificadas: 1/3")
    assert app.insignia_cifras({"evaluadas": 0, "verificadas": 0}) == ("gray", "Sin cifras que verificar")


def test_observabilidad_entrega_las_columnas_que_la_interfaz_muestra() -> None:
    usadas_slo = {"slo", "descripcion", "objetivo", "eventos_validos", "eventos_buenos", "cumplimiento",
                  "resultado", "fraccion_restante"}
    assert usadas_slo <= set(COLUMNAS_TABLA_SLOS)
    tabla = tabla_slos(evaluar_slos([]))
    assert len(tabla) == 3 and set(tabla["resultado"]) == {"sin eventos"}
    assert {"trace_id", "evento", "ms", "estado"} <= set(COLUMNAS_ULTIMAS_CONSULTAS)
    assert ultimas_consultas([]).empty


def test_insignias_de_integridad_cubren_los_tres_estados() -> None:
    assert set(app.INSIGNIAS_INTEGRIDAD) == {"coincide", "difiere", "no_aplica"}


def test_simbolos_citados_en_arquitectura_existen() -> None:
    simbolos = [s for _, nombres, _ in app.PATRONES for s in nombres]
    assert len(app.PATRONES) == 9
    for simbolo in simbolos:
        assert _resolver(simbolo) is not None, simbolo
    assert all(problema.strip() for _, _, problema in app.PATRONES)
    assert "REGISTRO = RegistroHerramientas(" in app.fuente_registro()
