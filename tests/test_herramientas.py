"""Pruebas del registro de herramientas: esquemas, validación, ejecución, contexto y enrutador.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import cargar_csv
from src.eda import VARIABLES_SEGMENTABLES, AnalisisIncidentes, correlaciones, perfil_calidad
from src.herramientas import (
    ORDEN_DESEMPATE,
    REGISTRO,
    ArgumentosInvalidos,
    Herramienta,
    HerramientaNoAutorizada,
    RegistroHerramientas,
    contexto_base,
    contexto_herramienta,
    enrutar_por_palabras,
    json_compacto,
    normalizar_texto,
    preparar_para_json,
    redondear_para_modelo,
)
from src.preguntas_demo import PREGUNTAS_DEMO
from src.verificacion import resolver_ruta
from tests.dobles_agente import analisis_agente

RUTA_CSV_REAL = Path(__file__).resolve().parents[1] / "data" / "incident_event_log.csv"


def contexto_legado(analisis: AnalisisIncidentes, herramienta: str) -> dict:
    """Copia literal de AnalisisIncidentes.contexto() anterior al registro, como referencia fija."""
    contexto = {
        "unidad_analisis": "Una fila por incidente; última actualización disponible",
        "origen": "CSV histórico de Incident Response Log; no son datos de TEUNO ni de ISSPOL",
        "control_archivo_completo": analisis.control,
        "filtro_prioridades": analisis.prioridades,
        "resumen_seleccion": analisis.resumen(),
        "limites": "Duración registrada no equivale a tiempo de caída ni prueba causalidad. No hay logs de aplicación ni despliegues.",
    }
    # Lista cerrada de herramientas: el LLM no ejecuta código libre.
    if herramienta == "prioridades":
        contexto["comparacion_prioridades"] = json.loads(analisis.prioridades_resumen().to_json(orient="records"))
    elif herramienta == "calidad":
        contexto["calidad_incidentes_seleccionados"] = perfil_calidad(analisis.datos).to_dict("records")
    elif herramienta == "relaciones":
        contexto["correlaciones_descriptivas"] = correlaciones(analisis.datos)
    elif herramienta != "resumen":
        raise ValueError("Herramienta no autorizada.")
    return contexto


@pytest.fixture
def analisis() -> AnalisisIncidentes:
    return analisis_agente()


@pytest.fixture(scope="module")
def analisis_real() -> AnalisisIncidentes:
    if not RUTA_CSV_REAL.exists():
        pytest.skip("Falta data/incident_event_log.csv")
    datos = cargar_csv(RUTA_CSV_REAL.read_bytes())
    prioridades = sorted(datos.incidentes["priority"].dropna().unique().tolist())
    return AnalisisIncidentes(datos.incidentes, datos.control, prioridades)


# Paso 1: registro y esquemas


def test_registro_tiene_cinco_herramientas_en_orden():
    assert REGISTRO.nombres() == ("resumen", "prioridades", "calidad", "relaciones", "segmentar")
    assert set(ORDEN_DESEMPATE) == set(REGISTRO.nombres())
    assert all(nombre in REGISTRO for nombre in REGISTRO.nombres())
    assert "shell" not in REGISTRO and ["resumen"] not in REGISTRO


def test_etiquetas_en_espanol_para_la_interfaz():
    etiquetas = REGISTRO.etiquetas()
    assert list(etiquetas) == list(REGISTRO.nombres())
    assert etiquetas["segmentar"] == "Segmentar por categoría o grupo"
    assert etiquetas["resumen"] == "Resumen general"


def test_esquemas_siguen_formato_ollama():
    esquemas = REGISTRO.esquemas_ollama()
    assert [e["function"]["name"] for e in esquemas] == list(REGISTRO.nombres())
    for esquema in esquemas:
        assert esquema["type"] == "function"
        funcion = esquema["function"]
        assert set(funcion) == {"name", "description", "parameters"}
        assert funcion["parameters"]["type"] == "object"
        assert isinstance(funcion["parameters"]["properties"], dict)
        assert isinstance(funcion["parameters"]["required"], list)
    sin_argumentos = next(e for e in esquemas if e["function"]["name"] == "resumen")
    assert sin_argumentos["function"]["parameters"] == {"type": "object", "properties": {}, "required": []}


def test_esquemas_caben_en_presupuesto():
    esquemas = REGISTRO.esquemas_ollama()
    assert len(json_compacto(esquemas)) <= 1600
    assert all(len(e["function"]["description"]) <= 100 for e in esquemas)


def test_enum_segmentar_deriva_de_eda():
    parametros = REGISTRO.obtener("segmentar").parametros
    assert parametros["properties"]["variable"]["enum"] == list(VARIABLES_SEGMENTABLES)
    assert parametros["required"] == ["variable"]
    assert (parametros["properties"]["top"]["minimum"], parametros["properties"]["top"]["maximum"]) == (3, 10)


def test_registro_rechaza_nombres_duplicados():
    herramienta = REGISTRO.obtener("resumen")
    with pytest.raises(ValueError):
        RegistroHerramientas([herramienta, herramienta])


# Paso 2: validación de nombre y argumentos


def test_validar_herramienta_inexistente():
    with pytest.raises(HerramientaNoAutorizada):
        REGISTRO.validar("shell", {})
    assert issubclass(HerramientaNoAutorizada, ValueError)
    assert issubclass(ArgumentosInvalidos, ValueError)
    with pytest.raises(HerramientaNoAutorizada):
        REGISTRO.obtener(None)  # type: ignore[arg-type]


def test_validar_enum_fuera_de_lista():
    with pytest.raises(ArgumentosInvalidos):
        REGISTRO.validar("segmentar", {"variable": "priority"})


@pytest.mark.parametrize("top, esperado", [
    (50, None), (-1, None), (True, None), ("cinco", None), (2, None), (11, None), (5.5, None),
    (np.bool_(True), None), (np.int64(50), None),
    ("5", 5), (5.0, 5), (3, 3), (10, 10), (np.int64(5), 5), (np.int32(10), 10), (np.float64(4.0), 4),
])
def test_validar_rangos_y_tipos_de_top(top, esperado):
    if esperado is None:
        with pytest.raises(ArgumentosInvalidos):
            REGISTRO.validar("segmentar", {"variable": "category", "top": top})
    else:
        argumentos, _ = REGISTRO.validar("segmentar", {"variable": "category", "top": top})
        assert argumentos["top"] == esperado
        assert type(argumentos["top"]) is int


def test_validar_aplica_defecto_y_descarta_claves_extra():
    assert REGISTRO.validar("segmentar", {"variable": "category", "pregunta": "x"}) == (
        {"variable": "category", "top": 5}, ("pregunta",))
    assert REGISTRO.validar("segmentar", {"variable": "category", "top": None}) == ({"variable": "category", "top": 5}, ())


def test_validar_argumentos_nulos_o_no_objeto():
    assert REGISTRO.validar("resumen", None) == ({}, ())
    assert REGISTRO.validar("calidad", {"extra": 1}) == ({}, ("extra",))
    with pytest.raises(ArgumentosInvalidos):
        REGISTRO.validar("segmentar", None)
    for invalido in ("no-json", ["category"], 5):
        with pytest.raises(ArgumentosInvalidos):
            REGISTRO.validar("resumen", invalido)


# Paso 3: contexto y paridad con la implementación legada


@pytest.mark.parametrize("herramienta", ["resumen", "prioridades", "calidad", "relaciones"])
@pytest.mark.parametrize("prioridades", [["1 - Critical"], []], ids=["critico", "vacio"])
def test_registro_reproduce_contexto_legado(herramienta, prioridades):
    analisis = analisis_agente(prioridades)
    assert contexto_herramienta(analisis, herramienta) == contexto_legado(analisis, herramienta)
    json.dumps(contexto_herramienta(analisis, herramienta), allow_nan=False)


def test_contexto_base_es_el_dict_legado(analisis):
    base = contexto_base(analisis)
    assert base == contexto_legado(analisis, "resumen")
    assert list(base) == ["unidad_analisis", "origen", "control_archivo_completo", "filtro_prioridades",
                          "resumen_seleccion", "limites"]


def test_eda_contexto_delega_en_registro(analisis):
    assert analisis.contexto("prioridades") == contexto_herramienta(analisis, "prioridades")
    with pytest.raises(ValueError):
        analisis.contexto("shell")
    with pytest.raises(ValueError):
        contexto_herramienta(analisis, "shell")


def test_ejecutar_segmentar_usa_clave_por_variable(analisis):
    resultado = REGISTRO.ejecutar("segmentar", {"variable": "category"}, analisis)
    assert resultado.clave == "segmentacion_category"
    assert resultado.argumentos == {"variable": "category", "top": 5}
    contexto = contexto_herramienta(analisis, "segmentar", {"variable": "assignment_group", "top": 3})
    assert "segmentacion_assignment_group" in contexto
    assert REGISTRO.obtener("segmentar").clave_seccion({"variable": "category", "top": 5}) == "segmentacion_category"
    assert REGISTRO.obtener("resumen").clave_seccion({}) is None


def test_segmentar_entrega_ranking_y_resto_sin_fila_otros(analisis):
    seccion = REGISTRO.ejecutar("segmentar", {"variable": "assignment_group", "top": 3}, analisis).seccion
    assert seccion["variable"] == "assignment_group"
    assert seccion["top"] == 3
    assert seccion["segmentos_totales"] == 5
    assert [s["segmento"] for s in seccion["segmentos"]] == ["Group 70", "Group 25", "(sin dato)"]
    assert [s["incidentes"] for s in seccion["segmentos"]] == [3, 2, 1]
    assert seccion["resto"] == {"segmentos": 2, "incidentes": 2, "porcentaje_incidentes": 25.0}
    assert set(seccion["segmentos"][0]) >= {"segmento", "incidentes", "porcentaje_incidentes", "mediana_horas"}
    json.dumps(seccion, allow_nan=False)


def test_segmentar_sin_resto_deja_resto_nulo(analisis):
    seccion = REGISTRO.ejecutar("segmentar", {"variable": "category", "top": 10}, analisis).seccion
    assert seccion["segmentos_totales"] == 4
    assert len(seccion["segmentos"]) == 4
    assert seccion["resto"] is None


def test_ejecutar_devuelve_ms(analisis):
    resultado = REGISTRO.ejecutar("calidad", None, analisis)
    assert resultado.ms >= 0
    assert resultado.nombre == "calidad"
    assert resultado.clave == "calidad_incidentes_seleccionados"
    with pytest.raises(HerramientaNoAutorizada):
        REGISTRO.ejecutar("shell", {}, analisis)


def test_herramientas_son_inmutables():
    herramienta = REGISTRO.obtener("calidad")
    assert isinstance(herramienta, Herramienta)
    with pytest.raises(AttributeError):
        herramienta.nombre = "shell"  # type: ignore[misc]


# Paso 4: serialización y redondeo para el modelo


def test_preparar_para_json_elimina_nan_y_tipos_numpy():
    valor = {
        "entero": np.int64(7), "flotante": np.float64(1.5), "nan": float("nan"), "inf": np.float64("inf"),
        "na": pd.NA, "nat": pd.NaT, "fecha": pd.Timestamp("2016-02-29 08:00"), "booleano": np.bool_(True),
        "tupla": (1, np.float32(2.5)), "arreglo": np.array([1, 2]), 3: "clave entera",
    }
    preparado = preparar_para_json(valor)
    texto = json.dumps(preparado, allow_nan=False)
    assert json.loads(texto) == {
        "entero": 7, "flotante": 1.5, "nan": None, "inf": None, "na": None, "nat": None,
        "fecha": "2016-02-29T08:00:00", "booleano": True, "tupla": [1, 2.5], "arreglo": [1, 2], "3": "clave entera",
    }
    assert type(preparado["entero"]) is int and type(preparado["booleano"]) is bool


def test_redondear_para_modelo():
    assert redondear_para_modelo(187.678) == 187.68
    assert redondear_para_modelo(0.20378) == 0.204
    assert redondear_para_modelo(-0.12345) == -0.123
    assert redondear_para_modelo(20769) == 20769 and type(redondear_para_modelo(20769)) is int
    assert redondear_para_modelo(True) is True
    assert redondear_para_modelo({"a": [1.23456, {"b": 0.98765}], "c": "texto"}) == {"a": [1.23, {"b": 0.988}], "c": "texto"}


def test_json_compacto_sin_espacios_ni_nan():
    assert json_compacto({"categoría": [1, 2.5]}) == '{"categoría":[1,2.5]}'
    with pytest.raises(ValueError):
        json_compacto({"x": math.nan})


def test_normalizar_texto():
    assert normalizar_texto("  ¿Qué  CATEGORÍAS\n tienen  más? ") == "¿que categorias tienen mas?"


# Paso 5: enrutador por palabras clave


@pytest.mark.parametrize("pregunta", [p for p in PREGUNTAS_DEMO if p.herramientas_esperadas], ids=lambda p: p.id)
def test_enrutador_palabras_clave(pregunta):
    ruta = enrutar_por_palabras(pregunta.texto)
    assert ruta is not None
    nombre, argumentos = ruta
    assert nombre in pregunta.herramientas_esperadas
    if len(pregunta.herramientas_esperadas) == 1:
        assert argumentos == pregunta.argumentos_esperados[nombre]


def test_enrutador_sin_coincidencias_devuelve_none():
    assert enrutar_por_palabras("Hola") is None
    assert enrutar_por_palabras("¿Qué tal estás hoy?") is None


def test_enrutador_desempate_por_especificidad():
    assert enrutar_por_palabras("Mediana por prioridad") == ("prioridades", {})
    assert enrutar_por_palabras("Mediana de las categorías") == ("segmentar", {"variable": "category", "top": 5})


def test_enrutador_variable_segun_orden_de_aparicion():
    assert enrutar_por_palabras("Ranking de grupos y categorías")[1]["variable"] == "assignment_group"
    assert enrutar_por_palabras("Ranking de categorías y grupos")[1]["variable"] == "category"
    assert enrutar_por_palabras("Ranking top de incidentes")[1]["variable"] == "category"


def test_enrutador_extrae_top_en_cifras_y_palabras():
    assert enrutar_por_palabras("Top 7 categorías")[1]["top"] == 7
    assert enrutar_por_palabras("Las cinco categorías con más incidentes")[1]["top"] == 5
    assert enrutar_por_palabras("Los diez grupos con más incidentes")[1]["top"] == 10
    assert enrutar_por_palabras("Top 50 categorías")[1]["top"] == 5
    assert enrutar_por_palabras("Top 2 o 8 grupos")[1]["top"] == 8


def test_enrutador_usa_el_registro_recibido():
    solo_calidad = RegistroHerramientas([REGISTRO.obtener("calidad")])
    assert enrutar_por_palabras("Compara prioridades", solo_calidad) is None
    assert enrutar_por_palabras("Valores faltantes", solo_calidad) == ("calidad", {})


# Paso 6: presupuesto de contexto con el CSV real


@pytest.mark.csv_real
def test_claves_cifras_de_preguntas_demo_resuelven_en_el_csv_real(analisis_real):
    esperados = {
        "resumen_seleccion.incidentes_seleccionados": 20769, "resumen_seleccion.mediana_horas": 24.6,
        "resumen_seleccion.media_horas": 187.68, "comparacion_prioridades[0].mediana_horas": 78.17,
        "comparacion_prioridades[2].incidentes": 19554, "correlaciones_descriptivas[1].spearman_con_duracion": 0.523,
        "segmentacion_assignment_group.segmentos[0].incidentes": 7369,
        "segmentacion_assignment_group.segmentos[0].mediana_horas": 0.35,
        "calidad_incidentes_seleccionados[3].nulos": 2157,
    }
    for pregunta in PREGUNTAS_DEMO:
        contexto = contexto_base(analisis_real)
        for nombre in pregunta.herramientas_esperadas:
            resultado = REGISTRO.ejecutar(nombre, pregunta.argumentos_esperados[nombre], analisis_real)
            if resultado.clave:
                contexto[resultado.clave] = resultado.seccion
        enviado = redondear_para_modelo(preparar_para_json(contexto))
        for ruta in pregunta.claves_cifras:
            assert resolver_ruta(enviado, ruta) == pytest.approx(esperados[ruta]), (pregunta.id, ruta)


@pytest.mark.csv_real
def test_contexto_peor_caso_cabe_en_presupuesto(analisis_real):
    contexto = contexto_base(analisis_real)
    for nombre, argumentos in (("calidad", {}), ("segmentar", {"variable": "assignment_group", "top": 10})):
        resultado = REGISTRO.ejecutar(nombre, argumentos, analisis_real)
        contexto[resultado.clave] = resultado.seccion
    assert len(json_compacto(redondear_para_modelo(preparar_para_json(contexto)))) <= 9000
