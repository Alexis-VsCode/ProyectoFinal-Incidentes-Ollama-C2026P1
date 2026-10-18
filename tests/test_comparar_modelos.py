"""Pruebas del comparador de modelos con el cliente falso: filas, resumen, decisión y archivos.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

import scripts.comparar_modelos as comparar
from scripts.comparar_modelos import (
    COLUMNAS_CSV,
    MODELOS_POR_DEFECTO,
    REPETICIONES_POR_DEFECTO,
    crear_parser,
    decidir_modelo,
    ejecutar_comparacion,
    fila_desde_registro,
    guardar,
    main,
    resumir,
)
from src.agent import AgenteIncidentes
from src.llm_client import ErrorOllama
from src.observabilidad import SLOS, MuestraConsulta, calcular_slis, leer_muestras, ruta_log
from src.preguntas_demo import PREGUNTAS_DEMO
from tests.dobles_agente import ClienteFalso, analisis_agente, respuesta_con_llamadas, respuesta_texto

POR_ID = {p.id: p for p in PREGUNTAS_DEMO}


def selector_perfecto(mensajes, herramientas):
    """Modelo ideal: llama exactamente a las herramientas esperadas de cada pregunta demo."""
    pregunta = next(p for p in PREGUNTAS_DEMO if p.texto == mensajes[-1]["content"])
    return respuesta_con_llamadas(*[(n, pregunta.argumentos_esperados[n]) for n in pregunta.herramientas_esperadas])


def selector_mudo(mensajes, herramientas):
    """Modelo que nunca llama herramientas: todo cae en el respaldo."""
    return respuesta_texto("No sé.")


def fabrica_falsa(selector=selector_perfecto, **opciones):
    def fabrica(modelo, *, seed=None):
        return ClienteFalso(selector, modelo=modelo, seed=seed, **opciones)
    return fabrica


def resumen_modelo(acierto, latencia, cifras=0.9, truncadas=0.0, errores=0):
    return {"slis": {"latencia_p50_s": latencia, "proporcion_cifras_verificadas": cifras,
                     "tasa_truncadas": truncadas, "errores": errores},
            "acierto": {"tasa_acierto_herramienta": acierto}}


@pytest.fixture
def ruta_comparacion(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "comparacion_prueba.jsonl"


# Paso 1: filas por consulta


def test_fila_desde_registro_sin_nan():
    q1 = POR_ID["Q1"]
    cliente = ClienteFalso([respuesta_con_llamadas(("resumen", {}))],
                           fragmentos=("Hay 8 incidentes; mediana 12 h y media 23,36 h.",), seed=101)
    registro = AgenteIncidentes(analisis_agente(), cliente).responder(q1.texto)
    fila = fila_desde_registro(registro, q1, "llama3.2:3b", 1, 101)
    assert tuple(fila) == COLUMNAS_CSV
    json.dumps(fila, allow_nan=False)
    assert (fila["pregunta_id"], fila["repeticion"], fila["semilla"], fila["estado"]) == ("Q1", 1, 101, "respondida")
    assert fila["herramientas_elegidas"] == fila["herramientas_esperadas"] == "resumen"
    assert fila["acierto_herramienta"] is True and fila["acierto_argumentos"] is True
    assert fila["cifras_evaluadas"] == fila["cifras_verificadas"] == 3
    assert fila["proporcion_verificada"] == 1.0
    assert fila["cobertura_cifras_clave"] == 1.0
    assert fila["digest_corto"] == "a80c4f17acd5"
    assert fila["latencia_total_s"] == registro.segundos
    assert fila["latencia_seleccion_s"] is not None and fila["latencia_interpretacion_s"] is not None
    assert fila["tokens_por_s"] == 7.0
    assert fila["trace_id"] == registro.trace_id
    assert fila["error"] == ""


def test_fila_marca_respaldo_como_desacierto():
    q4 = POR_ID["Q4"]
    registro = AgenteIncidentes(analisis_agente(), ClienteFalso([respuesta_texto("")])).responder(q4.texto)
    fila = fila_desde_registro(registro, q4, "llama3.2:3b", 1, 101)
    assert fila["origen"] == "respaldo_reglas"
    assert fila["herramientas_elegidas"] == "segmentar"
    assert fila["acierto_herramienta"] is False and fila["acierto_argumentos"] is False


def test_fila_de_pregunta_rechazada_no_mide_acierto():
    q6 = POR_ID["Q6"]
    registro = AgenteIncidentes(analisis_agente(), ClienteFalso()).responder(q6.texto)
    fila = fila_desde_registro(registro, q6, "llama3.2:3b", 1, 101)
    assert fila["estado"] == "rechazada"
    assert fila["acierto_herramienta"] is None and fila["acierto_argumentos"] is None
    assert fila["cobertura_cifras_clave"] is None


# Paso 2: resumen por modelo


def test_resumir_calcula_tasas_por_modelo():
    def fila(modelo, pregunta_id, acierto, estado="respondida", cobertura=None):
        return {"modelo": modelo, "pregunta_id": pregunta_id, "estado": estado, "acierto_herramienta": acierto,
                "acierto_argumentos": acierto, "cobertura_cifras_clave": cobertura}

    def muestra(modelo, i, estado="respondida", segundos=30.0):
        return MuestraConsulta(trace_id=f"{i:032x}", estado=estado, modelo=modelo, modo="automatico",
                               segundos=segundos, segundos_primer_token=5.0, tokens_por_segundo=7.0,
                               done_reason="stop", truncada=False,
                               origen_seleccion="modelo" if estado == "respondida" else None,
                               cifras_evaluadas=4 if estado == "respondida" else 0,
                               cifras_verificadas=3 if estado == "respondida" else 0,
                               tipo_error="timeout" if estado == "error" else None)

    filas = [fila("a", "Q1", True, cobertura=1.0), fila("a", "Q2", False, cobertura=0.5),
             fila("a", "Q6", None, estado="rechazada"),
             fila("b", "Q1", True, cobertura=0.0), fila("b", "Q2", True, estado="error")]
    muestras = [muestra("a", 1), muestra("a", 2), muestra("a", 3, estado="rechazada"),
                muestra("b", 4), muestra("b", 5, estado="error")]
    resumen = resumir(filas, muestras)
    assert set(resumen) == {"a", "b"}
    assert resumen["a"]["acierto"]["n"] == 3
    assert resumen["a"]["acierto"]["tasa_acierto_herramienta"] == 0.5
    assert resumen["a"]["acierto"]["media_cobertura"] == 0.75
    assert resumen["a"]["acierto"]["tasa_acierto_guarda"] == 1.0
    assert resumen["b"]["acierto"]["tasa_acierto_herramienta"] == 1.0
    assert resumen["b"]["acierto"]["tasa_acierto_guarda"] is None
    assert resumen["a"]["slis"] == calcular_slis(muestras[:3]).a_dict()
    assert resumen["b"]["slis"]["errores"] == 1
    assert [e["slo"] for e in resumen["a"]["slos"]] == [s.nombre for s in SLOS]


# Paso 3: decisión del modelo por defecto


def test_decidir_mantiene_referencia_sin_margen():
    resumen = {"llama3.2:3b": resumen_modelo(0.6, 30.0), "llama3.2:1b": resumen_modelo(0.7, 20.0)}
    decision = decidir_modelo(resumen)
    assert decision["estado"] == "mantener"
    assert decision["modelo_elegido"] == "llama3.2:3b"
    assert decision["cambia"] is False
    assert decision["motivos"]["llama3.2:1b"]


def test_decidir_cambia_con_margen_claro():
    resumen = {"llama3.2:3b": resumen_modelo(0.55, 30.0), "llama3.2:1b": resumen_modelo(0.7, 33.0, cifras=0.95)}
    decision = decidir_modelo(resumen)
    assert decision["estado"] == "cambiar"
    assert decision["modelo_elegido"] == "llama3.2:1b"
    assert decision["cambia"] is True
    assert decision["motivos"]["llama3.2:1b"] == []


@pytest.mark.parametrize("candidato", [
    resumen_modelo(0.9, 34.0),
    resumen_modelo(0.9, 20.0, errores=1),
    resumen_modelo(0.9, 20.0, truncadas=0.2),
    resumen_modelo(0.9, 20.0, cifras=0.8),
    resumen_modelo(None, 20.0),
], ids=["lento", "con_errores", "trunca_mas", "menos_cifras", "sin_acierto"])
def test_decidir_rechaza_candidato_lento_o_con_errores(candidato):
    decision = decidir_modelo({"llama3.2:3b": resumen_modelo(0.5, 30.0), "llama3.2:1b": candidato})
    assert decision["estado"] == "mantener"
    assert decision["modelo_elegido"] == "llama3.2:3b"
    assert len(decision["motivos"]["llama3.2:1b"]) == 1


def test_decidir_insuficiente_con_menos_de_dos_repeticiones():
    resumen = {"llama3.2:3b": resumen_modelo(0.5, 30.0), "llama3.2:1b": resumen_modelo(0.9, 20.0)}
    assert decidir_modelo(resumen, repeticiones=1)["estado"] == "insuficiente"
    assert decidir_modelo(resumen, repeticiones=2)["estado"] == "cambiar"
    assert decidir_modelo({"llama3.2:1b": resumen_modelo(0.9, 20.0)})["estado"] == "sin_referencia"


def test_valores_por_defecto_dos_modelos_y_dos_repeticiones():
    assert MODELOS_POR_DEFECTO == ("llama3.2:3b", "llama3.2:1b")
    assert REPETICIONES_POR_DEFECTO == 2
    argumentos = crear_parser().parse_args([])
    assert tuple(argumentos.modelos) == MODELOS_POR_DEFECTO
    assert (argumentos.repeticiones, argumentos.modo) == (2, "automatico")


# Paso 4: corrida completa con el cliente falso


def test_comparacion_con_cliente_falso_escribe_json_y_csv(tmp_path, ruta_comparacion):
    resultado = ejecutar_comparacion(MODELOS_POR_DEFECTO, 2, "automatico", fabrica_cliente=fabrica_falsa(),
                                     analisis=analisis_agente(), ruta_log=ruta_comparacion)
    assert set(resultado) == {"metadatos", "filas", "resumen", "decision"}
    assert len(resultado["filas"]) == 2 * 2 * len(PREGUNTAS_DEMO)
    assert {f["semilla"] for f in resultado["filas"]} == {101, 102}
    assert resultado["resumen"]["llama3.2:3b"]["acierto"]["tasa_acierto_herramienta"] == 1.0
    assert resultado["resumen"]["llama3.2:3b"]["acierto"]["tasa_acierto_guarda"] == 1.0
    assert resultado["decision"]["estado"] == "mantener"
    metadatos = resultado["metadatos"]
    assert metadatos["version_ollama"] == "0.40.6"
    assert metadatos["omitidos"] == {}
    assert [p["id"] for p in metadatos["preguntas"]] == [p.id for p in PREGUNTAS_DEMO]
    assert set(metadatos["modelos"]["llama3.2:1b"]) >= {"digest", "segundos_precalentamiento"}

    ruta_json, ruta_csv = guardar(resultado, tmp_path / "salida")
    assert re.fullmatch(r"comparacion_modelos_\d{8}T\d{6}Z\.json", ruta_json.name)
    assert ruta_csv.name == ruta_json.name.replace(".json", ".csv")
    assert json.loads(ruta_json.read_text(encoding="utf-8"))["decision"] == resultado["decision"]
    assert ruta_csv.read_bytes().startswith(b"\xef\xbb\xbf")
    with ruta_csv.open(encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    assert tuple(filas[0]) == COLUMNAS_CSV
    assert len(filas) == len(resultado["filas"])


def test_respaldo_baja_la_tasa_de_acierto(ruta_comparacion):
    resultado = ejecutar_comparacion(("llama3.2:3b",), 2, "automatico", fabrica_cliente=fabrica_falsa(selector_mudo),
                                     analisis=analisis_agente(), ruta_log=ruta_comparacion)
    resumen = resultado["resumen"]["llama3.2:3b"]
    assert resumen["acierto"]["tasa_acierto_herramienta"] == 0.0
    assert resumen["slis"]["tasa_respaldo"] == 1.0


def test_modelo_no_instalado_se_omite(ruta_comparacion):
    perfecta = fabrica_falsa()
    no_instalado = ErrorOllama("Modelo no disponible. Ejecute: ollama pull llama3.2:1b", tipo="no_instalado",
                               operacion="info_modelo", codigo_http=404)

    def fabrica(modelo, *, seed=None):
        if modelo == "llama3.2:1b":
            return ClienteFalso(selector_perfecto, modelo=modelo, seed=seed, error_info=no_instalado)
        return perfecta(modelo, seed=seed)

    resultado = ejecutar_comparacion(("llama3.2:3b", "llama3.2:1b"), 2, "automatico",
                                     fabrica_cliente=fabrica, analisis=analisis_agente(), ruta_log=ruta_comparacion)
    omitidos = resultado["metadatos"]["omitidos"]
    assert omitidos["llama3.2:1b"] == "omitido: ollama pull llama3.2:1b"
    assert set(resultado["resumen"]) == {"llama3.2:3b"}
    assert {f["modelo"] for f in resultado["filas"]} == {"llama3.2:3b"}

    resultado = ejecutar_comparacion(("mistral",), 2, "automatico", analisis=analisis_agente(), ruta_log=ruta_comparacion)
    assert resultado["metadatos"]["omitidos"]["mistral"].startswith("omitido:")
    assert resultado["filas"] == [] and resultado["decision"]["estado"] == "sin_referencia"


# Paso 5: observabilidad reutilizada (DRY)


def test_comparador_usa_calcular_slis_y_evaluar_slos(monkeypatch, ruta_comparacion):
    vistas: list[list[MuestraConsulta]] = []
    original = comparar.calcular_slis

    def espia(muestras):
        vistas.append(list(muestras))
        return original(vistas[-1])

    monkeypatch.setattr(comparar, "calcular_slis", espia)
    resultado = ejecutar_comparacion(MODELOS_POR_DEFECTO, 2, "automatico", fabrica_cliente=fabrica_falsa(),
                                     analisis=analisis_agente(), ruta_log=ruta_comparacion)
    assert sorted({m.modelo for grupo in vistas for m in grupo}) == sorted(MODELOS_POR_DEFECTO)
    for modelo in MODELOS_POR_DEFECTO:
        resumen = resultado["resumen"][modelo]
        assert resumen["slis"]["consultas"] == 2 * len(PREGUNTAS_DEMO)
        assert resumen["slis"]["rechazadas"] == 2
        assert [e["slo"] for e in resumen["slos"]] == [s.nombre for s in SLOS]


def test_muestras_del_log_igualan_a_las_de_memoria(ruta_comparacion):
    resultado = ejecutar_comparacion(MODELOS_POR_DEFECTO, 2, "automatico", fabrica_cliente=fabrica_falsa(),
                                     analisis=analisis_agente(), ruta_log=ruta_comparacion)
    muestras, invalidas = leer_muestras(ruta_comparacion)
    assert invalidas == 0
    assert len(muestras) == len(resultado["filas"])
    por_modelo: dict[str, list[MuestraConsulta]] = defaultdict(list)
    for muestra in muestras:
        por_modelo[muestra.modelo].append(muestra)
    for modelo, grupo in por_modelo.items():
        assert calcular_slis(grupo).a_dict() == resultado["resumen"][modelo]["slis"]
    assert {f["trace_id"] for f in resultado["filas"]} == {m.trace_id for m in muestras}


def test_comparador_registra_errores_como_muestras(ruta_comparacion):
    error = ErrorOllama("Ollama interrumpió la respuesta: sin memoria", tipo="stream", operacion="chat_stream")
    resultado = ejecutar_comparacion(("llama3.2:3b",), 1, "automatico", fabrica_cliente=fabrica_falsa(error_stream=error),
                                     analisis=analisis_agente(), ruta_log=ruta_comparacion)
    errores = [f for f in resultado["filas"] if f["estado"] == "error"]
    assert len(errores) == len(PREGUNTAS_DEMO) - 1
    assert all("interrumpió" in f["error"] and f["trace_id"] for f in errores)
    slis = resultado["resumen"]["llama3.2:3b"]["slis"]
    assert slis["errores"] == len(PREGUNTAS_DEMO) - 1
    assert slis["errores_por_tipo"] == {"stream": len(PREGUNTAS_DEMO) - 1}
    assert resultado["decision"]["estado"] == "insuficiente"


def test_comparador_no_aborta_por_un_fallo_inesperado(tmp_path, ruta_comparacion):
    texto_q2 = POR_ID["Q2"].texto

    def selector_con_fallo(mensajes, herramientas):
        if mensajes[-1]["content"] == texto_q2:
            raise RuntimeError("fallo inesperado del cliente")
        return selector_perfecto(mensajes, herramientas)

    resultado = ejecutar_comparacion(("llama3.2:3b",), 2, "automatico", fabrica_cliente=fabrica_falsa(selector_con_fallo),
                                     analisis=analisis_agente(), ruta_log=ruta_comparacion)
    assert len(resultado["filas"]) == 2 * len(PREGUNTAS_DEMO)
    fallidas = [f for f in resultado["filas"] if f["estado"] == "error"]
    assert [f["pregunta_id"] for f in fallidas] == ["Q2", "Q2"]
    assert all(f["error"] == "RuntimeError: fallo inesperado del cliente" and f["trace_id"] for f in fallidas)
    assert resultado["resumen"]["llama3.2:3b"]["slis"]["errores_por_tipo"] == {"RuntimeError": 2}
    ruta_json, _ = guardar(resultado, tmp_path / "salida")
    assert json.loads(ruta_json.read_text(encoding="utf-8"))["filas"][1]["estado"] == "error"


def test_comparador_escribe_log_separado_de_app(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path / "logs_app"))
    carpeta = tmp_path / "resultados"
    resultado = ejecutar_comparacion(("llama3.2:3b",), 1, "automatico", fabrica_cliente=fabrica_falsa(),
                                     analisis=analisis_agente(), carpeta=carpeta)
    ruta = Path(resultado["metadatos"]["ruta_log"])
    assert ruta.parent == carpeta / "logs"
    assert re.fullmatch(r"comparacion_\d{8}T\d{6}Z\.jsonl", ruta.name)
    assert len(ruta.read_text(encoding="utf-8").splitlines()) > 0
    assert not ruta_log().exists()
    ruta.unlink()


def test_main_guarda_resultados_y_devuelve_cero(monkeypatch, tmp_path, capsys):
    recibido = {}

    def ejecutar_falso(modelos, repeticiones, modo, **kwargs):
        recibido.update(modelos=tuple(modelos), repeticiones=repeticiones, modo=modo)
        return {"metadatos": {"omitidos": {}}, "filas": [], "resumen": {"llama3.2:3b": {}},
                "decision": {"estado": "mantener", "modelo_elegido": "llama3.2:3b"}}

    monkeypatch.setattr(comparar, "ejecutar_comparacion", ejecutar_falso)
    codigo = main(["--modelos", "llama3.2:3b", "--repeticiones", "3", "--carpeta", str(tmp_path)])
    assert codigo == 0
    assert recibido == {"modelos": ("llama3.2:3b",), "repeticiones": 3, "modo": "automatico"}
    assert len(list(tmp_path.glob("comparacion_modelos_*.json"))) == 1
    assert "mantener" in capsys.readouterr().out
