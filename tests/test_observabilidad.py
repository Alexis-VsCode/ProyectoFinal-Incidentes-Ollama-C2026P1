"""Pruebas de la observabilidad de la aplicación: logger JSONL, spans, muestras, SLI y SLO.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import itertools
import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import observabilidad as obs
from src.observabilidad import (
    CAMPOS,
    COLUMNAS_TABLA_SLOS,
    COLUMNAS_ULTIMAS_CONSULTAS,
    EVENTO_FIN,
    EVENTO_INICIO,
    NOMBRE_LOGGER,
    RUTA_LOGS,
    SLO,
    SLOS,
    SPAN_INTERPRETACION,
    SPAN_SELECCION,
    MuestraConsulta,
    calcular_slis,
    cerrar_logger,
    configurar_logger,
    evaluar_slo,
    evaluar_slos,
    leer_eventos,
    leer_muestras,
    nuevo_trace_id,
    percentil,
    registrar_evento,
    registrar_fin_consulta,
    ruta_log,
    span,
    tabla_slos,
    ultimas_consultas,
)

_contador = itertools.count(1)
UTC = timezone.utc
SLO_LATENCIA, SLO_DISPONIBILIDAD, SLO_CIFRAS = SLOS


@pytest.fixture
def logger_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path / "entorno"))
    ruta = tmp_path / "logs" / "app.jsonl"
    configurar_logger(ruta)
    yield ruta
    cerrar_logger()


def lineas(ruta: Path) -> list[dict]:
    return [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines()]


def manejadores_archivo() -> list[logging.FileHandler]:
    return [h for h in logging.getLogger(NOMBRE_LOGGER).handlers if isinstance(h, logging.FileHandler)]


def muestra(estado: str = "respondida", *, segundos: float | None = 30.0, primer: float | None = 8.0,
            tps: float | None = 7.0, truncada: bool = False, origen: str | None = "modelo",
            modo: str = "automatico", evaluadas: int = 0, verificadas: int = 0,
            tipo_error: str | None = None, modelo: str = "llama3.2:3b",
            trace_id: str | None = None) -> MuestraConsulta:
    return MuestraConsulta(
        trace_id=trace_id or f"{next(_contador):032x}", estado=estado, modelo=modelo, modo=modo,
        segundos=segundos, segundos_primer_token=primer, tokens_por_segundo=tps,
        done_reason="length" if truncada else "stop", truncada=truncada, origen_seleccion=origen,
        cifras_evaluadas=evaluadas, cifras_verificadas=verificadas, tipo_error=tipo_error)


def lote_s1() -> list[MuestraConsulta]:
    segundos = [30.0] * 10 + [40.0] * 6 + [60.0, 95.0]
    origenes = ["modelo"] * 12 + ["modelo_texto"] * 2 + ["respaldo_reglas"] * 2 + ["usuario"] * 2
    respondidas = []
    for i, (s, origen) in enumerate(zip(segundos, origenes)):
        evaluadas, verificadas = (10, 5) if i >= 16 else (5, 5)
        respondidas.append(muestra(segundos=s, truncada=i < 2, origen=origen,
                                   modo="manual" if origen == "usuario" else "automatico",
                                   evaluadas=evaluadas, verificadas=verificadas))
    otras = [
        muestra("error", segundos=90.0, primer=None, tps=None, origen="modelo", tipo_error="timeout"),
        muestra("error", segundos=0.5, primer=None, tps=None, origen=None, tipo_error="conexion"),
        muestra("rechazada", segundos=0.01, primer=None, tps=None, origen=None),
        muestra("cancelada", segundos=12.0, primer=5.0, tps=None, origen="modelo"),
    ]
    return respondidas + otras


def linea_fin(m: MuestraConsulta, ts: str) -> str:
    return json.dumps({"ts_utc": ts, "nivel": "INFO", "evento": EVENTO_FIN, "trace_id": m.trace_id,
                       "atributos": m.a_dict()})


# Paso 1: logger JSONL


def test_logger_escribe_una_linea_json_valida_por_evento(logger_tmp):
    for i in range(3):
        registrar_evento(EVENTO_INICIO, trace_id="t1", modo="automatico", indice=i)
    registros = lineas(logger_tmp)
    assert len(registros) == 3
    for registro in registros:
        assert tuple(registro) == CAMPOS
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00", registro["ts_utc"])
        assert registro["nivel"] == "INFO"
        assert registro["evento"] == EVENTO_INICIO
        assert registro["trace_id"] == "t1"
    assert [r["atributos"]["indice"] for r in registros] == [0, 1, 2]


def test_logger_serializa_atributos_no_json_y_nan(logger_tmp):
    fecha = datetime(2026, 10, 18, 20, 15, tzinfo=UTC)
    registrar_evento("prueba", ruta=Path("a") / "b.txt", fecha=fecha, valor=float("nan"),
                     lista=(1, float("inf")), entero=np.int64(5), flotante=np.float64("nan"))
    texto = logger_tmp.read_text(encoding="utf-8")
    assert "NaN" not in texto and "Infinity" not in texto
    atributos = json.loads(texto)["atributos"]
    assert atributos["ruta"] == str(Path("a") / "b.txt")
    assert atributos["fecha"] == str(fecha)
    assert atributos["valor"] is None
    assert atributos["lista"] == [1, None]
    assert atributos["entero"] == 5
    assert atributos["flotante"] is None


def test_logger_convierte_arreglos_numpy_en_listas(logger_tmp):
    devuelto = registrar_evento("prueba", valores=np.array([1, 2.5, np.nan]), enteros=np.array([[1, 2]]))
    atributos = lineas(logger_tmp)[0]["atributos"]
    assert atributos["valores"] == [1.0, 2.5, None]
    assert atributos["enteros"] == [[1, 2]]
    assert devuelto["atributos"] == atributos


def test_registro_de_terceros_se_formatea_como_json(logger_tmp):
    logging.getLogger(f"{NOMBRE_LOGGER}.terceros").warning("hola %s", "mundo")
    registro = lineas(logger_tmp)[0]
    assert tuple(registro) == CAMPOS
    assert registro["evento"] == "hola mundo"
    assert registro["nivel"] == "WARNING"
    assert registro["trace_id"] is None
    assert registro["atributos"] == {"logger": f"{NOMBRE_LOGGER}.terceros"}


def test_configurar_logger_es_idempotente(logger_tmp):
    configurar_logger(logger_tmp)
    configurar_logger(logger_tmp)
    assert len(manejadores_archivo()) == 1
    assert logging.getLogger(NOMBRE_LOGGER).propagate is False
    registrar_evento("unico")
    assert len(lineas(logger_tmp)) == 1


def test_configurar_logger_crea_carpeta(tmp_path):
    ruta = tmp_path / "a" / "b" / "c" / "app.jsonl"
    try:
        configurar_logger(ruta)
        assert ruta.parent.is_dir()
    finally:
        cerrar_logger()


def test_configurar_logger_cambia_de_ruta(tmp_path):
    primera, segunda = tmp_path / "uno.jsonl", tmp_path / "dos.jsonl"
    try:
        configurar_logger(primera)
        registrar_evento("primero")
        anterior = manejadores_archivo()[0]
        configurar_logger(segunda)
        registrar_evento("segundo")
        actuales = manejadores_archivo()
        assert len(actuales) == 1
        assert Path(actuales[0].baseFilename) == segunda
        assert anterior not in actuales
        assert anterior.stream is None
        assert [r["evento"] for r in lineas(primera)] == ["primero"]
        assert [r["evento"] for r in lineas(segunda)] == ["segundo"]
    finally:
        cerrar_logger()


def test_configurar_logger_concurrente_deja_un_solo_manejador(tmp_path, monkeypatch):
    original = logging.FileHandler

    class ManejadorLento(original):
        def __init__(self, *args, **kwargs):
            time.sleep(0.1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(logging, "FileHandler", ManejadorLento)
    ruta = tmp_path / "app.jsonl"
    barrera = threading.Barrier(4)

    def configurar():
        barrera.wait()
        configurar_logger(ruta)

    hilos = [threading.Thread(target=configurar) for _ in range(4)]
    try:
        for hilo in hilos:
            hilo.start()
        for hilo in hilos:
            hilo.join()
        assert len(manejadores_archivo()) == 1
        registrar_evento("unico")
        assert len(lineas(ruta)) == 1
    finally:
        cerrar_logger()


def test_cerrar_logger_libera_archivo(tmp_path):
    ruta = tmp_path / "app.jsonl"
    configurar_logger(ruta)
    registrar_evento("algo")
    cerrar_logger()
    assert manejadores_archivo() == []
    logger = logging.getLogger(NOMBRE_LOGGER)
    assert (logger.level, logger.propagate) == (logging.NOTSET, True)
    ruta.unlink()
    assert not ruta.exists()


def test_sin_configurar_no_escribe_ni_falla(tmp_path, monkeypatch):
    cerrar_logger()
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path))
    destino: list[dict] = []
    evento = registrar_evento("suelto", trace_id="t", destino=destino, valor=1)
    with span(SPAN_SELECCION, "t", destino=destino):
        pass
    assert evento["evento"] == "suelto"
    assert len(destino) == 2
    assert manejadores_archivo() == []
    assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger(NOMBRE_LOGGER).handlers)
    assert list(tmp_path.iterdir()) == []


def test_destino_recibe_el_mismo_dict_que_el_archivo(logger_tmp):
    destino: list[dict] = []
    devuelto = registrar_evento("evento", trace_id="abc", destino=destino, ruta=Path("x"), tupla=(1, 2),
                                valor=float("nan"), nivel_interno=3)
    assert destino[0] is devuelto
    assert destino[0] == lineas(logger_tmp)[0]


def test_ruta_log_lee_variable_de_entorno_en_cada_llamada(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path / "uno"))
    assert ruta_log() == tmp_path / "uno" / "app.jsonl"
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path / "dos"))
    assert ruta_log() == tmp_path / "dos" / "app.jsonl"
    monkeypatch.delenv("INCIDENTES_LOG_DIR")
    assert ruta_log() == RUTA_LOGS
    assert RUTA_LOGS.parts[-3:] == ("resultados_locales", "logs", "app.jsonl")


def test_configurar_logger_usa_ruta_log_por_defecto(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path))
    try:
        configurar_logger()
        registrar_evento("por_defecto")
        assert Path(manejadores_archivo()[0].baseFilename) == tmp_path / "app.jsonl"
        assert lineas(tmp_path / "app.jsonl")[0]["evento"] == "por_defecto"
    finally:
        cerrar_logger()


# Paso 2: spans y trace_id


def test_span_mide_ms_y_registra_ok(logger_tmp, monkeypatch):
    monkeypatch.setattr(obs, "perf_counter", iter([1.0, 1.25]).__next__)
    destino: list[dict] = []
    with span(SPAN_SELECCION, "t1", destino=destino, modelo="llama3.2:3b") as s:
        pass
    registro = lineas(logger_tmp)[0]
    assert registro["evento"] == SPAN_SELECCION
    assert registro["nivel"] == "INFO"
    assert registro["atributos"] == {"tipo": "span", "ms": 250.0, "estado": "ok", "modelo": "llama3.2:3b"}
    assert (s.ms, s.estado) == (250.0, "ok")
    assert destino == [registro]


def test_span_registra_aunque_haya_excepcion(logger_tmp):
    with pytest.raises(ValueError):
        with span(SPAN_SELECCION, "t1"):
            raise ValueError("x" * 500)
    registros = lineas(logger_tmp)
    assert len(registros) == 1
    atributos = registros[0]["atributos"]
    assert registros[0]["nivel"] == "ERROR"
    assert atributos["estado"] == "error"
    assert atributos["error_tipo"] == "ValueError"
    assert len(atributos["error"]) == 300
    assert "ms" in atributos


def test_span_en_generador_cancelado_marca_cancelado(logger_tmp):
    destino: list[dict] = []

    def generador():
        with span(SPAN_INTERPRETACION, "t1", destino=destino):
            yield "a"
            yield "b"

    gen = generador()
    assert next(gen) == "a"
    gen.close()
    assert len(destino) == 1
    assert destino[0]["atributos"]["estado"] == "cancelado"
    assert destino[0]["nivel"] == "INFO"


def test_span_permite_anotar_atributos(logger_tmp):
    with span(SPAN_INTERPRETACION, "t1", modelo="m") as s:
        s.atributos["eval_count"] = 186
        s.atributos["estado"] = "manipulado"
    atributos = lineas(logger_tmp)[0]["atributos"]
    assert atributos["eval_count"] == 186
    assert atributos["modelo"] == "m"
    assert atributos["estado"] == "ok"
    assert atributos["tipo"] == "span"


def test_trace_id_es_hex_de_32_y_unico():
    ids = [nuevo_trace_id() for _ in range(100)]
    assert len(set(ids)) == 100
    assert all(re.fullmatch(r"[0-9a-f]{32}", t) for t in ids)


# Paso 3: muestras de consulta


def test_muestra_ida_y_vuelta_dict():
    m = muestra(evaluadas=9, verificadas=8)
    assert MuestraConsulta.desde_dict(m.a_dict()) == m
    assert MuestraConsulta.desde_dict(m.a_dict() | {"extra": 1, "tipo": "x"}) == m
    with pytest.raises(ValueError):
        MuestraConsulta.desde_dict(m.a_dict() | {"estado": "terminada"})
    incompleta = m.a_dict()
    del incompleta["modelo"]
    with pytest.raises(ValueError):
        MuestraConsulta.desde_dict(incompleta)


def test_registrar_fin_consulta_nivel_y_fidelidad(logger_tmp):
    respondida = muestra(evaluadas=3, verificadas=2)
    fallida = muestra("error", segundos=1.0, tipo_error="timeout")
    destino: list[dict] = []
    registrar_fin_consulta(respondida, destino=destino)
    registrar_fin_consulta(fallida, destino=destino)
    registros = lineas(logger_tmp)
    assert [r["nivel"] for r in registros] == ["INFO", "WARNING"]
    assert [r["evento"] for r in registros] == [EVENTO_FIN, EVENTO_FIN]
    assert registros[0]["trace_id"] == respondida.trace_id
    assert registros[0]["atributos"] == respondida.a_dict()
    assert destino == registros
    assert leer_muestras(logger_tmp) == ([respondida, fallida], 0)


def test_leer_muestras_filtra_eventos_y_cuenta_lineas_invalidas(logger_tmp):
    m1, m2 = muestra(), muestra("rechazada", origen=None)
    registrar_evento(EVENTO_INICIO, trace_id=m1.trace_id, modo="automatico")
    with span(SPAN_SELECCION, m1.trace_id):
        pass
    registrar_fin_consulta(m1)
    registrar_fin_consulta(m2)
    with logger_tmp.open("a", encoding="utf-8") as archivo:
        archivo.write("{linea corrupta\n\n")
        archivo.write(json.dumps({"evento": EVENTO_FIN, "trace_id": "x", "atributos": {"estado": "rara"}}) + "\n")
    muestras, invalidas = leer_muestras(logger_tmp)
    assert muestras == [m1, m2]
    assert invalidas == 2
    assert leer_muestras(logger_tmp.parent / "no_existe.jsonl") == ([], 0)


@pytest.mark.parametrize(("campo", "valor"), [
    ("segundos", "abc"), ("segundos", True), ("segundos_primer_token", float("nan")),
    ("tokens_por_segundo", float("inf")), ("trace_id", None), ("trace_id", ""), ("modelo", 5),
    ("modo", None), ("done_reason", 1), ("origen_seleccion", ["modelo"]), ("tipo_error", 0),
])
def test_desde_dict_rechaza_tipos_invalidos(campo, valor):
    with pytest.raises(ValueError):
        MuestraConsulta.desde_dict(muestra().a_dict() | {campo: valor})


def test_leer_muestras_descarta_tipos_invalidos(tmp_path):
    valida = muestra(evaluadas=2, verificadas=2)
    ts = "2026-10-18T15:00:00.000+00:00"
    corruptas = [muestra().a_dict() | {"segundos": "abc"}, muestra().a_dict() | {"modelo": 5}]
    texto = "\n".join([linea_fin(valida, ts)]
                      + [json.dumps({"ts_utc": ts, "nivel": "INFO", "evento": EVENTO_FIN,
                                     "trace_id": a["trace_id"], "atributos": a}) for a in corruptas])
    texto += "\n" + linea_fin(muestra(), ts).replace('"segundos": 30.0', '"segundos": NaN')
    ruta = tmp_path / "app.jsonl"
    ruta.write_text(texto + "\n", encoding="utf-8")
    muestras, invalidas = leer_muestras(ruta)
    assert (muestras, invalidas) == ([valida], 3)
    assert calcular_slis(muestras).latencia_p50_s == 30.0


def test_leer_muestras_filtra_por_fecha_y_modelo(tmp_path):
    viejo, nuevo = muestra(), muestra(modelo="llama3.2:1b")
    ruta = tmp_path / "app.jsonl"
    ruta.write_text("\n".join([linea_fin(viejo, "2026-10-17T10:00:00.000+00:00"),
                               linea_fin(nuevo, "2026-10-18T15:00:00.000+00:00")]) + "\n", encoding="utf-8")
    assert leer_muestras(ruta, desde=datetime(2026, 10, 18, tzinfo=UTC)) == ([nuevo], 0)
    assert leer_muestras(ruta, desde=datetime(2026, 10, 18)) == ([nuevo], 0)
    assert leer_muestras(ruta, modelo="llama3.2:3b") == ([viejo], 0)
    assert leer_muestras(ruta, desde=datetime(2026, 10, 18, tzinfo=UTC), modelo="llama3.2:3b") == ([], 0)


def test_leer_muestras_trace_repetido_gana_el_ultimo(tmp_path):
    primero = muestra("error", tipo_error="timeout", trace_id="a" * 32)
    ultimo = muestra(trace_id="a" * 32)
    otro = muestra()
    ruta = tmp_path / "app.jsonl"
    ruta.write_text("\n".join(linea_fin(m, "2026-10-18T15:00:00.000+00:00") for m in (primero, otro, ultimo)),
                    encoding="utf-8")
    muestras, invalidas = leer_muestras(ruta)
    assert invalidas == 0
    assert len(muestras) == 2
    assert ultimo in muestras and primero not in muestras


def test_leer_eventos_ignora_corruptas_y_respeta_limite(tmp_path, monkeypatch):
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path))
    assert leer_eventos() == []
    try:
        configurar_logger()
        for i in range(5):
            registrar_evento("e", trace_id="t", indice=i)
    finally:
        cerrar_logger()
    with (tmp_path / "app.jsonl").open("a", encoding="utf-8") as archivo:
        archivo.write("no es json\n\n[1, 2]\n")
    eventos = leer_eventos()
    assert [e["atributos"]["indice"] for e in eventos] == [0, 1, 2, 3, 4]
    assert [e["atributos"]["indice"] for e in leer_eventos(tmp_path / "app.jsonl", limite=2)] == [3, 4]


def test_ultimas_consultas_una_fila_por_evento_de_las_ultimas_n():
    eventos: list[dict] = []
    for trace in ("a", "b", "c"):
        registrar_evento(EVENTO_INICIO, trace_id=trace, destino=eventos, modo="automatico")
        with span(SPAN_SELECCION, trace, destino=eventos):
            pass
        registrar_evento(EVENTO_FIN, trace_id=trace, destino=eventos, estado="respondida")
    registrar_evento("suelto", destino=eventos)
    tabla = ultimas_consultas(eventos, n=2)
    assert tuple(tabla.columns) == COLUMNAS_ULTIMAS_CONSULTAS == ("trace_id", "ts_utc", "evento", "ms", "estado")
    assert tabla["trace_id"].tolist() == ["b", "b", "b", "c", "c", "c"]
    assert tabla["evento"].tolist()[:3] == [EVENTO_INICIO, SPAN_SELECCION, EVENTO_FIN]
    assert pd.isna(tabla["ms"].iloc[0]) and tabla["ms"].iloc[1] >= 0
    assert tabla["estado"].tolist()[1:3] == ["ok", "respondida"]
    vacia = ultimas_consultas([], n=10)
    assert vacia.empty and tuple(vacia.columns) == COLUMNAS_ULTIMAS_CONSULTAS


def test_ultimas_consultas_ignora_trace_id_no_textual():
    eventos = [{"trace_id": ["x"], "evento": "lista"}, {"trace_id": {"a": 1}, "evento": "objeto"},
               {"trace_id": 5, "evento": "numero"}, {"trace_id": "t1", "evento": EVENTO_INICIO, "atributos": 7},
               "no es un evento"]
    tabla = ultimas_consultas(eventos)
    assert tabla.to_dict("records") == [{"trace_id": "t1", "ts_utc": None, "evento": EVENTO_INICIO,
                                         "ms": None, "estado": None}]


# Paso 4: SLI


@pytest.mark.parametrize(("valores", "q", "esperado"), [
    (list(range(1, 21)), 0.5, 10.0), (list(range(1, 21)), 0.95, 19.0), (list(range(20, 0, -1)), 0.95, 19.0),
    ([5.0], 0.5, 5.0), ([5.0], 0.95, 5.0), ([], 0.5, None),
])
def test_percentil_rango_mas_cercano(valores, q, esperado):
    assert percentil(valores, q) == esperado


def test_calcular_slis_lote_sintetico_s1():
    slis = calcular_slis(lote_s1())
    assert (slis.consultas, slis.validas, slis.respondidas, slis.rechazadas, slis.canceladas, slis.errores) == \
        (22, 20, 18, 1, 1, 2)
    assert slis.errores_por_tipo == {"timeout": 1, "conexion": 1}
    assert (slis.latencia_p50_s, slis.latencia_p95_s) == (30.0, 95.0)
    assert (slis.primer_token_p50_s, slis.primer_token_p95_s) == (8.0, 8.0)
    assert slis.tokens_por_segundo_p50 == 7.0
    assert slis.tasa_error == 0.1
    assert slis.tasa_truncadas == 0.1111
    assert slis.tasa_seleccion_modelo == 0.875
    assert slis.tasa_respaldo == 0.125
    assert (slis.cifras_evaluadas, slis.cifras_verificadas, slis.proporcion_cifras_verificadas) == (100, 90, 0.9)
    assert json.loads(json.dumps(slis.a_dict(), allow_nan=False))["errores"] == 2
    latencia = evaluar_slo(SLO_LATENCIA, lote_s1())
    assert (latencia.eventos_validos, latencia.eventos_buenos) == (18, 17)
    assert latencia.cumplimiento == 0.9444
    assert latencia.cumple is False
    assert latencia.presupuesto_total == 0.9
    assert latencia.presupuesto_consumido == 1
    assert latencia.presupuesto_restante == -0.1
    assert latencia.suficiente is False


def test_slis_excluyen_rechazadas_y_canceladas():
    slis = calcular_slis(lote_s1())
    assert slis.consultas == 22
    assert slis.validas == 20


def test_slis_vacios_devuelven_none():
    slis = calcular_slis([])
    assert (slis.consultas, slis.validas, slis.errores, slis.cifras_evaluadas) == (0, 0, 0, 0)
    assert slis.errores_por_tipo == {}
    for campo in ("latencia_p50_s", "latencia_p95_s", "primer_token_p50_s", "tokens_por_segundo_p50",
                  "tasa_error", "tasa_truncadas", "tasa_seleccion_modelo", "tasa_respaldo",
                  "proporcion_cifras_verificadas"):
        assert getattr(slis, campo) is None, campo


# Paso 5: SLO y presupuesto de error


def test_slo_latencia_cumple_en_el_limite():
    lote = [muestra(segundos=30.0) for _ in range(19)] + [muestra(segundos=95.0)]
    evaluacion = evaluar_slo(SLO_LATENCIA, lote)
    assert evaluacion.cumplimiento == 0.95
    assert evaluacion.cumple is True
    assert evaluacion.presupuesto_total == 1.0
    assert evaluacion.presupuesto_restante == pytest.approx(0.0)
    assert evaluacion.fraccion_restante == pytest.approx(0.0)
    assert evaluacion.suficiente is True


def test_slo_latencia_presupuesto_negativo():
    lote = [muestra(segundos=30.0) for _ in range(18)] + [muestra(segundos=95.0), muestra(segundos=120.0)]
    evaluacion = evaluar_slo(SLO_LATENCIA, lote)
    assert evaluacion.cumple is False
    assert evaluacion.presupuesto_consumido == 2
    assert evaluacion.presupuesto_restante == pytest.approx(-1.0)
    assert evaluacion.fraccion_restante == pytest.approx(-1.0)


def test_umbral_de_latencia_es_estricto():
    assert evaluar_slo(SLO_LATENCIA, [muestra(segundos=90.0)]).eventos_buenos == 0
    assert evaluar_slo(SLO_LATENCIA, [muestra(segundos=89.999)]).eventos_buenos == 1
    assert evaluar_slo(SLO_LATENCIA, [muestra(segundos=None)]).eventos_validos == 0


def test_slo_disponibilidad_cuenta_errores():
    lote = [muestra() for _ in range(19)] + [muestra("error", tipo_error="memoria"), muestra("rechazada")]
    evaluacion = evaluar_slo(SLO_DISPONIBILIDAD, lote)
    assert (evaluacion.eventos_validos, evaluacion.eventos_buenos) == (20, 19)
    assert evaluacion.cumple is True
    assert evaluacion.presupuesto_consumido == 1


def test_slo_cifras_usa_sumas():
    lote = [muestra(evaluadas=10, verificadas=9), muestra(evaluadas=10, verificadas=8),
            muestra(evaluadas=10, verificadas=10)]
    evaluacion = evaluar_slo(SLO_CIFRAS, lote)
    assert (evaluacion.eventos_validos, evaluacion.eventos_buenos) == (30, 27)
    assert evaluacion.cumplimiento == 0.9
    assert evaluacion.cumple is True


def test_slo_sin_eventos():
    for slo in SLOS:
        evaluacion = evaluar_slo(slo, [])
        assert evaluacion.cumple is None
        assert evaluacion.cumplimiento is None
        assert evaluacion.suficiente is False
        assert evaluacion.fraccion_restante is None
        assert evaluacion.presupuesto_total == 0.0


def test_suficiencia_por_minimo_de_eventos():
    evaluacion = evaluar_slo(SLO_LATENCIA, [muestra(segundos=10.0) for _ in range(19)])
    assert evaluacion.cumple is True
    assert evaluacion.suficiente is False


def test_evaluar_slos_devuelve_una_evaluacion_por_slo():
    evaluaciones = evaluar_slos(iter(lote_s1()))
    assert [e.slo for e in evaluaciones] == [s.nombre for s in SLOS]
    disponibilidad, cifras = evaluaciones[1], evaluaciones[2]
    assert (disponibilidad.cumple, disponibilidad.presupuesto_restante) == (False, -1.0)
    assert (cifras.cumple, cifras.presupuesto_restante) == (True, 0.0)
    assert json.loads(json.dumps([e.a_dict() for e in evaluaciones], allow_nan=False))[0]["slo"] == SLO_LATENCIA.nombre


def test_slos_declarados_son_validos():
    nombres = [s.nombre for s in SLOS]
    assert nombres == ["latencia_consulta", "disponibilidad_agente", "cifras_verificadas"]
    assert len(set(nombres)) == len(nombres)
    assert all(0 < s.objetivo < 1 for s in SLOS)
    assert SLO_LATENCIA.sli == "latencia" and SLO_LATENCIA.umbral_s == 90.0
    with pytest.raises(ValueError):
        SLO("x", "y", "latencia", 0.9)
    with pytest.raises(ValueError):
        SLO("x", "y", "cifras", 1.5)


def test_slos_no_mencionan_el_dataset():
    for slo in SLOS:
        texto = f"{slo.nombre} {slo.descripcion}".lower()
        for prohibida in ("incidente", "resolucion", "resolución", "mttr"):
            assert prohibida not in texto, (slo.nombre, prohibida)


def test_tabla_slos_columnas_para_mostrar():
    tabla = tabla_slos(evaluar_slos(lote_s1()))
    assert tuple(tabla.columns) == COLUMNAS_TABLA_SLOS
    assert tabla["slo"].tolist() == [s.nombre for s in SLOS]
    assert tabla["descripcion"].tolist() == [s.descripcion for s in SLOS]
    assert tabla["resultado"].tolist() == ["no cumple (indicativo)", "no cumple", "cumple"]
    sin_datos = tabla_slos(evaluar_slos([]))
    assert sin_datos["resultado"].tolist() == ["sin eventos"] * 3
    vacia = tabla_slos([])
    assert vacia.empty and tuple(vacia.columns) == COLUMNAS_TABLA_SLOS
