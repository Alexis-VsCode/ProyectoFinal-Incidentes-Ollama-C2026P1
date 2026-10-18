"""Pruebas vivas contra el Ollama local: formato real de tool calling, límites y SLI desde el log.

Requieren Ollama en 127.0.0.1:11434 con llama3.2:3b instalado. Se ejecutan con
`python -m pytest -m vivo -v` y guardan su evidencia en resultados_locales/evidencia_vivo/.
No exigen acierto del modelo: eso lo mide scripts.comparar_modelos.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from src.agent import (
    OPCIONES_INTERPRETACION,
    OPCIONES_SELECCION,
    SISTEMA_SELECCION,
    AgenteIncidentes,
    construir_mensajes_interpretacion,
)
from src.data_loader import cargar_csv
from src.eda import AnalisisIncidentes
from src.herramientas import REGISTRO, contexto_base, json_compacto, preparar_para_json
from src.llm_client import MODELO_POR_DEFECTO, MODELOS, NUM_CTX, ClienteOllama, ErrorOllama
from src.observabilidad import cerrar_logger, calcular_slis, configurar_logger, evaluar_slos, leer_muestras
from src.preguntas_demo import PREGUNTAS_DEMO

pytestmark = pytest.mark.vivo

BASE = Path(__file__).resolve().parents[1]
CARPETA_EVIDENCIA = BASE / "resultados_locales" / "evidencia_vivo"
PREGUNTAS_APP = [p for p in PREGUNTAS_DEMO if not p.solo_comparador]


def guardar_evidencia(nombre: str, datos: Any) -> Path:
    """Escribe `datos` como JSON legible en la carpeta de evidencia y devuelve la ruta."""
    CARPETA_EVIDENCIA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_EVIDENCIA / nombre
    ruta.write_text(json.dumps(preparar_para_json(datos), ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    return ruta


@pytest.fixture(scope="module")
def cliente_vivo() -> ClienteOllama:
    cliente = ClienteOllama(MODELO_POR_DEFECTO)
    try:
        cliente.version()
        cliente.info_modelo()
    except ErrorOllama as exc:
        pytest.skip(f"Ollama local no disponible: {exc}")
    return cliente


@pytest.fixture(scope="module")
def analisis_real() -> AnalisisIncidentes:
    ruta = BASE / "data" / "incident_event_log.csv"
    if not ruta.exists():
        pytest.skip("Falta data/incident_event_log.csv")
    datos = cargar_csv(ruta.read_bytes())
    prioridades = sorted(datos.incidentes["priority"].dropna().unique().tolist())
    return AnalisisIncidentes(datos.incidentes, datos.control, prioridades)


@pytest.fixture(scope="module")
def corrida_demo(cliente_vivo, analisis_real, tmp_path_factory) -> dict[str, Any]:
    """Responde una vez Q1-Q6 con el agente real y deja el log JSONL de esa corrida."""
    ruta = tmp_path_factory.mktemp("logs_vivo") / "vivo.jsonl"
    configurar_logger(ruta)
    muestras: list = []
    registros = {}
    try:
        agente = AgenteIncidentes(analisis_real, cliente_vivo, al_finalizar=muestras.append)
        for pregunta in PREGUNTAS_APP:
            registros[pregunta.id] = agente.responder(pregunta.texto)
    finally:
        cerrar_logger()
    for identificador, registro in registros.items():
        guardar_evidencia(f"consulta_{identificador}.json", registro.a_dict())
    return {"registros": registros, "muestras": muestras, "ruta_log": ruta}


def test_vivo_version_y_digest(cliente_vivo):
    info = cliente_vivo.info_modelo()
    assert cliente_vivo.version()
    assert re.fullmatch(r"[0-9a-f]{64}", info.digest)
    assert info.soporta_herramientas
    guardar_evidencia("version_y_digest.json", {"version": cliente_vivo.version(), "info": info.a_dict()})


def test_vivo_precalentar_carga_modelo(cliente_vivo):
    respuesta = cliente_vivo.precalentar()
    assert respuesta.done_reason == "load"


def test_vivo_tool_calling_devuelve_argumentos_objeto(cliente_vivo):
    pregunta = next(p for p in PREGUNTAS_DEMO if p.id == "Q4")
    mensajes = [{"role": "system", "content": SISTEMA_SELECCION}, {"role": "user", "content": pregunta.texto}]
    respuesta = cliente_vivo.chat(mensajes, REGISTRO.esquemas_ollama(), opciones=OPCIONES_SELECCION)
    guardar_evidencia("tool_calling.json", {"contenido": respuesta.contenido, "done_reason": respuesta.done_reason,
                                            "llamadas": [vars(ll) for ll in respuesta.llamadas],
                                            "metricas": respuesta.metricas.a_dict()})
    assert respuesta.llamadas, "el modelo no emitió tool_calls"
    assert respuesta.llamadas[0].nombre in REGISTRO
    assert isinstance(respuesta.llamadas[0].argumentos, dict)


def test_vivo_interpretacion_sin_tools_usa_resultado(corrida_demo):
    registro = corrida_demo["registros"]["Q1"]
    assert registro.respuesta.strip()
    assert registro.verificacion_cifras["verificadas"] >= 1


def test_vivo_done_reason_length_con_num_predict_bajo(cliente_vivo):
    mensajes = [{"role": "user", "content": "Describe en detalle qué es un incidente de TI."}]
    respuesta = cliente_vivo.chat(mensajes, opciones={"temperature": 0.0, "num_predict": 5})
    assert respuesta.done_reason == "length"
    assert respuesta.truncada


@pytest.mark.parametrize("pregunta", PREGUNTAS_APP, ids=lambda p: p.id)
def test_vivo_preguntas_demo(corrida_demo, pregunta):
    registro = corrida_demo["registros"][pregunta.id]
    if pregunta.herramientas_esperadas:
        assert registro.estado == "respondida"
        assert registro.metricas["interpretacion"]["eval_count"] > 0
    else:
        assert registro.estado == "rechazada"
        assert registro.metricas == {"seleccion": None, "interpretacion": None}


def test_vivo_prompt_peor_caso_cabe_en_num_ctx(cliente_vivo, analisis_real):
    ejecuciones = [REGISTRO.ejecutar("calidad", {}, analisis_real),
                   REGISTRO.ejecutar("segmentar", {"variable": "assignment_group", "top": 10}, analisis_real)]
    mensajes, contexto = construir_mensajes_interpretacion(
        "¿Qué campos tienen más valores faltantes y qué grupos de asignación concentran más incidentes?",
        ejecuciones, contexto_base(analisis_real))
    assert {"calidad_incidentes_seleccionados", "segmentacion_assignment_group"} <= set(contexto)
    flujo = cliente_vivo.chat_stream(mensajes, opciones=OPCIONES_INTERPRETACION)
    texto = "".join(flujo)
    metricas = flujo.respuesta.metricas
    guardar_evidencia("peor_caso_contexto.json", {
        "herramientas": [e.nombre for e in ejecuciones], "caracteres_contexto": len(json_compacto(contexto)),
        "done_reason": flujo.respuesta.done_reason, "metricas": metricas.a_dict()})
    assert texto.strip()
    assert metricas.prompt_eval_count + metricas.eval_count < 0.9 * NUM_CTX


def test_vivo_modelo_no_instalado_pide_pull(cliente_vivo):
    instalados = {m.get("name") for m in cliente_vivo.modelos()}
    faltante = next((m for m in MODELOS if m not in instalados), None)
    if faltante is None:
        pytest.skip("Todos los modelos de la lista blanca están instalados")
    with pytest.raises(ErrorOllama, match="ollama pull"):
        ClienteOllama(faltante).info_modelo()


def test_vivo_slis_calculables_desde_log(corrida_demo):
    muestras, invalidas = leer_muestras(corrida_demo["ruta_log"])
    assert invalidas == 0
    estados = [m.estado for m in muestras]
    assert estados.count("respondida") == 5 and estados.count("rechazada") == 1
    slis = calcular_slis(muestras)
    assert slis.a_dict() == calcular_slis(corrida_demo["muestras"]).a_dict()
    guardar_evidencia("slis_slos.json", {"slis": slis.a_dict(), "slos": [e.a_dict() for e in evaluar_slos(muestras)]})
