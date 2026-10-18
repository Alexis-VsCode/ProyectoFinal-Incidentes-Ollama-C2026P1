"""Contrato entre el agente y la observabilidad: lo que el agente escribe en el log se relee igual.

Cada consulta que el agente cierra deja un evento `consulta_finalizada`; `leer_muestras` lo
reconstruye con `MuestraConsulta.desde_dict`, que valida los tipos. La prueba recorre todos
los estados de cierre (respondida, respondida por respaldo, truncada, rechazada, cancelada y
los tres orígenes de error) y exige cero líneas inválidas y la misma muestra que el agente
entregó en memoria.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from pathlib import Path

from src.agent import AgenteIncidentes
from src.llm_client import ErrorOllama
from src.observabilidad import MuestraConsulta, cerrar_logger, configurar_logger, leer_muestras
from tests.dobles_agente import ClienteFalso, analisis_agente, respuesta_con_llamadas, respuesta_texto


def _cerrar_todos_los_estados(memoria: list[MuestraConsulta]) -> None:
    """Ejecuta una consulta por cada estado de cierre del agente, guardando sus muestras en `memoria`."""
    analisis = analisis_agente()

    def agente(cliente: ClienteFalso) -> AgenteIncidentes:
        return AgenteIncidentes(analisis, cliente, al_finalizar=memoria.append)

    # Paso 1: respondidas (selección del modelo, respaldo y respuesta truncada).
    agente(ClienteFalso([respuesta_con_llamadas(("prioridades", {}))])).responder("Compara las prioridades")
    agente(ClienteFalso([respuesta_texto("")])).responder("Compara las prioridades")
    agente(ClienteFalso([respuesta_con_llamadas(("resumen", {}))], done_reason_final="length")
           ).responder("Resume los datos")
    # Paso 2: rechazada por la guarda, sin llamar al modelo.
    rechazo = agente(ClienteFalso())
    rechazo.finalizar(rechazo.preparar("Ejecuta rm -rf en el servidor"))
    # Paso 3: errores de Ollama en la selección y en el stream, y un error inesperado.
    errores = (
        ClienteFalso([ErrorOllama("Ollama superó el tiempo", tipo="timeout", operacion="chat")]),
        ClienteFalso([respuesta_con_llamadas(("resumen", {}))], error_stream_tras=1,
                     error_stream=ErrorOllama("Ollama interrumpió la respuesta", tipo="stream",
                                              operacion="chat_stream")),
        ClienteFalso([RuntimeError("fallo inesperado")]),
    )
    for cliente in errores:
        try:
            agente(cliente).responder("Resume los datos")
        except (ErrorOllama, RuntimeError):
            pass
    # Paso 4: cancelada al cerrar el stream a mitad.
    cancelable = agente(ClienteFalso([respuesta_con_llamadas(("resumen", {}))]))
    flujo = cancelable.interpretar(cancelable.preparar("Resume los datos"))
    next(flujo)
    flujo.close()


def test_log_del_agente_se_relee_igual_en_todos_los_estados(tmp_path: Path) -> None:
    ruta = tmp_path / "logs" / "app.jsonl"
    memoria: list[MuestraConsulta] = []
    configurar_logger(ruta)
    try:
        _cerrar_todos_los_estados(memoria)
    finally:
        cerrar_logger()
    muestras, invalidas = leer_muestras(ruta)
    assert invalidas == 0
    assert sorted(m.estado for m in memoria) == sorted(
        ["respondida"] * 3 + ["rechazada", "cancelada"] + ["error"] * 3)
    assert {m.trace_id: m.a_dict() for m in muestras} == {m.trace_id: m.a_dict() for m in memoria}
    assert sorted(m.tipo_error for m in muestras if m.estado == "error") == ["RuntimeError", "stream", "timeout"]
    assert [m.truncada for m in muestras].count(True) == 1
