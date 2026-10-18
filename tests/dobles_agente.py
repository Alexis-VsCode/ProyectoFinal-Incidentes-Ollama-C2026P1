"""Dobles del agente: cliente de chat programable, flujo falso y un CSV sintético pequeño.

`ClienteFalso` cumple el contrato `ClienteChat` sin red ni modelo: devuelve las selecciones
programadas, transmite fragmentos fijos y registra cada llamada. `analisis_agente` construye
un `AnalisisIncidentes` real con `cargar_csv` sobre ocho incidentes ficticios, para que las
herramientas ejecuten la lógica de eda de verdad.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import csv
from collections.abc import Callable, Iterator, Sequence
from io import StringIO
from typing import Any

from src.data_loader import cargar_csv
from src.eda import AnalisisIncidentes
from src.llm_client import ErrorOllama, InfoModelo, LlamadaHerramienta, Metricas, RespuestaModelo

DIGEST_FALSO = "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"
VERSION_FALSA = "0.40.6"
FRAGMENTOS_POR_DEFECTO: tuple[str, ...] = ("**Hechos observados** ", "respuesta ", "simulada.")

COLUMNAS_CSV_AGENTE: tuple[str, ...] = (
    "number", "priority", "incident_state", "assignment_group", "category",
    "opened_at", "resolved_at", "closed_at", "sys_updated_at",
    "reassignment_count", "reopen_count", "sys_mod_count",
)
# Ocho incidentes ficticios: cinco grupos (uno desconocido), cuatro categorías (una desconocida),
# un incidente sin resolución y duraciones de 0,5 a 72 h.
FILAS_CSV_AGENTE: tuple[tuple[str, ...], ...] = (
    ("A1", "1 - Critical", "Closed", "Group 70", "Category 26", "01/03/2016 08:00", "01/03/2016 09:00", "01/03/2016 10:00", "01/03/2016 10:00", "0", "0", "2"),
    ("A2", "1 - Critical", "Closed", "Group 70", "Category 26", "02/03/2016 08:00", "03/03/2016 08:00", "03/03/2016 09:00", "03/03/2016 09:00", "1", "0", "4"),
    ("A3", "2 - High", "Closed", "Group 25", "Category 26", "03/03/2016 08:00", "03/03/2016 20:00", "03/03/2016 21:00", "03/03/2016 21:00", "2", "1", "6"),
    ("A4", "3 - Moderate", "Closed", "Group 25", "Category 42", "04/03/2016 08:00", "06/03/2016 08:00", "06/03/2016 09:00", "06/03/2016 09:00", "3", "0", "8"),
    ("A5", "3 - Moderate", "Active", "?", "Category 42", "05/03/2016 08:00", "?", "?", "05/03/2016 09:00", "4", "0", "10"),
    ("A6", "3 - Moderate", "Closed", "Group 39", "Category 53", "06/03/2016 08:00", "06/03/2016 14:00", "06/03/2016 15:00", "06/03/2016 15:00", "1", "0", "3"),
    ("A7", "4 - Low", "Closed", "Group 70", "?", "07/03/2016 08:00", "10/03/2016 08:00", "10/03/2016 09:00", "10/03/2016 09:00", "5", "2", "12"),
    ("A8", "4 - Low", "Closed", "Group 24", "Category 26", "08/03/2016 08:00", "08/03/2016 08:30", "08/03/2016 09:00", "08/03/2016 09:00", "0", "0", "1"),
)
PRIORIDADES_AGENTE: tuple[str, ...] = ("1 - Critical", "2 - High", "3 - Moderate", "4 - Low")


def csv_agente() -> bytes:
    """Bytes UTF-8 del CSV sintético del agente (8 eventos, 8 incidentes)."""
    salida = StringIO()
    escritor = csv.writer(salida, lineterminator="\n")
    escritor.writerow(COLUMNAS_CSV_AGENTE)
    escritor.writerows(FILAS_CSV_AGENTE)
    return salida.getvalue().encode("utf-8")


def analisis_agente(prioridades: Sequence[str] = PRIORIDADES_AGENTE) -> AnalisisIncidentes:
    """Fachada de análisis sobre el CSV sintético, filtrada por `prioridades`."""
    datos = cargar_csv(csv_agente())
    return AnalisisIncidentes(datos.incidentes, datos.control, list(prioridades))


def metricas_falsas(eval_count: int = 20, eval_s: float = 2.0, prompt_eval_count: int = 300) -> Metricas:
    """Métricas de Ollama en nanosegundos con `eval_count / eval_s` tokens por segundo."""
    return Metricas(
        total_duration_ns=int((eval_s + 0.5) * 1e9), load_duration_ns=int(0.1 * 1e9),
        prompt_eval_count=prompt_eval_count, prompt_eval_cached_count=None,
        prompt_eval_duration_ns=int(0.4 * 1e9), eval_count=eval_count, eval_duration_ns=int(eval_s * 1e9),
    )


def respuesta_con_llamadas(*pares: tuple[str, Any], modelo: str = "llama3.2:3b",
                           done_reason: str | None = "stop") -> RespuestaModelo:
    """Respuesta de selección con una `LlamadaHerramienta` por cada par (nombre, argumentos)."""
    llamadas = tuple(LlamadaHerramienta(nombre, argumentos, i) for i, (nombre, argumentos) in enumerate(pares))
    return RespuestaModelo(contenido="", llamadas=llamadas, done_reason=done_reason, modelo=modelo,
                           metricas=metricas_falsas(eval_count=25))


def respuesta_texto(texto: str, *, modelo: str = "llama3.2:3b", done_reason: str | None = "stop") -> RespuestaModelo:
    """Respuesta sin llamadas a herramientas: solo `contenido`."""
    return RespuestaModelo(contenido=texto, llamadas=(), done_reason=done_reason, modelo=modelo,
                           metricas=metricas_falsas(eval_count=25))


def info_falsa(modelo: str = "llama3.2:3b") -> InfoModelo:
    """Metadatos de un modelo instalado con soporte de herramientas."""
    return InfoModelo(nombre=modelo, digest=DIGEST_FALSO, tamano_bytes=2019393189, familia="llama",
                      parametros="3.2B", cuantizacion="Q4_K_M", capacidades=("completion", "tools"),
                      contexto_maximo=131072, modificado="2026-10-01T10:00:00-05:00")


class FlujoFalso:
    """Iterable de un solo uso que imita a `FlujoRespuesta`.

    Entrega `fragmentos` en orden. Si `error_tras` es un entero, lanza `error` después de
    entregar ese número de fragmentos. `respuesta` lanza RuntimeError hasta consumirlo entero.
    """

    def __init__(self, fragmentos: Sequence[str], *, modelo: str, done_reason: str | None = "stop",
                 error: ErrorOllama | None = None, error_tras: int | None = None) -> None:
        self.fragmentos = tuple(fragmentos)
        self.modelo = modelo
        self.done_reason = done_reason
        self.error = error
        self.error_tras = error_tras
        self._respuesta: RespuestaModelo | None = None
        self._usado = False

    def __iter__(self) -> Iterator[str]:
        if self._usado:
            raise RuntimeError("El flujo ya se consumió.")
        self._usado = True
        return self._generar()

    def _generar(self) -> Iterator[str]:
        for i, fragmento in enumerate(self.fragmentos):
            if self.error is not None and self.error_tras == i:
                raise self.error
            yield fragmento
        if self.error is not None and (self.error_tras is None or self.error_tras >= len(self.fragmentos)):
            raise self.error
        self._respuesta = RespuestaModelo(contenido="".join(self.fragmentos), llamadas=(),
                                          done_reason=self.done_reason, modelo=self.modelo,
                                          metricas=metricas_falsas(eval_count=140, eval_s=20.0, prompt_eval_count=900))

    @property
    def respuesta(self) -> RespuestaModelo:
        if self._respuesta is None:
            raise RuntimeError("Consuma el flujo completo antes de leer la respuesta.")
        return self._respuesta


Selector = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], RespuestaModelo]


class ClienteFalso:
    """Cliente de chat programable que cumple `ClienteChat` y registra todas sus llamadas.

    `selecciones` es una lista de `RespuestaModelo` o excepciones que `chat` consume en orden
    (vacía: responde texto vacío), o una función `(mensajes, herramientas) -> RespuestaModelo`.
    `error_stream` se lanza al llamar a `chat_stream` si `error_stream_tras` es None; si no,
    a mitad del flujo. `error_info` y `error_version` hacen fallar los metadatos.
    """

    def __init__(self, selecciones: Sequence[RespuestaModelo | BaseException] | Selector = (), *,
                 fragmentos: Sequence[str] = FRAGMENTOS_POR_DEFECTO, done_reason_final: str | None = "stop",
                 modelo: str = "llama3.2:3b", info: InfoModelo | None = None, version: str = VERSION_FALSA,
                 error_stream: ErrorOllama | None = None, error_stream_tras: int | None = None,
                 error_info: ErrorOllama | None = None, error_version: ErrorOllama | None = None,
                 seed: int | None = 42) -> None:
        self.modelo = modelo
        self.seed = seed
        self._selector: Selector | None = selecciones if callable(selecciones) else None
        self._selecciones: list[RespuestaModelo | BaseException] = [] if callable(selecciones) else list(selecciones)
        self.fragmentos = tuple(fragmentos)
        self.done_reason_final = done_reason_final
        self.info = info or info_falsa(modelo)
        self._version = version
        self.error_stream = error_stream
        self.error_stream_tras = error_stream_tras
        self.error_info = error_info
        self.error_version = error_version
        self.llamadas_chat: list[dict[str, Any]] = []
        self.llamadas_stream: list[dict[str, Any]] = []
        self.precalentamientos = 0

    def opciones_efectivas(self, opciones: dict[str, Any] | None = None) -> dict[str, Any]:
        """Opciones que enviaría el cliente real: num_ctx constante y semilla, más `opciones`."""
        efectivas: dict[str, Any] = {"num_ctx": 4096}
        if self.seed is not None:
            efectivas["seed"] = self.seed
        return efectivas | dict(opciones or {})

    def chat(self, mensajes: list[dict[str, Any]], herramientas: list[dict[str, Any]] | None = None,
             *, opciones: dict[str, Any] | None = None) -> RespuestaModelo:
        self.llamadas_chat.append({"mensajes": mensajes, "herramientas": herramientas, "opciones": opciones})
        if self._selector is not None:
            return self._selector(mensajes, herramientas)
        elemento = self._selecciones.pop(0) if self._selecciones else respuesta_texto("", modelo=self.modelo)
        if isinstance(elemento, BaseException):
            raise elemento
        return elemento

    def chat_stream(self, mensajes: list[dict[str, Any]], *, opciones: dict[str, Any] | None = None) -> FlujoFalso:
        self.llamadas_stream.append({"mensajes": mensajes, "opciones": opciones})
        if self.error_stream is not None and self.error_stream_tras is None:
            raise self.error_stream
        return FlujoFalso(self.fragmentos, modelo=self.modelo, done_reason=self.done_reason_final,
                          error=self.error_stream, error_tras=self.error_stream_tras)

    def info_modelo(self) -> InfoModelo:
        if self.error_info is not None:
            raise self.error_info
        return self.info

    def version(self) -> str:
        if self.error_version is not None:
            raise self.error_version
        return self._version

    def precalentar(self) -> RespuestaModelo:
        """Simula la carga del modelo en memoria (`done_reason == "load"`)."""
        self.precalentamientos += 1
        if self.error_info is not None:
            raise self.error_info
        return respuesta_texto("", modelo=self.modelo, done_reason="load")
