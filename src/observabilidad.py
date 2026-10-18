"""Observabilidad de la aplicación y su agente: logger JSONL, trazas, spans, SLI y SLO.

Solo mide la aplicación: latencia de la consulta, errores de Ollama y cifras verificadas.
La duración de apertura a resolución de los incidentes del CSV es un dato histórico de otra
organización; no es SLI, SLO ni MTTR. Los logs nunca guardan el texto de la pregunta, el
contexto ni la respuesta: solo longitudes, huellas y métricas.

Cada línea del log es un objeto con las claves de `CAMPOS`. Cada consulta emite un único
evento `consulta_finalizada` (una `MuestraConsulta`), del que se derivan todos los SLI. La
ventana de los SLO es el lote leído (un archivo de log o una corrida del comparador).

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json
import logging
import math
import numbers
import os
import threading
import uuid
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

NOMBRE_LOGGER = "incidentes"
VARIABLE_ENTORNO_LOGS = "INCIDENTES_LOG_DIR"
NOMBRE_ARCHIVO_LOG = "app.jsonl"
RUTA_LOGS: Path = Path(__file__).resolve().parents[1] / "resultados_locales" / "logs" / NOMBRE_ARCHIVO_LOG
CAMPOS = ("ts_utc", "nivel", "evento", "trace_id", "atributos")
SPAN_SELECCION = "seleccion_herramienta"
SPAN_EJECUCION = "ejecucion_herramienta"
SPAN_INTERPRETACION = "interpretacion"
EVENTO_INICIO = "consulta_iniciada"
EVENTO_FIN = "consulta_finalizada"
EVENTO_OLLAMA_ERROR = "ollama_error"
EVENTO_RESPALDO = "respaldo_activado"
EVENTO_RECHAZO = "guarda_rechazo"
EstadoSpan = Literal["ok", "error", "cancelado"]
EstadoConsulta = Literal["respondida", "rechazada", "error", "cancelada"]
ESTADOS_CONSULTA: tuple[str, ...] = ("respondida", "rechazada", "error", "cancelada")
ORIGENES_MODELO = frozenset({"modelo", "modelo_texto"})
ORIGENES_RESPALDO = frozenset({"respaldo_reglas", "respaldo_defecto"})
COLUMNAS_ULTIMAS_CONSULTAS = ("trace_id", "ts_utc", "evento", "ms", "estado")
COLUMNAS_TABLA_SLOS = ("slo", "descripcion", "objetivo", "eventos_validos", "eventos_buenos", "cumplimiento",
                       "resultado", "presupuesto_total", "presupuesto_consumido", "presupuesto_restante",
                       "fraccion_restante")
MAX_CARACTERES_ERROR = 300
_RESERVADOS_SPAN = frozenset({"tipo", "ms", "estado"})

_CAMPOS_REALES_OPCIONALES = ("segundos", "segundos_primer_token", "tokens_por_segundo")
_CAMPOS_TEXTO_OPCIONALES = ("done_reason", "origen_seleccion", "tipo_error")
_CANDADO_CONFIGURACION = threading.Lock()

_logger = logging.getLogger(NOMBRE_LOGGER)
if not any(isinstance(h, logging.NullHandler) for h in _logger.handlers):
    _logger.addHandler(logging.NullHandler())


# Paso 1: serialización y logger


def _a_json(valor: Any) -> Any:
    """Convierte a tipos JSON: numpy (escalares y arreglos) a Python, no finitos a None, tuplas a listas
    y lo demás a str."""
    if isinstance(valor, np.ndarray):
        valor = valor.tolist()
    if isinstance(valor, np.generic):
        valor = valor.item()
    if valor is None or isinstance(valor, (bool, str)):
        return valor
    if isinstance(valor, numbers.Integral):
        return int(valor)
    if isinstance(valor, numbers.Real):
        numero = float(valor)
        return numero if math.isfinite(numero) else None
    if isinstance(valor, dict):
        return {str(clave): _a_json(subvalor) for clave, subvalor in valor.items()}
    if isinstance(valor, (list, tuple, set, frozenset)):
        return [_a_json(subvalor) for subvalor in valor]
    return str(valor)


def _ahora_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class FormateadorJSON(logging.Formatter):
    """Una línea JSON por registro: el evento propio tal cual o, si es de terceros, uno construido."""

    def format(self, record: logging.LogRecord) -> str:
        evento = getattr(record, "evento_json", None)
        if not isinstance(evento, dict):
            evento = {
                "ts_utc": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                "nivel": record.levelname, "evento": record.getMessage(), "trace_id": None,
                "atributos": {"logger": record.name},
            }
        return json.dumps(_a_json(evento), ensure_ascii=False, allow_nan=False, default=str)


def ruta_log() -> Path:
    """Ruta del log de la aplicación: `<INCIDENTES_LOG_DIR>/app.jsonl` si la variable existe, si no `RUTA_LOGS`.

    La variable de entorno se lee en cada llamada.
    """
    carpeta = os.environ.get(VARIABLE_ENTORNO_LOGS)
    return Path(carpeta) / NOMBRE_ARCHIVO_LOG if carpeta else RUTA_LOGS


def _manejadores_archivo(logger: logging.Logger) -> list[logging.FileHandler]:
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def configurar_logger(ruta: Path | None = None, nivel: int = logging.INFO) -> logging.Logger:
    """Deja exactamente un FileHandler JSONL (utf-8, apertura diferida) apuntando a `ruta`.

    Es idempotente, también entre hilos (cada sesión de Streamlit ejecuta app.py en el suyo):
    si ya existe uno con la misma ruta lo conserva; si apunta a otra, lo cierra y lo
    sustituye. Crea la carpeta, fija el nivel y desactiva la propagación. Sin `ruta` usa
    `ruta_log()`.
    """
    destino = Path(ruta) if ruta is not None else ruta_log()
    objetivo = os.path.normcase(os.path.abspath(destino))
    logger = logging.getLogger(NOMBRE_LOGGER)
    with _CANDADO_CONFIGURACION:
        logger.setLevel(nivel)
        logger.propagate = False
        vigente = None
        for manejador in _manejadores_archivo(logger):
            if vigente is None and os.path.normcase(manejador.baseFilename) == objetivo:
                vigente = manejador
                continue
            logger.removeHandler(manejador)
            manejador.close()
        if vigente is None:
            destino.parent.mkdir(parents=True, exist_ok=True)
            vigente = logging.FileHandler(destino, encoding="utf-8", delay=True)
            logger.addHandler(vigente)
        vigente.setFormatter(FormateadorJSON())
    return logger


def cerrar_logger() -> None:
    """Cierra y quita los FileHandler (libera el archivo en Windows) y deja el logger como al importar.

    Conserva el NullHandler y restablece nivel NOTSET y propagación.
    """
    logger = logging.getLogger(NOMBRE_LOGGER)
    with _CANDADO_CONFIGURACION:
        for manejador in _manejadores_archivo(logger):
            logger.removeHandler(manejador)
            manejador.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


def nuevo_trace_id() -> str:
    """Identificador de consulta: 32 caracteres hexadecimales (uuid4)."""
    return uuid.uuid4().hex


def crear_evento(evento: str, trace_id: str | None, nivel: int, atributos: dict[str, Any]) -> dict[str, Any]:
    """Evento con las claves de `CAMPOS`; `ts_utc` en ISO-8601 UTC con milisegundos.

    Los atributos se convierten a tipos JSON al crearlo, de modo que el dict en memoria es
    igual a la línea que se escribe en el archivo.
    """
    return {"ts_utc": _ahora_utc(), "nivel": logging.getLevelName(nivel), "evento": evento,
            "trace_id": trace_id, "atributos": _a_json(dict(atributos))}


def _emitir(datos: dict[str, Any], nivel: int, destino: list[dict[str, Any]] | None) -> dict[str, Any]:
    if destino is not None:
        destino.append(datos)
    _logger.log(nivel, datos["evento"], extra={"evento_json": datos})
    return datos


def registrar_evento(evento: str, *, trace_id: str | None = None, nivel: int = logging.INFO,
                     destino: list[dict[str, Any]] | None = None, **atributos: Any) -> dict[str, Any]:
    """Crea el evento una vez, lo añade a `destino` y lo registra; devuelve ese mismo dict."""
    return _emitir(crear_evento(evento, trace_id, nivel, atributos), nivel, destino)


# Paso 2: spans


@dataclass
class Span:
    """Tramo medido de una consulta; el bloque puede añadir atributos (eval_count, etc.)."""

    nombre: str
    trace_id: str
    atributos: dict[str, Any] = field(default_factory=dict)
    ms: float | None = None
    estado: EstadoSpan = "ok"


@contextmanager
def span(nombre: str, trace_id: str, *, destino: list[dict[str, Any]] | None = None,
         **atributos: Any) -> Iterator[Span]:
    """Mide el bloque con `perf_counter` y siempre registra un evento `nombre` al salir.

    Una excepción marca "error" (nivel ERROR, con `error_tipo` y `error` de 300 caracteres
    como máximo) y se relanza. GeneratorExit y otras interrupciones marcan "cancelado" y se
    relanzan. Atributos: {"tipo": "span", "ms", "estado", **atributos}; los atributos del
    bloque no pueden sustituir `tipo`, `ms` ni `estado`.
    """
    actual = Span(nombre=nombre, trace_id=trace_id, atributos=dict(atributos))
    inicio = perf_counter()
    try:
        yield actual
    except Exception as exc:
        actual.estado = "error"
        actual.atributos["error_tipo"] = type(exc).__name__
        actual.atributos["error"] = str(exc)[:MAX_CARACTERES_ERROR]
        raise
    except BaseException:
        actual.estado = "cancelado"
        raise
    finally:
        actual.ms = round((perf_counter() - inicio) * 1000, 3)
        nivel = logging.ERROR if actual.estado == "error" else logging.INFO
        propios = {k: v for k, v in actual.atributos.items() if k not in _RESERVADOS_SPAN}
        _emitir(crear_evento(nombre, trace_id, nivel,
                             {"tipo": "span", "ms": actual.ms, "estado": actual.estado, **propios}),
                nivel, destino)


# Paso 3: muestras de consulta y lectura del log


@dataclass(frozen=True)
class MuestraConsulta:
    """Atributos del evento `consulta_finalizada`: una muestra por consulta, fuente de los SLI.

    Lanza ValueError si `estado` no es respondida, rechazada, error ni cancelada.
    """

    trace_id: str
    estado: EstadoConsulta
    modelo: str
    modo: str
    segundos: float | None
    segundos_primer_token: float | None
    tokens_por_segundo: float | None
    done_reason: str | None
    truncada: bool
    origen_seleccion: str | None
    cifras_evaluadas: int
    cifras_verificadas: int
    tipo_error: str | None

    def __post_init__(self) -> None:
        if self.estado not in ESTADOS_CONSULTA:
            raise ValueError(f"Estado de consulta no válido: {self.estado!r}")

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def desde_dict(cls, datos: dict[str, Any]) -> MuestraConsulta:
        """Reconstruye la muestra ignorando claves extra; ValueError si falta un campo o un valor no vale.

        Exige `trace_id`, `modelo` y `modo` de texto (el `trace_id` no vacío); `segundos`,
        `segundos_primer_token` y `tokens_por_segundo` None o números finitos (no bool);
        `done_reason`, `origen_seleccion` y `tipo_error` None o texto; las cifras, enteros no
        negativos; y `truncada`, booleano. Así una línea del log con tipos alterados se descarta
        en lugar de romper el cálculo de los SLI.
        """
        if not isinstance(datos, dict):
            raise ValueError("La muestra debe ser un objeto.")
        nombres = [f.name for f in fields(cls)]
        faltan = [n for n in nombres if n not in datos]
        if faltan:
            raise ValueError("Faltan campos de la muestra: " + ", ".join(faltan))
        # Paso 1: textos obligatorios y opcionales
        if not isinstance(datos["trace_id"], str) or not datos["trace_id"]:
            raise ValueError("trace_id debe ser un texto no vacío.")
        for nombre in ("modelo", "modo"):
            if not isinstance(datos[nombre], str):
                raise ValueError(f"{nombre} debe ser texto.")
        for nombre in _CAMPOS_TEXTO_OPCIONALES:
            if datos[nombre] is not None and not isinstance(datos[nombre], str):
                raise ValueError(f"{nombre} debe ser texto o None.")
        # Paso 2: números opcionales, cifras y bandera de truncado
        for nombre in _CAMPOS_REALES_OPCIONALES:
            valor = datos[nombre]
            if valor is not None and (isinstance(valor, bool) or not isinstance(valor, numbers.Real)
                                      or not math.isfinite(valor)):
                raise ValueError(f"{nombre} debe ser un número finito o None.")
        for nombre in ("cifras_evaluadas", "cifras_verificadas"):
            valor = datos[nombre]
            if isinstance(valor, bool) or not isinstance(valor, int) or valor < 0:
                raise ValueError(f"{nombre} debe ser un entero no negativo.")
        if not isinstance(datos["truncada"], bool):
            raise ValueError("truncada debe ser booleano.")
        return cls(**{n: datos[n] for n in nombres})


def registrar_fin_consulta(muestra: MuestraConsulta, *,
                           destino: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Emite `consulta_finalizada` con `muestra.a_dict()` como atributos (WARNING si hubo error)."""
    nivel = logging.WARNING if muestra.estado == "error" else logging.INFO
    return _emitir(crear_evento(EVENTO_FIN, muestra.trace_id, nivel, muestra.a_dict()), nivel, destino)


def _fecha_utc(texto: Any) -> datetime:
    fecha = datetime.fromisoformat(texto)
    return fecha if fecha.tzinfo is not None else fecha.replace(tzinfo=timezone.utc)


def _lineas_json(ruta: Path) -> Iterator[Any]:
    """Objetos JSON del archivo en orden; las líneas corruptas llegan como None y las vacías se omiten."""
    with ruta.open(encoding="utf-8", errors="replace") as archivo:
        for linea in archivo:
            if not linea.strip():
                continue
            try:
                yield json.loads(linea)
            except ValueError:
                yield None


def leer_muestras(ruta: Path | None = None, *, desde: datetime | None = None,
                  modelo: str | None = None) -> tuple[list[MuestraConsulta], int]:
    """Muestras de los eventos `consulta_finalizada` del log y número de líneas inválidas.

    Si un trace_id se repite, gana el último. `desde` sin zona se toma como UTC. Nunca lanza
    por una línea corrupta; sin archivo devuelve ([], 0). Sin `ruta` usa `ruta_log()`.
    """
    origen = Path(ruta) if ruta is not None else ruta_log()
    if not origen.exists():
        return [], 0
    if desde is not None and desde.tzinfo is None:
        desde = desde.replace(tzinfo=timezone.utc)
    por_trace: dict[str, tuple[datetime, MuestraConsulta]] = {}
    invalidas = 0
    for evento in _lineas_json(origen):
        if not isinstance(evento, dict):
            invalidas += 1
            continue
        if evento.get("evento") != EVENTO_FIN:
            continue
        try:
            muestra = MuestraConsulta.desde_dict({"trace_id": evento.get("trace_id"), **evento["atributos"]})
            fecha = _fecha_utc(evento["ts_utc"])
        except (KeyError, TypeError, ValueError):
            invalidas += 1
            continue
        por_trace.pop(muestra.trace_id, None)
        por_trace[muestra.trace_id] = (fecha, muestra)
    muestras = [m for fecha, m in por_trace.values()
                if (desde is None or fecha >= desde) and (modelo is None or m.modelo == modelo)]
    return muestras, invalidas


def leer_eventos(ruta: Path | None = None, limite: int = 2000) -> list[dict[str, Any]]:
    """Los últimos `limite` eventos válidos del log, en orden; ignora líneas corruptas y [] sin archivo."""
    if limite < 1:
        raise ValueError("limite debe ser al menos 1.")
    origen = Path(ruta) if ruta is not None else ruta_log()
    if not origen.exists():
        return []
    eventos: deque[dict[str, Any]] = deque(maxlen=limite)
    for evento in _lineas_json(origen):
        if isinstance(evento, dict):
            eventos.append(evento)
    return list(eventos)


def ultimas_consultas(eventos: Sequence[dict[str, Any]], n: int = 10) -> pd.DataFrame:
    """Una fila por evento o span de las últimas `n` consultas (por trace_id), en orden cronológico.

    Columnas `COLUMNAS_ULTIMAS_CONSULTAS`; `ms` y `estado` salen de los atributos si existen.
    Solo cuentan los eventos con `trace_id` de texto no vacío.
    """
    if n < 1:
        raise ValueError("n debe ser al menos 1.")
    validos = [e for e in eventos if isinstance(e, dict) and isinstance(e.get("trace_id"), str) and e["trace_id"]]
    recientes: dict[str, None] = {}
    for evento in validos:
        recientes.pop(evento["trace_id"], None)
        recientes[evento["trace_id"]] = None
    elegidos = set(list(recientes)[-n:])
    filas = []
    for evento in validos:
        if evento["trace_id"] not in elegidos:
            continue
        atributos = evento.get("atributos") if isinstance(evento.get("atributos"), dict) else {}
        filas.append({"trace_id": evento["trace_id"], "ts_utc": evento.get("ts_utc"),
                      "evento": evento.get("evento"), "ms": atributos.get("ms"), "estado": atributos.get("estado")})
    return pd.DataFrame(filas, columns=list(COLUMNAS_ULTIMAS_CONSULTAS))


# Paso 4: SLI


def percentil(valores: Sequence[float], q: float) -> float | None:
    """Percentil por rango más cercano: orden[ceil(q·n) − 1]; None si no hay valores."""
    if not 0 <= q <= 1:
        raise ValueError("q debe estar entre 0 y 1.")
    orden = sorted(float(v) for v in valores)
    if not orden:
        return None
    return orden[max(math.ceil(round(q * len(orden), 9)) - 1, 0)]


def _tasa(numerador: float, denominador: float, decimales: int = 4) -> float | None:
    return None if denominador == 0 else round(numerador / denominador, decimales)


def _redondear(valor: float | None, decimales: int) -> float | None:
    return None if valor is None else round(valor, decimales) + 0.0


@dataclass(frozen=True)
class SLIs:
    """Indicadores de la aplicación sobre un lote; tasas con 4 decimales y segundos con 3."""

    consultas: int
    validas: int
    respondidas: int
    rechazadas: int
    canceladas: int
    errores: int
    errores_por_tipo: dict[str, int]
    latencia_p50_s: float | None
    latencia_p95_s: float | None
    primer_token_p50_s: float | None
    primer_token_p95_s: float | None
    tokens_por_segundo_p50: float | None
    tasa_error: float | None
    tasa_truncadas: float | None
    tasa_seleccion_modelo: float | None
    tasa_respaldo: float | None
    cifras_evaluadas: int
    cifras_verificadas: int
    proporcion_cifras_verificadas: float | None

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def calcular_slis(muestras: Iterable[MuestraConsulta]) -> SLIs:
    """SLI del lote. Válidas = respondidas + error; rechazadas y canceladas no cuentan.

    Latencias, primer token y tokens/s salen de las respondidas; la selección del modelo y el
    respaldo, de las respondidas en modo automático; las cifras, de las sumas de las respondidas.
    """
    lista = list(muestras)
    respondidas = [m for m in lista if m.estado == "respondida"]
    errores = [m for m in lista if m.estado == "error"]
    latencias = [m.segundos for m in respondidas if m.segundos is not None]
    primeros = [m.segundos_primer_token for m in respondidas if m.segundos_primer_token is not None]
    velocidades = [m.tokens_por_segundo for m in respondidas if m.tokens_por_segundo is not None]
    automaticas = [m for m in respondidas
                   if m.modo == "automatico" and m.origen_seleccion in ORIGENES_MODELO | ORIGENES_RESPALDO]
    del_modelo = sum(1 for m in automaticas if m.origen_seleccion in ORIGENES_MODELO)
    evaluadas = sum(m.cifras_evaluadas for m in respondidas)
    verificadas = sum(m.cifras_verificadas for m in respondidas)
    validas = len(respondidas) + len(errores)
    return SLIs(
        consultas=len(lista), validas=validas, respondidas=len(respondidas),
        rechazadas=sum(1 for m in lista if m.estado == "rechazada"),
        canceladas=sum(1 for m in lista if m.estado == "cancelada"),
        errores=len(errores), errores_por_tipo=dict(Counter(m.tipo_error or "desconocido" for m in errores)),
        latencia_p50_s=_redondear(percentil(latencias, 0.5), 3),
        latencia_p95_s=_redondear(percentil(latencias, 0.95), 3),
        primer_token_p50_s=_redondear(percentil(primeros, 0.5), 3),
        primer_token_p95_s=_redondear(percentil(primeros, 0.95), 3),
        tokens_por_segundo_p50=_redondear(percentil(velocidades, 0.5), 2),
        tasa_error=_tasa(len(errores), validas),
        tasa_truncadas=_tasa(sum(1 for m in respondidas if m.truncada), len(respondidas)),
        tasa_seleccion_modelo=_tasa(del_modelo, len(automaticas)),
        tasa_respaldo=_tasa(len(automaticas) - del_modelo, len(automaticas)),
        cifras_evaluadas=evaluadas, cifras_verificadas=verificadas,
        proporcion_cifras_verificadas=_tasa(verificadas, evaluadas))


# Paso 5: SLO y presupuesto de error


@dataclass(frozen=True)
class SLO:
    """Objetivo sobre la proporción de eventos buenos; `umbral_s` es obligatorio en latencia."""

    nombre: str
    descripcion: str
    sli: Literal["latencia", "disponibilidad", "cifras"]
    objetivo: float
    umbral_s: float | None = None
    minimo_eventos: int = 20
    fuente: str = "supuesto inicial; calibrar con scripts.comparar_modelos"

    def __post_init__(self) -> None:
        if self.sli not in ("latencia", "disponibilidad", "cifras"):
            raise ValueError(f"SLI no válido: {self.sli!r}")
        if not 0 < self.objetivo < 1:
            raise ValueError("El objetivo debe estar entre 0 y 1 (sin incluirlos).")
        if self.sli == "latencia" and (self.umbral_s is None or self.umbral_s <= 0):
            raise ValueError("El SLO de latencia necesita un umbral_s positivo.")
        if self.minimo_eventos < 1:
            raise ValueError("minimo_eventos debe ser al menos 1.")


SLOS: tuple[SLO, ...] = (
    SLO("latencia_consulta", "95 % de las consultas respondidas terminan en menos de 90 s", "latencia", 0.95,
        umbral_s=90.0),
    SLO("disponibilidad_agente", "95 % de las consultas válidas terminan sin error de Ollama", "disponibilidad", 0.95),
    SLO("cifras_verificadas", "90 % de las cifras citadas existen en la evidencia enviada", "cifras", 0.90),
)


@dataclass(frozen=True)
class EvaluacionSLO:
    """Cumplimiento de un SLO en un lote y su presupuesto de error (puede quedar negativo)."""

    slo: str
    objetivo: float
    eventos_validos: int
    eventos_buenos: int
    cumplimiento: float | None
    cumple: bool | None
    suficiente: bool
    presupuesto_total: float
    presupuesto_consumido: int
    presupuesto_restante: float
    fraccion_restante: float | None

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluar_slo(slo: SLO, muestras: Sequence[MuestraConsulta]) -> EvaluacionSLO:
    """Evalúa un SLO: latencia (respondidas con segundos < umbral, estricto), disponibilidad
    (respondidas frente a respondidas + error) o cifras (suma de verificadas frente a evaluadas).
    """
    lista = list(muestras)
    respondidas = [m for m in lista if m.estado == "respondida"]
    if slo.sli == "latencia":
        tiempos = [m.segundos for m in respondidas if m.segundos is not None]
        validos, buenos = len(tiempos), sum(1 for s in tiempos if s < slo.umbral_s)
    elif slo.sli == "disponibilidad":
        validos = len(respondidas) + sum(1 for m in lista if m.estado == "error")
        buenos = len(respondidas)
    else:
        validos = sum(m.cifras_evaluadas for m in respondidas)
        buenos = sum(m.cifras_verificadas for m in respondidas)
    total = (1 - slo.objetivo) * validos
    consumido = validos - buenos
    restante = total - consumido
    return EvaluacionSLO(
        slo=slo.nombre, objetivo=slo.objetivo, eventos_validos=validos, eventos_buenos=buenos,
        cumplimiento=_tasa(buenos, validos),
        cumple=None if validos == 0 else buenos >= slo.objetivo * validos - 1e-9,
        suficiente=validos >= slo.minimo_eventos,
        presupuesto_total=_redondear(total, 3), presupuesto_consumido=consumido,
        presupuesto_restante=_redondear(restante, 3),
        fraccion_restante=_redondear(restante / total, 4) if total > 0 else None)


def evaluar_slos(muestras: Iterable[MuestraConsulta], slos: Sequence[SLO] = SLOS) -> list[EvaluacionSLO]:
    """Una evaluación por SLO, en el orden de `slos`."""
    lista = list(muestras)
    return [evaluar_slo(slo, lista) for slo in slos]


def tabla_slos(evaluaciones: Sequence[EvaluacionSLO], slos: Sequence[SLO] = SLOS) -> pd.DataFrame:
    """Tabla para mostrar con `COLUMNAS_TABLA_SLOS`; `resultado` es "cumple", "no cumple" o
    "sin eventos", con " (indicativo)" cuando hay menos eventos que `minimo_eventos`.
    """
    descripciones = {slo.nombre: slo.descripcion for slo in slos}
    filas = []
    for evaluacion in evaluaciones:
        if evaluacion.cumple is None:
            resultado = "sin eventos"
        else:
            resultado = "cumple" if evaluacion.cumple else "no cumple"
            if not evaluacion.suficiente:
                resultado += " (indicativo)"
        filas.append({"slo": evaluacion.slo, "descripcion": descripciones.get(evaluacion.slo, ""),
                      "objetivo": evaluacion.objetivo, "eventos_validos": evaluacion.eventos_validos,
                      "eventos_buenos": evaluacion.eventos_buenos, "cumplimiento": evaluacion.cumplimiento,
                      "resultado": resultado, "presupuesto_total": evaluacion.presupuesto_total,
                      "presupuesto_consumido": evaluacion.presupuesto_consumido,
                      "presupuesto_restante": evaluacion.presupuesto_restante,
                      "fraccion_restante": evaluacion.fraccion_restante})
    return pd.DataFrame(filas, columns=list(COLUMNAS_TABLA_SLOS))
