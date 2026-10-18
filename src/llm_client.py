"""Adaptador HTTP para el servicio local de Ollama: chat con herramientas, streaming y metadatos.

El cliente solo habla con un host local, con una lista blanca de modelos y un `num_ctx`
constante (si cambiara entre peticiones, Ollama recargaría el modelo). Traduce cada fallo a
un `ErrorOllama` clasificado por tipo y operación, con un mensaje legible para la interfaz.
No escribe logs ni conoce el `trace_id`: quien captura el error decide cómo registrarlo.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

import requests
from urllib3.exceptions import ReadTimeoutError

URL_OLLAMA = "http://127.0.0.1:11434"
HOSTS_LOCALES = frozenset({"127.0.0.1", "localhost", "[::1]"})
MODELOS: tuple[str, ...] = ("llama3.2:3b", "llama3.2:1b", "qwen2.5:3b")
MODELO_POR_DEFECTO = "llama3.2:3b"
KEEP_ALIVE = "10m"
NUM_CTX = 4096
SEMILLA: int | None = 42
TIMEOUT_CORTO = 3
TIMEOUT_SHOW = (3, 10)
TIMEOUT_CHAT = (3, 90)
TIMEOUT_STREAM = (3, 120)
TIMEOUT_PRECALENTAR = (3, 120)
TIMEOUT_GENERAR = (3, 180)
TIMEOUT_ESTADO = (0.5, 2)
OPCIONES_PROHIBIDAS = frozenset({"num_thread"})
OPCIONES_GENERAR = {"temperature": 0.2, "num_predict": 350}

TipoErrorOllama = Literal["conexion", "timeout", "no_instalado", "sin_herramientas", "memoria",
                          "http", "respuesta_invalida", "stream", "vacia", "desconocido"]

MENSAJE_CONEXION = "No se pudo conectar con Ollama en {host}. Abra Ollama en Windows y vuelva a verificar."
MENSAJE_TIMEOUT = "Ollama superó {s:g} s sin responder. Cierre aplicaciones que consuman RAM o pruebe llama3.2:1b."
MENSAJE_NO_INSTALADO = "Modelo no disponible. Ejecute: ollama pull {modelo}"
MENSAJE_SIN_HERRAMIENTAS = "El modelo {modelo} no admite tool calling; se usará el enrutador determinista."
MENSAJE_MEMORIA = "Ollama no tiene memoria suficiente para {modelo}. Cierre aplicaciones o use llama3.2:1b."
MENSAJE_HTTP = "Ollama rechazó la solicitud ({codigo}): {error}"
MENSAJE_INVALIDA = "No se obtuvo una respuesta válida de Ollama. Verifique el servicio y el modelo."
MENSAJE_STREAM = "Ollama interrumpió la respuesta: {error}"
MENSAJE_VACIA = "Ollama respondió sin texto; vuelva a intentar."


class ErrorOllama(RuntimeError):
    """Fallo de Ollama con mensaje para el usuario y clasificación para la observabilidad.

    `tipo` resume la causa, `operacion` dice qué método del cliente falló y `codigo_http` es
    el estado HTTP de la respuesta de error (None si no hubo respuesta HTTP de error).
    `ErrorOllama("x")` sigue siendo válido y queda con `tipo="desconocido"`.
    """

    def __init__(self, mensaje: str = "", *, tipo: TipoErrorOllama = "desconocido",
                 operacion: str = "", codigo_http: int | None = None) -> None:
        super().__init__(mensaje)
        self.tipo: TipoErrorOllama = tipo
        self.operacion = operacion
        self.codigo_http = codigo_http

    def atributos(self) -> dict[str, Any]:
        """Atributos serializables del error: tipo, operación, código HTTP y mensaje."""
        return {"tipo": self.tipo, "operacion": self.operacion, "codigo_http": self.codigo_http,
                "mensaje": str(self)}


class ErrorSinHerramientas(ErrorOllama):
    """El modelo rechaza `tools` (400 "does not support tools"); el agente usa su respaldo."""

    def __init__(self, mensaje: str = "", *, tipo: TipoErrorOllama = "sin_herramientas",
                 operacion: str = "", codigo_http: int | None = None) -> None:
        super().__init__(mensaje, tipo=tipo, operacion=operacion, codigo_http=codigo_http)


def _entero(valor: Any) -> int | None:
    return valor if isinstance(valor, int) and not isinstance(valor, bool) else None


def _texto(valor: Any) -> str | None:
    return valor if isinstance(valor, str) else None


def _lista_textos(valor: Any) -> tuple[str, ...] | None:
    if isinstance(valor, (list, tuple)) and all(isinstance(v, str) for v in valor):
        return tuple(valor)
    return None


@dataclass(frozen=True)
class Metricas:
    """Métricas que Ollama devuelve al terminar; las duraciones vienen en nanosegundos."""

    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_cached_count: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None

    @property
    def tokens_por_segundo(self) -> float | None:
        """Velocidad de generación: eval_count / (eval_duration_ns / 1e9); None si falta un dato."""
        if self.eval_count is None or not self.eval_duration_ns:
            return None
        return self.eval_count / (self.eval_duration_ns / 1e9)

    @classmethod
    def desde_json(cls, datos: dict[str, Any]) -> Metricas:
        """Lee las claves documentadas de /api/chat; las que faltan o no son enteras quedan en None."""
        return cls(
            total_duration_ns=_entero(datos.get("total_duration")),
            load_duration_ns=_entero(datos.get("load_duration")),
            prompt_eval_count=_entero(datos.get("prompt_eval_count")),
            prompt_eval_cached_count=_entero(datos.get("prompt_eval_cached_count")),
            prompt_eval_duration_ns=_entero(datos.get("prompt_eval_duration")),
            eval_count=_entero(datos.get("eval_count")),
            eval_duration_ns=_entero(datos.get("eval_duration")),
        )

    def a_dict(self) -> dict[str, Any]:
        """Campos crudos más `tokens_por_segundo` redondeado a 2 decimales."""
        tps = self.tokens_por_segundo
        return asdict(self) | {"tokens_por_segundo": None if tps is None else round(tps, 2)}


@dataclass(frozen=True)
class LlamadaHerramienta:
    """Llamada propuesta por el modelo; `argumentos` es dict si llegó como objeto o JSON válido."""

    nombre: str
    argumentos: Any
    indice: int | None = None


@dataclass(frozen=True)
class RespuestaModelo:
    """Respuesta completa del modelo: texto, llamadas a herramientas, motivo de fin y métricas."""

    contenido: str
    llamadas: tuple[LlamadaHerramienta, ...]
    done_reason: str | None
    modelo: str
    metricas: Metricas

    @property
    def truncada(self) -> bool:
        """True si el modelo se detuvo por alcanzar `num_predict` (done_reason == "length")."""
        return self.done_reason == "length"


@dataclass(frozen=True)
class InfoModelo:
    """Metadatos del modelo instalado, combinando /api/tags (digest) y /api/show (detalles)."""

    nombre: str
    digest: str
    tamano_bytes: int | None
    familia: str | None
    parametros: str | None
    cuantizacion: str | None
    capacidades: tuple[str, ...]
    contexto_maximo: int | None
    modificado: str | None

    @property
    def digest_corto(self) -> str:
        """Los 12 primeros caracteres del digest."""
        return self.digest[:12]

    @property
    def soporta_herramientas(self) -> bool:
        """True si el modelo declara la capacidad "tools"."""
        return "tools" in self.capacidades

    def a_dict(self) -> dict[str, Any]:
        """Campos serializables más `digest_corto` y `soporta_herramientas`."""
        datos = {f.name: getattr(self, f.name) for f in fields(self)}
        datos["capacidades"] = list(self.capacidades)
        return datos | {"digest_corto": self.digest_corto, "soporta_herramientas": self.soporta_herramientas}


class FlujoRespuesta:
    """Iterable de un solo uso que da fragmentos de texto; sirve para `st.write_stream`.

    Lee el NDJSON de /api/chat línea a línea y cierra la conexión al terminar o al cortarse.
    `respuesta` existe solo después de consumir el flujo entero; antes lanza RuntimeError.
    """

    def __init__(self, respuesta_http: Any, modelo: str, *, timeout: Any = TIMEOUT_STREAM,
                 operacion: str = "chat_stream") -> None:
        self._http = respuesta_http
        self._modelo = modelo
        self._timeout = timeout
        self._operacion = operacion
        self._iniciado = False
        self._respuesta: RespuestaModelo | None = None

    def __iter__(self) -> Iterator[str]:
        if self._iniciado:
            raise RuntimeError("El flujo de respuesta es de un solo uso.")
        self._iniciado = True
        return self._fragmentos()

    @property
    def respuesta(self) -> RespuestaModelo:
        if self._respuesta is None:
            raise RuntimeError("El flujo no se consumió entero; la respuesta aún no está disponible.")
        return self._respuesta

    def _error(self, mensaje: str, tipo: TipoErrorOllama) -> ErrorOllama:
        return ErrorOllama(mensaje, tipo=tipo, operacion=self._operacion)

    def _fragmentos(self) -> Iterator[str]:
        partes: list[str] = []
        final: dict[str, Any] | None = None
        try:
            # Paso 1: cada línea es un objeto JSON; un `error` corta el flujo con su texto
            for linea in self._http.iter_lines():
                if not linea:
                    continue
                try:
                    texto = linea.decode("utf-8") if isinstance(linea, bytes) else str(linea)
                    datos = json.loads(texto)
                except ValueError as exc:
                    raise self._error(MENSAJE_INVALIDA, "respuesta_invalida") from exc
                if not isinstance(datos, dict):
                    raise self._error(MENSAJE_INVALIDA, "respuesta_invalida")
                if "error" in datos:
                    raise self._error(MENSAJE_STREAM.format(error=datos["error"]), "stream")
                # Paso 2: se entrega cada fragmento de texto apenas llega
                mensaje = datos.get("message")
                fragmento = mensaje.get("content") if isinstance(mensaje, dict) else None
                if isinstance(fragmento, str) and fragmento:
                    partes.append(fragmento)
                    yield fragmento
                if datos.get("done") is True:
                    final = datos
                    break
        except requests.Timeout as exc:
            raise self._error(MENSAJE_TIMEOUT.format(s=_segundos_timeout(self._timeout, exc)), "timeout") from exc
        except requests.RequestException as exc:
            if exc.args and isinstance(exc.args[0], ReadTimeoutError):
                raise self._error(MENSAJE_TIMEOUT.format(s=_segundos_timeout(self._timeout, exc)), "timeout") from exc
            raise self._error(MENSAJE_STREAM.format(error=exc), "stream") from exc
        finally:
            self._http.close()
        # Paso 3: la línea `done` aporta motivo de fin y métricas; sin texto es un error
        if final is None:
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida")
        contenido = "".join(partes)
        if not contenido.strip():
            raise self._error(MENSAJE_VACIA, "vacia")
        done_reason = final.get("done_reason")
        modelo = final.get("model")
        self._respuesta = RespuestaModelo(
            contenido=contenido, llamadas=(),
            done_reason=done_reason if isinstance(done_reason, str) else None,
            modelo=modelo if isinstance(modelo, str) and modelo else self._modelo,
            metricas=Metricas.desde_json(final))


@runtime_checkable
class ClienteChat(Protocol):
    """Contrato del que depende el agente; lo cumplen el cliente real y los dobles de prueba."""

    modelo: str

    def chat(self, mensajes: list[dict[str, Any]], herramientas: list[dict[str, Any]] | None = None,
             *, opciones: dict[str, Any] | None = None) -> RespuestaModelo: ...

    def chat_stream(self, mensajes: list[dict[str, Any]], *,
                    opciones: dict[str, Any] | None = None) -> FlujoRespuesta: ...

    def info_modelo(self) -> InfoModelo: ...

    def version(self) -> str: ...

    def opciones_efectivas(self, opciones: dict[str, Any] | None = None) -> dict[str, Any]: ...


def _segundos_timeout(timeout: Any, exc: BaseException) -> float:
    if isinstance(timeout, tuple):
        return timeout[0] if isinstance(exc, requests.ConnectTimeout) else timeout[1]
    return timeout


def _a_utc(texto: Any) -> str | None:
    if not isinstance(texto, str):
        return None
    try:
        fecha = datetime.fromisoformat(texto)
    except ValueError:
        return None
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)
    return fecha.astimezone(timezone.utc).isoformat(timespec="seconds")


class ClienteOllama:
    """Cliente HTTP de Ollama restringido a un host local y a una lista blanca de modelos.

    Lanza ValueError si el modelo no está permitido o la URL no es local. La sesión ignora
    los proxies del entorno (`trust_env = False`).
    """

    def __init__(self, modelo: str = MODELO_POR_DEFECTO, *, modelos_permitidos: Sequence[str] = MODELOS,
                 url: str = URL_OLLAMA, keep_alive: str = KEEP_ALIVE, num_ctx: int = NUM_CTX,
                 seed: int | None = SEMILLA, sesion: requests.Session | None = None) -> None:
        if modelo not in tuple(modelos_permitidos):
            raise ValueError("Seleccione uno de los modelos locales documentados: " + ", ".join(modelos_permitidos))
        partes = urlsplit(url)
        host = partes.hostname or ""
        if partes.scheme not in ("http", "https") or (host not in HOSTS_LOCALES and f"[{host}]" not in HOSTS_LOCALES):
            raise ValueError("Ollama debe estar en este equipo (127.0.0.1, localhost o [::1]).")
        self.modelo = modelo
        self.url = url.rstrip("/")
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.seed = seed
        self.sesion = sesion if sesion is not None else requests.Session()
        self.sesion.trust_env = False
        self._host = partes.netloc
        self._version: str | None = None
        self._info: InfoModelo | None = None

    # Paso 1: transporte y traducción de errores

    def _error(self, mensaje: str, tipo: TipoErrorOllama, operacion: str,
               codigo_http: int | None = None) -> ErrorOllama:
        return ErrorOllama(mensaje, tipo=tipo, operacion=operacion, codigo_http=codigo_http)

    def _solicitar(self, metodo: str, ruta: str, operacion: str, timeout: Any, **kwargs: Any) -> Any:
        try:
            r = getattr(self.sesion, metodo)(f"{self.url}{ruta}", timeout=timeout, **kwargs)
        except requests.Timeout as exc:
            raise self._error(MENSAJE_TIMEOUT.format(s=_segundos_timeout(timeout, exc)), "timeout", operacion) from exc
        except requests.ConnectionError as exc:
            raise self._error(MENSAJE_CONEXION.format(host=self._host), "conexion", operacion) from exc
        except requests.RequestException as exc:
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion) from exc
        self._verificar_estado(r, operacion, con_modelo=metodo == "post")
        return r

    def _verificar_estado(self, r: Any, operacion: str, *, con_modelo: bool) -> None:
        """Traduce un código >= 400 a ErrorOllama y cierra la respuesta.

        Un 404 solo significa "modelo no instalado" cuando la solicitud nombra un modelo (los
        POST a /api/chat y /api/show); en un GET indica una ruta inexistente y se trata como
        cualquier otro código HTTP.
        """
        codigo = r.status_code
        if codigo == 404 and con_modelo:
            r.close()
            raise self._error(MENSAJE_NO_INSTALADO.format(modelo=self.modelo), "no_instalado", operacion, 404)
        if codigo < 400:
            return
        try:
            detalle = self._detalle_error(r)
        finally:
            r.close()
        if codigo == 400 and "does not support tools" in detalle:
            raise ErrorSinHerramientas(MENSAJE_SIN_HERRAMIENTAS.format(modelo=self.modelo),
                                       operacion=operacion, codigo_http=400)
        if codigo == 500 and "memory" in detalle.lower():
            raise self._error(MENSAJE_MEMORIA.format(modelo=self.modelo), "memoria", operacion, 500)
        raise self._error(MENSAJE_HTTP.format(codigo=codigo, error=detalle), "http", operacion, codigo)

    @staticmethod
    def _detalle_error(r: Any) -> str:
        try:
            datos = r.json()
            if isinstance(datos, dict) and isinstance(datos.get("error"), str):
                return datos["error"]
        except ValueError:
            pass
        return str(r.text or "")[:300]

    def _leer_json(self, r: Any, operacion: str) -> dict[str, Any]:
        try:
            datos = r.json()
        except ValueError as exc:
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion) from exc
        finally:
            r.close()
        if not isinstance(datos, dict):
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion)
        return datos

    # Paso 2: lectura de respuestas de /api/chat

    def _leer_llamadas(self, crudas: Any, operacion: str) -> tuple[LlamadaHerramienta, ...]:
        if crudas is None:
            return ()
        if not isinstance(crudas, list):
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion)
        llamadas = []
        for cruda in crudas:
            funcion = cruda.get("function") if isinstance(cruda, dict) else None
            if not isinstance(funcion, dict) or not isinstance(funcion.get("name"), str):
                raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion)
            argumentos = funcion.get("arguments")
            if isinstance(argumentos, str):
                try:
                    decodificados = json.loads(argumentos)
                except ValueError:
                    decodificados = None
                if isinstance(decodificados, dict):
                    argumentos = decodificados
            indice = _entero(funcion.get("index", cruda.get("index")))
            llamadas.append(LlamadaHerramienta(funcion["name"], argumentos, indice))
        return tuple(llamadas)

    def _respuesta_desde_json(self, datos: dict[str, Any], operacion: str, *,
                              exigir_mensaje: bool = True) -> RespuestaModelo:
        mensaje = datos.get("message")
        if mensaje is None and not exigir_mensaje:
            mensaje = {}
        if not isinstance(mensaje, dict):
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion)
        contenido = mensaje.get("content") or ""
        if not isinstance(contenido, str):
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion)
        done_reason = datos.get("done_reason")
        modelo = datos.get("model")
        return RespuestaModelo(
            contenido=contenido,
            llamadas=self._leer_llamadas(mensaje.get("tool_calls"), operacion),
            done_reason=done_reason if isinstance(done_reason, str) else None,
            modelo=modelo if isinstance(modelo, str) and modelo else self.modelo,
            metricas=Metricas.desde_json(datos))

    # Paso 3: payload y chat

    def opciones_efectivas(self, opciones: dict[str, Any] | None = None) -> dict[str, Any]:
        """Opciones que se envían: {"num_ctx", "seed" si hay semilla} combinadas con `opciones`.

        Lanza ValueError si `opciones` intenta cambiar `num_ctx` o incluye `num_thread`.
        """
        extra = dict(opciones or {})
        if "num_ctx" in extra and extra["num_ctx"] != self.num_ctx:
            raise ValueError(f"num_ctx es constante ({self.num_ctx}); cambiarlo obliga a recargar el modelo.")
        prohibidas = OPCIONES_PROHIBIDAS & set(extra)
        if prohibidas:
            raise ValueError("Opciones no permitidas: " + ", ".join(sorted(prohibidas)))
        base: dict[str, Any] = {"num_ctx": self.num_ctx}
        if self.seed is not None:
            base["seed"] = self.seed
        return base | extra

    def _payload(self, mensajes: list[dict[str, Any]], *, stream: bool,
                 herramientas: list[dict[str, Any]] | None = None,
                 opciones: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.modelo, "messages": mensajes, "stream": stream,
                                   "keep_alive": self.keep_alive, "options": self.opciones_efectivas(opciones)}
        if herramientas:
            payload["tools"] = herramientas
        return payload

    def _chat(self, mensajes: list[dict[str, Any]], herramientas: list[dict[str, Any]] | None,
              opciones: dict[str, Any] | None, timeout: Any) -> RespuestaModelo:
        payload = self._payload(mensajes, stream=False, herramientas=herramientas, opciones=opciones)
        r = self._solicitar("post", "/api/chat", "chat", timeout, json=payload)
        return self._respuesta_desde_json(self._leer_json(r, "chat"), "chat")

    def chat(self, mensajes: list[dict[str, Any]], herramientas: list[dict[str, Any]] | None = None,
             *, opciones: dict[str, Any] | None = None) -> RespuestaModelo:
        """POST /api/chat sin streaming; `tools` solo se envía si hay herramientas.

        Una respuesta vacía no es error: en la selección significa que el agente usa su respaldo.
        """
        return self._chat(mensajes, herramientas, opciones, TIMEOUT_CHAT)

    def chat_stream(self, mensajes: list[dict[str, Any]], *,
                    opciones: dict[str, Any] | None = None) -> FlujoRespuesta:
        """POST /api/chat con streaming y sin `tools`; la conexión se abre al llamar al método."""
        payload = self._payload(mensajes, stream=True, opciones=opciones)
        r = self._solicitar("post", "/api/chat", "chat_stream", TIMEOUT_STREAM, json=payload, stream=True)
        return FlujoRespuesta(r, self.modelo, timeout=TIMEOUT_STREAM)

    def generar(self, sistema: str, mensaje: str) -> str:
        """Compatibilidad: una consulta system+user sin herramientas; error si llega sin texto."""
        respuesta = self._chat([{"role": "system", "content": sistema}, {"role": "user", "content": mensaje}],
                               None, OPCIONES_GENERAR, TIMEOUT_GENERAR)
        if not respuesta.contenido.strip():
            raise self._error(MENSAJE_VACIA, "vacia", "chat")
        return respuesta.contenido.strip()

    def precalentar(self) -> RespuestaModelo:
        """Carga el modelo en memoria con `messages: []` y el mismo `num_ctx` (done_reason "load")."""
        payload = {"model": self.modelo, "messages": [], "stream": False, "keep_alive": self.keep_alive,
                   "options": {"num_ctx": self.num_ctx}}
        r = self._solicitar("post", "/api/chat", "precalentar", TIMEOUT_PRECALENTAR, json=payload)
        return self._respuesta_desde_json(self._leer_json(r, "precalentar"), "precalentar", exigir_mensaje=False)

    # Paso 4: metadatos del servicio y del modelo

    def modelos(self) -> list[dict[str, Any]]:
        """Modelos instalados según GET /api/tags."""
        datos = self._leer_json(self._solicitar("get", "/api/tags", "modelos", TIMEOUT_CORTO), "modelos")
        modelos = datos.get("models")
        if not isinstance(modelos, list):
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", "modelos")
        return modelos

    def _entrada_instalada(self, modelos: list[Any]) -> dict[str, Any] | None:
        return next((m for m in modelos if isinstance(m, dict) and self.modelo in (m.get("name"), m.get("model"))),
                    None)

    def version(self) -> str:
        """Versión de Ollama (GET /api/version), con caché en la instancia."""
        if self._version is None:
            datos = self._leer_json(self._solicitar("get", "/api/version", "version", TIMEOUT_CORTO), "version")
            version = datos.get("version")
            if not isinstance(version, str) or not version:
                raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", "version")
            self._version = version
        return self._version

    def info_modelo(self) -> InfoModelo:
        """Metadatos del modelo: digest y tamaño de /api/tags; detalles y capacidades de /api/show.

        Lanza ErrorOllama de tipo "no_instalado" si el modelo no aparece en /api/tags.
        """
        if self._info is not None:
            return self._info
        operacion = "info_modelo"
        datos = self._leer_json(self._solicitar("get", "/api/tags", operacion, TIMEOUT_CORTO), operacion)
        modelos = datos.get("models")
        if not isinstance(modelos, list):
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion)
        entrada = self._entrada_instalada(modelos)
        if entrada is None:
            raise self._error(MENSAJE_NO_INSTALADO.format(modelo=self.modelo), "no_instalado", operacion)
        if not isinstance(entrada.get("digest"), str):
            raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", operacion)
        r = self._solicitar("post", "/api/show", operacion, TIMEOUT_SHOW, json={"model": self.modelo})
        show = self._leer_json(r, operacion)
        detalles_tags = entrada.get("details") if isinstance(entrada.get("details"), dict) else {}
        detalles_show = show.get("details") if isinstance(show.get("details"), dict) else {}
        detalles = detalles_tags | detalles_show
        capacidades = _lista_textos(show.get("capabilities")) or _lista_textos(entrada.get("capabilities")) or ()
        model_info = show.get("model_info") if isinstance(show.get("model_info"), dict) else {}
        arquitectura = _texto(model_info.get("general.architecture"))
        contexto = _entero(model_info.get(f"{arquitectura}.context_length")) if arquitectura else None
        if contexto is None:
            contexto = next((_entero(v) for k, v in model_info.items() if k.endswith(".context_length")), None)
        if contexto is None:
            contexto = _entero(detalles.get("context_length"))
        self._info = InfoModelo(
            nombre=self.modelo, digest=entrada["digest"], tamano_bytes=_entero(entrada.get("size")),
            familia=_texto(detalles.get("family")), parametros=_texto(detalles.get("parameter_size")),
            cuantizacion=_texto(detalles.get("quantization_level")),
            capacidades=capacidades, contexto_maximo=contexto,
            modificado=_texto(entrada.get("modified_at")))
        return self._info

    # Paso 5: diagnóstico para la interfaz

    def estado(self) -> dict[str, Any]:
        """Sonda rápida que nunca lanza: versión, latencia y si el modelo está cargado (/api/ps).

        Devuelve {"accesible", "version", "modelo", "modelo_cargado", "expira_utc",
        "latencia_ms", "error"}. Si /api/version falla, `accesible` es False y `error` lleva el
        mensaje. Si falla solo /api/ps, se conservan `accesible`, `version` y `latencia_ms`, y
        `modelo_cargado` y `expira_utc` quedan en None junto al mensaje en `error`.
        """
        resultado: dict[str, Any] = {"accesible": False, "version": None, "modelo": self.modelo,
                                     "modelo_cargado": None, "expira_utc": None, "latencia_ms": None,
                                     "error": None}
        try:
            inicio = perf_counter()
            datos = self._leer_json(self._solicitar("get", "/api/version", "version", TIMEOUT_ESTADO), "version")
            version = datos.get("version")
            if not isinstance(version, str) or not version:
                raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", "version")
            resultado.update(accesible=True, version=version, latencia_ms=round((perf_counter() - inicio) * 1000, 1))
            cargados = self._leer_json(self._solicitar("get", "/api/ps", "estado", TIMEOUT_ESTADO), "estado")
            modelos = cargados.get("models")
            if not isinstance(modelos, list):
                raise self._error(MENSAJE_INVALIDA, "respuesta_invalida", "estado")
            entrada = self._entrada_instalada(modelos)
            resultado["modelo_cargado"] = entrada is not None
            if entrada is not None:
                resultado["expira_utc"] = _a_utc(entrada.get("expires_at"))
        except ErrorOllama as exc:
            resultado["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - la sonda de estado nunca debe romper la interfaz
            resultado["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return resultado

    def verificar(self, precalentar: bool = True) -> dict[str, Any]:
        """Comprueba el servicio y el modelo; si está instalado y se pide, lo precalienta.

        Devuelve {"accesible", "modelo_instalado", "modelos", "digest", "precalentado",
        "segundos_precalentamiento"}. Lanza ErrorOllama si Ollama no responde o falla la carga.
        """
        modelos = self.modelos()
        entrada = self._entrada_instalada(modelos)
        resultado: dict[str, Any] = {
            "accesible": True, "modelo_instalado": entrada is not None,
            "modelos": [m.get("name") or m.get("model") for m in modelos if isinstance(m, dict)],
            "digest": entrada.get("digest") if entrada is not None else None,
            "precalentado": False, "segundos_precalentamiento": None}
        if precalentar and entrada is not None:
            inicio = perf_counter()
            self.precalentar()
            resultado.update(precalentado=True, segundos_precalentamiento=round(perf_counter() - inicio, 3))
        return resultado
