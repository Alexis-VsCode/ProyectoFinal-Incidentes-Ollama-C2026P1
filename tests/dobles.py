"""Dobles HTTP para probar el cliente de Ollama sin red ni modelo cargado.

`SesionFalsa` sustituye a `requests.Session` y registra cada solicitud; `RespuestaHTTPFalsa`
imita la parte de `requests.Response` que usa el cliente (status_code, json, text, iter_lines
y close). Los ayudantes construyen cuerpos con el formato documentado de /api/chat.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any


class RespuestaHTTPFalsa:
    """Respuesta HTTP mínima: cuerpo JSON (o texto crudo) y líneas NDJSON para streaming.

    `cuerpo` puede ser un objeto serializable o un `str` que se entrega tal cual a `json()`
    (sirve para simular JSON inválido). Cada elemento de `lineas` puede ser un dict (se
    serializa), un `str`, `bytes` o una excepción, que se lanza al llegar a ella.
    """

    def __init__(self, status_code: int = 200, cuerpo: Any = None,
                 lineas: Iterable[Any] | None = None) -> None:
        self.status_code = status_code
        self.cuerpo = cuerpo
        self.lineas = list(lineas) if lineas is not None else []
        self.cerrada = False

    @property
    def text(self) -> str:
        if isinstance(self.cuerpo, str):
            return self.cuerpo
        return "" if self.cuerpo is None else json.dumps(self.cuerpo, ensure_ascii=False)

    def json(self) -> Any:
        if isinstance(self.cuerpo, str):
            return json.loads(self.cuerpo)
        if self.cuerpo is None:
            raise ValueError("Cuerpo vacío")
        return self.cuerpo

    def iter_lines(self) -> Iterator[bytes]:
        for linea in self.lineas:
            if isinstance(linea, BaseException):
                raise linea
            if isinstance(linea, bytes):
                yield linea
            elif isinstance(linea, str):
                yield linea.encode("utf-8")
            else:
                yield json.dumps(linea, ensure_ascii=False).encode("utf-8")

    def close(self) -> None:
        self.cerrada = True


class SesionFalsa:
    """Sesión que responde por ruta ("GET /api/tags", "POST /api/chat") y registra las llamadas.

    El valor de cada ruta es una `RespuestaHTTPFalsa`, una excepción (se lanza) o una lista
    de ellas, que se consume en orden. Una ruta sin respuesta configurada lanza AssertionError.
    """

    def __init__(self, rutas: dict[str, Any] | None = None) -> None:
        self.rutas: dict[str, Any] = dict(rutas or {})
        self.llamadas: list[dict[str, Any]] = []
        self.trust_env = True

    def _responder(self, metodo: str, url: str, **kwargs: Any) -> RespuestaHTTPFalsa:
        ruta = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        self.llamadas.append({"metodo": metodo, "ruta": ruta, "url": url, **kwargs})
        clave = f"{metodo} {ruta}"
        if clave not in self.rutas:
            raise AssertionError(f"Solicitud no esperada: {clave}")
        valor = self.rutas[clave]
        if isinstance(valor, list):
            if not valor:
                raise AssertionError(f"Sin más respuestas para {clave}")
            valor = valor.pop(0)
        if isinstance(valor, BaseException):
            raise valor
        return valor

    def get(self, url: str, **kwargs: Any) -> RespuestaHTTPFalsa:
        return self._responder("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> RespuestaHTTPFalsa:
        return self._responder("POST", url, **kwargs)

    def ultima(self, clave: str | None = None) -> dict[str, Any]:
        """Devuelve la última llamada registrada, opcionalmente filtrada por "METODO /ruta"."""
        candidatas = [c for c in self.llamadas if clave is None or f"{c['metodo']} {c['ruta']}" == clave]
        if not candidatas:
            raise AssertionError(f"No se registró ninguna llamada {clave or ''}".strip())
        return candidatas[-1]


def cuerpo_chat(contenido: str = "", llamadas: Iterable[dict[str, Any]] = (),
                done_reason: str | None = "stop", modelo: str = "llama3.2:3b",
                **metricas: Any) -> dict[str, Any]:
    """Cuerpo de /api/chat sin streaming; `llamadas` son entradas de `message.tool_calls`."""
    mensaje: dict[str, Any] = {"role": "assistant", "content": contenido}
    llamadas = list(llamadas)
    if llamadas:
        mensaje["tool_calls"] = llamadas
    cuerpo: dict[str, Any] = {"model": modelo, "message": mensaje, "done": True}
    if done_reason is not None:
        cuerpo["done_reason"] = done_reason
    cuerpo.update(metricas)
    return cuerpo


def llamada_ollama(nombre: str, argumentos: Any, indice: int | None = 0) -> dict[str, Any]:
    """Entrada de `tool_calls` con el formato documentado: `function.name/arguments/index`."""
    funcion: dict[str, Any] = {"name": nombre, "arguments": argumentos}
    if indice is not None:
        funcion["index"] = indice
    return {"function": funcion}


def lineas_stream(*fragmentos: str, done_reason: str = "stop", modelo: str = "llama3.2:3b",
                  **metricas: Any) -> list[dict[str, Any]]:
    """Líneas NDJSON de un stream: un fragmento por línea y la línea final `done: true`."""
    lineas: list[dict[str, Any]] = [
        {"model": modelo, "message": {"role": "assistant", "content": f}, "done": False}
        for f in fragmentos
    ]
    final: dict[str, Any] = {"model": modelo, "message": {"role": "assistant", "content": ""},
                             "done": True, "done_reason": done_reason}
    final.update(metricas)
    lineas.append(final)
    return lineas
