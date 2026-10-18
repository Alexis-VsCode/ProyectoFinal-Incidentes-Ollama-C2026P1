"""Pruebas del adaptador HTTP de Ollama con dobles: payload, lectura, stream y errores.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from src.llm_client import (
    KEEP_ALIVE,
    MODELO_POR_DEFECTO,
    MODELOS,
    NUM_CTX,
    ClienteOllama,
    ErrorOllama,
    ErrorSinHerramientas,
    LlamadaHerramienta,
)
from tests.dobles import (
    RespuestaHTTPFalsa,
    SesionFalsa,
    cuerpo_chat,
    llamada_ollama,
    lineas_stream,
)

DIGEST = "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"
ESQUEMAS = [{"type": "function", "function": {
    "name": "segmentar", "description": "Ranking de segmentos.",
    "parameters": {"type": "object", "properties": {"variable": {"type": "string", "enum": ["category"]}},
                   "required": ["variable"]}}}]
MENSAJES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def cliente_con(rutas: dict, **kwargs) -> tuple[ClienteOllama, SesionFalsa]:
    sesion = SesionFalsa(rutas)
    return ClienteOllama(sesion=sesion, **kwargs), sesion


def tags(*modelos: dict) -> RespuestaHTTPFalsa:
    return RespuestaHTTPFalsa(200, {"models": list(modelos)})


def entrada_tags(nombre: str = "llama3.2:3b") -> dict:
    return {"name": nombre, "model": nombre, "size": 2019393189, "digest": DIGEST,
            "modified_at": "2026-10-01T10:00:00-05:00",
            "details": {"family": "llama", "parameter_size": "3.2B", "quantization_level": "Q4_K_M"},
            "capabilities": ["completion", "tools"]}


# Paso 1: constantes y payload


def test_constantes_publicas_del_cliente():
    assert MODELOS == ("llama3.2:3b", "llama3.2:1b", "qwen2.5:3b")
    assert MODELO_POR_DEFECTO == "llama3.2:3b"
    assert KEEP_ALIVE == "10m"
    assert NUM_CTX == 4096


def test_chat_envia_payload_oficial_con_tools():
    cliente, sesion = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo_chat("ok"))})
    cliente.chat(MENSAJES, ESQUEMAS)
    payload = sesion.ultima("POST /api/chat")["json"]
    assert payload["model"] == "llama3.2:3b"
    assert payload["messages"] == MENSAJES
    assert payload["tools"] == ESQUEMAS
    assert payload["stream"] is False
    assert payload["keep_alive"] == "10m"
    assert payload["options"]["num_ctx"] == 4096
    assert payload["options"]["seed"] == 42


def test_chat_sin_herramientas_omite_tools_y_nunca_envia_think():
    cliente, sesion = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo_chat("ok"))})
    cliente.chat(MENSAJES, opciones={"temperature": 0.0, "num_predict": 128})
    payload = sesion.ultima("POST /api/chat")["json"]
    for clave in ("tools", "think", "format"):
        assert clave not in payload
    assert "num_thread" not in payload["options"]
    assert payload["options"] == {"num_ctx": 4096, "seed": 42, "temperature": 0.0, "num_predict": 128}


def test_opciones_no_pueden_cambiar_num_ctx():
    cliente, sesion = cliente_con({})
    with pytest.raises(ValueError):
        cliente.chat(MENSAJES, opciones={"num_ctx": 2048})
    with pytest.raises(ValueError):
        cliente.opciones_efectivas({"num_thread": 4})
    assert cliente.opciones_efectivas({"num_ctx": 4096}) == {"num_ctx": 4096, "seed": 42}
    assert sesion.llamadas == []


def test_semilla_none_no_se_envia():
    cliente, _ = cliente_con({}, seed=None)
    assert cliente.opciones_efectivas() == {"num_ctx": 4096}


def test_cliente_desactiva_proxy_del_entorno():
    cliente, sesion = cliente_con({})
    assert sesion.trust_env is False
    assert cliente.sesion is sesion


# Paso 2: lectura de la respuesta


def test_chat_parsea_tool_calls_con_argumentos_objeto():
    cuerpo = cuerpo_chat("", [llamada_ollama("segmentar", {"variable": "category", "top": 5}, 0)])
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo)})
    respuesta = cliente.chat(MENSAJES, ESQUEMAS)
    assert respuesta.llamadas[0] == LlamadaHerramienta("segmentar", {"variable": "category", "top": 5}, 0)
    assert respuesta.contenido == ""


def test_chat_decodifica_argumentos_en_texto_json():
    cuerpo = cuerpo_chat("", [llamada_ollama("segmentar", '{"variable":"category"}', None)])
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo)})
    llamada = cliente.chat(MENSAJES, ESQUEMAS).llamadas[0]
    assert llamada.argumentos == {"variable": "category"}
    assert llamada.indice is None


def test_chat_conserva_argumentos_texto_no_json():
    cuerpo = cuerpo_chat("", [llamada_ollama("segmentar", "no-json"), llamada_ollama("calidad", "[1, 2]", 1)])
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo)})
    llamadas = cliente.chat(MENSAJES, ESQUEMAS).llamadas
    assert llamadas[0].argumentos == "no-json"
    assert llamadas[1].argumentos == "[1, 2]"


def test_chat_vacio_no_es_error():
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo_chat(""))})
    respuesta = cliente.chat(MENSAJES, ESQUEMAS)
    assert respuesta.contenido == ""
    assert respuesta.llamadas == ()


def test_metricas_y_tokens_por_segundo():
    cuerpo = cuerpo_chat("texto", total_duration=21_000_000_000, load_duration=1_000_000,
                         prompt_eval_count=700, prompt_eval_duration=900_000_000,
                         eval_count=140, eval_duration=20_000_000_000)
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo)})
    respuesta = cliente.chat(MENSAJES)
    assert respuesta.metricas.tokens_por_segundo == 7.0
    assert respuesta.metricas.eval_count == 140
    assert respuesta.metricas.total_duration_ns == 21_000_000_000
    assert respuesta.metricas.prompt_eval_cached_count is None
    assert respuesta.metricas.a_dict()["tokens_por_segundo"] == 7.0
    assert respuesta.done_reason == "stop"
    assert respuesta.truncada is False
    assert respuesta.modelo == "llama3.2:3b"


def test_done_reason_length_marca_truncada():
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo_chat("x", done_reason="length"))})
    assert cliente.chat(MENSAJES).truncada is True


# Paso 3: streaming


def test_chat_stream_concatena_fragmentos_y_expone_metricas():
    lineas = lineas_stream("Ho", "la", eval_count=2, eval_duration=1_000_000_000)
    respuesta_http = RespuestaHTTPFalsa(200, lineas=lineas)
    cliente, sesion = cliente_con({"POST /api/chat": respuesta_http})
    flujo = cliente.chat_stream(MENSAJES, opciones={"temperature": 0.2, "num_predict": 300})
    llamada = sesion.ultima("POST /api/chat")
    assert llamada["stream"] is True
    assert llamada["json"]["stream"] is True
    assert "tools" not in llamada["json"]
    assert "".join(flujo) == "Hola"
    assert flujo.respuesta.metricas.eval_count == 2
    assert flujo.respuesta.contenido == "Hola"
    assert flujo.respuesta.done_reason == "stop"
    assert respuesta_http.cerrada is True


def test_chat_stream_error_a_mitad_lanza_error_ollama():
    lineas = [{"message": {"content": "Hola "}, "done": False}, {"error": "model runner crashed"}]
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, lineas=lineas)})
    recibidos: list[str] = []
    with pytest.raises(ErrorOllama, match="model runner crashed") as info:
        for fragmento in cliente.chat_stream(MENSAJES):
            recibidos.append(fragmento)
    assert recibidos == ["Hola "]
    assert info.value.tipo == "stream"
    assert "Ollama interrumpió la respuesta" in str(info.value)


def test_chat_stream_vacio_es_error():
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, lineas=lineas_stream())})
    with pytest.raises(ErrorOllama, match="sin texto") as info:
        list(cliente.chat_stream(MENSAJES))
    assert info.value.tipo == "vacia"


def test_chat_stream_timeout_entre_fragmentos_es_timeout():
    corte = requests.ConnectionError(ReadTimeoutError(None, None, "Read timed out."))
    lineas = [{"message": {"content": "Ho"}, "done": False}, corte]
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, lineas=lineas)})
    with pytest.raises(ErrorOllama, match="120 s") as info:
        list(cliente.chat_stream(MENSAJES))
    assert info.value.tipo == "timeout"
    assert info.value.operacion == "chat_stream"


def test_chat_stream_bytes_no_utf8_es_respuesta_invalida():
    respuesta_http = RespuestaHTTPFalsa(200, lineas=[{"message": {"content": "Ho"}, "done": False}, b"\xff\xfe{}"])
    cliente, _ = cliente_con({"POST /api/chat": respuesta_http})
    recibidos: list[str] = []
    with pytest.raises(ErrorOllama, match="No se obtuvo una respuesta válida de Ollama") as info:
        for fragmento in cliente.chat_stream(MENSAJES):
            recibidos.append(fragmento)
    assert recibidos == ["Ho"]
    assert (info.value.tipo, info.value.operacion) == ("respuesta_invalida", "chat_stream")
    assert respuesta_http.cerrada is True


def test_flujo_sin_consumir_no_expone_respuesta():
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, lineas=lineas_stream("a", "b"))})
    flujo = cliente.chat_stream(MENSAJES)
    with pytest.raises(RuntimeError):
        _ = flujo.respuesta
    iterador = iter(flujo)
    next(iterador)
    with pytest.raises(RuntimeError):
        _ = flujo.respuesta


def test_flujo_es_de_un_solo_uso():
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, lineas=lineas_stream("a"))})
    flujo = cliente.chat_stream(MENSAJES)
    assert list(flujo) == ["a"]
    with pytest.raises(RuntimeError):
        list(flujo)


def test_chat_stream_conecta_al_llamarlo():
    cliente, _ = cliente_con({"POST /api/chat": requests.ConnectionError()})
    with pytest.raises(ErrorOllama, match="No se pudo conectar"):
        cliente.chat_stream(MENSAJES)


# Paso 4: errores


@pytest.mark.parametrize(("respuesta", "fragmento", "clase"), [
    (requests.ConnectionError(), "No se pudo conectar con Ollama en 127.0.0.1:11434. Abra Ollama en Windows", ErrorOllama),
    (requests.Timeout(), "Ollama superó 90 s sin responder. Cierre aplicaciones que consuman RAM o pruebe llama3.2:1b.", ErrorOllama),
    (RespuestaHTTPFalsa(404, "<cuerpo no leido>"), "Modelo no disponible. Ejecute: ollama pull llama3.2:3b", ErrorOllama),
    (RespuestaHTTPFalsa(400, {"error": "registry.ollama.ai/library/x does not support tools"}),
     "El modelo llama3.2:3b no admite tool calling; se usará el enrutador determinista.", ErrorSinHerramientas),
    (RespuestaHTTPFalsa(500, {"error": "model requires more system memory (5.2 GiB)"}),
     "Ollama no tiene memoria suficiente para llama3.2:3b. Cierre aplicaciones o use llama3.2:1b.", ErrorOllama),
    (RespuestaHTTPFalsa(502, {"error": "bad gateway"}), "Ollama rechazó la solicitud (502): bad gateway", ErrorOllama),
], ids=["conexion", "timeout", "404", "sin_tools", "memoria", "otro_codigo"])
def test_errores_http_traducidos(respuesta, fragmento, clase):
    cliente, _ = cliente_con({"POST /api/chat": respuesta})
    with pytest.raises(clase) as info:
        cliente.chat(MENSAJES, ESQUEMAS)
    assert fragmento in str(info.value)
    if isinstance(respuesta, RespuestaHTTPFalsa):
        assert respuesta.cerrada is True


@pytest.mark.parametrize("cuerpo", ["{no es json", {"done": True}, ["lista"], {"message": "texto"},
                                    {"message": {"content": "", "tool_calls": [{"function": {}}]}}],
                         ids=["json_invalido", "sin_message", "no_objeto", "message_no_objeto", "llamada_sin_nombre"])
def test_respuesta_invalida_es_error_legible(cuerpo):
    cliente, _ = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo)})
    with pytest.raises(ErrorOllama, match="No se obtuvo una respuesta válida de Ollama") as info:
        cliente.chat(MENSAJES)
    assert info.value.tipo == "respuesta_invalida"


@pytest.mark.parametrize(("tipo", "operacion", "codigo", "preparar"), [
    ("conexion", "chat", None, lambda: {"POST /api/chat": requests.ConnectionError()}),
    ("timeout", "chat", None, lambda: {"POST /api/chat": requests.Timeout()}),
    ("no_instalado", "chat", 404, lambda: {"POST /api/chat": RespuestaHTTPFalsa(404)}),
    ("sin_herramientas", "chat", 400,
     lambda: {"POST /api/chat": RespuestaHTTPFalsa(400, {"error": "x does not support tools"})}),
    ("memoria", "chat", 500, lambda: {"POST /api/chat": RespuestaHTTPFalsa(500, {"error": "out of memory"})}),
    ("http", "chat", 503, lambda: {"POST /api/chat": RespuestaHTTPFalsa(503, "servicio no disponible")}),
    ("respuesta_invalida", "chat", None, lambda: {"POST /api/chat": RespuestaHTTPFalsa(200, "{roto")}),
    ("stream", "chat_stream", None,
     lambda: {"POST /api/chat": RespuestaHTTPFalsa(200, lineas=[{"error": "fallo interno"}])}),
    ("vacia", "chat_stream", None, lambda: {"POST /api/chat": RespuestaHTTPFalsa(200, lineas=lineas_stream())}),
])
def test_error_ollama_lleva_tipo_operacion_y_codigo(tipo, operacion, codigo, preparar):
    cliente, _ = cliente_con(preparar())
    with pytest.raises(ErrorOllama) as info:
        if operacion == "chat":
            cliente.chat(MENSAJES, ESQUEMAS)
        else:
            list(cliente.chat_stream(MENSAJES))
    error = info.value
    assert (error.tipo, error.operacion, error.codigo_http) == (tipo, operacion, codigo)
    assert error.atributos() == {"tipo": tipo, "operacion": operacion, "codigo_http": codigo,
                                 "mensaje": str(error)}


@pytest.mark.parametrize(("ruta", "llamar"), [
    ("GET /api/version", lambda c: c.version()),
    ("GET /api/tags", lambda c: c.modelos()),
    ("GET /api/tags", lambda c: c.verificar()),
], ids=["version", "modelos", "verificar"])
def test_404_sin_modelo_en_la_solicitud_no_pide_pull(ruta, llamar):
    cliente, _ = cliente_con({ruta: RespuestaHTTPFalsa(404, "404 page not found")})
    with pytest.raises(ErrorOllama) as info:
        llamar(cliente)
    assert (info.value.tipo, info.value.codigo_http) == ("http", 404)
    assert "ollama pull" not in str(info.value)
    assert "Ollama rechazó la solicitud (404)" in str(info.value)


def test_estado_con_404_no_pide_pull():
    cliente, _ = cliente_con({"GET /api/version": RespuestaHTTPFalsa(404, "404 page not found")})
    estado = cliente.estado()
    assert estado["accesible"] is False
    assert "ollama pull" not in estado["error"]


def test_error_ollama_legado_sin_atributos():
    error = ErrorOllama("x")
    assert str(error) == "x"
    assert (error.tipo, error.operacion, error.codigo_http) == ("desconocido", "", None)
    assert isinstance(error, RuntimeError)
    assert ErrorSinHerramientas("y").tipo == "sin_herramientas"
    assert issubclass(ErrorSinHerramientas, ErrorOllama)
    assert ErrorOllama(tipo="timeout", operacion="chat").atributos()["tipo"] == "timeout"


# Paso 5: lista blanca y host local


def test_modelo_fuera_de_lista_blanca_rechazado():
    with pytest.raises(ValueError):
        ClienteOllama("mistral", sesion=SesionFalsa())
    assert ClienteOllama("qwen2.5:3b", sesion=SesionFalsa()).modelo == "qwen2.5:3b"


def test_lista_blanca_configurable():
    cliente = ClienteOllama("phi3:mini", modelos_permitidos=("phi3:mini",), sesion=SesionFalsa())
    assert cliente.modelo == "phi3:mini"
    with pytest.raises(ValueError):
        ClienteOllama("llama3.2:3b", modelos_permitidos=("phi3:mini",), sesion=SesionFalsa())


@pytest.mark.parametrize("url", ["http://10.0.0.5:11434", "http://ollama.example.com:11434", "ftp://127.0.0.1:11434"])
def test_url_no_local_rechazada(url):
    with pytest.raises(ValueError):
        ClienteOllama(url=url, sesion=SesionFalsa())


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://[::1]:11434", "http://127.0.0.1:11434/"])
def test_url_local_aceptada(url):
    cliente, sesion = cliente_con({"GET /api/version": RespuestaHTTPFalsa(200, {"version": "0.40.6"})}, url=url)
    cliente.version()
    assert sesion.ultima()["url"].endswith(":11434/api/version")


# Paso 6: metadatos, precarga y compatibilidad


def test_info_modelo_combina_tags_y_show():
    show = {"details": {"family": "llama", "parameter_size": "3.2B", "quantization_level": "Q4_K_M"},
            "capabilities": ["completion", "tools"],
            "model_info": {"general.architecture": "llama", "llama.context_length": 131072}}
    cliente, sesion = cliente_con({"GET /api/tags": tags(entrada_tags("llama3.2:1b"), entrada_tags()),
                                   "POST /api/show": RespuestaHTTPFalsa(200, show)})
    info = cliente.info_modelo()
    assert info.digest_corto == "a80c4f17acd5"
    assert info.digest == DIGEST
    assert info.cuantizacion == "Q4_K_M"
    assert info.parametros == "3.2B"
    assert info.soporta_herramientas is True
    assert info.contexto_maximo == 131072
    assert info.tamano_bytes == 2019393189
    assert sesion.ultima("POST /api/show")["json"] == {"model": "llama3.2:3b"}
    assert info.a_dict()["digest_corto"] == "a80c4f17acd5"
    assert cliente.info_modelo() is info
    assert len(sesion.llamadas) == 2


def test_info_modelo_usa_capacidades_de_tags_si_show_no_las_trae():
    show = {"details": {"quantization_level": "Q8_0"}, "model_info": {}}
    entrada = entrada_tags() | {"details": {"context_length": 8192}}
    cliente, _ = cliente_con({"GET /api/tags": tags(entrada), "POST /api/show": RespuestaHTTPFalsa(200, show)})
    info = cliente.info_modelo()
    assert info.capacidades == ("completion", "tools")
    assert info.contexto_maximo == 8192
    assert info.cuantizacion == "Q8_0"


def test_info_modelo_tolera_campos_con_tipo_inesperado():
    show = {"capabilities": 5, "details": {"family": 3, "parameter_size": ["3.2B"], "quantization_level": None},
            "model_info": {"general.architecture": ["llama"]}}
    cliente, _ = cliente_con({"GET /api/tags": tags(entrada_tags() | {"details": {}}),
                              "POST /api/show": RespuestaHTTPFalsa(200, show)})
    info = cliente.info_modelo()
    assert info.capacidades == ("completion", "tools")
    assert (info.familia, info.parametros, info.cuantizacion, info.contexto_maximo) == (None, None, None, None)
    texto = {"capabilities": "tools"}
    cliente, _ = cliente_con({"GET /api/tags": tags(entrada_tags() | {"capabilities": None}),
                              "POST /api/show": RespuestaHTTPFalsa(200, texto)})
    assert cliente.info_modelo().capacidades == ()


def test_info_modelo_no_instalado_pide_pull():
    cliente, sesion = cliente_con({"GET /api/tags": tags(entrada_tags("llama3.2:1b"))})
    with pytest.raises(ErrorOllama, match="ollama pull llama3.2:3b") as info:
        cliente.info_modelo()
    assert info.value.tipo == "no_instalado"
    assert info.value.operacion == "info_modelo"
    assert len(sesion.llamadas) == 1


def test_version_devuelve_cadena():
    cliente, sesion = cliente_con({"GET /api/version": RespuestaHTTPFalsa(200, {"version": "0.40.6"})})
    assert cliente.version() == "0.40.6"
    assert cliente.version() == "0.40.6"
    assert len(sesion.llamadas) == 1


def test_precalentar_envia_mensajes_vacios():
    cuerpo = {"model": "llama3.2:3b", "message": {"role": "assistant", "content": ""},
              "done": True, "done_reason": "load"}
    cliente, sesion = cliente_con({"POST /api/chat": RespuestaHTTPFalsa(200, cuerpo)})
    respuesta = cliente.precalentar()
    payload = sesion.ultima("POST /api/chat")["json"]
    assert payload["messages"] == []
    assert payload["options"]["num_ctx"] == 4096
    assert payload["keep_alive"] == "10m"
    assert payload["stream"] is False
    assert respuesta.done_reason == "load"


def test_generar_legado_sigue_funcionando():
    cliente, sesion = cliente_con({"POST /api/chat": [RespuestaHTTPFalsa(200, cuerpo_chat("  texto  ")),
                                                      RespuestaHTTPFalsa(200, cuerpo_chat("   "))]})
    assert cliente.generar("sistema", "mensaje") == "texto"
    payload = sesion.ultima("POST /api/chat")["json"]
    assert payload["messages"] == [{"role": "system", "content": "sistema"}, {"role": "user", "content": "mensaje"}]
    assert sesion.ultima("POST /api/chat")["timeout"] == (3, 180)
    with pytest.raises(ErrorOllama, match="sin texto"):
        cliente.generar("sistema", "mensaje")


def test_modelos_devuelve_lista_de_tags():
    cliente, _ = cliente_con({"GET /api/tags": tags(entrada_tags())})
    assert cliente.modelos()[0]["name"] == "llama3.2:3b"


# Paso 7: estado y verificación para la UI


def test_estado_reporta_version_modelo_cargado_y_expiracion():
    ps = {"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b",
                      "expires_at": "2026-10-18T15:25:03.123456789-05:00"}]}
    cliente, sesion = cliente_con({"GET /api/version": RespuestaHTTPFalsa(200, {"version": "0.40.6"}),
                                   "GET /api/ps": RespuestaHTTPFalsa(200, ps)})
    estado = cliente.estado()
    assert set(estado) == {"accesible", "version", "modelo", "modelo_cargado", "expira_utc", "latencia_ms", "error"}
    assert estado["accesible"] is True
    assert estado["version"] == "0.40.6"
    assert estado["modelo"] == "llama3.2:3b"
    assert estado["modelo_cargado"] is True
    assert estado["expira_utc"] == "2026-10-18T20:25:03+00:00"
    assert estado["latencia_ms"] >= 0
    assert estado["error"] is None
    assert all(llamada["timeout"] == (0.5, 2) for llamada in sesion.llamadas)


def test_estado_modelo_no_cargado():
    cliente, _ = cliente_con({"GET /api/version": RespuestaHTTPFalsa(200, {"version": "0.40.6"}),
                              "GET /api/ps": RespuestaHTTPFalsa(200, {"models": []})})
    estado = cliente.estado()
    assert estado["modelo_cargado"] is False
    assert estado["expira_utc"] is None


def test_estado_sin_api_ps_conserva_version_y_accesible():
    cliente, _ = cliente_con({"GET /api/version": RespuestaHTTPFalsa(200, {"version": "0.40.6"}),
                              "GET /api/ps": requests.Timeout()})
    estado = cliente.estado()
    assert (estado["accesible"], estado["version"]) == (True, "0.40.6")
    assert estado["latencia_ms"] >= 0
    assert (estado["modelo_cargado"], estado["expira_utc"]) == (None, None)
    assert "superó 2 s" in estado["error"]


@pytest.mark.parametrize("fallo", [requests.ConnectionError(), requests.Timeout(), RuntimeError("inesperado"),
                                   RespuestaHTTPFalsa(200, "{roto")])
def test_estado_nunca_lanza(fallo):
    cliente, _ = cliente_con({"GET /api/version": fallo})
    estado = cliente.estado()
    assert estado["accesible"] is False
    assert estado["version"] is None
    assert estado["modelo_cargado"] is None
    assert isinstance(estado["error"], str) and estado["error"]


def test_verificar_lista_modelos_y_precalienta():
    cuerpo_carga = {"model": "llama3.2:3b", "message": {"role": "assistant", "content": ""},
                    "done": True, "done_reason": "load"}
    cliente, sesion = cliente_con({"GET /api/tags": tags(entrada_tags(), entrada_tags("qwen2.5:3b")),
                                   "POST /api/chat": RespuestaHTTPFalsa(200, cuerpo_carga)})
    resultado = cliente.verificar()
    assert set(resultado) == {"accesible", "modelo_instalado", "modelos", "digest", "precalentado",
                              "segundos_precalentamiento"}
    assert resultado["accesible"] is True
    assert resultado["modelo_instalado"] is True
    assert resultado["modelos"] == ["llama3.2:3b", "qwen2.5:3b"]
    assert resultado["digest"] == DIGEST
    assert resultado["precalentado"] is True
    assert resultado["segundos_precalentamiento"] >= 0
    assert sesion.ultima("POST /api/chat")["json"]["messages"] == []


def test_verificar_sin_precalentar_o_sin_modelo_no_llama_chat():
    cliente, sesion = cliente_con({"GET /api/tags": [tags(entrada_tags()), tags(entrada_tags("llama3.2:1b"))]})
    sin_precalentar = cliente.verificar(precalentar=False)
    assert (sin_precalentar["precalentado"], sin_precalentar["segundos_precalentamiento"]) == (False, None)
    sin_modelo = cliente.verificar()
    assert (sin_modelo["modelo_instalado"], sin_modelo["digest"], sin_modelo["precalentado"]) == (False, None, False)
    assert all(llamada["ruta"] == "/api/tags" for llamada in sesion.llamadas)


def test_verificar_sin_servicio_lanza_error_ollama():
    cliente, _ = cliente_con({"GET /api/tags": requests.ConnectionError()})
    with pytest.raises(ErrorOllama) as info:
        cliente.verificar()
    assert info.value.tipo == "conexion"
