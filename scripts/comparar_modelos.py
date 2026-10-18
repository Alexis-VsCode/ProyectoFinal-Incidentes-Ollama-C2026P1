"""Compara modelos locales de Ollama con las preguntas de demostración y decide el modelo por defecto.

Ejecutar desde la raíz:
    python -m scripts.comparar_modelos --modelos llama3.2:3b llama3.2:1b --repeticiones 2 --modo automatico

Usa el agente real sobre el CSV completo con todas las prioridades. Latencia, tokens/s,
errores, truncadas, origen de la selección y cifras verificadas salen de las mismas funciones
de observabilidad que la aplicación (`calcular_slis` y `evaluar_slos` sobre una muestra por
consulta); el script solo añade el acierto de herramienta y argumentos y la cobertura de las
cifras clave frente a lo esperado. El log de la corrida va a un archivo propio, separado de
app.jsonl, para que el tráfico de prueba no se mezcle con el uso real. Las latencias miden la
herramienta, no los incidentes del CSV.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any

from src.agent import OPCIONES_INTERPRETACION, OPCIONES_SELECCION, AgenteIncidentes, RegistroConsulta
from src.data_loader import cargar_csv
from src.eda import SIN_DATO, AnalisisIncidentes
from src.herramientas import json_compacto, preparar_para_json
from src.llm_client import ClienteOllama, ErrorOllama
from src.observabilidad import (
    SPAN_INTERPRETACION,
    SPAN_SELECCION,
    MuestraConsulta,
    calcular_slis,
    cerrar_logger,
    configurar_logger,
    evaluar_slos,
)
from src.preguntas_demo import PREGUNTAS_DEMO, PreguntaDemo
from src.verificacion import cobertura_cifras, verificar_cifras

BASE = Path(__file__).resolve().parents[1]
RUTA_CSV = BASE / "data" / "incident_event_log.csv"
CARPETA_RESULTADOS = BASE / "resultados_locales"
MODELOS_POR_DEFECTO: tuple[str, ...] = ("llama3.2:3b", "llama3.2:1b")
MODELO_REFERENCIA = "llama3.2:3b"
REPETICIONES_POR_DEFECTO = 2
MINIMO_REPETICIONES = 2
SEMILLA_BASE = 100
MODOS: tuple[str, ...] = ("automatico", "manual")
ORIGENES_MODELO = frozenset({"modelo", "modelo_texto"})
COLUMNAS_CSV: tuple[str, ...] = (
    "fecha_utc", "modelo", "digest_corto", "version_ollama", "modo", "pregunta_id", "repeticion", "semilla",
    "estado", "herramientas_elegidas", "argumentos", "origen", "herramientas_esperadas", "acierto_herramienta",
    "acierto_argumentos", "discrepancia_enrutador", "cifras_evaluadas", "cifras_verificadas",
    "proporcion_verificada", "cobertura_cifras_clave", "truncada", "done_reason", "latencia_seleccion_s",
    "latencia_interpretacion_s", "latencia_total_s", "tokens_por_s", "prompt_tokens_interpretacion",
    "eval_tokens_interpretacion", "error", "trace_id",
)

FabricaCliente = Callable[..., Any]


def _marca_tiempo() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _proporcion(buenos: int, total: int) -> float | None:
    return None if total == 0 else round(buenos / total, 4)


def cargar_analisis_completo(ruta: Path = RUTA_CSV) -> AnalisisIncidentes:
    """Fachada de análisis sobre el CSV completo con todas las prioridades."""
    datos = cargar_csv(Path(ruta).read_bytes())
    prioridades = sorted(datos.incidentes["priority"].fillna(SIN_DATO).unique().tolist())
    return AnalisisIncidentes(datos.incidentes, datos.control, prioridades)


# Paso 1: una fila por consulta


def _segundos_span(traza: Sequence[dict[str, Any]], nombre: str) -> float | None:
    """Duración en segundos del primer span `nombre` de la traza; None si no hubo."""
    for evento in traza:
        atributos = evento.get("atributos") or {}
        if evento.get("evento") == nombre and atributos.get("tipo") == "span" and atributos.get("ms") is not None:
            return round(atributos["ms"] / 1000, 3)
    return None


def fila_desde_registro(registro: RegistroConsulta, pregunta: PreguntaDemo, modelo: str, repeticion: int,
                        semilla: int) -> dict[str, Any]:
    """Fila del CSV (claves en el orden de COLUMNAS_CSV) para una consulta terminada, sin NaN.

    `acierto_herramienta` es verdadero solo si eligió el modelo (origen modelo o modelo_texto)
    y el conjunto de herramientas coincide con el esperado; en preguntas sin herramientas
    esperadas (la guarda) acierto y cobertura quedan en None.
    """
    esperadas = pregunta.herramientas_esperadas
    acierto_herramienta = acierto_argumentos = cobertura = None
    if esperadas:
        acierto_herramienta = (registro.origen_seleccion in ORIGENES_MODELO
                               and set(registro.herramientas) == set(esperadas))
        acierto_argumentos = acierto_herramienta and all(
            dict(argumentos) == pregunta.argumentos_esperados[nombre]
            for nombre, argumentos in zip(registro.herramientas, registro.argumentos))
        if registro.estado == "respondida" and pregunta.claves_cifras:
            verificacion = verificar_cifras(registro.respuesta, registro.contexto, registro.pregunta)
            cobertura = cobertura_cifras(verificacion, registro.contexto, pregunta.claves_cifras)
    muestra = registro.a_muestra()
    interpretacion = registro.metricas.get("interpretacion") or {}
    return {
        "fecha_utc": registro.fecha_utc, "modelo": modelo,
        "digest_corto": registro.digest[:12] if registro.digest else None,
        "version_ollama": registro.version_ollama, "modo": registro.modo, "pregunta_id": pregunta.id,
        "repeticion": repeticion, "semilla": semilla, "estado": registro.estado,
        "herramientas_elegidas": "|".join(registro.herramientas),
        "argumentos": json_compacto([dict(a) for a in registro.argumentos]),
        "origen": registro.origen_seleccion, "herramientas_esperadas": "|".join(esperadas),
        "acierto_herramienta": acierto_herramienta, "acierto_argumentos": acierto_argumentos,
        "discrepancia_enrutador": registro.discrepancia_enrutador,
        "cifras_evaluadas": muestra.cifras_evaluadas, "cifras_verificadas": muestra.cifras_verificadas,
        "proporcion_verificada": registro.verificacion_cifras.get("proporcion"),
        "cobertura_cifras_clave": cobertura, "truncada": registro.truncada, "done_reason": registro.done_reason,
        "latencia_seleccion_s": _segundos_span(registro.traza, SPAN_SELECCION),
        "latencia_interpretacion_s": _segundos_span(registro.traza, SPAN_INTERPRETACION),
        "latencia_total_s": registro.segundos, "tokens_por_s": interpretacion.get("tokens_por_segundo"),
        "prompt_tokens_interpretacion": interpretacion.get("prompt_eval_count"),
        "eval_tokens_interpretacion": interpretacion.get("eval_count"),
        "error": "", "trace_id": registro.trace_id,
    }


def fila_de_error(pregunta: PreguntaDemo, modelo: str, repeticion: int, semilla: int, modo: str,
                  error: str, trace_id: str | None) -> dict[str, Any]:
    """Fila del CSV para una consulta que terminó con error: sin respuesta, con el mensaje en `error`."""
    fallo = False if pregunta.herramientas_esperadas else None
    fila = dict.fromkeys(COLUMNAS_CSV)
    fila.update(
        fecha_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), modelo=modelo, modo=modo,
        pregunta_id=pregunta.id, repeticion=repeticion, semilla=semilla, estado="error", herramientas_elegidas="",
        argumentos="[]", herramientas_esperadas="|".join(pregunta.herramientas_esperadas),
        acierto_herramienta=fallo, acierto_argumentos=fallo, cifras_evaluadas=0, cifras_verificadas=0,
        truncada=False, error=error, trace_id=trace_id)
    return fila


def _consultar(agente: AgenteIncidentes, pregunta: PreguntaDemo, modelo: str, repeticion: int, semilla: int,
               modo: str, muestras: list[MuestraConsulta]) -> dict[str, Any]:
    """Responde una pregunta y devuelve su fila; cualquier excepción queda como fila de error.

    Un ErrorOllama o un ValueError guardan su mensaje; un fallo inesperado, "Tipo: mensaje".
    Así una consulta fallida no aborta la corrida ni pierde las filas ya medidas.
    """
    previas = len(muestras)
    try:
        if modo == "manual":
            nombre = pregunta.herramientas_esperadas[0] if pregunta.herramientas_esperadas else "resumen"
            registro = agente.responder(pregunta.texto, nombre, argumentos=pregunta.argumentos_esperados.get(nombre))
        else:
            registro = agente.responder(pregunta.texto, modo="automatico")
    except (ErrorOllama, ValueError) as exc:
        mensaje = str(exc)
    except Exception as exc:  # noqa: BLE001 - la corrida sigue y el fallo queda en su fila
        mensaje = f"{type(exc).__name__}: {exc}"
    else:
        return fila_desde_registro(registro, pregunta, modelo, repeticion, semilla)
    trace_id = muestras[-1].trace_id if len(muestras) > previas else None
    return fila_de_error(pregunta, modelo, repeticion, semilla, modo, mensaje, trace_id)


# Paso 2: corrida completa


def ejecutar_comparacion(modelos: Sequence[str] = MODELOS_POR_DEFECTO,
                         repeticiones: int = REPETICIONES_POR_DEFECTO, modo: str = "automatico",
                         preguntas: Sequence[PreguntaDemo] = PREGUNTAS_DEMO,
                         fabrica_cliente: FabricaCliente = ClienteOllama, *,
                         analisis: AnalisisIncidentes | None = None, ruta_log: Path | None = None,
                         carpeta: Path = CARPETA_RESULTADOS) -> dict[str, Any]:
    """Responde cada pregunta `repeticiones` veces con cada modelo y resume y decide.

    `fabrica_cliente(modelo, seed=...)` crea el cliente; cada repetición usa la semilla
    SEMILLA_BASE + repetición (1, 2, ...). Un modelo que falta o no está permitido se omite
    con su motivo en metadatos["omitidos"]. La precarga se mide aparte y no entra en la
    latencia. El log va a `ruta_log` o a <carpeta>/logs/comparacion_<marca>.jsonl y se cierra
    al terminar. Devuelve {"metadatos", "filas", "resumen", "decision"}.
    """
    if repeticiones < 1:
        raise ValueError("repeticiones debe ser 1 o mayor.")
    if modo not in MODOS:
        raise ValueError("El modo debe ser automatico o manual.")
    destino_log = Path(ruta_log) if ruta_log is not None else Path(carpeta) / "logs" / f"comparacion_{_marca_tiempo()}.jsonl"
    analisis = analisis if analisis is not None else cargar_analisis_completo()
    filas: list[dict[str, Any]] = []
    muestras: list[MuestraConsulta] = []
    omitidos: dict[str, str] = {}
    info_modelos: dict[str, dict[str, Any]] = {}
    configurar_logger(destino_log)
    try:
        for modelo in modelos:
            # Paso 2.1: metadatos y precarga del modelo, fuera de la latencia medida.
            try:
                cliente = fabrica_cliente(modelo, seed=SEMILLA_BASE + 1)
                version = cliente.version()
                info = cliente.info_modelo()
                inicio = perf_counter()
                cliente.precalentar()
                precarga = round(perf_counter() - inicio, 3)
            except ErrorOllama as exc:
                omitidos[modelo] = f"omitido: ollama pull {modelo}" if exc.tipo == "no_instalado" else f"omitido: {exc}"
                continue
            except ValueError as exc:
                omitidos[modelo] = f"omitido: {exc}"
                continue
            info_modelos[modelo] = {"digest": info.digest, "version_ollama": version,
                                    "segundos_precalentamiento": precarga}
            # Paso 2.2: repeticiones con semilla propia.
            for repeticion in range(1, repeticiones + 1):
                semilla = SEMILLA_BASE + repeticion
                cliente_rep = cliente if repeticion == 1 else fabrica_cliente(modelo, seed=semilla)
                agente = AgenteIncidentes(analisis, cliente_rep, al_finalizar=muestras.append)
                for pregunta in preguntas:
                    filas.append(_consultar(agente, pregunta, modelo, repeticion, semilla, modo, muestras))
    finally:
        cerrar_logger()
    resumen = resumir(filas, muestras)
    metadatos = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sistema": platform.platform(),
        "procesador": platform.processor() or platform.machine(), "nucleos_logicos": os.cpu_count(),
        "python": platform.python_version(),
        "version_ollama": next((i["version_ollama"] for i in info_modelos.values()), None),
        "opciones": {"seleccion": OPCIONES_SELECCION, "interpretacion": OPCIONES_INTERPRETACION},
        "preguntas": [{"id": p.id, "texto": p.texto, "herramientas_esperadas": list(p.herramientas_esperadas),
                       "solo_comparador": p.solo_comparador} for p in preguntas],
        "modelos": info_modelos, "omitidos": omitidos, "repeticiones": repeticiones, "modo": modo,
        "ruta_log": str(destino_log),
    }
    return {"metadatos": metadatos, "filas": filas, "resumen": resumen,
            "decision": decidir_modelo(resumen, repeticiones=repeticiones)}


# Paso 3: resumen y decisión


def _acierto(filas: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Acierto frente a lo esperado: herramienta, argumentos, cobertura y guarda (filas sin acierto medible)."""
    medibles = [f for f in filas if f.get("acierto_herramienta") is not None]
    guarda = [f for f in filas if f.get("acierto_herramienta") is None]
    coberturas = [f["cobertura_cifras_clave"] for f in medibles if f.get("cobertura_cifras_clave") is not None]
    return {
        "n": len(filas), "evaluadas": len(medibles),
        "tasa_acierto_herramienta": _proporcion(sum(bool(f["acierto_herramienta"]) for f in medibles), len(medibles)),
        "tasa_acierto_argumentos": _proporcion(sum(bool(f.get("acierto_argumentos")) for f in medibles), len(medibles)),
        "media_cobertura": round(fmean(coberturas), 4) if coberturas else None,
        "tasa_acierto_guarda": _proporcion(sum(f.get("estado") == "rechazada" for f in guarda), len(guarda)),
    }


def resumir(filas: Sequence[dict[str, Any]], muestras: Iterable[MuestraConsulta]) -> dict[str, dict[str, Any]]:
    """Por modelo: {"slis", "slos", "acierto"}; SLI y SLO con las funciones de observabilidad."""
    muestras_por_modelo: dict[str, list[MuestraConsulta]] = defaultdict(list)
    for muestra in muestras:
        muestras_por_modelo[muestra.modelo].append(muestra)
    filas_por_modelo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fila in filas:
        filas_por_modelo[fila["modelo"]].append(fila)
    resumen: dict[str, dict[str, Any]] = {}
    for modelo in dict.fromkeys([*filas_por_modelo, *muestras_por_modelo]):
        grupo = muestras_por_modelo.get(modelo, [])
        resumen[modelo] = {"slis": calcular_slis(grupo).a_dict(), "slos": [e.a_dict() for e in evaluar_slos(grupo)],
                           "acierto": _acierto(filas_por_modelo.get(modelo, []))}
    return resumen


def _motivos_rechazo(candidato: dict[str, Any], referencia: dict[str, Any], margen_acierto: float,
                     tolerancia_latencia: float) -> list[str]:
    """Criterios que el candidato no cumple frente a la referencia; lista vacía si los cumple todos."""
    c, r = candidato["slis"], referencia["slis"]
    acierto_c = candidato["acierto"].get("tasa_acierto_herramienta")
    acierto_r = referencia["acierto"].get("tasa_acierto_herramienta")
    motivos = []
    if c.get("errores") != 0:
        motivos.append(f"tiene {c.get('errores')} errores")
    if c.get("tasa_truncadas") is None or r.get("tasa_truncadas") is None or c["tasa_truncadas"] > r["tasa_truncadas"]:
        motivos.append("trunca más respuestas que la referencia")
    if acierto_c is None or acierto_r is None or round(acierto_c - acierto_r, 6) < margen_acierto:
        motivos.append(f"acierto de herramienta a menos de {margen_acierto * 100:g} puntos de la referencia")
    if (c.get("proporcion_cifras_verificadas") is None or r.get("proporcion_cifras_verificadas") is None
            or c["proporcion_cifras_verificadas"] < r["proporcion_cifras_verificadas"]):
        motivos.append("menor proporción de cifras verificadas")
    if (c.get("latencia_p50_s") is None or r.get("latencia_p50_s") is None
            or c["latencia_p50_s"] > round(tolerancia_latencia * r["latencia_p50_s"], 6)):
        motivos.append(f"latencia mediana mayor que {tolerancia_latencia:g} veces la de la referencia")
    return motivos


def decidir_modelo(resumen: dict[str, dict[str, Any]], referencia: str = MODELO_REFERENCIA,
                   margen_acierto: float = 0.15, tolerancia_latencia: float = 1.10, *,
                   repeticiones: int = REPETICIONES_POR_DEFECTO) -> dict[str, Any]:
    """Mantiene la referencia salvo que un candidato cumpla todos los criterios.

    Criterios: 0 errores, tasa de truncadas no mayor, acierto de herramienta al menos
    `margen_acierto` por encima, proporción de cifras verificadas igual o mayor y latencia
    mediana no mayor que `tolerancia_latencia` veces la de la referencia. Estados: "cambiar",
    "mantener", "insuficiente" (menos de MINIMO_REPETICIONES) o "sin_referencia".
    """
    criterio = {"referencia": referencia, "margen_acierto": margen_acierto, "tolerancia_latencia": tolerancia_latencia,
                "repeticiones": repeticiones, "minimo_repeticiones": MINIMO_REPETICIONES}
    decision: dict[str, Any] = {"estado": "mantener", "modelo_elegido": referencia, "cambia": False,
                                "motivos": {}, "criterio": criterio}
    if referencia not in resumen:
        return decision | {"estado": "sin_referencia"}
    if repeticiones < MINIMO_REPETICIONES:
        return decision | {"estado": "insuficiente"}
    motivos = {modelo: _motivos_rechazo(datos, resumen[referencia], margen_acierto, tolerancia_latencia)
               for modelo, datos in resumen.items() if modelo != referencia}
    ganadores = [modelo for modelo, lista in motivos.items() if not lista]
    if not ganadores:
        return decision | {"motivos": motivos}
    elegido = max(ganadores, key=lambda m: (resumen[m]["acierto"]["tasa_acierto_herramienta"],
                                            -resumen[m]["slis"]["latencia_p50_s"]))
    return decision | {"estado": "cambiar", "modelo_elegido": elegido, "cambia": True, "motivos": motivos}


# Paso 4: archivos de salida y línea de comandos


def guardar(resultado: dict[str, Any], carpeta: Path) -> tuple[Path, Path]:
    """Escribe comparacion_modelos_<AAAAMMDDTHHMMSSZ>.json y .csv (utf-8-sig) en `carpeta`."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_json = carpeta / f"comparacion_modelos_{_marca_tiempo()}.json"
    ruta_csv = ruta_json.with_suffix(".csv")
    ruta_json.write_text(json.dumps(preparar_para_json(resultado), ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    with ruta_csv.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS_CSV, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(resultado["filas"])
    return ruta_json, ruta_csv


def crear_parser() -> argparse.ArgumentParser:
    """Argumentos de la línea de comandos con los valores por defecto del comparador."""
    parser = argparse.ArgumentParser(prog="python -m scripts.comparar_modelos",
                                     description="Compara modelos locales de Ollama con las preguntas de demostración.")
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS_POR_DEFECTO), help="Modelos a comparar.")
    parser.add_argument("--repeticiones", type=int, default=REPETICIONES_POR_DEFECTO,
                        help=f"Repeticiones por pregunta (mínimo {MINIMO_REPETICIONES} para decidir).")
    parser.add_argument("--modo", choices=MODOS, default="automatico", help="Selección de herramientas.")
    parser.add_argument("--carpeta", type=Path, default=CARPETA_RESULTADOS, help="Carpeta de resultados.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la comparación, guarda JSON y CSV e imprime la decisión; 1 si no se evaluó ningún modelo."""
    argumentos = crear_parser().parse_args(argv)
    resultado = ejecutar_comparacion(argumentos.modelos, argumentos.repeticiones, argumentos.modo,
                                     carpeta=argumentos.carpeta)
    ruta_json, ruta_csv = guardar(resultado, argumentos.carpeta)
    for modelo, motivo in resultado["metadatos"].get("omitidos", {}).items():
        print(f"{modelo}: {motivo}")
    decision = resultado["decision"]
    print(f"Decisión: {decision['estado']} (modelo por defecto: {decision['modelo_elegido']})")
    print(f"Resultados: {ruta_json}")
    print(f"Filas: {ruta_csv}")
    return 0 if resultado["resumen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
