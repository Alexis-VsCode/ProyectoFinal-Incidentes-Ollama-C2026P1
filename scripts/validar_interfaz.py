"""Recorre la interfaz sin navegador ni Ollama con Streamlit AppTest y guarda los tiempos de cada paso.

Ejecutar desde la raíz:
    python -m scripts.validar_interfaz

Busca los widgets por su clave estable (`pestana`, `filtro_prioridades`, `modelo`), no por su
posición ni por su rótulo. Comprueba la carga con 20.769 incidentes, el filtro 1 - Critical
con 220 incidentes y mediana 78,17 h, el aviso de selección vacía, las siete pestañas, un
rerun y el cambio de modelo. Si un paso falla (por ejemplo, porque app.py todavía no expone
una clave), lo registra, detiene el recorrido sin traza de error y sale con código 1; la
medición se guarda igual en resultados_locales/medicion_interfaz.json.

No evalúa respuestas de Ollama ni la apariencia en el navegador. Los eventos que la app
registre durante el recorrido van a una carpeta de logs propia, separada de app.jsonl.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Final

from streamlit.testing.v1 import AppTest

from scripts.validar_datos import numero_es
from src.observabilidad import VARIABLE_ENTORNO_LOGS

RAIZ: Final = Path(__file__).resolve().parents[1]
RUTA_APP: Final = RAIZ / "app.py"
RUTA_MEDICION: Final = RAIZ / "resultados_locales" / "medicion_interfaz.json"
CARPETA_LOGS: Final = RAIZ / "resultados_locales" / "logs_validacion_interfaz"
TIMEOUT_S: Final = 60
PESTANAS: Final = ("Datos", "Resumen", "Explorar", "Relaciones y segmentos",
                   "Agente local", "Observabilidad", "Arquitectura")
CRITICAL: Final = "1 - Critical"
KPI_TOTAL: Final = "20.769"
KPI_CRITICAL: Final = "220"
MEDIANA_CRITICAL: Final = "78,17"
MODELO_ALTERNATIVO: Final = "llama3.2:1b"
ALCANCE: Final = ("Mide Streamlit AppTest sin navegador ni inferencia: no evalúa respuestas de Ollama "
                  "ni la apariencia de la página.")

Paso = tuple[str, Callable[[], None]]


def ejecutar_pasos(pasos: Sequence[Paso]) -> dict[str, Any]:
    """Ejecuta los pasos en orden, mide cada uno y se detiene en el primero que lanza una excepción.

    Devuelve {"estado": "completa" | "incompleta", "total_segundos", "pasos"}; cada paso es
    {"paso", "estado": "ok" | "fallo", "segundos", "detalle"}, con los segundos redondeados a
    milésimas y el detalle "Tipo: mensaje" solo en el paso que falló. Nunca relanza la excepción.
    """
    registros: list[dict[str, Any]] = []
    estado = "completa"
    inicio_total = perf_counter()
    for nombre, accion in pasos:
        inicio = perf_counter()
        try:
            accion()
        except Exception as exc:  # noqa: BLE001 - cualquier fallo se registra y detiene el recorrido
            registros.append({"paso": nombre, "estado": "fallo", "segundos": round(perf_counter() - inicio, 3),
                              "detalle": f"{type(exc).__name__}: {exc}"})
            estado = "incompleta"
            break
        registros.append({"paso": nombre, "estado": "ok", "segundos": round(perf_counter() - inicio, 3),
                          "detalle": ""})
    return {"estado": estado, "total_segundos": round(perf_counter() - inicio_total, 3), "pasos": registros}


def guardar_medicion(medicion: dict[str, Any], ruta: Path = RUTA_MEDICION) -> Path:
    """Escribe la medición como JSON UTF-8 legible, crea la carpeta si falta y devuelve la ruta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(medicion, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return ruta


def _sin_excepcion(vista: AppTest) -> None:
    """AssertionError con el primer mensaje si la última ejecución dejó una excepción en la página."""
    if vista.exception:
        raise AssertionError(f"La interfaz mostró una excepción: {vista.exception[0].message}")


def fijar_valor(vista: AppTest, clave: str, valor: Any) -> None:
    """Asigna `valor` al widget con `clave`, vuelve a ejecutar el script y exige que no haya excepción.

    Usa set_value si el nodo lo admite (radio, selectbox, multiselect, segmented_control); si la
    clave es de un contenedor sin interacción en AppTest (st.tabs con key) o solo existe en
    session_state, la fija por session_state. KeyError si la clave no está ni en la página ni
    en session_state.
    """
    try:
        nodo = vista.get_by_key(clave)
    except KeyError:
        nodo = None
    if nodo is not None and hasattr(nodo, "set_value"):
        nodo.set_value(valor)
    elif nodo is not None or clave in vista.session_state:
        vista.session_state[clave] = valor
    else:
        raise KeyError(clave)
    vista.run()
    _sin_excepcion(vista)


def exigir_metrica(vista: AppTest, texto: str, exacta: bool = True) -> None:
    """AssertionError si ninguna métrica visible vale `texto` (o lo contiene, si exacta es False)."""
    valores = [str(m.value) for m in vista.metric]
    if not any(v == texto if exacta else texto in v for v in valores):
        raise AssertionError(f"Ninguna métrica muestra {texto!r}; visibles: {valores}")


def pasos_interfaz(vista: AppTest) -> list[Paso]:
    """Recorrido completo de la interfaz sobre `vista`, en el orden en que debe ejecutarse."""
    def carga_inicial() -> None:
        vista.run()
        _sin_excepcion(vista)

    def resumen_total() -> None:
        fijar_valor(vista, "pestana", "Resumen")
        exigir_metrica(vista, KPI_TOTAL)

    def filtro_critical() -> None:
        fijar_valor(vista, "filtro_prioridades", [CRITICAL])
        exigir_metrica(vista, KPI_CRITICAL)
        exigir_metrica(vista, MEDIANA_CRITICAL, exacta=False)

    def seleccion_vacia() -> None:
        fijar_valor(vista, "filtro_prioridades", [])
        if not vista.warning:
            raise AssertionError("Una selección vacía debe mostrar un aviso.")

    def restaurar_filtro() -> None:
        fijar_valor(vista, "filtro_prioridades", list(vista.get_by_key("filtro_prioridades").options))
        exigir_metrica(vista, KPI_TOTAL)

    def rerun() -> None:
        vista.run()
        _sin_excepcion(vista)

    pasos: list[Paso] = [
        ("carga inicial", carga_inicial),
        (f"pestaña Resumen con {KPI_TOTAL} incidentes", resumen_total),
        (f"filtro {CRITICAL}: {KPI_CRITICAL} incidentes y mediana {MEDIANA_CRITICAL} h", filtro_critical),
        ("selección vacía muestra aviso", seleccion_vacia),
        ("todas las prioridades de nuevo", restaurar_filtro),
    ]
    pasos += [(f"pestaña {nombre}", lambda n=nombre: fijar_valor(vista, "pestana", n)) for nombre in PESTANAS]
    pasos += [
        ("rerun sin cambios", rerun),
        (f"cambio de modelo a {MODELO_ALTERNATIVO}", lambda: fijar_valor(vista, "modelo", MODELO_ALTERNATIVO)),
    ]
    return pasos


def main() -> int:
    """Recorre app.py, guarda la medición e imprime cada paso; 0 si el recorrido es completo, 1 si no."""
    # Paso 1: los eventos del recorrido van a una carpeta propia, no al log de uso real.
    os.environ.setdefault(VARIABLE_ENTORNO_LOGS, str(CARPETA_LOGS))
    vista = AppTest.from_file(str(RUTA_APP), default_timeout=TIMEOUT_S)
    medicion = ejecutar_pasos(pasos_interfaz(vista))
    # Paso 2: contexto de la medición y escritura del JSON.
    salida = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sistema": platform.system(),
        "python": platform.python_version(),
        "streamlit": importlib.metadata.version("streamlit"),
        "app": RUTA_APP.name,
        **medicion,
        "alcance": ALCANCE,
    }
    ruta = guardar_medicion(salida)
    # Paso 3: informe en consola con los tiempos en formato es-EC.
    for paso in medicion["pasos"]:
        linea = f"[{paso['estado'].upper():5}] {paso['paso']}: {numero_es(paso['segundos'], 2)} s"
        print(linea + (f" | {paso['detalle']}" if paso["detalle"] else ""))
    print(f"Validación {medicion['estado']} en {numero_es(medicion['total_segundos'], 2)} s. Medición: "
          f"{ruta.relative_to(RAIZ)}")
    if medicion["estado"] != "completa":
        print("Si app.py todavía está en construcción, vuelva a ejecutar este comando al terminarla.")
    print(ALCANCE)
    return 0 if medicion["estado"] == "completa" else 1


if __name__ == "__main__":
    raise SystemExit(main())
