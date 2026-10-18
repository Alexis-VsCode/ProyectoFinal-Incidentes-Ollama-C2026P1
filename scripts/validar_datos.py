"""Valida el CSV incluido sin interfaz ni Ollama y guarda la evidencia en resultados_locales/validacion_datos.json.

Ejecutar desde la raíz:
    python -m scripts.validar_datos

Reporta el entorno (sistema, Python y versiones de las dependencias directas), el control de
carga con la integridad SHA-256, el resumen por prioridad, el perfil de calidad de los
incidentes y las correlaciones de Spearman con las horas hasta resolución. Las horas son
percentiles descriptivos del histórico: no son SLO, SLA ni MTTR. Al final imprime un resumen
con los números en formato es-EC.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any, Final

import pandas as pd

from src.data_loader import SHA256_CSV_INCLUIDO, DatosIncidentes, cargar_ruta, verificar_integridad
from src.eda import SIN_DATO, AnalisisIncidentes, perfil_calidad
from src.visualizations import AVISO_DESCRIPTIVO

RAIZ: Final = Path(__file__).resolve().parents[1]
RUTA_CSV: Final = RAIZ / "data" / "incident_event_log.csv"
RUTA_SALIDA: Final = RAIZ / "resultados_locales" / "validacion_datos.json"
DEPENDENCIAS: Final = ("streamlit", "pandas", "numpy", "plotly", "requests", "matplotlib", "seaborn")
ESTADO_OLLAMA: Final = "No evaluado por este comando. Validar desde la aplicación local."


def numero_es(valor: float, decimales: int = 0) -> str:
    """Número con punto de miles y coma decimal (es-EC): 20769 -> "20.769"; 78.1666 con 2 -> "78,17"."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def _registros(tabla: pd.DataFrame) -> list[dict[str, Any]]:
    """Filas de la tabla como diccionarios JSON nativos; NaN pasa a None."""
    return json.loads(tabla.to_json(orient="records"))


def construir_salida(datos: DatosIncidentes, sha256_esperado: str | None = None) -> dict[str, Any]:
    """Evidencia de validación de `datos` con todas las prioridades seleccionadas.

    Claves: fecha_utc, sistema, python, dependencias (DEPENDENCIAS con su versión instalada),
    control, integridad (verificar_integridad contra `sha256_esperado`; "no_aplica" si es
    None), resumen, nota_horas, prioridades, calidad (perfil_calidad de los incidentes),
    correlaciones (Pearson y Spearman con las horas), matriz_spearman y estado_ollama. El
    resultado se serializa con json.dumps(allow_nan=False).
    """
    # Paso 1: fachada con todas las prioridades, los nulos como SIN_DATO.
    prioridades = sorted(datos.incidentes["priority"].fillna(SIN_DATO).unique().tolist())
    analisis = AnalisisIncidentes(datos.incidentes, datos.control, prioridades)
    # Paso 2: entorno, control y cálculos de la fachada, ya en tipos JSON.
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sistema": platform.system(),
        "python": platform.python_version(),
        "dependencias": {paquete: importlib.metadata.version(paquete) for paquete in DEPENDENCIAS},
        "control": datos.control,
        "integridad": verificar_integridad(datos.control, sha256_esperado),
        "resumen": analisis.resumen(),
        "nota_horas": AVISO_DESCRIPTIVO,
        "prioridades": _registros(analisis.prioridades_resumen()),
        "calidad": _registros(perfil_calidad(analisis.datos)),
        "correlaciones": analisis.relaciones(),
        "matriz_spearman": json.loads(analisis.matriz_relaciones().to_json(orient="index")),
        "estado_ollama": ESTADO_OLLAMA,
    }


def _horas(valor: float | None) -> str:
    """Horas en es-EC con dos decimales, o "sin dato"."""
    return "sin dato" if valor is None else f"{numero_es(valor, 2)} h"


def resumen_legible(salida: dict[str, Any]) -> str:
    """Texto breve de la salida con los números en formato es-EC, para leer en la consola."""
    control, resumen = salida["control"], salida["resumen"]
    lineas = [
        "Resumen de la validación (formato es-EC)",
        f"  Eventos: {numero_es(control['eventos'])} | Incidentes: {numero_es(control['incidentes'])}"
        f" | Integridad SHA-256: {salida['integridad']['estado']}",
        f"  Horas hasta resolución: mediana {_horas(resumen['mediana_horas'])},"
        f" p90 {_horas(resumen['p90_horas'])}. {salida['nota_horas']}",
        "  Por prioridad:",
    ]
    for fila in salida["prioridades"]:
        lineas.append(f"    {fila['priority']}: {numero_es(fila['incidentes'])} incidentes,"
                      f" mediana {_horas(fila['mediana_horas'])}")
    lineas.append("  Spearman con las horas (asociación, no causa):")
    for fila in salida["correlaciones"]:
        rho = fila["spearman_con_duracion"]
        valor = "sin definir" if rho is None else numero_es(rho, 3)
        lineas.append(f"    {fila['variable']}: {valor} (pares: {numero_es(fila['pares_validos'])})")
    return "\n".join(lineas)


def main() -> None:
    """Carga el CSV incluido, escribe validacion_datos.json e imprime la evidencia y el resumen."""
    datos = cargar_ruta(RUTA_CSV)
    salida = construir_salida(datos, SHA256_CSV_INCLUIDO)
    RUTA_SALIDA.parent.mkdir(exist_ok=True)
    texto = json.dumps(salida, ensure_ascii=False, indent=2, allow_nan=False)
    RUTA_SALIDA.write_text(texto, encoding="utf-8")
    print(texto)
    print()
    print(resumen_legible(salida))


if __name__ == "__main__":
    main()
