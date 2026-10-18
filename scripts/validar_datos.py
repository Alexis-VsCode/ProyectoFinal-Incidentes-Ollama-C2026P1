"""Ejecutar desde la raíz: python -m scripts.validar_datos."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

from src.data_loader import cargar_csv
from src.eda import AnalisisIncidentes, SIN_DATO


def main():
    raiz = Path(__file__).resolve().parents[1]
    datos = cargar_csv((raiz / "data/incident_event_log.csv").read_bytes())
    prioridades = sorted(datos.incidentes["priority"].fillna(SIN_DATO).unique().tolist())
    analisis = AnalisisIncidentes(datos.incidentes, datos.control, prioridades)
    salida = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "sistema": platform.system(), "python": platform.python_version(),
        "dependencias": {p: importlib.metadata.version(p) for p in ["streamlit", "pandas", "numpy", "plotly", "requests"]},
        "control": datos.control, "resumen": analisis.resumen(),
        "prioridades": json.loads(analisis.prioridades_resumen().to_json(orient="records")),
        "estado_ollama": "No evaluado por este comando. Validar desde la aplicación local.",
    }
    carpeta = raiz / "resultados_locales"
    carpeta.mkdir(exist_ok=True)
    texto = json.dumps(salida, ensure_ascii=False, indent=2, allow_nan=False)
    (carpeta / "validacion_datos.json").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
