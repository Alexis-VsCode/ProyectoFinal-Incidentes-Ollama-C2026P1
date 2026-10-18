"""Pruebas de los scripts de apoyo: diagrama de arquitectura, validación de datos y de interfaz.

No levantan Streamlit ni Ollama: el diagrama se dibuja en memoria, la validación de datos usa el
CSV sintético de conftest y la medición de la interfaz se prueba con pasos simulados.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json
import struct
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from scripts.generar_arquitectura import LEYENDA, crear_figura, exportar, textos_figura
from scripts.validar_datos import construir_salida, numero_es
from scripts.validar_interfaz import ejecutar_pasos, fijar_valor, guardar_medicion
from src.data_loader import DatosIncidentes

COMPONENTES_OBLIGATORIOS: tuple[str, ...] = (
    "Usuario", "app.py", "Streamlit", "7 pestañas", "AnalisisIncidentes", "Strategy", "Factory",
    "data_loader", "Incident Response Log", "SHA-256", "visualizations", "Plotly", "Seaborn",
    "AgenteIncidentes", "RegistroHerramientas", "lista blanca", "5 herramientas", "ClienteOllama",
    "Adapter", "Ollama local", "llama3.2:3b", "127.0.0.1:11434", "verificación de cifras",
    "observabilidad", "JSONL", "trace_id", "SLIs", "SLO",
)


def _ancho_png(ruta: Path) -> int:
    """Ancho en píxeles leído de la cabecera IHDR del PNG."""
    cabecera = ruta.read_bytes()[:24]
    assert cabecera[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">I", cabecera[16:20])[0]


def test_diagrama_incluye_componentes_y_leyenda_de_responsabilidades() -> None:
    texto = "\n".join(textos_figura(crear_figura()))
    faltantes = [c for c in COMPONENTES_OBLIGATORIOS if c not in texto]
    assert faltantes == []
    assert LEYENDA == "Python calcula; el modelo interpreta; la persona verifica"
    assert LEYENDA in texto


def test_diagrama_sin_emojis() -> None:
    texto = "".join(textos_figura(crear_figura()))
    simbolos = {c for c in texto if unicodedata.category(c) == "So" or ord(c) >= 0x1F000}
    assert simbolos == set()


def test_exportar_png_de_unos_1800_px_y_svg_sin_figuras_abiertas(tmp_path: Path) -> None:
    abiertas_antes = plt.get_fignums()
    png, svg = exportar(tmp_path)
    assert png == tmp_path / "arquitectura.png" and svg == tmp_path / "arquitectura.svg"
    assert 1700 <= _ancho_png(png) <= 1900
    assert svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert "<svg" in svg.read_text(encoding="utf-8")
    assert plt.get_fignums() == abiertas_antes


def test_numero_es_usa_punto_de_miles_y_coma_decimal() -> None:
    assert numero_es(20769) == "20.769"
    assert numero_es(119998) == "119.998"
    assert numero_es(78.1666, 2) == "78,17"
    assert numero_es(1234.5, 1) == "1.234,5"


def test_salida_de_validar_datos_reporta_versiones_calidad_y_spearman(datos_sinteticos: DatosIncidentes) -> None:
    salida = construir_salida(datos_sinteticos)
    assert {"matplotlib", "seaborn", "streamlit", "pandas", "plotly"} <= set(salida["dependencias"])
    assert salida["resumen"]["incidentes_seleccionados"] == 8
    critical = next(p for p in salida["prioridades"] if p["priority"] == "1 - Critical")
    assert critical["incidentes"] == 2
    variables_calidad = [fila["variable"] for fila in salida["calidad"]]
    assert "priority" in variables_calidad and "tiempo_resolucion_horas" in variables_calidad
    assert [c["variable"] for c in salida["correlaciones"]] == ["sys_mod_count", "reassignment_count", "reopen_count"]
    assert all("spearman_con_duracion" in c for c in salida["correlaciones"])
    assert "no son SLO" in salida["nota_horas"]
    json.dumps(salida, allow_nan=False)


def test_ejecutar_pasos_mide_tiempos_y_se_detiene_en_el_primer_fallo(tmp_path: Path) -> None:
    ejecutados: list[str] = []

    def falla() -> None:
        ejecutados.append("falla")
        raise KeyError("filtro_prioridades")

    pasos = [
        ("carga inicial", lambda: ejecutados.append("carga")),
        ("filtro 1 - Critical", falla),
        ("seleccion vacia", lambda: ejecutados.append("no debe correr")),
    ]
    medicion = ejecutar_pasos(pasos)
    assert ejecutados == ["carga", "falla"]
    assert medicion["estado"] == "incompleta"
    assert [p["estado"] for p in medicion["pasos"]] == ["ok", "fallo"]
    assert "filtro_prioridades" in medicion["pasos"][1]["detalle"]
    assert all(isinstance(p["segundos"], float) and p["segundos"] >= 0 for p in medicion["pasos"])
    ruta = guardar_medicion(medicion, tmp_path / "medicion_interfaz.json")
    assert json.loads(ruta.read_text(encoding="utf-8"))["estado"] == "incompleta"


def test_ejecutar_pasos_completa_cuando_todo_pasa() -> None:
    medicion = ejecutar_pasos([("uno", lambda: None), ("dos", lambda: None)])
    assert medicion["estado"] == "completa"
    assert medicion["total_segundos"] >= 0
    assert [p["paso"] for p in medicion["pasos"]] == ["uno", "dos"]


class _NodoConValor:
    """Widget simulado que admite set_value."""

    def __init__(self) -> None:
        self.valor: object = None

    def set_value(self, valor: object) -> _NodoConValor:
        self.valor = valor
        return self


class _VistaFalsa:
    """Lo mínimo de AppTest que usa fijar_valor: get_by_key, session_state, run y exception."""

    def __init__(self, nodos: dict[str, object], estado: dict[str, object]) -> None:
        self.nodos = nodos
        self.session_state = estado
        self.exception: list[object] = []
        self.ejecuciones = 0

    def get_by_key(self, clave: str) -> object:
        return self.nodos[clave]

    def run(self) -> _VistaFalsa:
        self.ejecuciones += 1
        return self


def test_fijar_valor_usa_set_value_o_session_state_y_falla_si_la_clave_no_existe() -> None:
    filtro = _NodoConValor()
    vista = _VistaFalsa({"filtro_prioridades": filtro}, {"pestana": "Datos"})
    fijar_valor(vista, "filtro_prioridades", ["1 - Critical"])
    fijar_valor(vista, "pestana", "Resumen")
    assert filtro.valor == ["1 - Critical"]
    assert vista.session_state["pestana"] == "Resumen"
    assert vista.ejecuciones == 2
    with pytest.raises(KeyError):
        fijar_valor(vista, "clave_inexistente", 1)
    assert "clave_inexistente" not in vista.session_state


@pytest.mark.parametrize("valor, esperado", [(0, "0"), (220, "220"), (0.5, "0,50")])
def test_numero_es_casos_borde(valor: float, esperado: str) -> None:
    decimales = 2 if isinstance(valor, float) else 0
    assert numero_es(valor, decimales) == esperado
