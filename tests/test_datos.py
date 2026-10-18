"""Pruebas de carga por ruta, firma del archivo, tipos de la tabla de incidentes e integridad.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src import data_loader
from src.data_loader import (
    SHA256_CSV_INCLUIDO, DatosIncidentes, cargar_csv, cargar_ruta, firma_archivo,
    verificar_integridad,
)
from src.diccionario import DURACION


def _prohibir_lectura(monkeypatch: pytest.MonkeyPatch) -> None:
    def lectura_prohibida(self: Path) -> bytes:
        raise AssertionError("no debe leer el contenido del archivo")
    monkeypatch.setattr(Path, "read_bytes", lectura_prohibida)


def test_firma_archivo_no_lee_contenido_y_cambia_con_mtime(
    ruta_csv_sintetico: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prohibir_lectura(monkeypatch)
    ruta, tamano, mtime_ns = firma_archivo(ruta_csv_sintetico)
    assert ruta == str(ruta_csv_sintetico.resolve())
    assert tamano == ruta_csv_sintetico.stat().st_size
    os.utime(ruta_csv_sintetico, ns=(mtime_ns + 5_000_000_000, mtime_ns + 5_000_000_000))
    ruta_2, tamano_2, mtime_ns_2 = firma_archivo(ruta_csv_sintetico)
    assert (ruta_2, tamano_2) == (ruta, tamano)
    assert mtime_ns_2 != mtime_ns


def test_cargar_ruta_equivale_a_cargar_csv(ruta_csv_sintetico: Path, csv_sintetico: bytes) -> None:
    por_ruta = cargar_ruta(ruta_csv_sintetico)
    por_bytes = cargar_csv(csv_sintetico)
    assert isinstance(por_ruta, DatosIncidentes)
    assert por_ruta.control["sha256_csv"] == por_bytes.control["sha256_csv"]
    assert por_ruta.incidentes.equals(por_bytes.incidentes)


def test_cargar_ruta_rechaza_archivo_grande_antes_de_leer(
    ruta_csv_sintetico: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_loader, "MAX_BYTES", 10)
    _prohibir_lectura(monkeypatch)
    with pytest.raises(ValueError, match="50 MiB"):
        cargar_ruta(ruta_csv_sintetico)


def test_cargar_ruta_archivo_vacio_lanza_value_error(tmp_path: Path) -> None:
    ruta = tmp_path / "vacio.csv"
    ruta.write_bytes(b"")
    with pytest.raises(ValueError, match="50 MiB"):
        cargar_ruta(ruta)


def test_cargar_ruta_inexistente_lanza_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        cargar_ruta(tmp_path / "no_existe.csv")
    with pytest.raises(OSError):
        firma_archivo(tmp_path / "no_existe.csv")


def test_tipos_de_la_tabla_incidentes_cumplen_el_contrato(datos_sinteticos: DatosIncidentes) -> None:
    tipos = datos_sinteticos.incidentes.dtypes
    for columna in ("number", "priority", "incident_state", "assignment_group", "category"):
        assert isinstance(tipos[columna], pd.StringDtype), columna
    for columna in ("opened_at", "resolved_at", "closed_at", "sys_updated_at"):
        assert str(tipos[columna]) == "datetime64[ns]", columna
    for columna in ("reassignment_count", "reopen_count", "sys_mod_count"):
        assert str(tipos[columna]) == "Float64", columna
    assert str(tipos[DURACION]) == "float64"
    assert datos_sinteticos.incidentes["assignment_group"].isna().sum() == 1


def test_verificar_integridad_coincide_difiere_y_no_aplica(datos_sinteticos: DatosIncidentes) -> None:
    control = datos_sinteticos.control
    calculado = control["sha256_csv"]
    assert verificar_integridad(control, calculado) == {
        "estado": "coincide", "esperado": calculado, "calculado": calculado,
    }
    assert verificar_integridad(control, SHA256_CSV_INCLUIDO) == {
        "estado": "difiere", "esperado": SHA256_CSV_INCLUIDO, "calculado": calculado,
    }
    assert verificar_integridad(control, None) == {
        "estado": "no_aplica", "esperado": None, "calculado": calculado,
    }


def test_verificar_integridad_ignora_mayusculas_del_hash_esperado(datos_sinteticos: DatosIncidentes) -> None:
    calculado = datos_sinteticos.control["sha256_csv"]
    assert verificar_integridad(datos_sinteticos.control, calculado.upper())["estado"] == "coincide"


def test_sha256_csv_incluido_es_el_hash_del_taller1() -> None:
    assert SHA256_CSV_INCLUIDO == "5d77ea4853e443cf8447fca0f3089b7689e5469225cb68f72b2b9feea5f2d0d2"
