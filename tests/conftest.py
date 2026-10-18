"""Fixtures compartidas: CSV sintético congelado de 9 eventos y aislamiento del registro JSONL.

El CSV sintético tiene 9 eventos y 8 incidentes: INC01 con un evento previo
«New», un incidente sin resolución (INC05), una duración de 0 h (INC07), un
grupo desconocido (INC07) y una categoría desconocida (INC08). Las pruebas por
nombre dependen de estas filas exactas. Sirve para verificar reglas, nunca
para resultados académicos.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest

from src.data_loader import DatosIncidentes, cargar_csv

COLUMNAS_CSV_SINTETICO: tuple[str, ...] = (
    "number", "priority", "incident_state", "assignment_group", "category",
    "opened_at", "resolved_at", "closed_at", "sys_updated_at",
    "reassignment_count", "reopen_count", "sys_mod_count",
)

FILAS_CSV_SINTETICO: tuple[tuple[str, ...], ...] = (
    ("INC01", "1 - Critical", "New", "G1", "C1", "05/01/2016 08:00", "?", "?", "05/01/2016 08:05", "0", "0", "1"),
    ("INC01", "1 - Critical", "Closed", "G1", "C1", "05/01/2016 08:00", "05/01/2016 10:00", "05/01/2016 11:00", "05/01/2016 11:00", "0", "0", "2"),
    ("INC02", "1 - Critical", "Closed", "G1", "C1", "10/01/2016 08:00", "10/01/2016 12:00", "10/01/2016 13:00", "10/01/2016 13:00", "1", "0", "4"),
    ("INC03", "2 - High", "Closed", "G2", "C1", "15/01/2016 08:00", "15/01/2016 16:00", "15/01/2016 16:00", "15/01/2016 16:00", "2", "0", "6"),
    ("INC04", "3 - Moderate", "Closed", "G2", "C2", "20/01/2016 08:00", "21/01/2016 08:00", "21/01/2016 08:00", "21/01/2016 08:00", "3", "1", "8"),
    ("INC05", "3 - Moderate", "Active", "G3", "C2", "03/03/2016 08:00", "?", "?", "03/03/2016 09:00", "4", "0", "10"),
    ("INC06", "3 - Moderate", "Closed", "G3", "C3", "04/03/2016 08:00", "06/03/2016 08:00", "06/03/2016 08:00", "06/03/2016 08:00", "5", "1", "12"),
    ("INC07", "4 - Low", "Closed", "?", "C3", "05/03/2016 08:00", "05/03/2016 08:00", "05/03/2016 08:00", "05/03/2016 08:00", "0", "0", "1"),
    ("INC08", "4 - Low", "Closed", "G4", "?", "06/03/2016 08:00", "10/03/2016 08:00", "10/03/2016 08:00", "10/03/2016 08:00", "6", "2", "14"),
)


def construir_csv(filas: tuple[tuple[str, ...], ...] = FILAS_CSV_SINTETICO) -> bytes:
    """Devuelve el CSV UTF-8 con la cabecera COLUMNAS_CSV_SINTETICO y las filas dadas."""
    salida = StringIO()
    escritor = csv.writer(salida, lineterminator="\n")
    escritor.writerow(COLUMNAS_CSV_SINTETICO)
    escritor.writerows(filas)
    return salida.getvalue().encode("utf-8")


@pytest.fixture(autouse=True)
def aislar_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Dirige el registro JSONL de la app a un directorio temporal en cada prueba."""
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(scope="session")
def csv_sintetico() -> bytes:
    """Bytes del CSV sintético congelado: 9 eventos, 8 incidentes."""
    return construir_csv()


@pytest.fixture
def ruta_csv_sintetico(tmp_path: Path, csv_sintetico: bytes) -> Path:
    """Ruta de un archivo temporal con el CSV sintético."""
    ruta = tmp_path / "incidentes_sinteticos.csv"
    ruta.write_bytes(csv_sintetico)
    return ruta


@pytest.fixture
def datos_sinteticos(csv_sintetico: bytes) -> DatosIncidentes:
    """Resultado de cargar_csv sobre el CSV sintético; nuevo en cada prueba."""
    return cargar_csv(csv_sintetico)
