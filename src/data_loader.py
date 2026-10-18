"""Carga del CSV de Taller 1, consolidación determinista por incidente y control de integridad.

No importa Streamlit ni el módulo de observabilidad: la composición con la caché
y con los spans ocurre en la interfaz.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import os
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.diccionario import DURACION

MAX_BYTES = 50 * 1024 * 1024
MAX_FILAS = 300_000
FECHAS = ("opened_at", "sys_created_at", "sys_updated_at", "resolved_at", "closed_at")
CONTADORES = ("sys_mod_count", "reassignment_count", "reopen_count")
COLUMNAS = (
    "number", "priority", "incident_state", "assignment_group", "category",
    "opened_at", "resolved_at", "closed_at", "sys_updated_at",
    "reassignment_count", "reopen_count", "sys_mod_count",
)
SHA256_CSV_INCLUIDO: Final = "5d77ea4853e443cf8447fca0f3089b7689e5469225cb68f72b2b9feea5f2d0d2"
MENSAJE_TAMANO: Final = "Seleccione un CSV no vacío de hasta 50 MiB."


@dataclass
class DatosIncidentes:
    """Eventos originales, una fila por incidente y el control del archivo cargado."""
    eventos: pd.DataFrame
    incidentes: pd.DataFrame
    control: dict


def cargar_csv(contenido: bytes) -> DatosIncidentes:
    """Lee bytes UTF-8; '?' significa desconocido y nunca se imputa.

    Lanza ValueError si el archivo está vacío, supera MAX_BYTES, no es un CSV
    UTF-8 legible, le faltan columnas de COLUMNAS o tiene eventos sin number.
    """
    if not contenido or len(contenido) > MAX_BYTES:
        raise ValueError(MENSAJE_TAMANO)
    # Paso 1: cabecera única y lectura como texto, con '?' y vacío como nulos.
    try:
        texto = contenido.decode("utf-8-sig")
        cabecera = next(csv.reader(StringIO(texto)), [])
        nombres = [c.strip() for c in cabecera]
        if not nombres or any(not c for c in nombres) or len(nombres) != len(set(nombres)):
            raise ValueError("La cabecera necesita nombres únicos y no vacíos.")
        eventos = pd.read_csv(
            BytesIO(contenido), dtype="string", encoding="utf-8-sig",
            keep_default_na=False, na_values=["?", ""], nrows=MAX_FILAS + 1,
        )
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("No se pudo leer el CSV UTF-8 separado por comas.") from exc
    eventos.columns = nombres
    faltantes = sorted(set(COLUMNAS) - set(eventos.columns))
    if faltantes:
        raise ValueError(f"Se requiere el esquema de Incident Response Log. Faltan: {faltantes}")
    if eventos.empty or len(eventos) > MAX_FILAS:
        raise ValueError("El CSV debe contener entre 1 y 300.000 eventos.")
    for columna in eventos:
        eventos[columna] = eventos[columna].str.strip().replace({"?": pd.NA, "": pd.NA})
    if eventos["number"].isna().any():
        raise ValueError("Hay eventos sin identificador number; revise el archivo.")
    duplicados = int(eventos.duplicated().sum())
    # Paso 2: fechas día/mes/año; las inválidas quedan nulas y se cuentan.
    fechas_invalidas = {}
    for columna in FECHAS:
        if columna not in eventos:
            continue
        original = eventos[columna]
        convertida = pd.to_datetime(original, format="%d/%m/%Y %H:%M", errors="coerce")
        fechas_invalidas[columna] = int((original.notna() & convertida.isna()).sum())
        eventos[columna] = convertida
    # Paso 3: contadores enteros no negativos; lo demás queda nulo y se cuenta.
    contadores_invalidos = {}
    for columna in CONTADORES:
        original = eventos[columna]
        numeros = pd.to_numeric(original, errors="coerce")
        invalidos = numeros.isna() | ~np.isfinite(numeros) | (numeros < 0) | (numeros % 1 != 0)
        contadores_invalidos[columna] = int((original.notna() & invalidos).sum())
        eventos[columna] = numeros.mask(invalidos).astype("Float64")
    # Paso 4: regla del Taller 1, última actualización, contador y fila de origen.
    ordenados = eventos.assign(_fila_origen=np.arange(len(eventos))).sort_values(
        ["number", "sys_updated_at", "sys_mod_count", "_fila_origen"],
        kind="stable", na_position="first",
    )
    incidentes = ordenados.drop_duplicates("number", keep="last")[list(COLUMNAS)].copy()
    # Paso 5: horas hasta resolución; las negativas quedan sin dato.
    horas = (incidentes["resolved_at"] - incidentes["opened_at"]).dt.total_seconds() / 3600
    negativas = int((horas < 0).sum())
    incidentes[DURACION] = horas.mask(horas < 0)
    incidentes = incidentes.reset_index(drop=True)
    control = {
        "sha256_csv": sha256(contenido).hexdigest(),
        "eventos": int(len(eventos)), "columnas_originales": int(len(eventos.columns)),
        "incidentes": int(len(incidentes)), "columnas_analiticas": int(len(incidentes.columns)),
        "duplicados_exactos_eventos": duplicados,
        "fechas_invalidas": fechas_invalidas,
        "contadores_invalidos": contadores_invalidos,
        "duraciones_negativas": negativas,
        "incidentes_sin_duracion": int(incidentes[DURACION].isna().sum()),
        "resolucion_posterior_actualizacion": int((incidentes["resolved_at"] > incidentes["sys_updated_at"]).sum()),
        "cierre_anterior_resolucion": int((incidentes["closed_at"] < incidentes["resolved_at"]).sum()),
    }
    return DatosIncidentes(eventos, incidentes, control)


def firma_archivo(ruta: Path) -> tuple[str, int, int]:
    """(ruta resuelta, tamaño en bytes, st_mtime_ns) usando solo os.stat; no lee el contenido.

    Lanza OSError si el archivo no existe o no se puede consultar.
    """
    estado = os.stat(ruta)
    return str(Path(ruta).resolve()), int(estado.st_size), int(estado.st_mtime_ns)


def cargar_ruta(ruta: Path) -> DatosIncidentes:
    """Carga el CSV de una ruta comprobando el tamaño antes de leerlo.

    Si st_size supera MAX_BYTES lanza ValueError sin leer el archivo; si no,
    equivale a cargar_csv(ruta.read_bytes()). OSError si la ruta no existe.
    """
    ruta = Path(ruta)
    if ruta.stat().st_size > MAX_BYTES:
        raise ValueError(MENSAJE_TAMANO)
    return cargar_csv(ruta.read_bytes())


def verificar_integridad(control: dict, esperado: str | None) -> dict:
    """Compara el SHA-256 calculado al cargar con el esperado, sin recalcularlo.

    Devuelve {"estado": "coincide" | "difiere" | "no_aplica", "esperado", "calculado"};
    "no_aplica" cuando esperado es None (por ejemplo, un CSV cargado por la persona).
    """
    calculado = control["sha256_csv"]
    if esperado is None:
        estado = "no_aplica"
    elif esperado.strip().lower() == str(calculado).lower():
        estado = "coincide"
    else:
        estado = "difiere"
    return {"estado": estado, "esperado": esperado, "calculado": calculado}
