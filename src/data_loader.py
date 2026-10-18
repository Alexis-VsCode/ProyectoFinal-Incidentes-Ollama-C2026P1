"""Carga del CSV de Taller 1 y consolidación determinista por incidente."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO

import numpy as np
import pandas as pd

MAX_BYTES = 50 * 1024 * 1024
MAX_FILAS = 300_000
FECHAS = ("opened_at", "sys_created_at", "sys_updated_at", "resolved_at", "closed_at")
CONTADORES = ("sys_mod_count", "reassignment_count", "reopen_count")
COLUMNAS = (
    "number", "priority", "incident_state", "assignment_group", "category",
    "opened_at", "resolved_at", "closed_at", "sys_updated_at",
    "reassignment_count", "reopen_count", "sys_mod_count",
)


@dataclass
class DatosIncidentes:
    eventos: pd.DataFrame
    incidentes: pd.DataFrame
    control: dict


def cargar_csv(contenido: bytes) -> DatosIncidentes:
    """Lee bytes UTF-8; '?' significa desconocido y nunca se imputa."""
    if not contenido or len(contenido) > MAX_BYTES:
        raise ValueError("Seleccione un CSV no vacío de hasta 50 MiB.")
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
    fechas_invalidas = {}
    for columna in FECHAS:
        if columna not in eventos:
            continue
        original = eventos[columna]
        convertida = pd.to_datetime(original, format="%d/%m/%Y %H:%M", errors="coerce")
        fechas_invalidas[columna] = int((original.notna() & convertida.isna()).sum())
        eventos[columna] = convertida
    contadores_invalidos = {}
    for columna in CONTADORES:
        original = eventos[columna]
        numeros = pd.to_numeric(original, errors="coerce")
        invalidos = numeros.isna() | ~np.isfinite(numeros) | (numeros < 0) | (numeros % 1 != 0)
        contadores_invalidos[columna] = int((original.notna() & invalidos).sum())
        eventos[columna] = numeros.mask(invalidos).astype("Float64")
    # Es la regla del Taller 1: última actualización, contador y fila de origen.
    ordenados = eventos.assign(_fila_origen=np.arange(len(eventos))).sort_values(
        ["number", "sys_updated_at", "sys_mod_count", "_fila_origen"],
        kind="stable", na_position="first",
    )
    incidentes = ordenados.drop_duplicates("number", keep="last")[list(COLUMNAS)].copy()
    horas = (incidentes["resolved_at"] - incidentes["opened_at"]).dt.total_seconds() / 3600
    negativas = int((horas < 0).sum())
    incidentes["tiempo_resolucion_horas"] = horas.mask(horas < 0)
    incidentes = incidentes.reset_index(drop=True)
    control = {
        "sha256_csv": sha256(contenido).hexdigest(),
        "eventos": int(len(eventos)), "columnas_originales": int(len(eventos.columns)),
        "incidentes": int(len(incidentes)), "columnas_analiticas": int(len(incidentes.columns)),
        "duplicados_exactos_eventos": duplicados,
        "fechas_invalidas": fechas_invalidas,
        "contadores_invalidos": contadores_invalidos,
        "duraciones_negativas": negativas,
        "incidentes_sin_duracion": int(incidentes["tiempo_resolucion_horas"].isna().sum()),
        "resolucion_posterior_actualizacion": int((incidentes["resolved_at"] > incidentes["sys_updated_at"]).sum()),
        "cierre_anterior_resolucion": int((incidentes["closed_at"] < incidentes["resolved_at"]).sum()),
    }
    return DatosIncidentes(eventos, incidentes, control)
