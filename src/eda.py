"""Funciones de análisis y fachada; no depende de Streamlit ni de Ollama."""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import math

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

DURACION = "tiempo_resolucion_horas"
SIN_DATO = "(sin dato)"


def numero(valor) -> float | None:
    return float(valor) if pd.notna(valor) and math.isfinite(float(valor)) else None


def perfil_calidad(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame({
        "variable": df.columns, "tipo": df.dtypes.astype(str).to_numpy(),
        "nulos": df.isna().sum().to_numpy(),
        "porcentaje_nulos": (df.isna().mean().to_numpy() * 100).round(3) if n else 0,
        "valores_unicos": df.nunique().to_numpy(),
    })


def indicadores(df: pd.DataFrame) -> dict:
    s = df[DURACION].dropna()
    return {
        "incidentes_seleccionados": int(len(df)), "duraciones_validas": int(len(s)),
        "incidentes_sin_duracion": int(df[DURACION].isna().sum()),
        "media_horas": numero(s.mean()), "mediana_horas": numero(s.median()),
        "p90_horas": numero(s.quantile(.9)) if len(s) else None,
    }


def comparar_prioridades(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(priority=df["priority"].fillna(SIN_DATO)).groupby("priority").agg(
        incidentes=("number", "size"), duraciones_validas=(DURACION, "count"),
        media_horas=(DURACION, "mean"), mediana_horas=(DURACION, "median"),
    ).reset_index().sort_values("priority").reset_index(drop=True)


def correlaciones(df: pd.DataFrame) -> list[dict]:
    resultados = []
    for columna in ("reassignment_count", "reopen_count"):
        pares = df[[columna, DURACION]].dropna()
        r = None
        if len(pares) >= 3 and pares.nunique().min() > 1:
            r = numero(pares[columna].corr(pares[DURACION], method="pearson"))
        resultados.append({"variable": columna, "pearson_con_duracion": r, "pares_validos": len(pares)})
    return resultados


class AnalizadorVariable(ABC):
    """Strategy: contrato común heredado del trabajo con POO."""
    @abstractmethod
    def analizar(self, serie: pd.Series) -> pd.DataFrame:
        raise NotImplementedError


class AnalizadorNumerico(AnalizadorVariable):
    def analizar(self, serie: pd.Series) -> pd.DataFrame:
        s = serie.dropna()
        return pd.DataFrame({
            "estadistico": ["validos", "nulos", "media", "mediana", "desviacion_muestral", "minimo", "maximo"],
            "valor": [len(s), int(serie.isna().sum()), numero(s.mean()), numero(s.median()),
                      numero(s.std(ddof=1)), numero(s.min()), numero(s.max())],
        })


class AnalizadorCategorico(AnalizadorVariable):
    def analizar(self, serie: pd.Series) -> pd.DataFrame:
        salida = serie.fillna(SIN_DATO).value_counts().rename_axis("categoria").reset_index(name="frecuencia")
        salida["porcentaje_sobre_total"] = 100 * salida["frecuencia"] / len(serie) if len(serie) else 0
        return salida


class AnalizadorFactory:
    @staticmethod
    def crear(serie: pd.Series) -> AnalizadorVariable:
        if is_numeric_dtype(serie) and not is_bool_dtype(serie):
            return AnalizadorNumerico()
        return AnalizadorCategorico()


class AnalisisIncidentes:
    """Facade: la UI y el agente piden resultados sin repetir los cálculos."""
    def __init__(self, incidentes: pd.DataFrame, control: dict, prioridades: list[str]):
        self.prioridades = list(prioridades)
        mascara = incidentes["priority"].fillna(SIN_DATO).isin(prioridades)
        self.datos = incidentes.loc[mascara].copy()
        self.control = dict(control)

    def resumen(self) -> dict:
        return indicadores(self.datos)

    def prioridades_resumen(self) -> pd.DataFrame:
        return comparar_prioridades(self.datos)

    def explorar(self, variable: str) -> pd.DataFrame:
        return AnalizadorFactory.crear(self.datos[variable]).analizar(self.datos[variable])

    def contexto(self, herramienta: str) -> dict:
        contexto = {
            "unidad_analisis": "Una fila por incidente; última actualización disponible",
            "origen": "CSV histórico de Incident Response Log; no son datos de TEUNO ni de ISSPOL",
            "control_archivo_completo": self.control,
            "filtro_prioridades": self.prioridades,
            "resumen_seleccion": self.resumen(),
            "limites": "Duración registrada no equivale a tiempo de caída ni prueba causalidad. No hay logs de aplicación ni despliegues.",
        }
        # Lista cerrada de herramientas: el LLM no ejecuta código libre.
        if herramienta == "prioridades":
            contexto["comparacion_prioridades"] = json.loads(self.prioridades_resumen().to_json(orient="records"))
        elif herramienta == "calidad":
            contexto["calidad_incidentes_seleccionados"] = perfil_calidad(self.datos).to_dict("records")
        elif herramienta == "relaciones":
            contexto["correlaciones_descriptivas"] = correlaciones(self.datos)
        elif herramienta != "resumen":
            raise ValueError("Herramienta no autorizada.")
        return contexto
