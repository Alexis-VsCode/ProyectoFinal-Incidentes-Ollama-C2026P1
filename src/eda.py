"""Funciones de análisis descriptivo y fachada AnalisisIncidentes; no depende de Streamlit ni de Ollama.

Las horas hasta resolución son estadísticos descriptivos del histórico: no son
SLO, SLA ni MTTR. Las funciones de módulo son puras: no modifican el DataFrame
que reciben.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import math
from numbers import Integral
from typing import Any, Final

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from src.diccionario import (
    DURACION, MIN_MUESTRA, OTROS, SIN_DATO, VARIABLES_RELACION, VARIABLES_SEGMENTO, tabla_diccionario,
)

VARIABLES_SEGMENTABLES: Final = VARIABLES_SEGMENTO
COLUMNAS_PRIORIDADES: Final = (
    "priority", "incidentes", "porcentaje_incidentes", "duraciones_validas", "media_horas", "mediana_horas",
)
COLUMNAS_SEGMENTO: Final = (
    "segmento", "incidentes", "porcentaje_incidentes", "duraciones_validas",
    "mediana_horas", "p90_horas", "porcentaje_sin_duracion", "muestra_suficiente",
)
TIPOS_SEGMENTO: Final = {
    "segmento": "object", "incidentes": "int64", "porcentaje_incidentes": "float64",
    "duraciones_validas": "int64", "mediana_horas": "float64", "p90_horas": "float64",
    "porcentaje_sin_duracion": "float64", "muestra_suficiente": "bool",
}


def numero(valor: Any) -> float | None:
    """Convierte a float; devuelve None si el valor es nulo o no finito."""
    return float(valor) if pd.notna(valor) and math.isfinite(float(valor)) else None


def perfil_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por columna: variable, tipo, nulos, porcentaje_nulos (redondeado a 3) y valores_unicos."""
    n = len(df)
    return pd.DataFrame({
        "variable": df.columns, "tipo": df.dtypes.astype(str).to_numpy(),
        "nulos": df.isna().sum().to_numpy(),
        "porcentaje_nulos": (df.isna().mean().to_numpy() * 100).round(3) if n else 0,
        "valores_unicos": df.nunique().to_numpy(),
    })


def indicadores(df: pd.DataFrame) -> dict:
    """Totales de la selección y media, mediana y p90 de las horas válidas (None si no hay)."""
    s = df[DURACION].dropna()
    return {
        "incidentes_seleccionados": int(len(df)), "duraciones_validas": int(len(s)),
        "incidentes_sin_duracion": int(df[DURACION].isna().sum()),
        "media_horas": numero(s.mean()), "mediana_horas": numero(s.median()),
        "p90_horas": numero(s.quantile(.9)) if len(s) else None,
    }


def comparar_prioridades(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por prioridad (nulos como SIN_DATO), ordenada por etiqueta, con COLUMNAS_PRIORIDADES.

    porcentaje_incidentes va de 0 a 100 sobre len(df), sin redondear.
    """
    tabla = df.assign(priority=df["priority"].fillna(SIN_DATO)).groupby("priority").agg(
        incidentes=("number", "size"), duraciones_validas=(DURACION, "count"),
        media_horas=(DURACION, "mean"), mediana_horas=(DURACION, "median"),
    ).reset_index().sort_values("priority").reset_index(drop=True)
    tabla["porcentaje_incidentes"] = 100 * tabla["incidentes"] / max(len(df), 1)
    return tabla[list(COLUMNAS_PRIORIDADES)]


def segmentar(df: pd.DataFrame, variable: str, top: int = 10) -> pd.DataFrame:
    """Ranking de segmentos de `variable` con conteos, mediana y p90 de horas.

    Contrato:
    - variable debe estar en VARIABLES_SEGMENTO y top debe ser un entero
      (no bool) mayor o igual que 1; si no, ValueError.
    - Columnas exactas COLUMNAS_SEGMENTO con los tipos de TIPOS_SEGMENTO; los
      porcentajes van de 0 a 100 sin redondear y muestra_suficiente es
      duraciones_validas >= MIN_MUESTRA.
    - Los nulos cuentan como SIN_DATO. Orden: incidentes descendente y, en
      empate, segmento ascendente.
    - Tras las `top` primeras filas se agrega una sola fila OTROS si sobran
      segmentos; su mediana y p90 salen de las horas agrupadas del resto.
    - La suma de incidentes es len(df). Con df vacío devuelve una tabla vacía
      con las mismas columnas.
    - attrs["segmentos_totales"] guarda el número de segmentos distintos antes
      de agrupar el resto.
    """
    if variable not in VARIABLES_SEGMENTO:
        raise ValueError(f"Variable no segmentable: {variable}. Use una de {list(VARIABLES_SEGMENTO)}.")
    if isinstance(top, bool) or not isinstance(top, Integral) or top < 1:
        raise ValueError("El número de segmentos (top) debe ser un entero de 1 o más.")
    # Paso 1: etiqueta de segmento con los nulos como SIN_DATO y horas en float64.
    base = pd.DataFrame({
        "segmento": df[variable].astype("string").fillna(SIN_DATO).to_numpy(dtype=object),
        "horas": df[DURACION].astype("float64").to_numpy(),
    })
    # Paso 2: estadísticos por segmento y orden por volumen con desempate por nombre.
    grupos = base.groupby("segmento", sort=False)["horas"]
    resumen = pd.DataFrame({
        "incidentes": grupos.size(), "duraciones_validas": grupos.count(),
        "mediana_horas": grupos.median(), "p90_horas": grupos.quantile(0.9),
    }).rename_axis("segmento").reset_index()
    resumen = resumen.sort_values(["incidentes", "segmento"], ascending=[False, True], kind="stable")
    principales = resumen.head(top)
    registros = list(principales.itertuples(index=False, name=None))
    # Paso 3: una sola fila OTROS con las horas agrupadas del resto, si sobran segmentos.
    resto = base.loc[~base["segmento"].isin(principales["segmento"]), "horas"]
    if len(resto):
        validas = resto.dropna()
        registros.append((OTROS, len(resto), len(validas), validas.median(), validas.quantile(0.9)))
    # Paso 4: porcentajes, bandera de muestra y tipos del contrato.
    salida = pd.DataFrame(registros, columns=["segmento", "incidentes", "duraciones_validas", "mediana_horas", "p90_horas"])
    salida["porcentaje_incidentes"] = 100 * salida["incidentes"] / max(len(df), 1)
    salida["porcentaje_sin_duracion"] = 100 * (salida["incidentes"] - salida["duraciones_validas"]) / salida["incidentes"]
    salida["muestra_suficiente"] = salida["duraciones_validas"] >= MIN_MUESTRA
    salida = salida[list(COLUMNAS_SEGMENTO)].astype(TIPOS_SEGMENTO).reset_index(drop=True)
    salida.attrs["segmentos_totales"] = int(len(resumen))
    return salida


def spearman(x: pd.Series, y: pd.Series) -> float | None:
    """Correlación de Spearman como Pearson sobre rangos promedio, sin scipy.

    Recibe series alineadas y sin nulos. Devuelve None con menos de 3 pares, si
    alguna serie es constante o si el resultado no es finito.
    """
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return None
    return numero(x.rank().corr(y.rank()))


def _pearson(x: pd.Series, y: pd.Series) -> float | None:
    """Pearson con las mismas condiciones de definición que spearman."""
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return None
    return numero(x.corr(y, method="pearson"))


def correlaciones(df: pd.DataFrame) -> list[dict]:
    """Pearson y Spearman de cada contador de VARIABLES_RELACION con las horas, en ese orden.

    Cada elemento: {"variable", "pearson_con_duracion", "spearman_con_duracion",
    "pares_validos"}; los pares son las filas con ambos valores no nulos.
    Describe asociación, no causa.
    """
    resultados = []
    for variable in VARIABLES_RELACION:
        pares = df[[variable, DURACION]].dropna().astype("float64")
        x, y = pares[variable], pares[DURACION]
        resultados.append({
            "variable": variable, "pearson_con_duracion": _pearson(x, y),
            "spearman_con_duracion": spearman(x, y), "pares_validos": int(len(pares)),
        })
    return resultados


def matriz_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz 4x4 float64 de Spearman con índice y columnas [*VARIABLES_RELACION, DURACION].

    Usa pares completos por cada par de columnas y al menos 3 observaciones;
    queda NaN donde la relación no está definida.
    """
    columnas = [*VARIABLES_RELACION, DURACION]
    return df[columnas].astype("float64").corr(method="spearman", min_periods=3)


def diccionario_con_calidad(incidentes: pd.DataFrame) -> pd.DataFrame:
    """tabla_diccionario() unida por la izquierda con perfil_calidad(incidentes), en el orden del diccionario."""
    return tabla_diccionario().merge(perfil_calidad(incidentes), on="variable", how="left")


class AnalizadorVariable(ABC):
    """Strategy: contrato común para describir una sola variable de la selección."""
    @abstractmethod
    def analizar(self, serie: pd.Series) -> pd.DataFrame:
        """Tabla descriptiva de la serie; no la modifica."""
        raise NotImplementedError


class AnalizadorNumerico(AnalizadorVariable):
    """Estadísticos descriptivos de una variable numérica, ignorando nulos."""
    def analizar(self, serie: pd.Series) -> pd.DataFrame:
        """Columnas estadistico y valor: validos, nulos, media, mediana, desviacion_muestral,
        minimo y maximo; los estadísticos no definidos valen None."""
        s = serie.dropna()
        return pd.DataFrame({
            "estadistico": ["validos", "nulos", "media", "mediana", "desviacion_muestral", "minimo", "maximo"],
            "valor": [len(s), int(serie.isna().sum()), numero(s.mean()), numero(s.median()),
                      numero(s.std(ddof=1)), numero(s.min()), numero(s.max())],
        })


class AnalizadorCategorico(AnalizadorVariable):
    """Frecuencias de una variable categórica, con los nulos como SIN_DATO."""
    def analizar(self, serie: pd.Series) -> pd.DataFrame:
        """Columnas categoria, frecuencia y porcentaje_sobre_total (0 a 100), de mayor a menor frecuencia."""
        salida =serie.fillna(SIN_DATO).value_counts().rename_axis("categoria").reset_index(name="frecuencia")
        salida["porcentaje_sobre_total"] = 100 * salida["frecuencia"] / len(serie) if len(serie) else 0
        return salida


class AnalizadorFactory:
    """Factory: elige la Strategy según el tipo de la serie."""
    @staticmethod
    def crear(serie: pd.Series) -> AnalizadorVariable:
        """AnalizadorNumerico si la serie es numérica y no booleana; AnalizadorCategorico en otro caso."""
        if is_numeric_dtype(serie) and not is_bool_dtype(serie):
            return AnalizadorNumerico()
        return AnalizadorCategorico()


class AnalisisIncidentes:
    """Facade: la UI y el agente piden resultados sobre la selección sin repetir los cálculos."""
    def __init__(self, incidentes: pd.DataFrame, control: dict, prioridades: list[str]) -> None:
        """Guarda una copia de los incidentes cuyas prioridades (nulos como SIN_DATO) están en `prioridades`."""
        self.prioridades = list(prioridades)
        mascara = incidentes["priority"].fillna(SIN_DATO).isin(prioridades)
        self.datos = incidentes.loc[mascara].copy()
        self.control = dict(control)

    def resumen(self) -> dict:
        """indicadores(self.datos)."""
        return indicadores(self.datos)

    def prioridades_resumen(self) -> pd.DataFrame:
        """comparar_prioridades(self.datos)."""
        return comparar_prioridades(self.datos)

    def explorar(self, variable: str) -> pd.DataFrame:
        """Tabla de la Strategy que AnalizadorFactory elige para la variable; KeyError si no existe."""
        return AnalizadorFactory.crear(self.datos[variable]).analizar(self.datos[variable])

    def segmentar(self, variable: str, top: int = 5) -> pd.DataFrame:
        """segmentar(self.datos, variable, top): ValueError si la variable no es segmentable o top < 1."""
        return segmentar(self.datos, variable, top)

    def relaciones(self) -> list[dict]:
        """correlaciones(self.datos)."""
        return correlaciones(self.datos)

    def matriz_relaciones(self) -> pd.DataFrame:
        """matriz_spearman(self.datos)."""
        return matriz_spearman(self.datos)

    def estrategia(self, variable: str) -> str:
        """Nombre de la clase Strategy que AnalizadorFactory elige para la variable."""
        return type(AnalizadorFactory.crear(self.datos[variable])).__name__

    def contexto(self, herramienta: str) -> dict:
        """Compatibilidad: la fuente única es src.herramientas.REGISTRO."""
        from src.herramientas import contexto_herramienta  # import diferido para evitar el ciclo
        return contexto_herramienta(self, herramienta)
