"""Gráficos pertinentes al tipo de variable; sin operaciones de inferencia."""
import pandas as pd
import plotly.express as px
from pandas.api.types import is_numeric_dtype

from src.eda import DURACION, SIN_DATO


def grafico_prioridades(segmentos: pd.DataFrame):
    return px.bar(segmentos, x="priority", y="incidentes", text="incidentes",
                  labels={"priority": "Prioridad", "incidentes": "Incidentes únicos"},
                  title="Incidentes por prioridad", template="plotly_white")


def grafico_duracion(df: pd.DataFrame):
    return px.histogram(df.dropna(subset=[DURACION]), x=DURACION, nbins=40,
                        labels={DURACION: "Tiempo registrado hasta resolución (h)"},
                        title="Distribución de la duración válida", template="plotly_white")


def grafico_variable(df: pd.DataFrame, variable: str):
    if is_numeric_dtype(df[variable]):
        return px.histogram(df, x=variable, nbins=35, template="plotly_white")
    conteos = df[variable].fillna(SIN_DATO).value_counts().head(10).rename_axis("categoria").reset_index(name="frecuencia")
    return px.bar(conteos, x="categoria", y="frecuencia", title="Diez categorías más frecuentes",
                  template="plotly_white")
