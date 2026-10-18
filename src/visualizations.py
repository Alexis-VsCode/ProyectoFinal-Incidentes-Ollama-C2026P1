"""Catálogo de gráficos descriptivos: Plotly para la interfaz y Seaborn para las figuras estáticas.

Cada función recibe una estructura del contrato de datos, lee solo las columnas que usa, no muta
la entrada y, si no hay datos que dibujar, devuelve una figura con el aviso AVISO_VACIO en lugar
de lanzar. Solo depende de src.diccionario; las figuras Matplotlib se crean con
matplotlib.figure.Figure y nunca se registran en pyplot.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import math
from io import BytesIO
from typing import Final

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pandas.api.types import is_numeric_dtype

from src.diccionario import DURACION, MIN_MUESTRA, PISO_LOG_HORAS, SIN_DATO, TIPOS, etiqueta

PLANTILLA: Final = "plotly_white"
COLOR_PRINCIPAL: Final = "#166B7A"
COLOR_CAUTELA: Final = "#9AA9B3"
AVISO_VACIO: Final = "Sin datos para la selección actual"
TOP_CATEGORIAS: Final = 10
AVISO_DESCRIPTIVO: Final = "Percentiles descriptivos del histórico; no son SLO, SLA ni MTTR."


def _numero_es(valor: float, decimales: int = 1) -> str:
    """Número con punto de miles y coma decimal: 1234.5 -> "1.234,5"."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def _horas_es(valor: float) -> str:
    """Horas con un decimal en formato español: 24.6 -> "24,6"; 1000 -> "1.000,0"."""
    return _numero_es(float(valor), 1)


def _tick_log_es(valor: float) -> str:
    """Rótulo de un tick de potencia de 10 en formato español: 0.01 -> "0,01"; 10000 -> "10.000"."""
    if valor >= 1:
        return _numero_es(valor, 0)
    return f"{valor:g}".replace(".", ",")


def _estilo(fig: go.Figure, titulo: str, subtitulo: str | None = None) -> go.Figure:
    """Aplica plantilla, separadores españoles (",.") y título con subtítulo opcional; devuelve la misma figura.

    El subtítulo puede partirse en líneas con "<br>"; el margen superior crece con cada línea.
    """
    lineas = subtitulo.count("<br>") + 1 if subtitulo else 0
    fig.update_layout(
        template=PLANTILLA, separators=",.",
        title={"text": titulo, "subtitle": {"text": subtitulo or ""}},
        margin={"t": 65 + 25 * lineas if lineas else 60},
    )
    return fig


def _vacia_plotly(titulo: str) -> go.Figure:
    """Figura Plotly sin trazas, con ejes ocultos y la anotación AVISO_VACIO centrada."""
    fig = go.Figure()
    fig.add_annotation(text=AVISO_VACIO, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _estilo(fig, titulo)


def _vacia_matplotlib(titulo: str) -> Figure:
    """Figura Matplotlib con un eje apagado, el título y el texto AVISO_VACIO centrado."""
    fig = Figure(figsize=(8, 4.5))
    eje = fig.add_subplot()
    eje.axis("off")
    eje.set_title(titulo)
    eje.text(0.5, 0.5, AVISO_VACIO, ha="center", va="center", fontsize=13, transform=eje.transAxes)
    return fig


def _horas_para_log(df: pd.DataFrame) -> tuple[pd.Series, int, int]:
    """(horas válidas en float64 con 0 h llevadas a PISO_LOG_HORAS, n con 0 h, n sin dato).

    Válidas son las horas no nulas y no negativas; una hora negativa cuenta como sin dato,
    igual que en la preparación de los datos. La serie conserva el orden de las filas de df.
    """
    horas = df[DURACION].astype("float64")
    horas = horas.mask(horas < 0)
    validas = horas.dropna()
    ceros = validas == 0
    return validas.mask(ceros, PISO_LOG_HORAS), int(ceros.sum()), int(horas.isna().sum())


def _nota_horas(n_validas: int, n_ceros: int, n_sin_dato: int) -> str:
    """Resumen de cobertura de las horas: válidas, ceros dibujados en el piso y sin dato excluidas."""
    return (f"Válidas: {_numero_es(n_validas, 0)} · con 0 h: {_numero_es(n_ceros, 0)} (dibujadas en 1 min)"
            f" · sin dato: {_numero_es(n_sin_dato, 0)} (excluidas)")


def grafico_prioridades(segmentos: pd.DataFrame) -> go.Figure:
    """Barras verticales de incidentes por prioridad en el orden recibido.

    Entrada: salida de comparar_prioridades (priority, incidentes, duraciones_validas,
    media_horas, mediana_horas). Ejes «Prioridad» e «Incidentes únicos».
    """
    titulo = "Incidentes por prioridad"
    if segmentos.empty:
        return _vacia_plotly(titulo)
    # Paso 1: conservar el orden ordinal recibido y llevar la mediana al hover.
    prioridades = segmentos["priority"].astype("string").fillna(SIN_DATO).tolist()
    fig = go.Figure(go.Bar(
        x=prioridades, y=segmentos["incidentes"].tolist(), marker_color=COLOR_PRINCIPAL,
        texttemplate="%{y:,}", textposition="outside",
        customdata=np.column_stack([segmentos["duraciones_validas"].to_numpy(dtype="float64"),
                                    segmentos["mediana_horas"].to_numpy(dtype="float64")]),
        hovertemplate=("%{x}<br>Incidentes únicos: %{y:,}<br>Duraciones válidas: %{customdata[0]:,}"
                       "<br>Mediana: %{customdata[1]:,.1f} h<extra></extra>"),
    ))
    fig.update_xaxes(title_text="Prioridad", type="category", categoryorder="array", categoryarray=prioridades)
    fig.update_yaxes(title_text="Incidentes únicos")
    return _estilo(fig, titulo)


def grafico_duracion(df: pd.DataFrame) -> go.Figure:
    """ECDF de las horas hasta resolución en eje x logarítmico, con mediana y p90 marcadas.

    Los 0 h se dibujan en PISO_LOG_HORAS (1 min), así que la proporción acumulada cubre todas
    las duraciones válidas. El eje y usa formato ".0%" y hay dos líneas horizontales en 0,5 y 0,9
    rotuladas con los percentiles calculados sobre las horas reales válidas. El subtítulo tiene dos
    líneas: AVISO_DESCRIPTIVO (no son SLO, SLA ni MTTR) y la cobertura de ceros y sin dato.
    """
    titulo = "Proporción acumulada de incidentes según horas hasta resolución"
    horas, n_ceros, n_sin_dato = _horas_para_log(df)
    if horas.empty:
        return _vacia_plotly(titulo)
    # Paso 1: percentiles sobre las horas reales válidas (los ceros cuentan como 0 h).
    reales = df[DURACION].astype("float64")
    reales = reales[reales >= 0]
    mediana, p90 = float(reales.median()), float(reales.quantile(0.9))
    # Paso 2: ECDF normalizada en proporción sobre el eje logarítmico.
    fig = px.ecdf(pd.DataFrame({"horas": horas.to_numpy()}), x="horas", log_x=True,
                  ecdfnorm="probability", color_discrete_sequence=[COLOR_PRINCIPAL])
    fig.update_traces(hovertemplate="Horas: %{x:,.2f}<br>Proporción acumulada: %{y:.1%}<extra></extra>")
    # Paso 3: líneas de referencia descriptivas en 0,5 y 0,9.
    for altura, rotulo in ((0.5, f"Mediana: {_horas_es(mediana)} h"), (0.9, f"P90: {_horas_es(p90)} h")):
        fig.add_hline(y=altura, line_dash="dash", line_color=COLOR_CAUTELA, line_width=1,
                      annotation_text=rotulo, annotation_position="top left")
    fig.update_xaxes(title_text="Horas hasta resolución (escala logarítmica)")
    fig.update_yaxes(title_text="Proporción acumulada de incidentes", tickformat=".0%", range=[0, 1.02])
    subtitulo = f"{AVISO_DESCRIPTIVO}<br>{_nota_horas(len(horas), n_ceros, n_sin_dato)}"
    return _estilo(fig, titulo, subtitulo)


def _histograma_log_horas(df: pd.DataFrame, titulo: str) -> go.Figure:
    """Histograma de log10(horas) con ticks en potencias de 10 rotulados en horas."""
    horas, n_ceros, n_sin_dato = _horas_para_log(df)
    if horas.empty:
        return _vacia_plotly(titulo)
    # Paso 1: binning uniforme en log10, diez intervalos por década.
    logaritmos = np.log10(horas.to_numpy())
    desde = min(-2, math.floor(logaritmos.min()))
    hasta = max(4, math.ceil(logaritmos.max()))
    potencias = list(range(desde, hasta + 1))
    fig = go.Figure(go.Histogram(x=logaritmos, xbins={"size": 0.1}, marker_color=COLOR_PRINCIPAL,
                                 hovertemplate="Incidentes: %{y:,}<extra></extra>"))
    # Paso 2: los ticks se rotulan en horas, no en logaritmos.
    fig.update_xaxes(title_text="Horas hasta resolución (escala logarítmica)", tickvals=potencias,
                     ticktext=[_tick_log_es(10.0 ** p) for p in potencias])
    fig.update_yaxes(title_text="Incidentes")
    return _estilo(fig, titulo, _nota_horas(len(horas), n_ceros, n_sin_dato))


def _barras_por_valor(serie: pd.Series, titulo: str, nombre: str) -> go.Figure:
    """Barras de frecuencia por valor de un conteo discreto, en orden ascendente de valor."""
    validos = serie.dropna().astype("float64")
    if validos.empty:
        return _vacia_plotly(titulo)
    conteos = validos.value_counts().sort_index()
    valores = [int(v) if float(v).is_integer() else float(v) for v in conteos.index]
    fig = go.Figure(go.Bar(x=valores, y=conteos.tolist(), marker_color=COLOR_PRINCIPAL,
                           hovertemplate=f"{nombre}: %{{x}}<br>Incidentes: %{{y:,}}<extra></extra>"))
    fig.update_xaxes(title_text=nombre)
    fig.update_yaxes(title_text="Incidentes")
    return _estilo(fig, titulo, f"Una barra por valor observado · sin dato: {_numero_es(int(serie.isna().sum()), 0)}")


def _barras_categoricas(serie: pd.Series, titulo: str, nombre: str, ordinal: bool = False) -> go.Figure:
    """Barras horizontales de las TOP_CATEGORIAS más frecuentes; los nulos cuentan como SIN_DATO.

    Con ordinal=False las barras van por frecuencia descendente (desempate alfabético). Con
    ordinal=True las mismas categorías van en el orden de sus etiquetas, con SIN_DATO al final.
    """
    # Paso 1: frecuencia descendente con desempate alfabético estable.
    conteos = serie.astype("string").fillna(SIN_DATO).value_counts().sort_index()
    conteos = conteos.sort_values(ascending=False, kind="stable")
    total = len(conteos)
    visibles = conteos.head(TOP_CATEGORIAS)
    # Paso 2: una variable ordinal conserva el orden de sus etiquetas.
    if ordinal:
        orden = sorted(c for c in visibles.index if c != SIN_DATO)
        visibles = visibles.reindex(orden + [c for c in visibles.index if c == SIN_DATO])
    fig = go.Figure(go.Bar(x=visibles.tolist(), y=[str(c) for c in visibles.index], orientation="h",
                           marker_color=COLOR_PRINCIPAL,
                           hovertemplate="%{y}<br>Incidentes: %{x:,}<extra></extra>"))
    # Paso 3: la primera categoría queda arriba.
    fig.update_yaxes(title_text=nombre, autorange="reversed", type="category")
    fig.update_xaxes(title_text="Incidentes")
    subtitulo = f"{total} categorías en total"
    if total > TOP_CATEGORIAS:
        subtitulo += f"; se muestran las {TOP_CATEGORIAS} más frecuentes"
    return _estilo(fig, titulo, subtitulo)


def grafico_variable(df: pd.DataFrame, variable: str) -> go.Figure:
    """Gráfico pertinente al tipo de la variable.

    - DURACION: histograma de log10(horas) con ticks rotulados en horas («0,01» a «10.000»).
    - Numérica (conteos discretos): barras por valor observado.
    - Resto (categóricas): barras horizontales de las 10 más frecuentes y subtítulo
      «k categorías en total». Si TIPOS marca la variable como «categórica ordinal», esas
      barras siguen el orden de las etiquetas (SIN_DATO al final) en lugar de la frecuencia.
    ValueError si variable no es una columna de df.
    """
    if variable not in df.columns:
        raise ValueError(f"La variable {variable!r} no es una columna de los datos.")
    nombre = etiqueta(variable)
    titulo = f"Distribución de {nombre.lower()}"
    if df.empty:
        return _vacia_plotly(titulo)
    if variable == DURACION:
        return _histograma_log_horas(df, titulo)
    if is_numeric_dtype(df[variable]):
        return _barras_por_valor(df[variable], titulo, nombre)
    return _barras_categoricas(df[variable], titulo, nombre, TIPOS.get(variable) == "categórica ordinal")


def grafico_segmentos(segmentos: pd.DataFrame, variable: str) -> go.Figure:
    """Barras horizontales de una traza con la mediana de horas por segmento, en el orden de la tabla.

    Entrada: salida de segmentar (COLUMNAS_SEGMENTO). Las barras con muestra_suficiente False
    van en COLOR_CAUTELA y el subtítulo cita el umbral MIN_MUESTRA; el hover muestra incidentes,
    p90 y porcentaje sin duración. Sin filas, o si ningún segmento tiene mediana, devuelve el
    aviso AVISO_VACIO.
    """
    titulo = f"Mediana de horas hasta resolución por {etiqueta(variable).lower()}"
    if segmentos.empty or segmentos["mediana_horas"].isna().all():
        return _vacia_plotly(titulo)
    # Paso 1: color por suficiencia de la muestra y datos de hover por fila.
    nombres = [str(s) for s in segmentos["segmento"]]
    colores = [COLOR_PRINCIPAL if bool(ok) else COLOR_CAUTELA for ok in segmentos["muestra_suficiente"]]
    extra = np.column_stack([segmentos[c].to_numpy(dtype="float64")
                             for c in ("incidentes", "duraciones_validas", "p90_horas", "porcentaje_sin_duracion")])
    fig = go.Figure(go.Bar(
        x=segmentos["mediana_horas"].to_numpy(dtype="float64"), y=nombres, orientation="h",
        marker={"color": colores}, customdata=extra,
        hovertemplate=("%{y}<br>Mediana: %{x:,.1f} h<br>Incidentes: %{customdata[0]:,}"
                       "<br>Duraciones válidas: %{customdata[1]:,}<br>P90: %{customdata[2]:,.1f} h"
                       "<br>Sin duración: %{customdata[3]:.1f} %<extra></extra>"),
    ))
    # Paso 2: el primer segmento de la tabla queda arriba.
    fig.update_yaxes(title_text=etiqueta(variable), type="category", autorange="reversed")
    fig.update_xaxes(title_text="Mediana de horas hasta resolución")
    return _estilo(fig, titulo, f"En gris: menos de {MIN_MUESTRA} duraciones válidas (muestra insuficiente)."
                                "<br>Percentiles descriptivos del histórico.")


def figura_boxplot_prioridad(df: pd.DataFrame) -> Figure:
    """Boxplot Seaborn de horas por prioridad con eje y logarítmico.

    Orden: prioridades ascendentes y SIN_DATO al final; cada tick dice «prioridad\\n(n=k)» con k
    duraciones válidas. La nota al pie aclara que n cuenta duraciones válidas, informa los sin dato
    excluidos y, si hay 0 h, que se dibujan en PISO_LOG_HORAS. Alinea filas por posición, así que
    admite índices repetidos.
    """
    titulo = "Horas hasta resolución por prioridad"
    horas, n_ceros, n_sin_dato = _horas_para_log(df)
    if horas.empty:
        return _vacia_matplotlib(titulo)
    # Paso 1: tabla mínima con prioridad rotulada y horas ya llevadas al piso logarítmico.
    con_horas = (df[DURACION].astype("float64") >= 0).to_numpy()
    prioridad = df["priority"].astype("string").fillna(SIN_DATO).astype(str).to_numpy()[con_horas]
    datos = pd.DataFrame({"prioridad": prioridad, "horas": horas.to_numpy()})
    orden = sorted(p for p in datos["prioridad"].unique() if p != SIN_DATO)
    orden += [SIN_DATO] if (datos["prioridad"] == SIN_DATO).any() else []
    n_por_prioridad = datos["prioridad"].value_counts()
    # Paso 2: boxplot sobre una Figure propia, sin pyplot.
    with sns.axes_style("whitegrid"):
        fig = Figure(figsize=(8, 4.8))
        eje = fig.add_subplot()
    sns.boxplot(data=datos, x="prioridad", y="horas", order=orden, log_scale=True,
                color=COLOR_PRINCIPAL, fliersize=2, ax=eje)
    eje.set_xticks(range(len(orden)))
    eje.set_xticklabels([f"{p}\n(n={_numero_es(int(n_por_prioridad[p]), 0)})" for p in orden])
    eje.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _tick_log_es(v)))
    eje.set_xlabel("Prioridad")
    eje.set_ylabel("Horas hasta resolución (escala logarítmica)")
    eje.set_title(titulo)
    # Paso 3: nota al pie sobre el significado de n, los sin dato y los ceros.
    lineas = [f"n = duraciones válidas por prioridad; sin dato excluidos: {_numero_es(n_sin_dato, 0)}."]
    if n_ceros:
        lineas.append(f"{_numero_es(n_ceros, 0)} incidentes con 0 h se dibujan en 1 min (piso del eje logarítmico).")
    fig.text(0.01, 0.01, "\n".join(lineas), fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03 * (len(lineas) + 1), 1, 1))
    return fig


def figura_mapa_spearman(matriz: pd.DataFrame) -> Figure:
    """Mapa de calor Seaborn del triángulo inferior (con diagonal) de la matriz de Spearman.

    Escala fija vmin=-1, vmax=1, centro 0 y cmap RdBu_r; anotaciones con coma decimal y barra de
    color «ρ de Spearman». Ejes rotulados con las etiquetas del diccionario.
    """
    titulo = "Asociación monótona (ρ de Spearman)"
    if matriz.empty or matriz.isna().all().all():
        return _vacia_matplotlib(titulo)
    # Paso 1: máscara del triángulo superior estricto y anotaciones en formato español.
    valores = matriz.astype("float64")
    mascara = np.triu(np.ones(valores.shape, dtype=bool), k=1)
    anotaciones = valores.map(lambda v: "" if pd.isna(v) else _numero_es(v, 2))
    fig = Figure(figsize=(7, 5.6))
    eje = fig.add_subplot()
    sns.heatmap(valores, mask=mascara, annot=anotaciones.to_numpy(), fmt="", vmin=-1, vmax=1, center=0,
                cmap="RdBu_r", square=True, linewidths=0.5, ax=eje,
                xticklabels=[etiqueta(c) for c in valores.columns],
                yticklabels=[etiqueta(i) for i in valores.index],
                cbar_kws={"label": "ρ de Spearman"})
    eje.set_title(titulo)
    # Paso 2: rótulos legibles y barra de color con coma decimal.
    eje.tick_params(axis="y", labelrotation=0)
    for rotulo in eje.get_xticklabels():
        rotulo.set_rotation(30)
        rotulo.set_horizontalalignment("right")
    eje.collections[0].colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _numero_es(v, 2)))
    fig.tight_layout()
    return fig


def figura_a_png(figura: Figure, dpi: int = 150) -> bytes:
    """Serializa una Figure de Matplotlib a bytes PNG, sin tocar pyplot."""
    salida = BytesIO()
    figura.savefig(salida, format="png", dpi=dpi, bbox_inches="tight")
    return salida.getvalue()
