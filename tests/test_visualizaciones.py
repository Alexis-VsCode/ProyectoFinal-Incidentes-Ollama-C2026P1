"""Pruebas del catalogo de graficos Plotly y de las figuras Seaborn sobre fixtures construidas a mano.

Las fixtures siguen los dtypes del contrato de la tabla de incidentes: texto en `string`,
contadores en `Float64` nullable y horas en `float64`, con al menos un 0 h y un NaN.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from src.diccionario import DURACION, ETIQUETAS, OTROS, PISO_LOG_HORAS, SIN_DATO, VARIABLES_RELACION
from src import visualizations as vis

COLUMNAS_SEGMENTO = ("segmento", "incidentes", "porcentaje_incidentes", "duraciones_validas",
                     "mediana_horas", "p90_horas", "porcentaje_sin_duracion", "muestra_suficiente")
COLUMNAS_PRIORIDADES = ("priority", "incidentes", "duraciones_validas", "media_horas", "mediana_horas")
ORDEN_PRIORIDADES = ("1 - Critical", "2 - High", "3 - Moderate", "4 - Low", SIN_DATO)


@pytest.fixture
def incidentes() -> pd.DataFrame:
    """Nueve incidentes con un 0 h (INC07), un NaN de horas (INC05) y una prioridad nula (INC09)."""
    fechas = pd.to_datetime(["2016-01-05 08:00", "2016-01-10 08:00", "2016-01-15 08:00",
                             "2016-01-20 08:00", "2016-03-03 08:00", "2016-03-04 08:00",
                             "2016-03-05 08:00", "2016-03-06 08:00", "2016-03-07 08:00"])
    horas = [2.0, 4.0, 8.0, 24.0, np.nan, 48.0, 0.0, 96.0, 12.0]
    resueltas = pd.Series([f + pd.Timedelta(hours=h) if not math.isnan(h) else pd.NaT
                           for f, h in zip(fechas, horas)], dtype="datetime64[ns]")
    return pd.DataFrame({
        "number": pd.array([f"INC0{i}" for i in range(1, 10)], dtype="string"),
        "priority": pd.array(["1 - Critical", "1 - Critical", "2 - High", "3 - Moderate", "3 - Moderate",
                              "3 - Moderate", "4 - Low", "4 - Low", None], dtype="string"),
        "incident_state": pd.array(["Closed"] * 4 + ["Active"] + ["Closed"] * 4, dtype="string"),
        "assignment_group": pd.array(["G1", "G1", "G2", "G2", "G3", "G3", None, "G4", "G4"], dtype="string"),
        "category": pd.array(["C1", "C1", "C1", "C2", "C2", "C3", "C3", None, "C1"], dtype="string"),
        "opened_at": pd.Series(fechas, dtype="datetime64[ns]"),
        "resolved_at": resueltas,
        "closed_at": resueltas,
        "sys_updated_at": resueltas.fillna(pd.Timestamp("2016-03-03 09:00")),
        "reassignment_count": pd.array([0, 1, 2, 3, 4, 5, 0, 6, 2], dtype="Float64"),
        "reopen_count": pd.array([0, 0, 0, 1, 0, 1, 0, 2, None], dtype="Float64"),
        "sys_mod_count": pd.array([2, 4, 6, 8, 10, 12, 1, 14, 5], dtype="Float64"),
        DURACION: pd.Series(horas, dtype="float64"),
    })


@pytest.fixture
def prioridades() -> pd.DataFrame:
    """Forma de salida de `comparar_prioridades`."""
    return pd.DataFrame({
        "priority": ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"],
        "incidentes": [220, 337, 19554, 658],
        "duraciones_validas": [220, 330, 18226, 568],
        "media_horas": [150.2, 120.4, 190.1, 175.0],
        "mediana_horas": [78.17, 48.32, 24.29, 22.85],
    })


@pytest.fixture
def segmentos() -> pd.DataFrame:
    """Forma de salida de `segmentar`, con una fila de cautela y la fila del resto."""
    return pd.DataFrame({
        "segmento": ["C1", "C2", "C3", OTROS],
        "incidentes": pd.array([40, 35, 5, 3], dtype="int64"),
        "porcentaje_incidentes": [48.19, 42.17, 6.02, 3.61],
        "duraciones_validas": pd.array([38, 31, 4, 3], dtype="int64"),
        "mediana_horas": [10.5, 20.25, 30.0, np.nan],
        "p90_horas": [50.0, 80.0, 60.0, np.nan],
        "porcentaje_sin_duracion": [5.0, 11.43, 20.0, 0.0],
        "muestra_suficiente": [True, True, False, False],
    }).loc[:, list(COLUMNAS_SEGMENTO)]


@pytest.fixture
def matriz() -> pd.DataFrame:
    """Matriz de Spearman 4x4 simetrica, con diagonal 1 y el indice del contrato."""
    nombres = [*VARIABLES_RELACION, DURACION]
    valores = np.array([[1.0, 0.65, 0.31, 0.92],
                        [0.65, 1.0, 0.12, 0.55],
                        [0.31, 0.12, 1.0, 0.27],
                        [0.92, 0.55, 0.27, 1.0]])
    return pd.DataFrame(valores, index=nombres, columns=nombres, dtype="float64")


def _anotaciones(fig: go.Figure) -> list[str]:
    return [a.text for a in fig.layout.annotations]


def _subtitulo(fig: go.Figure) -> str:
    return fig.layout.title.subtitle.text or ""


def test_grafico_prioridades_barras_con_etiquetas_es(prioridades):
    fig = vis.grafico_prioridades(prioridades)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 and fig.data[0].type == "bar"
    assert list(fig.data[0].x) == list(prioridades["priority"])
    assert list(fig.data[0].y) == list(prioridades["incidentes"])
    assert fig.data[0].texttemplate
    assert fig.layout.xaxis.title.text == "Prioridad"
    assert fig.layout.yaxis.title.text == "Incidentes únicos"


def test_grafico_duracion_ecdf_con_eje_log_y_porcentaje(incidentes):
    fig = vis.grafico_duracion(incidentes)
    assert fig.data[0].type == "scatter"
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.tickformat == ".0%"
    # Paso 1: las 8 duraciones validas entran en la ECDF y el 0 h se dibuja en el piso logaritmico.
    assert len(fig.data[0].x) == 8
    assert min(fig.data[0].x) == pytest.approx(PISO_LOG_HORAS)


def test_grafico_duracion_marca_mediana_y_p90(incidentes):
    fig = vis.grafico_duracion(incidentes)
    alturas = sorted(forma.y0 for forma in fig.layout.shapes)
    assert alturas == [0.5, 0.9]
    assert all(forma.y0 == forma.y1 for forma in fig.layout.shapes)
    textos = _anotaciones(fig)
    # Paso 1: mediana de [0,2,4,8,12,24,48,96] = 10 y p90 lineal = 62,4, en formato espanol.
    assert "Mediana: 10,0 h" in textos
    assert "P90: 62,4 h" in textos


def test_grafico_duracion_informa_ceros_y_sin_dato(incidentes):
    subtitulo = _subtitulo(vis.grafico_duracion(incidentes))
    assert "con 0 h: 1" in subtitulo and "1 min" in subtitulo
    assert "sin dato: 1" in subtitulo
    assert "descriptivo" in subtitulo
    assert "no son SLO, SLA ni MTTR" in subtitulo
    assert "cumple" not in subtitulo.lower() and "objetivo" not in subtitulo.lower()


def test_grafico_variable_duracion_usa_bins_logaritmicos_con_ticks_en_horas(incidentes):
    fig = vis.grafico_variable(incidentes, DURACION)
    assert fig.data[0].type == "histogram"
    textos = list(fig.layout.xaxis.ticktext)
    assert {"0,01", "1", "10", "100"} <= set(textos)
    valores = sorted(fig.data[0].x)
    assert len(valores) == 8
    assert valores[0] == pytest.approx(math.log10(PISO_LOG_HORAS))
    assert valores[-1] == pytest.approx(math.log10(96.0))
    assert "sin dato: 1" in _subtitulo(fig)


def test_grafico_variable_conteo_discreto_usa_barras_por_valor(incidentes):
    fig = vis.grafico_variable(incidentes, "reassignment_count")
    assert len(fig.data) == 1 and fig.data[0].type == "bar"
    assert list(fig.data[0].x) == [0, 1, 2, 3, 4, 5, 6]
    assert list(fig.data[0].y) == [2, 1, 2, 1, 1, 1, 1]
    con_nulo = vis.grafico_variable(incidentes, "reopen_count")
    assert list(con_nulo.data[0].x) == [0, 1, 2] and list(con_nulo.data[0].y) == [5, 2, 1]
    assert "sin dato: 1" in _subtitulo(con_nulo)


def test_grafico_variable_categorica_limita_a_diez_e_informa_total():
    nombres = [f"C{i:02d}" for i in range(1, 13)]
    serie = [nombre for nombre, veces in zip(nombres, range(12, 0, -1)) for _ in range(veces)]
    df = pd.DataFrame({"category": pd.array(serie, dtype="string")})
    fig = vis.grafico_variable(df, "category")
    traza = fig.data[0]
    assert traza.type == "bar" and traza.orientation == "h"
    assert list(traza.y) == nombres[:10]
    assert list(traza.x) == list(range(12, 2, -1))
    assert fig.layout.yaxis.autorange == "reversed"
    assert "12 categorías en total" in _subtitulo(fig)


def test_grafico_variable_categorica_cuenta_nulos_como_sin_dato(incidentes):
    fig = vis.grafico_variable(incidentes, "category")
    pares = dict(zip(fig.data[0].y, fig.data[0].x))
    assert pares == {"C1": 4, "C2": 2, "C3": 2, SIN_DATO: 1}
    assert "4 categorías en total" in _subtitulo(fig)


def test_grafico_segmentos_horizontal_respeta_orden_y_marca_cautela(segmentos):
    fig = vis.grafico_segmentos(segmentos, "category")
    assert len(fig.data) == 1
    traza = fig.data[0]
    assert traza.type == "bar" and traza.orientation == "h"
    assert list(traza.y) == list(segmentos["segmento"])
    assert list(traza.x[:3]) == [10.5, 20.25, 30.0]
    assert fig.layout.yaxis.autorange == "reversed"
    assert list(traza.marker.color) == [vis.COLOR_PRINCIPAL, vis.COLOR_PRINCIPAL,
                                        vis.COLOR_CAUTELA, vis.COLOR_CAUTELA]
    assert ETIQUETAS["category"].lower() in fig.layout.title.text.lower()
    assert fig.layout.yaxis.title.text == ETIQUETAS["category"]


def test_figura_boxplot_prioridad_eje_log_orden_y_n(incidentes):
    fig = vis.figura_boxplot_prioridad(incidentes)
    assert isinstance(fig, Figure)
    eje = fig.axes[0]
    assert eje.get_yscale() == "log"
    etiquetas = [t.get_text() for t in eje.get_xticklabels()]
    assert etiquetas == ["1 - Critical\n(n=2)", "2 - High\n(n=1)", "3 - Moderate\n(n=2)",
                         "4 - Low\n(n=2)", f"{SIN_DATO}\n(n=1)"]
    assert all(e.startswith(p) for e, p in zip(etiquetas, ORDEN_PRIORIDADES))


def test_figura_boxplot_anota_ceros_en_piso_log(incidentes):
    fig = vis.figura_boxplot_prioridad(incidentes)
    notas = [t.get_text() for t in fig.texts]
    assert any("0 h" in nota and "1 min" in nota for nota in notas)
    sin_ceros = vis.figura_boxplot_prioridad(incidentes[incidentes[DURACION] != 0])
    assert not any("0 h" in t.get_text() for t in sin_ceros.texts)


def test_figura_mapa_spearman_triangulo_inferior_y_rango_fijo(matriz):
    fig = vis.figura_mapa_spearman(matriz)
    assert isinstance(fig, Figure)
    eje = fig.axes[0]
    assert len(eje.texts) == 10
    assert tuple(eje.collections[0].get_clim()) == (-1.0, 1.0)
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "ρ de Spearman"
    esperadas = [ETIQUETAS[v] for v in matriz.columns]
    assert [t.get_text() for t in eje.get_xticklabels()] == esperadas
    assert [t.get_text() for t in eje.get_yticklabels()] == esperadas
    anotaciones = {t.get_text() for t in eje.texts}
    assert "0,92" in anotaciones and "1,00" in anotaciones
    # Paso 1: el triangulo superior no se dibuja, asi que cada coeficiente aparece una sola vez.
    assert sum(t.get_text() == "0,65" for t in eje.texts) == 1


def test_figuras_matplotlib_no_dejan_figuras_en_pyplot(incidentes, matriz):
    import matplotlib.pyplot as plt

    plt.close("all")
    vis.figura_a_png(vis.figura_boxplot_prioridad(incidentes))
    vis.figura_a_png(vis.figura_mapa_spearman(matriz))
    vis.figura_boxplot_prioridad(incidentes.iloc[0:0])
    assert plt.get_fignums() == []


def test_figura_a_png_devuelve_bytes_png(matriz):
    contenido = vis.figura_a_png(vis.figura_mapa_spearman(matriz))
    assert isinstance(contenido, bytes)
    assert contenido.startswith(b"\x89PNG")


CASOS_VACIOS = {
    "prioridades": lambda d: vis.grafico_prioridades(pd.DataFrame(columns=list(COLUMNAS_PRIORIDADES))),
    "duracion": lambda d: vis.grafico_duracion(d.iloc[0:0]),
    "duracion_sin_validas": lambda d: vis.grafico_duracion(d.assign(**{DURACION: np.nan})),
    "variable_duracion": lambda d: vis.grafico_variable(d.iloc[0:0], DURACION),
    "variable_conteo": lambda d: vis.grafico_variable(d.iloc[0:0], "reassignment_count"),
    "variable_categorica": lambda d: vis.grafico_variable(d.iloc[0:0], "category"),
    "segmentos": lambda d: vis.grafico_segmentos(pd.DataFrame(columns=list(COLUMNAS_SEGMENTO)), "category"),
    "boxplot": lambda d: vis.figura_boxplot_prioridad(d.iloc[0:0]),
    "boxplot_sin_validas": lambda d: vis.figura_boxplot_prioridad(d.assign(**{DURACION: np.nan})),
    "mapa": lambda d: vis.figura_mapa_spearman(pd.DataFrame()),
}


@pytest.mark.parametrize("caso", list(CASOS_VACIOS))
def test_entradas_vacias_devuelven_aviso_sin_excepcion(caso, incidentes):
    fig = CASOS_VACIOS[caso](incidentes)
    if isinstance(fig, go.Figure):
        assert vis.AVISO_VACIO in _anotaciones(fig)
        assert len(fig.data) == 0
    else:
        assert isinstance(fig, Figure)
        textos = [t.get_text() for t in fig.texts] + [t.get_text() for e in fig.axes for t in e.texts]
        assert vis.AVISO_VACIO in textos


def test_visualizaciones_no_mutan_la_entrada(incidentes, prioridades, segmentos, matriz):
    copias = {n: d.copy(deep=True) for n, d in
              {"incidentes": incidentes, "prioridades": prioridades, "segmentos": segmentos, "matriz": matriz}.items()}
    vis.grafico_prioridades(prioridades)
    vis.grafico_duracion(incidentes)
    for variable in (DURACION, "reassignment_count", "reopen_count", "category", "priority"):
        vis.grafico_variable(incidentes, variable)
    vis.grafico_segmentos(segmentos, "category")
    vis.figura_boxplot_prioridad(incidentes)
    vis.figura_mapa_spearman(matriz)
    for nombre, actual in {"incidentes": incidentes, "prioridades": prioridades,
                           "segmentos": segmentos, "matriz": matriz}.items():
        pd.testing.assert_frame_equal(actual, copias[nombre])


def test_separadores_numericos_en_espanol(incidentes, prioridades, segmentos):
    figuras = [vis.grafico_prioridades(prioridades), vis.grafico_duracion(incidentes),
               vis.grafico_variable(incidentes, DURACION), vis.grafico_variable(incidentes, "category"),
               vis.grafico_variable(incidentes, "sys_mod_count"), vis.grafico_segmentos(segmentos, "category"),
               vis.grafico_duracion(incidentes.iloc[0:0])]
    for fig in figuras:
        assert fig.layout.separators == ",."
        assert fig.layout.title.text


def test_horas_en_formato_espanol():
    assert vis._horas_es(24.6) == "24,6"
    assert vis._horas_es(1000) == "1.000,0"
    assert vis._horas_es(397.37) == "397,4"


def test_grafico_variable_ordinal_conserva_orden_de_prioridad(incidentes):
    fig = vis.grafico_variable(incidentes, "priority")
    traza = fig.data[0]
    assert list(traza.y) == list(ORDEN_PRIORIDADES)
    assert list(traza.x) == [2, 1, 3, 2, 1]
    assert fig.layout.yaxis.autorange == "reversed"


def test_grafico_segmentos_sin_medianas_devuelve_aviso(segmentos):
    sin_horas = segmentos.assign(mediana_horas=np.nan, p90_horas=np.nan, duraciones_validas=0,
                                 porcentaje_sin_duracion=100.0, muestra_suficiente=False)
    fig = vis.grafico_segmentos(sin_horas, "category")
    assert vis.AVISO_VACIO in _anotaciones(fig)
    assert len(fig.data) == 0


def test_grafico_segmentos_rotula_el_umbral_de_muestra_del_diccionario(segmentos, monkeypatch):
    monkeypatch.setattr(vis, "MIN_MUESTRA", 50, raising=False)
    assert "menos de 50 duraciones" in _subtitulo(vis.grafico_segmentos(segmentos, "category"))


def test_figura_boxplot_aclara_que_n_cuenta_duraciones_validas(incidentes):
    sin_ceros = vis.figura_boxplot_prioridad(incidentes[incidentes[DURACION] != 0])
    notas = " ".join(t.get_text() for t in sin_ceros.texts)
    assert "n = duraciones válidas" in notas
    assert "sin dato excluidos: 1" in notas


def test_figura_boxplot_tolera_indice_repetido(incidentes):
    repetido = incidentes.set_axis([0, 0, 1, 1, 2, 2, 3, 3, 4])
    fig = vis.figura_boxplot_prioridad(repetido)
    etiquetas = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert etiquetas == ["1 - Critical\n(n=2)", "2 - High\n(n=1)", "3 - Moderate\n(n=2)",
                         "4 - Low\n(n=2)", f"{SIN_DATO}\n(n=1)"]


def test_subtitulo_de_duracion_parte_en_lineas_cortas(incidentes):
    lineas = _subtitulo(vis.grafico_duracion(incidentes)).split("<br>")
    assert len(lineas) == 2
    assert lineas[0] == vis.AVISO_DESCRIPTIVO
    assert all(len(linea) <= 90 for linea in lineas)


def test_horas_negativas_cuentan_como_sin_dato_no_como_cero():
    df = pd.DataFrame({DURACION: pd.Series([0.0, -5.0, 2.0, np.nan, 10.0], dtype="float64")})
    fig = vis.grafico_duracion(df)
    assert len(fig.data[0].x) == 3
    assert "con 0 h: 1" in _subtitulo(fig) and "sin dato: 2" in _subtitulo(fig)
    # Paso 1: la mediana sale de [0, 2, 10], sin la negativa.
    assert "Mediana: 2,0 h" in _anotaciones(fig)
