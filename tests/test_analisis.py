"""Pruebas de segmentación, correlaciones, matriz de Spearman, diccionario con calidad y fachada.

Usan el CSV sintético del conftest (S) o DataFrames construidos a mano (F).

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import diccionario
from src.data_loader import DatosIncidentes
from src.eda import (
    COLUMNAS_PRIORIDADES, COLUMNAS_SEGMENTO, DURACION, MIN_MUESTRA, OTROS, SIN_DATO,
    VARIABLES_RELACION, VARIABLES_SEGMENTABLES, VARIABLES_SEGMENTO, AnalisisIncidentes,
    comparar_prioridades, correlaciones, diccionario_con_calidad, indicadores,
    matriz_spearman, perfil_calidad, segmentar, spearman,
)

TODAS = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"]
RAIZ = Path(__file__).resolve().parents[1]
MODULOS_DEL_PAQUETE = ("src.diccionario", "src.data_loader", "src.eda")


def _analisis(datos: DatosIncidentes, prioridades: list[str] = TODAS) -> AnalisisIncidentes:
    return AnalisisIncidentes(datos.incidentes, datos.control, prioridades)


def _fila(tabla: pd.DataFrame, segmento: str) -> pd.Series:
    return tabla.set_index("segmento").loc[segmento]


def test_eda_reexporta_constantes_del_diccionario() -> None:
    assert DURACION == diccionario.DURACION == "tiempo_resolucion_horas"
    assert SIN_DATO == diccionario.SIN_DATO
    assert OTROS == diccionario.OTROS
    assert VARIABLES_SEGMENTABLES is VARIABLES_SEGMENTO
    assert VARIABLES_SEGMENTABLES == ("category", "assignment_group")
    assert COLUMNAS_SEGMENTO == (
        "segmento", "incidentes", "porcentaje_incidentes", "duraciones_validas",
        "mediana_horas", "p90_horas", "porcentaje_sin_duracion", "muestra_suficiente",
    )


def test_comparar_prioridades_conteos_mediana_y_porcentaje_de_incidentes(datos_sinteticos: DatosIncidentes) -> None:
    tabla = comparar_prioridades(datos_sinteticos.incidentes)
    assert tuple(tabla.columns) == COLUMNAS_PRIORIDADES
    assert tabla["priority"].tolist() == TODAS
    por_prioridad = tabla.set_index("priority")
    assert por_prioridad["incidentes"].tolist() == [2, 1, 3, 2]
    assert por_prioridad["duraciones_validas"].tolist() == [2, 1, 2, 2]
    assert por_prioridad["mediana_horas"].tolist() == [3.0, 8.0, 36.0, 48.0]
    assert por_prioridad["porcentaje_incidentes"].tolist() == [25.0, 12.5, 37.5, 25.0]
    assert por_prioridad["porcentaje_incidentes"].sum() == pytest.approx(100.0)


def test_comparar_prioridades_vacio_conserva_columnas(datos_sinteticos: DatosIncidentes) -> None:
    tabla = comparar_prioridades(datos_sinteticos.incidentes.iloc[0:0])
    assert tuple(tabla.columns) == COLUMNAS_PRIORIDADES
    assert tabla.empty


def test_segmentar_categoria_conteos_mediana_p90_y_sin_duracion(datos_sinteticos: DatosIncidentes) -> None:
    tabla = segmentar(datos_sinteticos.incidentes, "category")
    assert tuple(tabla.columns) == COLUMNAS_SEGMENTO
    assert tabla["segmento"].tolist() == ["C1", "C2", "C3", SIN_DATO]
    c1, c2, c3 = _fila(tabla, "C1"), _fila(tabla, "C2"), _fila(tabla, "C3")
    assert (c1["incidentes"], c1["duraciones_validas"]) == (3, 3)
    assert c1["mediana_horas"] == pytest.approx(4.0)
    assert c1["p90_horas"] == pytest.approx(7.2)
    assert c1["porcentaje_incidentes"] == pytest.approx(37.5)
    assert (c2["incidentes"], c2["duraciones_validas"]) == (2, 1)
    assert c2["mediana_horas"] == pytest.approx(24.0)
    assert c2["porcentaje_sin_duracion"] == pytest.approx(50.0)
    assert c3["mediana_horas"] == pytest.approx(24.0)
    assert c3["p90_horas"] == pytest.approx(43.2)
    assert c3["porcentaje_sin_duracion"] == pytest.approx(0.0)


def test_segmentar_top_agrupa_resto_en_otros(datos_sinteticos: DatosIncidentes) -> None:
    tabla = segmentar(datos_sinteticos.incidentes, "category", top=2)
    assert tabla["segmento"].tolist() == ["C1", "C2", OTROS]
    otros = _fila(tabla, OTROS)
    assert otros["incidentes"] == 3
    assert otros["duraciones_validas"] == 3
    assert otros["mediana_horas"] == pytest.approx(48.0)
    assert otros["p90_horas"] == pytest.approx(86.4)
    assert otros["porcentaje_incidentes"] == pytest.approx(37.5)
    assert tabla["incidentes"].sum() == 8


def test_segmentar_sin_resto_no_agrega_otros(datos_sinteticos: DatosIncidentes) -> None:
    tabla = segmentar(datos_sinteticos.incidentes, "category", top=4)
    assert OTROS not in tabla["segmento"].tolist()
    assert len(tabla) == 4


def test_segmentar_nulos_como_sin_dato(datos_sinteticos: DatosIncidentes) -> None:
    tabla = segmentar(datos_sinteticos.incidentes, "assignment_group")
    assert tabla["segmento"].tolist() == ["G1", "G2", "G3", SIN_DATO, "G4"]
    assert _fila(tabla, SIN_DATO)["incidentes"] == 1
    assert _fila(tabla, SIN_DATO)["mediana_horas"] == pytest.approx(0.0)


def test_segmentar_desempata_por_segmento_ascendente(datos_sinteticos: DatosIncidentes) -> None:
    tabla = segmentar(datos_sinteticos.incidentes, "assignment_group", top=3)
    assert tabla["segmento"].tolist() == ["G1", "G2", "G3", OTROS]
    otros = _fila(tabla, OTROS)
    assert (otros["incidentes"], otros["duraciones_validas"]) == (2, 2)
    assert otros["mediana_horas"] == pytest.approx(48.0)


def test_segmentar_rechaza_variable_no_autorizada_y_top_invalido(datos_sinteticos: DatosIncidentes) -> None:
    for variable in ("number", "priority", "incident_state"):
        with pytest.raises(ValueError):
            segmentar(datos_sinteticos.incidentes, variable)
    with pytest.raises(ValueError):
        segmentar(datos_sinteticos.incidentes, "category", top=0)


@pytest.mark.parametrize("top", [2.5, True, "3", None], ids=["decimal", "booleano", "texto", "nulo"])
def test_segmentar_rechaza_top_no_entero_con_value_error(datos_sinteticos: DatosIncidentes, top: Any) -> None:
    with pytest.raises(ValueError, match="top"):
        segmentar(datos_sinteticos.incidentes, "category", top=top)
    with pytest.raises(ValueError, match="top"):
        _analisis(datos_sinteticos).segmentar("category", top=top)


def test_segmentar_acepta_top_entero_de_numpy(datos_sinteticos: DatosIncidentes) -> None:
    tabla = segmentar(datos_sinteticos.incidentes, "category", top=np.int64(2))
    assert tabla["segmento"].tolist() == ["C1", "C2", OTROS]


def test_segmentar_vacio_conserva_columnas_del_contrato(datos_sinteticos: DatosIncidentes) -> None:
    tabla = segmentar(datos_sinteticos.incidentes.iloc[0:0], "category")
    assert tuple(tabla.columns) == COLUMNAS_SEGMENTO
    assert tabla.empty
    with pytest.raises(ValueError):
        segmentar(datos_sinteticos.incidentes.iloc[0:0], "number")


def test_segmentar_marca_muestra_suficiente_desde_30() -> None:
    df = pd.DataFrame({
        "category": pd.array(["A"] * 30 + ["B"] * 31, dtype="string"),
        DURACION: [1.0] * 30 + [2.0] * 29 + [np.nan] * 2,
    })
    tabla = segmentar(df, "category")
    assert tabla["segmento"].tolist() == ["B", "A"]
    assert _fila(tabla, "A")["duraciones_validas"] == MIN_MUESTRA
    assert bool(_fila(tabla, "A")["muestra_suficiente"]) is True
    assert bool(_fila(tabla, "B")["muestra_suficiente"]) is False


def test_segmentar_tipos_de_columnas_y_total_de_segmentos(datos_sinteticos: DatosIncidentes) -> None:
    tabla = segmentar(datos_sinteticos.incidentes, "category", top=2)
    assert str(tabla["incidentes"].dtype) == "int64"
    assert str(tabla["duraciones_validas"].dtype) == "int64"
    assert str(tabla["muestra_suficiente"].dtype) == "bool"
    for columna in ("porcentaje_incidentes", "mediana_horas", "p90_horas", "porcentaje_sin_duracion"):
        assert str(tabla[columna].dtype) == "float64", columna
    assert tabla.attrs["segmentos_totales"] == 4
    assert list(tabla.index) == [0, 1, 2]


def test_correlaciones_incluye_spearman_en_tres_contadores(datos_sinteticos: DatosIncidentes) -> None:
    resultado = correlaciones(datos_sinteticos.incidentes)
    assert [r["variable"] for r in resultado] == list(VARIABLES_RELACION)
    for registro in resultado:
        assert set(registro) == {"variable", "pearson_con_duracion", "spearman_con_duracion", "pares_validos"}
        assert registro["pares_validos"] == 7
    mod, reasignaciones, reaperturas = resultado
    assert mod["spearman_con_duracion"] == pytest.approx(1.0)
    assert mod["pearson_con_duracion"] == pytest.approx(0.920494, abs=1e-6)
    assert mod["pearson_con_duracion"] < 0.99
    assert reasignaciones["spearman_con_duracion"] == pytest.approx(0.991031, abs=1e-6)
    assert reasignaciones["pearson_con_duracion"] == pytest.approx(0.929949, abs=1e-6)
    assert reaperturas["spearman_con_duracion"] == pytest.approx(0.896421, abs=1e-6)


def test_correlacion_con_variable_constante_devuelve_none(datos_sinteticos: DatosIncidentes) -> None:
    resultado = _analisis(datos_sinteticos, ["1 - Critical", "2 - High"]).relaciones()
    reaperturas = next(r for r in resultado if r["variable"] == "reopen_count")
    assert reaperturas["pares_validos"] == 3
    assert reaperturas["pearson_con_duracion"] is None
    assert reaperturas["spearman_con_duracion"] is None


def test_correlacion_con_menos_de_tres_pares_devuelve_none(datos_sinteticos: DatosIncidentes) -> None:
    for registro in _analisis(datos_sinteticos, ["1 - Critical"]).relaciones():
        assert registro["pares_validos"] == 2
        assert registro["pearson_con_duracion"] is None
        assert registro["spearman_con_duracion"] is None


def test_spearman_es_pearson_sobre_rangos_y_devuelve_none_si_no_esta_definido() -> None:
    assert spearman(pd.Series([1.0, 2.0, 3.0]), pd.Series([30.0, 20.0, 10.0])) == pytest.approx(-1.0)
    assert spearman(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 4.0, 100.0])) == pytest.approx(1.0)
    assert spearman(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])) is None
    assert spearman(pd.Series([5.0, 5.0, 5.0]), pd.Series([1.0, 2.0, 3.0])) is None
    resultado = spearman(pd.Series([0, 1, 2, 3, 5, 0, 6.0]), pd.Series([2, 4, 8, 24, 48, 0, 96.0]))
    assert isinstance(resultado, float)
    assert resultado == pytest.approx(0.991031, abs=1e-6)


def test_matriz_spearman_simetrica_diagonal_uno_y_coherente_con_lista(datos_sinteticos: DatosIncidentes) -> None:
    matriz = matriz_spearman(datos_sinteticos.incidentes)
    orden = [*VARIABLES_RELACION, DURACION]
    assert list(matriz.index) == orden
    assert list(matriz.columns) == orden
    assert matriz.shape == (4, 4)
    assert (matriz.dtypes == "float64").all()
    np.testing.assert_allclose(matriz.to_numpy(), matriz.to_numpy().T)
    np.testing.assert_allclose(np.diag(matriz.to_numpy()), np.ones(4))
    for registro in correlaciones(datos_sinteticos.incidentes):
        assert matriz.loc[registro["variable"], DURACION] == pytest.approx(registro["spearman_con_duracion"])


def test_matriz_spearman_deja_nan_donde_no_esta_definida(datos_sinteticos: DatosIncidentes) -> None:
    matriz = _analisis(datos_sinteticos, ["1 - Critical", "2 - High"]).matriz_relaciones()
    assert np.isnan(matriz.loc["reopen_count", DURACION])
    assert matriz.loc["sys_mod_count", DURACION] == pytest.approx(1.0)


def test_spearman_funciona_sin_scipy(monkeypatch: pytest.MonkeyPatch, datos_sinteticos: DatosIncidentes) -> None:
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    resultado = correlaciones(datos_sinteticos.incidentes)
    assert resultado[1]["spearman_con_duracion"] == pytest.approx(0.991031, abs=1e-6)
    matriz = matriz_spearman(datos_sinteticos.incidentes)
    assert matriz.loc["reassignment_count", DURACION] == pytest.approx(0.991031, abs=1e-6)


def test_diccionario_con_calidad_une_tipo_y_nulos(datos_sinteticos: DatosIncidentes) -> None:
    tabla = diccionario_con_calidad(datos_sinteticos.incidentes)
    assert list(tabla.columns) == [
        "variable", "etiqueta", "tipo_analitico", "descripcion", "origen",
        "tipo", "nulos", "porcentaje_nulos", "valores_unicos",
    ]
    assert len(tabla) == 13
    assert tabla["variable"].tolist() == diccionario.tabla_diccionario()["variable"].tolist()
    duracion = tabla.set_index("variable").loc[DURACION]
    assert duracion["nulos"] == 1
    assert duracion["porcentaje_nulos"] == pytest.approx(12.5)
    assert duracion["etiqueta"] == "Horas hasta resolución"
    assert tabla.set_index("variable").loc["priority", "tipo"] == "string"


def _iguales(a: Any, b: Any) -> None:
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize("funcion", [
    indicadores,
    comparar_prioridades,
    perfil_calidad,
    correlaciones,
    matriz_spearman,
    diccionario_con_calidad,
    lambda df: segmentar(df, "category", top=2),
    lambda df: segmentar(df, "assignment_group"),
], ids=["indicadores", "comparar_prioridades", "perfil_calidad", "correlaciones",
        "matriz_spearman", "diccionario_con_calidad", "segmentar_category", "segmentar_grupo"])
def test_funciones_de_analisis_no_mutan_la_entrada(
    funcion: Callable[[pd.DataFrame], Any], datos_sinteticos: DatosIncidentes,
) -> None:
    entrada = datos_sinteticos.incidentes
    copia = entrada.copy(deep=True)
    primera = funcion(entrada)
    pd.testing.assert_frame_equal(entrada, copia)
    _iguales(primera, funcion(entrada))


def test_fachada_expone_segmentar_relaciones_matriz_y_estrategia(datos_sinteticos: DatosIncidentes) -> None:
    analisis = _analisis(datos_sinteticos)
    tabla = analisis.segmentar("category")
    pd.testing.assert_frame_equal(tabla, segmentar(analisis.datos, "category", top=5))
    assert analisis.relaciones() == correlaciones(analisis.datos)
    pd.testing.assert_frame_equal(analisis.matriz_relaciones(), matriz_spearman(analisis.datos))
    assert analisis.estrategia("priority") == "AnalizadorCategorico"
    assert analisis.estrategia(DURACION) == "AnalizadorNumerico"
    assert analisis.estrategia("reopen_count") == "AnalizadorNumerico"


def test_fachada_segmentar_respeta_el_filtro_y_valida(datos_sinteticos: DatosIncidentes) -> None:
    analisis = _analisis(datos_sinteticos, ["3 - Moderate", "4 - Low"])
    assert analisis.segmentar("assignment_group")["incidentes"].sum() == 5
    assert analisis.segmentar("category", top=1)["segmento"].tolist() == ["C2", OTROS]
    with pytest.raises(ValueError):
        analisis.segmentar("priority")
    with pytest.raises(ValueError):
        analisis.segmentar("category", top=0)


def test_contexto_delega_en_herramientas_con_import_diferido(
    monkeypatch: pytest.MonkeyPatch, datos_sinteticos: DatosIncidentes,
) -> None:
    llamadas: list[tuple[AnalisisIncidentes, str]] = []

    def contexto_herramienta(analisis: AnalisisIncidentes, nombre: str) -> dict:
        if nombre == "shell":
            raise ValueError("Herramienta no autorizada.")
        llamadas.append((analisis, nombre))
        return {"herramienta": nombre}

    modulo_falso = types.ModuleType("src.herramientas")
    modulo_falso.contexto_herramienta = contexto_herramienta  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.herramientas", modulo_falso)
    analisis = _analisis(datos_sinteticos)
    assert analisis.contexto("prioridades") == {"herramienta": "prioridades"}
    assert llamadas == [(analisis, "prioridades")]
    with pytest.raises(ValueError):
        analisis.contexto("shell")


def test_contexto_herramientas_legadas_serializables_y_lista_cerrada(datos_sinteticos: DatosIncidentes) -> None:
    pytest.importorskip("src.herramientas")
    for prioridades in (TODAS, []):
        analisis = _analisis(datos_sinteticos, prioridades)
        for herramienta in ("resumen", "prioridades", "calidad", "relaciones"):
            contexto = analisis.contexto(herramienta)
            json.dumps(contexto, allow_nan=False)
            assert contexto["filtro_prioridades"] == prioridades
        with pytest.raises(ValueError):
            analisis.contexto("shell")


def test_seleccion_vacia_deja_relaciones_indefinidas_y_tablas_con_contrato(
    datos_sinteticos: DatosIncidentes,
) -> None:
    analisis = _analisis(datos_sinteticos, [])
    orden = [*VARIABLES_RELACION, DURACION]
    matriz = analisis.matriz_relaciones()
    assert (list(matriz.index), list(matriz.columns)) == (orden, orden)
    assert bool(matriz.isna().all().all())
    for registro in analisis.relaciones():
        assert (registro["pares_validos"], registro["pearson_con_duracion"], registro["spearman_con_duracion"]) == (0, None, None)
    tabla = analisis.segmentar("assignment_group")
    assert tuple(tabla.columns) == COLUMNAS_SEGMENTO
    assert tabla.empty
    assert tabla.attrs["segmentos_totales"] == 0
    assert len(diccionario_con_calidad(analisis.datos)) == 13


def test_modulos_de_datos_no_cargan_interfaz_cliente_http_ni_herramientas() -> None:
    ajenos = ("streamlit", "requests", "src.llm_client", "src.observabilidad", "src.herramientas")
    codigo = (
        "import sys\n"
        "import src.diccionario, src.data_loader, src.eda\n"
        f"print(sorted(m for m in {ajenos!r} if m in sys.modules))\n"
    )
    salida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, cwd=RAIZ, check=True)
    assert salida.stdout.strip() == "[]"


def _api_publica(modulo: types.ModuleType) -> list[tuple[str, Callable[..., Any]]]:
    """Funciones públicas definidas en el módulo y métodos públicos de sus clases."""
    elementos: list[tuple[str, Callable[..., Any]]] = []
    for nombre, objeto in vars(modulo).items():
        if nombre.startswith("_") or getattr(objeto, "__module__", None) != modulo.__name__:
            continue
        if inspect.isfunction(objeto):
            elementos.append((nombre, objeto))
        elif inspect.isclass(objeto):
            for nombre_metodo, metodo in vars(objeto).items():
                funcion = metodo.__func__ if isinstance(metodo, (staticmethod, classmethod)) else metodo
                if not nombre_metodo.startswith("_") and inspect.isfunction(funcion):
                    elementos.append((f"{nombre}.{nombre_metodo}", funcion))
    return elementos


@pytest.mark.parametrize("nombre_modulo", MODULOS_DEL_PAQUETE)
def test_api_publica_del_paquete_documentada_y_tipada(nombre_modulo: str) -> None:
    modulo = importlib.import_module(nombre_modulo)
    assert modulo.__doc__ and "Autor: Kevin Alexis Barrera Llerena 2026" in modulo.__doc__
    faltantes = []
    for nombre, funcion in _api_publica(modulo):
        firma = inspect.signature(funcion)
        sin_tipo = [p.name for p in firma.parameters.values()
                    if p.name not in {"self", "cls"} and p.annotation is inspect.Parameter.empty]
        if not (funcion.__doc__ or "").strip():
            faltantes.append(f"{nombre}: sin docstring")
        if sin_tipo or firma.return_annotation is inspect.Signature.empty:
            faltantes.append(f"{nombre}: sin anotación en {sin_tipo or 'retorno'}")
    assert faltantes == []
