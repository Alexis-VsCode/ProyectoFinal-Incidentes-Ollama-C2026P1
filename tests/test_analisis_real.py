"""Pruebas sobre el CSV real del Taller 1 (data/incident_event_log.csv).

Reproducen las cifras ya validadas del Taller 1. Las horas de resolución son
estadísticos descriptivos del histórico: no son SLO, SLA ni MTTR.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data_loader import SHA256_CSV_INCLUIDO, DatosIncidentes, cargar_ruta, verificar_integridad
from src.eda import (
    DURACION, OTROS, SIN_DATO, VARIABLES_RELACION, VARIABLES_SEGMENTABLES, AnalisisIncidentes,
    matriz_spearman,
)

pytestmark = pytest.mark.csv_real

RUTA_CSV_REAL = Path(__file__).resolve().parents[1] / "data" / "incident_event_log.csv"
PRIORIDADES = ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"]
PARES_MAXIMOS = 20_769 - 1_418


@pytest.fixture(scope="module")
def datos_reales() -> DatosIncidentes:
    return cargar_ruta(RUTA_CSV_REAL)


@pytest.fixture(scope="module")
def analisis_real(datos_reales: DatosIncidentes) -> AnalisisIncidentes:
    return AnalisisIncidentes(datos_reales.incidentes, datos_reales.control, PRIORIDADES)


def test_integridad_del_csv_incluido_coincide(datos_reales: DatosIncidentes) -> None:
    resultado = verificar_integridad(datos_reales.control, SHA256_CSV_INCLUIDO)
    assert resultado["estado"] == "coincide"
    assert (datos_reales.control["eventos"], datos_reales.control["incidentes"]) == (119_998, 20_769)


def test_comparar_prioridades_real_reproduce_taller1(analisis_real: AnalisisIncidentes) -> None:
    tabla = analisis_real.prioridades_resumen().set_index("priority")
    assert tabla.index.tolist() == PRIORIDADES
    assert tabla["incidentes"].tolist() == [220, 337, 19_554, 658]
    assert tabla["mediana_horas"].tolist() == pytest.approx([78.17, 48.32, 24.29, 22.85], abs=0.01)
    sin_duracion = tabla["incidentes"] - tabla["duraciones_validas"]
    porcentaje_sin_duracion = 100 * sin_duracion / tabla["incidentes"]
    assert porcentaje_sin_duracion["1 - Critical"] == 0
    assert porcentaje_sin_duracion["3 - Moderate"] == pytest.approx(6.79, abs=0.01)
    assert porcentaje_sin_duracion["4 - Low"] == pytest.approx(13.68, abs=0.01)
    assert sin_duracion.sum() == 1_418
    assert tabla["porcentaje_incidentes"].sum() == pytest.approx(100.0)


@pytest.mark.parametrize("variable", VARIABLES_SEGMENTABLES)
@pytest.mark.parametrize("top", [1, 5, 10])
def test_segmentar_real_suma_20769_en_cada_variable(
    analisis_real: AnalisisIncidentes, variable: str, top: int,
) -> None:
    tabla = analisis_real.segmentar(variable, top=top)
    assert tabla["incidentes"].sum() == 20_769
    assert tabla["porcentaje_incidentes"].sum() == pytest.approx(100.0)
    assert len(tabla) == top + 1
    assert tabla["segmento"].iloc[-1] == OTROS


def test_segmentar_real_grupos_de_asignacion_top5(analisis_real: AnalisisIncidentes) -> None:
    tabla = analisis_real.segmentar("assignment_group", top=5)
    assert tabla["segmento"].tolist() == ["Group 70", SIN_DATO, "Group 25", "Group 39", "Group 24", OTROS]
    assert tabla["incidentes"].tolist() == [7_369, 2_157, 1_080, 995, 907, 8_261]
    assert tabla["porcentaje_incidentes"].round(2).tolist()[:2] == [35.48, 10.39]
    assert tabla["porcentaje_incidentes"].iloc[-1] == pytest.approx(39.78, abs=0.01)
    assert tabla["mediana_horas"].tolist()[:5] == pytest.approx([0.35, 50.52, 101.82, 21.73, 25.43], abs=0.01)
    assert tabla.attrs["segmentos_totales"] == 65


def test_spearman_real_definido_y_acotado(analisis_real: AnalisisIncidentes) -> None:
    resultado = analisis_real.relaciones()
    assert [r["variable"] for r in resultado] == list(VARIABLES_RELACION)
    for registro in resultado:
        assert registro["spearman_con_duracion"] is not None
        assert -1.0 <= registro["spearman_con_duracion"] <= 1.0
        assert -1.0 <= registro["pearson_con_duracion"] <= 1.0
        assert 3 <= registro["pares_validos"] <= PARES_MAXIMOS
    por_variable = {r["variable"]: r for r in resultado}
    assert por_variable["reassignment_count"]["pearson_con_duracion"] == pytest.approx(0.204, abs=0.001)
    assert por_variable["reassignment_count"]["spearman_con_duracion"] == pytest.approx(0.523, abs=0.001)
    assert por_variable["reopen_count"]["pearson_con_duracion"] == pytest.approx(0.055, abs=0.001)
    assert por_variable["reopen_count"]["spearman_con_duracion"] == pytest.approx(0.106, abs=0.001)


def test_matriz_spearman_real_coherente_con_lista(analisis_real: AnalisisIncidentes) -> None:
    matriz = matriz_spearman(analisis_real.datos)
    np.testing.assert_allclose(matriz.to_numpy(), matriz.to_numpy().T)
    np.testing.assert_allclose(np.diag(matriz.to_numpy()), np.ones(4))
    for registro in analisis_real.relaciones():
        assert matriz.loc[registro["variable"], DURACION] == pytest.approx(registro["spearman_con_duracion"])


def test_indicadores_reales_sin_cambios(analisis_real: AnalisisIncidentes) -> None:
    resumen = analisis_real.resumen()
    assert resumen["incidentes_seleccionados"] == 20_769
    assert resumen["incidentes_sin_duracion"] == 1_418
    assert resumen["duraciones_validas"] == PARES_MAXIMOS
    assert resumen["media_horas"] == pytest.approx(187.68, abs=0.01)
    assert resumen["mediana_horas"] == pytest.approx(24.60, abs=0.01)
    assert resumen["p90_horas"] == pytest.approx(397.37, abs=0.01)
