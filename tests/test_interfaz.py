"""Pruebas de la interfaz Streamlit: ayudantes puros de app.py y recorridos con AppTest.

Los recorridos usan el CSV real incluido (marcador `interfaz`); la carga se cachea con
st.cache_resource y se comparte entre pruebas del mismo proceso. Ollama nunca se consulta:
la sonda de salud se parchea con un servicio caído y el agente usa el doble ClienteFalso.
El registro JSONL va a un directorio temporal por prueba.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app
from src.diccionario import etiqueta
from src.llm_client import ClienteOllama, ErrorOllama
from src.observabilidad import cerrar_logger
from src.preguntas_demo import PREGUNTAS_SUGERIDAS
from tests.dobles_agente import DIGEST_FALSO, ClienteFalso, respuesta_con_llamadas

RUTA_APP = Path(__file__).resolve().parents[1] / "app.py"
PESTANAS_ESPERADAS = ("Datos", "Resumen", "Explorar", "Relaciones y segmentos",
                      "Agente local", "Observabilidad", "Arquitectura")
ESTADO_SIN_SERVICIO: dict[str, Any] = {
    "accesible": False, "version": None, "modelo": "llama3.2:3b", "modelo_cargado": None,
    "expira_utc": None, "latencia_ms": None,
    "error": "No se pudo conectar con Ollama en 127.0.0.1:11434.",
}
FRAGMENTOS_VERIFICABLES = ("**Hechos observados** Critical tiene 220 incidentes ", "y una mediana de 78,17 h.")


class FabricaClientes:
    """Reemplaza a ClienteOllama: cada llamada crea un ClienteFalso con las mismas opciones y lo guarda."""

    def __init__(self, **opciones: Any) -> None:
        self.opciones = opciones
        self.creados: list[ClienteFalso] = []

    def __call__(self, modelo: str = "llama3.2:3b", **_: Any) -> ClienteFalso:
        cliente = ClienteFalso(modelo=modelo, **self.opciones)
        self.creados.append(cliente)
        return cliente


@pytest.fixture
def at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[AppTest]:
    """AppTest de app.py con logs en tmp_path, sonda de salud sin servicio y caché de datos limpia."""
    monkeypatch.setenv("INCIDENTES_LOG_DIR", str(tmp_path))
    st.cache_data.clear()
    with patch.object(ClienteOllama, "estado", return_value=dict(ESTADO_SIN_SERVICIO)):
        yield AppTest.from_file(str(RUTA_APP), default_timeout=120)
    cerrar_logger()


def abrir(prueba: AppTest, pestana: str) -> AppTest:
    """Fija la pestaña activa por su etiqueta y ejecuta la app."""
    prueba.session_state["pestana"] = pestana
    return prueba.run()


def metrica(prueba: AppTest, etiqueta: str) -> str:
    """Valor mostrado de la métrica con esa etiqueta."""
    return next(m.value for m in prueba.metric if m.label == etiqueta)


def insignias(prueba: AppTest) -> list[str]:
    """Cuerpos markdown de las insignias st.badge renderizadas (":color-badge[texto]")."""
    return [m.value for m in prueba.markdown if "-badge[" in m.value]


def textos_de_tablas(prueba: AppTest) -> list[str]:
    """Nombres de columna y valores de todas las tablas renderizadas, como texto."""
    textos: list[str] = []
    for tabla in prueba.dataframe:
        datos = tabla.value
        textos.extend(str(c) for c in datos.columns)
        textos.extend(str(v) for v in datos.to_numpy().ravel())
    return textos


def consultar(prueba: AppTest, fabrica: FabricaClientes, pregunta: str | None = None) -> AppTest:
    """Abre «Agente local» y envía el formulario con ClienteOllama sustituido por la fábrica de dobles."""
    with patch("src.llm_client.ClienteOllama", fabrica):
        abrir(prueba, "Agente local")
        if pregunta is not None:
            prueba.text_area(key="pregunta_agente").input(pregunta)
        prueba.button(key="consultar_agente").click()
        prueba.run()
    return prueba


# Paso 1: ayudantes puros de formato y estado


def test_numero_es_usa_punto_de_miles_y_coma_decimal() -> None:
    assert app.numero_es(20769, 0) == "20.769"
    assert app.numero_es(78.1666666, 2) == "78,17"
    assert app.numero_es(1234567.891, 2) == "1.234.567,89"
    assert app.numero_es(-0.5, 1) == "-0,5"


def test_mostrar_horas_en_formato_es_ec_o_sin_duracion() -> None:
    assert app.mostrar_horas(78.16666666) == "78,17 h"
    assert app.mostrar_horas(24.6) == "24,60 h"
    assert app.mostrar_horas(None) == "Sin duración válida"


def test_texto_es_formatea_enteros_reales_nulos_y_booleanos() -> None:
    assert app.texto_es(19554) == "19.554"
    assert app.texto_es(0.2041, 3) == "0,204"
    assert app.texto_es(None) == "Sin dato"
    assert app.texto_es(float("nan")) == "Sin dato"
    assert app.texto_es(True) == "Sí"
    assert app.texto_es("Group 70") == "Group 70"


def test_texto_es_formatea_fechas_dia_mes_y_nat_como_sin_dato() -> None:
    assert app.texto_es(pd.Timestamp("2016-02-29 01:16")) == "29/02/2016 01:16"
    assert app.texto_es(pd.NaT) == "Sin dato"


def test_arquitectura_sin_diagrama_indica_como_regenerarlo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    simulado = MagicMock()
    monkeypatch.setattr(app, "st", simulado)
    monkeypatch.setattr(app, "RUTA_DIAGRAMA", tmp_path / "no_existe.png")
    app.render_arquitectura()
    simulado.image.assert_not_called()
    avisos = " ".join(str(llamada.args[0]) for llamada in simulado.warning.call_args_list)
    assert "python -m scripts.generar_arquitectura" in avisos


def test_firma_estado_cambia_con_filtro_y_modelo_pero_no_con_el_orden() -> None:
    base = app.firma_estado("abc", ["1 - Critical", "2 - High"], "llama3.2:3b")
    assert len(base) == 64
    assert base == app.firma_estado("abc", ["2 - High", "1 - Critical"], "llama3.2:3b")
    assert base != app.firma_estado("abc", ["1 - Critical"], "llama3.2:3b")
    assert base != app.firma_estado("abc", ["1 - Critical", "2 - High"], "llama3.2:1b")
    assert base != app.firma_estado("abd", ["1 - Critical", "2 - High"], "llama3.2:3b")


# Paso 2: recorridos de la interfaz con el CSV real


@pytest.mark.interfaz
def test_inicio_sin_excepcion_y_kpi_total_en_formato_es_ec(at: AppTest) -> None:
    at.run()
    assert not at.exception
    assert metrica(at, "Incidentes seleccionados") == "20.769"
    assert metrica(at, "Eventos del archivo completo") == "119.998"
    assert metrica(at, "Sin duración en la selección") == "1.418"
    assert tuple(t.label for t in at.tabs) == PESTANAS_ESPERADAS


@pytest.mark.interfaz
def test_filtro_critical_muestra_220_y_mediana_78_17(at: AppTest) -> None:
    at.run()
    at.multiselect(key="filtro_prioridades").set_value(["1 - Critical"]).run()
    assert not at.exception
    assert metrica(at, "Incidentes seleccionados") == "220"
    assert metrica(at, "Mediana hasta resolución") == "78,17 h"


@pytest.mark.interfaz
def test_seleccion_vacia_muestra_aviso_sin_excepcion(at: AppTest) -> None:
    at.run()
    at.multiselect(key="filtro_prioridades").set_value([]).run()
    assert not at.exception
    assert metrica(at, "Incidentes seleccionados") == "0"
    assert any("Seleccione al menos una prioridad" in w.value for w in at.warning)


@pytest.mark.interfaz
def test_recorre_las_siete_pestanas_sin_excepcion(at: AppTest) -> None:
    for pestana in PESTANAS_ESPERADAS:
        abrir(at, pestana)
        assert not at.exception, pestana
        assert at.session_state["pestana"] == pestana


@pytest.mark.interfaz
def test_datos_muestra_diccionario_en_espanol_e_integridad(at: AppTest) -> None:
    abrir(at, "Datos")
    assert not at.exception
    assert "Horas hasta resolución" in textos_de_tablas(at)
    assert ":green-badge[SHA-256 coincide con el CSV incluido]" in insignias(at)
    assert {"descargar_incidentes", "descargar_diccionario"} <= {b.key for b in at.download_button}


@pytest.mark.interfaz
def test_resumen_rotula_percentiles_como_descriptivos(at: AppTest) -> None:
    abrir(at, "Resumen")
    assert not at.exception
    assert len(at.get("plotly_chart")) >= 2
    assert any("no son SLO, SLA ni MTTR" in c.value for c in at.caption)


@pytest.mark.interfaz
def test_explorar_muestra_la_estrategia_elegida_por_la_factory(at: AppTest) -> None:
    abrir(at, "Explorar")
    assert any("AnalizadorNumerico" in i.value for i in at.info)
    at.selectbox(key="variable_explorar").set_value("priority").run()
    assert not at.exception
    assert any("AnalizadorCategorico" in i.value for i in at.info)


@pytest.mark.interfaz
def test_relaciones_muestra_figuras_seaborn_y_cambia_segmento(at: AppTest) -> None:
    abrir(at, "Relaciones y segmentos")
    assert not at.exception
    assert len(at.get("image")) >= 2
    assert {"descargar_boxplot", "descargar_mapa"} <= {b.key for b in at.download_button}
    at.session_state["variable_segmento"] = "assignment_group"
    at.run()
    assert not at.exception
    assert "Group 70" in textos_de_tablas(at)


@pytest.mark.interfaz
def test_observabilidad_sin_consultas_muestra_aviso_y_slos_sin_datos(at: AppTest) -> None:
    abrir(at, "Observabilidad")
    assert not at.exception
    assert any("Aún no hay consultas registradas" in i.value for i in at.info)
    assert any("Sin datos" in i.value for i in at.info)
    assert any("no describen los incidentes del CSV" in w.value for w in at.warning)
    slos = next(t.value for t in at.dataframe if "Resultado" in t.value.columns)
    assert len(slos) == 3
    assert set(slos["Resultado"]) == {"sin eventos"}
    assert ":red-badge[Servicio no accesible]" in insignias(at)
    assert ":green-badge[SHA-256 coincide con el CSV incluido]" in insignias(at)
    assert any(m.label == "Carga de datos" for m in at.metric)


@pytest.mark.interfaz
def test_verificar_ollama_sin_servicio_muestra_error(at: AppTest) -> None:
    error = ErrorOllama("No se pudo conectar con Ollama en 127.0.0.1:11434.", tipo="conexion", operacion="modelos")
    with patch.object(ClienteOllama, "verificar", side_effect=error):
        abrir(at, "Agente local")
        at.button(key="btn_verificar").click().run()
    assert not at.exception
    assert any("No se pudo conectar con Ollama" in e.value for e in at.error)
    assert "estado_ollama" not in at.session_state


@pytest.mark.interfaz
def test_verificar_ollama_con_servicio_muestra_version_modelo_y_digest(at: AppTest) -> None:
    resultado = {"accesible": True, "modelo_instalado": True, "modelos": ["llama3.2:3b"], "digest": DIGEST_FALSO,
                 "precalentado": True, "segundos_precalentamiento": 8.25}
    with patch.object(ClienteOllama, "verificar", return_value=resultado), \
            patch.object(ClienteOllama, "version", return_value="0.40.6"):
        abrir(at, "Agente local")
        at.button(key="btn_verificar").click().run()
    assert not at.exception
    exito = " ".join(s.value for s in at.success)
    assert "0.40.6" in exito and "llama3.2:3b" in exito and DIGEST_FALSO[:12] in exito
    assert ":green-badge[Modelo en memoria]" in insignias(at)


@pytest.mark.interfaz
def test_modo_manual_muestra_selector_de_herramienta_autorizada(at: AppTest) -> None:
    abrir(at, "Agente local")
    at.toggle(key="modo_automatico").set_value(False).run()
    assert not at.exception
    selector = at.selectbox(key="herramienta_manual")
    assert "Comparar prioridades" in selector.options


@pytest.mark.interfaz
def test_pregunta_sugerida_rellena_el_formulario(at: AppTest) -> None:
    abrir(at, "Agente local")
    assert at.text_area(key="pregunta_agente").value == PREGUNTAS_SUGERIDAS[0]
    at.button(key="sugerida_1").click().run()
    assert not at.exception
    assert at.text_area(key="pregunta_agente").value == PREGUNTAS_SUGERIDAS[1]


@pytest.mark.interfaz
def test_consulta_con_agente_simulado_muestra_insignia_traza_y_descarga(at: AppTest) -> None:
    fabrica = FabricaClientes(selecciones=[respuesta_con_llamadas(("prioridades", {}))],
                              fragmentos=FRAGMENTOS_VERIFICABLES)
    consultar(at, fabrica)
    assert not at.exception
    assert not at.error
    registro = at.session_state["respuesta_agente"]
    assert registro.estado == "respondida" and registro.herramientas == ("prioridades",)
    assert ":green-badge[Cifras verificadas: 2/2]" in insignias(at)
    assert ":blue-badge[Herramienta elegida por el modelo]" in insignias(at)
    assert any(s.label.startswith("Herramientas ejecutadas") for s in at.status)
    assert {"Traza", "Evidencia enviada al modelo"} <= {e.label for e in at.expander}
    assert "prioridades" in textos_de_tablas(at)
    assert "descargar_consulta" in {b.key for b in at.download_button}
    assert any("requiere revisión humana" in c.value and registro.trace_id[:8] in c.value for c in at.caption)
    assert len(at.session_state["muestras"]) == 1
    # Paso 1: la misma consulta queda trazada en Observabilidad.
    abrir(at, "Observabilidad")
    assert not at.exception
    assert registro.trace_id[:8] in textos_de_tablas(at)


@pytest.mark.interfaz
def test_consulta_con_respaldo_muestra_insignia_naranja(at: AppTest) -> None:
    fabrica = FabricaClientes(selecciones=[], fragmentos=FRAGMENTOS_VERIFICABLES)
    consultar(at, fabrica)
    assert not at.exception
    assert at.session_state["respuesta_agente"].origen_seleccion.startswith("respaldo")
    assert ":orange-badge[Herramienta elegida por respaldo determinista]" in insignias(at)


@pytest.mark.interfaz
def test_pregunta_fuera_de_alcance_se_rechaza_sin_llamar_al_modelo(at: AppTest) -> None:
    fabrica = FabricaClientes(selecciones=[respuesta_con_llamadas(("resumen", {}))])
    consultar(at, fabrica, PREGUNTAS_SUGERIDAS[5])
    assert not at.exception
    assert at.session_state["respuesta_agente"].estado == "rechazada"
    assert any("No puedo hacerlo" in w.value for w in at.warning)
    assert all(not c.llamadas_chat and not c.llamadas_stream for c in fabrica.creados)


@pytest.mark.interfaz
def test_error_de_ollama_en_el_stream_muestra_error_sin_respuesta(at: AppTest) -> None:
    fabrica = FabricaClientes(selecciones=[respuesta_con_llamadas(("resumen", {}))],
                              error_stream=ErrorOllama("Ollama interrumpió la respuesta: prueba", tipo="stream"))
    consultar(at, fabrica)
    assert not at.exception
    assert any("interrumpió" in e.value for e in at.error)
    assert "respuesta_agente" not in at.session_state
    assert at.session_state["muestras"][0].estado == "error"


@pytest.mark.interfaz
def test_cambio_de_filtro_invalida_respuesta_del_agente(at: AppTest) -> None:
    fabrica = FabricaClientes(selecciones=[respuesta_con_llamadas(("prioridades", {}))],
                              fragmentos=FRAGMENTOS_VERIFICABLES)
    consultar(at, fabrica)
    assert "respuesta_agente" in at.session_state
    at.multiselect(key="filtro_prioridades").set_value(["1 - Critical"]).run()
    assert not at.exception
    assert "respuesta_agente" not in at.session_state


@pytest.mark.interfaz
def test_carga_datos_emite_un_span_por_carga_real(at: AppTest, tmp_path: Path) -> None:
    st.cache_resource.clear()
    at.run()
    at.run()
    assert not at.exception
    lineas = (tmp_path / "app.jsonl").read_text(encoding="utf-8").splitlines()
    cargas = [e for e in map(json.loads, lineas) if e["evento"] == "carga_datos"]
    assert len(cargas) == 1
    assert cargas[0]["atributos"]["estado"] == "ok"
    assert cargas[0]["atributos"]["eventos"] == 119998


# Paso 3: formato es-EC en todas las tablas, avisos claros y orden de la respuesta


@pytest.mark.interfaz
def test_vista_previa_muestra_numeros_y_fechas_en_formato_es_ec(at: AppTest) -> None:
    abrir(at, "Datos")
    assert not at.exception
    tablas = [t.value for t in at.dataframe if etiqueta("number") in t.value.columns]
    assert tablas, "La vista previa debe rotular sus columnas en español."
    primera = tablas[0].iloc[0]
    assert primera[etiqueta("number")] == "INC0000045"
    assert primera[etiqueta("opened_at")] == "29/02/2016 01:16"
    assert primera[etiqueta("reassignment_count")] == "0"
    assert primera[etiqueta("tiempo_resolucion_horas")] == "10,22"
    assert not [v for v in tablas[0].to_numpy().ravel() if re.fullmatch(r"-?\d+\.\d+", str(v))]


@pytest.mark.interfaz
def test_explorar_muestra_conteos_sin_decimales(at: AppTest) -> None:
    abrir(at, "Explorar")
    assert not at.exception
    tabla = next(t.value for t in at.dataframe if "Estadístico" in t.value.columns)
    valores = dict(zip(tabla["Estadístico"], tabla["Valor"]))
    assert valores["Válidos"] == "19.351"
    assert valores["Nulos"] == "1.418"
    assert valores["Mediana"] == "24,60"


@pytest.mark.interfaz
def test_observabilidad_sin_eventos_no_reporta_slos_incumplidos(at: AppTest) -> None:
    abrir(at, "Observabilidad")
    assert not at.exception
    assert metrica(at, "SLOs cumplidos") == "Sin dato"
    assert any("Use «Agente local» y vuelva aquí" in i.value for i in at.info)


@pytest.mark.interfaz
def test_ultimas_consultas_muestran_el_momento_hasta_los_segundos(at: AppTest, tmp_path: Path) -> None:
    evento = {"ts_utc": "2026-10-18T22:08:33.583+00:00", "nivel": "INFO", "evento": "consulta_iniciada",
              "trace_id": "abcdef0123456789", "atributos": {"modo": "automatico"}}
    (tmp_path / "app.jsonl").write_text(json.dumps(evento) + "\n", encoding="utf-8")
    abrir(at, "Observabilidad")
    assert not at.exception
    tabla = next(t.value for t in at.dataframe if "Momento (UTC)" in t.value.columns)
    assert tabla["Momento (UTC)"].tolist() == ["2026-10-18 22:08:33"]


@pytest.mark.interfaz
def test_csv_incluido_ausente_muestra_mensaje_en_espanol(at: AppTest) -> None:
    with patch("src.data_loader.firma_archivo", side_effect=FileNotFoundError(2, "No such file or directory")):
        at.run()
    assert not at.exception
    mensaje = " ".join(e.value for e in at.error)
    assert "No se encontró o no se pudo leer el CSV incluido" in mensaje
    assert "Cargar otro CSV compatible" in mensaje
    assert "No such file" not in mensaje


@pytest.mark.interfaz
def test_agente_muestra_el_reparto_python_modelo_persona(at: AppTest) -> None:
    abrir(at, "Agente local")
    assert not at.exception
    assert any("Python calcula; el modelo interpreta; la persona verifica" in c.value for c in at.caption)


@pytest.mark.interfaz
def test_respuesta_del_agente_aparece_bajo_su_encabezado_y_rotula_el_modo(at: AppTest) -> None:
    fabrica = FabricaClientes(selecciones=[respuesta_con_llamadas(("prioridades", {}))],
                              fragmentos=FRAGMENTOS_VERIFICABLES)
    consultar(at, fabrica)
    assert not at.exception
    orden = [str(n.value) for n in at.main if getattr(n, "type", None) in ("subheader", "markdown")]
    assert orden.count("Respuesta del agente") == 1
    texto = next(i for i, valor in enumerate(orden) if valor.startswith("**Hechos observados**"))
    assert orden.index("Respuesta del agente") < texto
    assert any("modo automático" in c.value for c in at.caption)
