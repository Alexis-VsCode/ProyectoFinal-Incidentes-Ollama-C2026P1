"""Interfaz del proyecto. Ejecutar: python -m streamlit run app.py"""
from hashlib import sha256
import json
from pathlib import Path

import streamlit as st

from src.agent import AgenteIncidentes, HERRAMIENTAS
from src.data_loader import cargar_csv
from src.eda import AnalisisIncidentes, DURACION, SIN_DATO, correlaciones, perfil_calidad
from src.llm_client import ClienteOllama, ErrorOllama, MODELOS
from src.visualizations import grafico_duracion, grafico_prioridades, grafico_variable

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Incidentes TI", page_icon="📊", layout="wide")
st.title("Análisis de incidentes TI con un agente local")
st.caption("Proyecto integrador · Kevin Alexis Barrera Llerena · Universidad Casa Grande")


@st.cache_data(show_spinner=False, max_entries=2)
def preparar(contenido: bytes):
    return cargar_csv(contenido)


def mostrar_horas(valor):
    return f"{valor:,.2f} h" if valor is not None else "Sin duración válida"


with st.sidebar:
    st.header("Datos y modelo")
    fuente = st.radio("Fuente del CSV", ["Dataset del Taller 1", "Cargar otro CSV compatible"])
    subido = st.file_uploader("CSV con el esquema de incidentes", type=["csv"]) if fuente.startswith("Cargar") else None
    modelo = st.selectbox("Modelo local", MODELOS)
    st.caption("Streamlit muestra y coordina. Python calcula. Ollama interpreta.")

try:
    if fuente.startswith("Cargar"):
        if subido is None:
            st.info("Cargue el CSV de Incident Response Log para continuar.")
            st.stop()
        contenido = subido.getvalue()
    else:
        contenido = (BASE / "data" / "incident_event_log.csv").read_bytes()
    with st.spinner("Cargando eventos y preparando incidentes..."):
        datos = preparar(contenido)
except (OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    opciones = sorted(datos.incidentes["priority"].fillna(SIN_DATO).unique().tolist())
    prioridades = st.multiselect("Prioridades a analizar", opciones, default=opciones)

analisis = AnalisisIncidentes(datos.incidentes, datos.control, prioridades)
firma = sha256((datos.control["sha256_csv"] + json.dumps(sorted(prioridades)) + modelo).encode()).hexdigest()
if st.session_state.get("firma") != firma:
    st.session_state.pop("respuesta_agente", None)
    st.session_state.pop("estado_ollama", None)
    st.session_state["firma"] = firma

st.info("La unidad de análisis es el incidente único. Las horas describen apertura–resolución; no miden por sí solas disponibilidad ni causa raíz.")
resumen_tab, explorar_tab, agente_tab, aprender_tab = st.tabs(["Resumen", "Explorar", "Agente local", "Cómo entenderlo"])

with resumen_tab:
    resumen = analisis.resumen()
    columnas = st.columns(4)
    columnas[0].metric("Eventos del archivo completo", f"{datos.control['eventos']:,}")
    columnas[1].metric("Incidentes seleccionados", f"{resumen['incidentes_seleccionados']:,}")
    columnas[2].metric("Mediana hasta resolución", mostrar_horas(resumen["mediana_horas"]))
    columnas[3].metric("Sin duración en la selección", resumen["incidentes_sin_duracion"])
    st.caption(f"Incidentes únicos del archivo: {datos.control['incidentes']:,}. Con duración válida en la selección: {resumen['duraciones_validas']:,}.")
    if analisis.datos.empty:
        st.warning("Seleccione alguna prioridad para obtener resultados.")
    else:
        st.plotly_chart(grafico_prioridades(analisis.prioridades_resumen()), width="stretch")
        st.dataframe(analisis.prioridades_resumen(), hide_index=True, width="stretch")
        st.plotly_chart(grafico_duracion(analisis.datos), width="stretch")
    with st.expander("Control del archivo y reglas de preparación"):
        st.json(datos.control)
        st.write("Se reconoce ? como nulo. Se conserva la última actualización por number; los desempates usan sys_mod_count y la posición original. Las duraciones negativas se excluyen solo del cálculo de horas.")
    st.download_button("Descargar incidentes seleccionados", analisis.datos.to_csv(index=False).encode("utf-8-sig"),
                       "incidentes_seleccionados.csv", "text/csv")

with explorar_tab:
    if not analisis.datos.empty:
        variable = st.selectbox("Variable", [DURACION, "reassignment_count", "reopen_count", "priority", "category", "assignment_group", "incident_state"])
        st.dataframe(analisis.explorar(variable), hide_index=True, width="stretch")
        st.plotly_chart(grafico_variable(analisis.datos, variable), width="stretch")
        st.caption("En variables categóricas el gráfico limita la vista a diez categorías; la tabla conserva todas.")
        st.write("Correlaciones descriptivas con la duración")
        st.dataframe(correlaciones(analisis.datos), hide_index=True)
        st.caption("Pearson usa pares no nulos. Una relación estadística no demuestra causa.")
    with st.expander("Calidad de los eventos originales"):
        st.dataframe(perfil_calidad(datos.eventos), hide_index=True, width="stretch")
    with st.expander("Vista de incidentes seleccionados"):
        st.dataframe(analisis.datos.head(100), hide_index=True, width="stretch")

with agente_tab:
    cliente = ClienteOllama(modelo)
    st.write("Elige qué función analítica consultará el agente y formula una pregunta. La evidencia visible permite revisar su respuesta.")
    if st.button("Verificar Ollama"):
        try:
            st.session_state["estado_ollama"] = cliente.modelos()
        except ErrorOllama as exc:
            st.session_state.pop("estado_ollama", None)
            st.error(str(exc))
    if "estado_ollama" in st.session_state:
        instalados = st.session_state["estado_ollama"]
        if modelo in [m.get("name") for m in instalados]:
            st.success(f"Servicio accesible y modelo {modelo} instalado.")
        else:
            st.warning(f"Falta el modelo seleccionado. Ejecute: ollama pull {modelo}")
        with st.expander("Modelos y versiones instalados"):
            st.json(instalados)
    with st.form("consulta_agente"):
        herramienta = st.selectbox("Herramienta autorizada", list(HERRAMIENTAS), format_func=HERRAMIENTAS.get)
        pregunta = st.text_area("Pregunta", "Resume los resultados y explica qué no se puede concluir.", max_chars=600)
        consultar = st.form_submit_button("Consultar al modelo local", disabled=analisis.datos.empty)
    if consultar:
        st.session_state.pop("respuesta_agente", None)
        try:
            with st.spinner("Ollama está generando una explicación local..."):
                st.session_state["respuesta_agente"] = AgenteIncidentes(analisis, cliente).responder(pregunta, herramienta)
        except (ErrorOllama, ValueError) as exc:
            st.error(str(exc))
            st.caption("El análisis estadístico sigue disponible. Esta consulta no generó una respuesta de IA.")
    if resultado := st.session_state.get("respuesta_agente"):
        st.subheader("Explicación generada por el modelo")
        st.markdown(resultado["respuesta"])
        st.caption(f"Modelo: {resultado['modelo']} · Tiempo de la consulta: {resultado['segundos']} s · Requiere revisión humana.")
        with st.expander("Evidencia exacta enviada al modelo", expanded=True):
            st.json(resultado["contexto"])
        st.download_button("Descargar consulta y evidencia", json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False),
                           "consulta_agente.json", "application/json")

with aprender_tab:
    st.subheader("Del notebook a una aplicación")
    st.write("Taller 1: las funciones reciben datos y devuelven resultados. Taller 2: las clases distribuyen responsabilidades. Taller 3: Strategy intercambia analizadores, Factory los crea y Facade reúne el flujo.")
    st.write("app.py contiene la interfaz. src/data_loader.py prepara datos. src/eda.py calcula. src/visualizations.py dibuja. src/agent.py coordina la consulta. src/llm_client.py conecta con Ollama.")
    st.write("Ejercicio: selecciona una prioridad, compara el número de incidentes y la mediana; luego revisa que el contexto del agente contenga exactamente esa selección.")
    st.write("Este CSV tiene registros históricos de atención. La propuesta anterior sobre métricas, logs y fallos de despliegue queda como una futura ampliación.")
