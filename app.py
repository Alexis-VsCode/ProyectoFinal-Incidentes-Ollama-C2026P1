"""Interfaz Streamlit del análisis de incidentes TI con un agente local de Ollama.

La interfaz solo muestra y coordina: los cálculos viven en src.eda, los gráficos en
src.visualizations, el agente en src.agent y las métricas de la propia aplicación en
src.observabilidad. Las horas hasta resolución del CSV son percentiles descriptivos del
histórico; no son SLO, SLA ni MTTR. Los números se muestran en formato es-EC.

Ejecutar: python -m streamlit run app.py

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import inspect
import json
import math
import numbers
import textwrap
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
import streamlit as st

from src import herramientas
from src.agent import HERRAMIENTAS, MAX_CARACTERES, PREGUNTAS_SUGERIDAS, AgenteIncidentes, RegistroConsulta
from src.data_loader import (
    SHA256_CSV_INCLUIDO, DatosIncidentes, cargar_csv, cargar_ruta, firma_archivo, verificar_integridad,
)
from src.diccionario import DURACION, MIN_MUESTRA, TIPOS, VARIABLES_EXPLORAR, VARIABLES_SEGMENTO, etiqueta
from src.eda import AnalisisIncidentes, AnalizadorFactory, comparar_prioridades, diccionario_con_calidad, perfil_calidad
from src.llm_client import MODELO_POR_DEFECTO, MODELOS, ClienteOllama, ErrorOllama
from src.observabilidad import (
    EVENTO_FIN, EVENTO_INICIO, EVENTO_OLLAMA_ERROR, EVENTO_RECHAZO, EVENTO_RESPALDO, SLOS, SPAN_EJECUCION,
    SPAN_INTERPRETACION, SPAN_SELECCION, MuestraConsulta, SLIs, calcular_slis, configurar_logger, evaluar_slos,
    leer_eventos, leer_muestras, nuevo_trace_id, ruta_log, span, tabla_slos, ultimas_consultas,
)
from src.visualizations import (
    figura_a_png, figura_boxplot_prioridad, figura_mapa_spearman, grafico_duracion, grafico_prioridades,
    grafico_segmentos, grafico_variable,
)

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

BASE: Final = Path(__file__).resolve().parent
RUTA_CSV: Final = BASE / "data" / "incident_event_log.csv"
RUTA_DIAGRAMA: Final = BASE / "assets" / "arquitectura.png"
PESTANAS: Final = ("Datos", "Resumen", "Explorar", "Relaciones y segmentos",
                   "Agente local", "Observabilidad", "Arquitectura")
N_CONSULTAS: Final = 10
SPAN_CARGA: Final = "carga_datos"
SIN_DATO_UI: Final = "Sin dato"
AVISO_SELECCION_VACIA: Final = "Seleccione al menos una prioridad para obtener resultados."
NOTA_DESCRIPTIVA: Final = ("Las horas hasta resolución son percentiles descriptivos del histórico; "
                           "no son SLO, SLA ni MTTR.")
AVISO_METRICAS_APP: Final = "Métricas de esta aplicación; no describen los incidentes del CSV."
LEMA_AGENTE: Final = "Python calcula; el modelo interpreta; la persona verifica."
MENSAJE_SIN_CSV_INCLUIDO: Final = (
    "No se encontró o no se pudo leer el CSV incluido en data/incident_event_log.csv. Restáurelo desde el "
    "repositorio o elija «Cargar otro CSV compatible» en la barra lateral."
)
AVISO_SIN_DIAGRAMA: Final = ("No se encontró assets/arquitectura.png. Regenérelo con: "
                             "python -m scripts.generar_arquitectura")
FORMATO_FECHA_ES: Final = "%d/%m/%Y %H:%M"
MODOS_UI: Final = {"automatico": "automático", "manual": "manual"}
ESTADISTICOS_CONTEO: Final = frozenset({"validos", "nulos"})
REGLAS_PREPARACION: Final = (
    "Se reconoce «?» como nulo y nunca se imputa. Se conserva la última actualización por number; los "
    "desempates usan sys_mod_count y la posición original. Las duraciones negativas se excluyen solo del "
    "cálculo de horas."
)
FUENTES: Final = {"incluido": "CSV incluido (Incident Response Log)", "subir": "Cargar otro CSV compatible"}
VENTANAS: Final = {"sesion": "Esta sesión", "registro": "Registro local completo"}
INSIGNIAS_INTEGRIDAD: Final = {
    "coincide": ("green", "SHA-256 coincide con el CSV incluido"),
    "difiere": ("red", "SHA-256 difiere del CSV incluido"),
    "no_aplica": ("gray", "SHA-256 no aplica (CSV cargado)"),
}
ORIGENES: Final = {
    "modelo": ("blue", "Herramienta elegida por el modelo"),
    "modelo_texto": ("blue", "Herramienta elegida por el modelo (llamada escrita como texto)"),
    "usuario": ("blue", "Herramienta elegida por la persona (modo manual)"),
    "respaldo_reglas": ("orange", "Herramienta elegida por respaldo determinista"),
    "respaldo_defecto": ("orange", "Herramienta elegida por respaldo determinista"),
}
ETIQUETAS_EVENTO: Final = {
    EVENTO_INICIO: "Consulta iniciada", EVENTO_RECHAZO: "Rechazo de la guarda",
    SPAN_SELECCION: "Selección de herramienta", EVENTO_RESPALDO: "Respaldo determinista",
    SPAN_EJECUCION: "Ejecución en Python", SPAN_INTERPRETACION: "Interpretación en streaming",
    EVENTO_OLLAMA_ERROR: "Error de Ollama", EVENTO_FIN: "Consulta finalizada",
}
COLUMNAS_TRAZA: Final = ("Paso", "Herramienta", "Argumentos", "Origen", "Estado", "ms")
ESTADISTICOS: Final = {
    "validos": "Válidos", "nulos": "Nulos", "media": "Media", "mediana": "Mediana",
    "desviacion_muestral": "Desviación estándar muestral", "minimo": "Mínimo", "maximo": "Máximo",
}
# Cada columna mostrada: nombre del contrato -> (rótulo en español, decimales, sufijo).
COLUMNAS_PRIORIDADES_ES: Final = {
    "priority": ("Prioridad", 0, ""), "incidentes": ("Incidentes", 0, ""),
    "porcentaje_incidentes": ("% de la selección", 2, " %"), "duraciones_validas": ("Duraciones válidas", 0, ""),
    "media_horas": ("Media (h)", 2, ""), "mediana_horas": ("Mediana (h)", 2, ""),
}
COLUMNAS_SEGMENTO_ES: Final = {
    "segmento": ("Segmento", 0, ""), "incidentes": ("Incidentes", 0, ""),
    "porcentaje_incidentes": ("% de la selección", 2, " %"), "duraciones_validas": ("Duraciones válidas", 0, ""),
    "mediana_horas": ("Mediana (h)", 2, ""), "p90_horas": ("P90 (h)", 2, ""),
    "porcentaje_sin_duracion": ("% sin duración", 2, " %"), "muestra_suficiente": ("Muestra suficiente", 0, ""),
}
COLUMNAS_CORRELACION_ES: Final = {
    "variable": ("Variable", 0, ""), "pearson_con_duracion": ("Pearson", 3, ""),
    "spearman_con_duracion": ("Spearman", 3, ""), "pares_validos": ("Pares válidos", 0, ""),
}
COLUMNAS_CATEGORICA_ES: Final = {
    "categoria": ("Categoría", 0, ""), "frecuencia": ("Incidentes", 0, ""),
    "porcentaje_sobre_total": ("% del total", 2, " %"),
}
COLUMNAS_DICCIONARIO_ES: Final = {
    "variable": ("Variable", 0, ""), "etiqueta": ("Etiqueta", 0, ""), "tipo_analitico": ("Tipo analítico", 0, ""),
    "descripcion": ("Descripción", 0, ""), "origen": ("Origen", 0, ""), "tipo": ("Tipo en pandas", 0, ""),
    "nulos": ("Nulos", 0, ""), "porcentaje_nulos": ("% nulos", 3, " %"), "valores_unicos": ("Valores únicos", 0, ""),
}
COLUMNAS_CALIDAD_ES: Final = {
    "variable": ("Variable", 0, ""), "tipo": ("Tipo en pandas", 0, ""), "nulos": ("Nulos", 0, ""),
    "porcentaje_nulos": ("% nulos", 3, " %"), "valores_unicos": ("Valores únicos", 0, ""),
}
# SLI mostrados: campo de SLIs -> rótulo y unidad ("s" segundos, "%" fracción, "" conteo o número).
SLIS_MOSTRADOS: Final = (
    ("consultas", "Consultas", ""), ("respondidas", "Respondidas", ""),
    ("rechazadas", "Rechazadas por la guarda", ""), ("errores", "Con error de Ollama", ""),
    ("latencia_p50_s", "Latencia p50", "s"), ("latencia_p95_s", "Latencia p95", "s"),
    ("primer_token_p50_s", "Primer token p50", "s"), ("primer_token_p95_s", "Primer token p95", "s"),
    ("tokens_por_segundo_p50", "Tokens por segundo p50", ""), ("tasa_error", "Tasa de error", "%"),
    ("tasa_truncadas", "Respuestas truncadas", "%"), ("tasa_seleccion_modelo", "Selección por el modelo", "%"),
    ("tasa_respaldo", "Selección por respaldo", "%"), ("proporcion_cifras_verificadas", "Cifras verificadas", "%"),
)
PATRONES: Final = (
    ("Strategy", ("src.eda.AnalizadorVariable", "src.eda.AnalizadorNumerico", "src.eda.AnalizadorCategorico"),
     "Las variables numéricas y las categóricas se resumen distinto, sin un if por tipo en la interfaz."),
    ("Factory", ("src.eda.AnalizadorFactory.crear",),
     "Concentra en un solo punto la decisión de qué estrategia usar; Explorar muestra cuál eligió."),
    ("Facade", ("src.eda.AnalisisIncidentes",),
     "La interfaz y el agente piden resultados por una sola puerta, sin conocer pandas."),
    ("Adapter", ("src.llm_client.ClienteOllama", "src.llm_client.ClienteChat"),
     "Aísla HTTP y errores de Ollama detrás de un protocolo (inversión de dependencias) y permite probar "
     "el agente con un doble."),
    ("Registro (lista blanca)", ("src.herramientas.RegistroHerramientas", "src.herramientas.REGISTRO"),
     "El modelo solo puede pedir herramientas de lectura registradas; es la fuente única para el agente "
     "y la interfaz."),
    ("Funcional", ("src.eda.segmentar", "src.eda.spearman", "src.eda.correlaciones",
                   "src.verificacion.verificar_cifras", "src.observabilidad.calcular_slis"),
     "Funciones puras que se reutilizan en la interfaz, el agente y las pruebas sin efectos secundarios."),
    ("Imperativo", ("src.data_loader.cargar_csv", "app.main"),
     "Validación y orquestación paso a paso, con control explícito de cada error."),
    ("POO", ("src.data_loader.DatosIncidentes", "src.agent.RegistroConsulta", "src.observabilidad.MuestraConsulta"),
     "Contratos de datos con estado: el archivo cargado, la evidencia de cada consulta y su muestra de SLI."),
    ("Eventos", ("app.usar_pregunta_sugerida", "app.invalidar_si_cambia"),
     "La interfaz reacciona a cada interacción: reruns, callbacks, pestañas con on_change y una firma "
     "que invalida respuestas de un contexto anterior."),
)
TEXTO_CICLO: Final = (
    "- Cada interacción vuelve a ejecutar `app.py` de arriba abajo (rerun).\n"
    "- `st.cache_resource` conserva los datos cargados: el CSV se lee una vez por ruta, tamaño y fecha de "
    "modificación, y esa carga real deja un span `carga_datos` en el registro.\n"
    "- Las pestañas siguen su estado (`on_change=\"rerun\"`) y solo se dibuja la abierta.\n"
    "- Los botones de preguntas sugeridas usan un callback (`on_click`) que rellena el formulario antes del rerun.\n"
    "- La firma SHA-256 de (CSV, prioridades, modelo) invalida la respuesta del agente cuando cambia el contexto."
)
EJERCICIO: Final = (
    "Con el CSV incluido, deje solo «1 - Critical» en la barra lateral: los indicadores pasan a 220 incidentes "
    "y 78,17 h de mediana. Luego consulte la segunda pregunta sugerida en «Agente local» y compruebe en "
    "«Evidencia enviada al modelo» que el filtro y las cifras coinciden con lo calculado."
)


# Paso 1: formato es-EC y ayudantes puros


def numero_es(valor: float, decimales: int = 2) -> str:
    """Número con punto de miles y coma decimal: 20769 -> "20.769"; 78.1667 con 2 -> "78,17"."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def _es_finito(valor: Any) -> bool:
    """True si el valor es un número real finito (no bool, no nulo)."""
    return isinstance(valor, numbers.Real) and not isinstance(valor, bool) and math.isfinite(float(valor))


def texto_es(valor: Any, decimales: int = 2) -> str:
    """Texto para mostrar: enteros con punto de miles, reales con `decimales` y coma decimal.

    Nulos, NaT y no finitos dan SIN_DATO_UI; los booleanos, «Sí» o «No»; las fechas, día/mes/año
    con hora y minuto ("29/02/2016 01:16"); el resto, str(valor).
    """
    if isinstance(valor, (bool, np.bool_)):
        return "Sí" if valor else "No"
    if (valor is None or valor is pd.NA or valor is pd.NaT
            or (isinstance(valor, numbers.Real) and not _es_finito(valor))):
        return SIN_DATO_UI
    if isinstance(valor, datetime):
        return valor.strftime(FORMATO_FECHA_ES)
    if isinstance(valor, numbers.Integral):
        return numero_es(int(valor), 0)
    if isinstance(valor, numbers.Real):
        return numero_es(float(valor), decimales)
    return str(valor)


def mostrar_horas(valor: float | None) -> str:
    """Horas con dos decimales en formato es-EC ("78,17 h"); None da "Sin duración válida"."""
    return "Sin duración válida" if valor is None else f"{numero_es(valor, 2)} h"


def porcentaje_es(fraccion: float | None, decimales: int = 1) -> str:
    """Fracción de 0 a 1 como porcentaje es-EC ("95,0 %"); nula o no finita da SIN_DATO_UI."""
    return f"{numero_es(float(fraccion) * 100, decimales)} %" if _es_finito(fraccion) else SIN_DATO_UI


def filas_es(tabla: pd.DataFrame | Iterable[dict[str, Any]],
             columnas: dict[str, tuple[str, int, str]]) -> list[dict[str, str]]:
    """Filas para st.dataframe con rótulos en español y valores en formato es-EC.

    `columnas` mapea cada columna del contrato a (rótulo, decimales, sufijo); el sufijo no se
    añade a un valor sin dato. Solo da formato: no calcula nada.
    """
    registros = tabla.to_dict("records") if isinstance(tabla, pd.DataFrame) else list(tabla)
    filas = []
    for registro in registros:
        fila = {}
        for nombre, (rotulo, decimales, sufijo) in columnas.items():
            texto = texto_es(registro.get(nombre), decimales)
            fila[rotulo] = texto + sufijo if sufijo and texto != SIN_DATO_UI else texto
        filas.append(fila)
    return filas


def columnas_vista_previa(columnas: Iterable[str]) -> dict[str, tuple[str, int, str]]:
    """Columnas para filas_es de la vista previa: etiqueta en español, dos decimales solo en las horas.

    Los contadores y el resto de columnas se muestran sin decimales.
    """
    return {columna: (etiqueta(columna), 2 if columna == DURACION else 0, "") for columna in columnas}


def firma_estado(sha256_csv: str, prioridades: Sequence[str], modelo: str) -> str:
    """SHA-256 hexadecimal de (CSV, prioridades ordenadas, modelo); el orden del filtro no la cambia."""
    contenido = json.dumps([sha256_csv, sorted(prioridades), modelo], ensure_ascii=False)
    return sha256(contenido.encode("utf-8")).hexdigest()


def insignia_cifras(verificacion: dict[str, Any]) -> tuple[str, str]:
    """(color, texto) de la insignia de cifras: verde si todas coinciden, naranja si es parcial, gris sin cifras."""
    evaluadas, verificadas = int(verificacion.get("evaluadas", 0)), int(verificacion.get("verificadas", 0))
    if evaluadas == 0:
        return "gray", "Sin cifras que verificar"
    return ("green" if verificadas == evaluadas else "orange"), f"Cifras verificadas: {verificadas}/{evaluadas}"


def _texto_o_vacio(valor: Any) -> str:
    """str(valor), o "" si es None o un número no finito."""
    if valor is None or (isinstance(valor, numbers.Real) and not isinstance(valor, bool) and not _es_finito(valor)):
        return ""
    return str(valor)


def momento_utc(valor: Any) -> str:
    """Marca ISO 8601 en UTC recortada a segundos y sin la «T» ("2026-10-18 22:08:33"); "" si no hay valor."""
    return _texto_o_vacio(valor)[:19].replace("T", " ")


def filas_traza(pasos: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Una fila por evento de la traza del agente, con COLUMNAS_TRAZA.

    Muestra la herramienta, sus argumentos validados, el origen de la selección, el estado o
    motivo y los milisegundos del span (o los segundos totales de `consulta_finalizada`).
    """
    filas = []
    for evento in pasos:
        atributos = evento.get("atributos") if isinstance(evento.get("atributos"), dict) else {}
        argumentos = atributos.get("argumentos")
        ms = atributos.get("ms")
        if ms is None and _es_finito(atributos.get("segundos")):
            ms = float(atributos["segundos"]) * 1000
        filas.append(dict(zip(COLUMNAS_TRAZA, (
            ETIQUETAS_EVENTO.get(evento.get("evento"), str(evento.get("evento"))),
            _texto_o_vacio(atributos.get("herramienta")),
            json.dumps(argumentos, ensure_ascii=False) if argumentos else "",
            _texto_o_vacio(atributos.get("origen") or atributos.get("origen_seleccion")),
            _texto_o_vacio(atributos.get("estado") or atributos.get("motivo") or atributos.get("categoria")),
            texto_es(ms, 1) if _es_finito(ms) else "",
        ))))
    return filas


def fuente_registro() -> str:
    """Código de la definición de REGISTRO en src.herramientas leído con inspect.

    Si el bloque `REGISTRO = ...` no se ubica, devuelve el código de RegistroHerramientas.
    """
    lineas, _ = inspect.getsourcelines(herramientas)
    inicio = next((i for i, linea in enumerate(lineas) if linea.startswith("REGISTRO = ")), None)
    fin = None if inicio is None else next((i for i in range(inicio, len(lineas)) if lineas[i].rstrip() == "])"), None)
    if inicio is None or fin is None:
        return inspect.getsource(herramientas.RegistroHerramientas)
    return "".join(lineas[inicio:fin + 1])


# Paso 2: carga cacheada con su span y estado de la sesión


def _cargar_con_span(cargar: Callable[[], DatosIncidentes], fuente: str, tamano: int,
                     esperado: str | None) -> tuple[DatosIncidentes, dict[str, Any]]:
    """Ejecuta `cargar` dentro del span `carga_datos` y devuelve los datos con su medición.

    La medición tiene duracion_ms, trace_id, momento_utc, fuente e integridad (verificar_integridad
    contra `esperado`). Si la carga falla, el span queda en error y la excepción se propaga.
    """
    momento = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with span(SPAN_CARGA, nuevo_trace_id(), fuente=fuente, bytes=tamano) as tramo:
        datos = cargar()
        tramo.atributos.update(eventos=datos.control["eventos"], incidentes=datos.control["incidentes"])
    medicion = {"duracion_ms": tramo.ms, "trace_id": tramo.trace_id, "momento_utc": momento, "fuente": fuente,
                "integridad": verificar_integridad(datos.control, esperado)}
    return datos, medicion


@st.cache_resource(max_entries=1, show_spinner="Preparando incidentes (solo la primera vez)…")
def datos_incluidos(ruta: str, tamano: int, mtime_ns: int) -> tuple[DatosIncidentes, dict[str, Any]]:
    """CSV incluido, cargado una vez por (ruta, tamaño, mtime) y comparado con SHA256_CSV_INCLUIDO."""
    return _cargar_con_span(lambda: cargar_ruta(Path(ruta)), "incluido", tamano, SHA256_CSV_INCLUIDO)


@st.cache_resource(max_entries=1, scope="session", show_spinner="Preparando el CSV cargado…")
def datos_subidos(huella: str, _contenido: bytes) -> tuple[DatosIncidentes, dict[str, Any]]:
    """CSV subido, cargado una vez por huella "file_id:tamaño"; su integridad no aplica."""
    return _cargar_con_span(lambda: cargar_csv(_contenido), "subido", len(_contenido), None)


@st.cache_resource(max_entries=2, show_spinner=False)
def calidad_eventos(sha256_csv: str, _eventos: pd.DataFrame) -> pd.DataFrame:
    """perfil_calidad de los eventos originales, calculado una vez por archivo."""
    return perfil_calidad(_eventos)


@st.cache_data(ttl=10, show_spinner=False)
def salud_ollama(modelo: str) -> dict[str, Any]:
    """ClienteOllama(modelo).estado(), como mucho una sonda cada 10 s; nunca lanza."""
    return ClienteOllama(modelo).estado()


def configurar_barra_lateral() -> tuple[str, UploadedFile | None, str]:
    """Fuente del CSV, archivo subido (o None) y modelo elegido de la lista blanca."""
    with st.sidebar:
        st.header("Datos y modelo")
        fuente = st.radio("Fuente del CSV", tuple(FUENTES), format_func=FUENTES.get, key="fuente")
        subido = (st.file_uploader("CSV con el esquema de incidentes", type=["csv"], key="csv_subido")
                  if fuente == "subir" else None)
        modelo = st.selectbox("Modelo local", MODELOS, index=MODELOS.index(MODELO_POR_DEFECTO), key="modelo",
                              help="Lista blanca del cliente de Ollama.")
        st.caption("Streamlit muestra y coordina. Python calcula. Ollama interpreta.")
    return fuente, subido, modelo


def obtener_datos(fuente: str, subido: UploadedFile | None) -> tuple[DatosIncidentes, dict[str, Any]]:
    """Datos y medición de la fuente elegida; ante OSError o ValueError muestra el error y detiene la ejecución.

    Si el CSV incluido falta o no se puede leer, muestra MENSAJE_SIN_CSV_INCLUIDO en lugar del
    mensaje del sistema operativo; los ValueError de la carga ya vienen en español.
    """
    try:
        if fuente == "subir":
            if subido is None:
                st.info("Cargue un CSV con el esquema de Incident Response Log para continuar.")
                st.stop()
            return datos_subidos(f"{subido.file_id}:{subido.size}", subido.getvalue())
        return datos_incluidos(*firma_archivo(RUTA_CSV))
    except (OSError, ValueError) as exc:
        st.error(MENSAJE_SIN_CSV_INCLUIDO if isinstance(exc, OSError) and fuente != "subir" else str(exc))
        st.stop()
        raise


def elegir_prioridades(datos: DatosIncidentes) -> list[str]:
    """Multiselect de prioridades (clave filtro_prioridades); al inicio, y si cambia el CSV, todas."""
    opciones = comparar_prioridades(datos.incidentes)["priority"].tolist()
    actual = st.session_state.get("filtro_prioridades")
    if actual is None or any(p not in opciones for p in actual):
        validas = [p for p in actual if p in opciones] if actual else []
        st.session_state["filtro_prioridades"] = validas or list(opciones)
    return st.sidebar.multiselect("Prioridades a analizar", opciones, key="filtro_prioridades")


def invalidar_si_cambia(firma: str) -> None:
    """Si la firma de (CSV, prioridades, modelo) cambió, borra respuesta_agente y estado_ollama."""
    if st.session_state.get("firma") != firma:
        for clave in ("respuesta_agente", "estado_ollama"):
            st.session_state.pop(clave, None)
        st.session_state["firma"] = firma


def usar_pregunta_sugerida(pregunta: str) -> None:
    """Callback on_click: copia la pregunta sugerida al formulario del agente."""
    st.session_state["pregunta_agente"] = pregunta


def acumular_muestra(muestra: MuestraConsulta) -> None:
    """Observador al_finalizar del agente: guarda cada muestra (también errores) para los SLI de la sesión."""
    st.session_state.setdefault("muestras", []).append(muestra)


# Paso 3: franja de indicadores y pestañas de análisis


def render_kpis(datos: DatosIncidentes, analisis: AnalisisIncidentes) -> None:
    """Cuatro métricas siempre visibles y un aviso si la selección está vacía."""
    resumen = analisis.resumen()
    columnas = st.columns(4)
    columnas[0].metric("Eventos del archivo completo", numero_es(datos.control["eventos"], 0))
    columnas[1].metric("Incidentes seleccionados", numero_es(resumen["incidentes_seleccionados"], 0))
    columnas[2].metric("Mediana hasta resolución", mostrar_horas(resumen["mediana_horas"]))
    columnas[3].metric("Sin duración en la selección", numero_es(resumen["incidentes_sin_duracion"], 0))
    st.caption(f"Incidentes únicos del archivo: {numero_es(datos.control['incidentes'], 0)}. Con duración válida "
               f"en la selección: {numero_es(resumen['duraciones_validas'], 0)}. La unidad de análisis es el "
               "incidente único.")
    if analisis.datos.empty:
        st.warning(AVISO_SELECCION_VACIA)


def render_datos(datos: DatosIncidentes, analisis: AnalisisIncidentes, medicion: dict[str, Any]) -> None:
    """Control e integridad del archivo, diccionario con calidad, calidad de eventos y vista previa."""
    control = datos.control
    st.subheader("Control del archivo")
    columnas = st.columns(4)
    columnas[0].metric("Eventos", numero_es(control["eventos"], 0))
    columnas[1].metric("Columnas originales", numero_es(control["columnas_originales"], 0))
    columnas[2].metric("Incidentes únicos", numero_es(control["incidentes"], 0))
    columnas[3].metric("Sin duración (archivo)", numero_es(control["incidentes_sin_duracion"], 0))
    color, texto = INSIGNIAS_INTEGRIDAD[medicion["integridad"]["estado"]]
    st.badge(texto, color=color, help=f"SHA-256 calculado al cargar: {control['sha256_csv']}")
    st.caption(f"SHA-256 calculado al cargar: {control['sha256_csv'][:12]}… · {REGLAS_PREPARACION}")
    with st.expander("Control completo del archivo (JSON)"):
        st.json(control)
    st.subheader("Diccionario de variables y calidad")
    diccionario = diccionario_con_calidad(datos.incidentes)
    st.dataframe(filas_es(diccionario, COLUMNAS_DICCIONARIO_ES), hide_index=True, width="stretch")
    st.download_button("Descargar diccionario (CSV)", data=lambda: diccionario.to_csv(index=False).encode("utf-8-sig"),
                       file_name="diccionario_variables.csv", mime="text/csv", key="descargar_diccionario",
                       on_click="ignore")
    with st.expander("Calidad de los eventos originales"):
        st.dataframe(filas_es(calidad_eventos(control["sha256_csv"], datos.eventos), COLUMNAS_CALIDAD_ES),
                     hide_index=True, width="stretch")
    st.subheader("Vista previa de la selección")
    st.caption("Primeras 100 filas de los incidentes seleccionados.")
    st.dataframe(filas_es(analisis.datos.head(100), columnas_vista_previa(analisis.datos.columns)),
                 hide_index=True, width="stretch")
    st.download_button("Descargar incidentes seleccionados (CSV)",
                       data=lambda: analisis.datos.to_csv(index=False).encode("utf-8-sig"),
                       file_name="incidentes_seleccionados.csv", mime="text/csv", key="descargar_incidentes",
                       on_click="ignore", disabled=analisis.datos.empty)


def render_resumen(analisis: AnalisisIncidentes) -> None:
    """Media, p90 y duraciones válidas; incidentes por prioridad y ECDF de las horas, rotulada como descriptiva."""
    if analisis.datos.empty:
        st.info(AVISO_SELECCION_VACIA)
        return
    resumen = analisis.resumen()
    columnas = st.columns(3)
    columnas[0].metric("Media hasta resolución", mostrar_horas(resumen["media_horas"]))
    columnas[1].metric("Percentil 90", mostrar_horas(resumen["p90_horas"]))
    columnas[2].metric("Duraciones válidas", numero_es(resumen["duraciones_validas"], 0))
    if (resumen["media_horas"] or 0) > (resumen["mediana_horas"] or 0):
        st.caption("La media supera a la mediana: pocos incidentes muy largos elevan el promedio, por eso la "
                   "mediana describe mejor el caso típico.")
    tabla = analisis.prioridades_resumen()
    st.plotly_chart(grafico_prioridades(tabla), width="stretch")
    st.dataframe(filas_es(tabla, COLUMNAS_PRIORIDADES_ES), hide_index=True, width="stretch")
    st.plotly_chart(grafico_duracion(analisis.datos), width="stretch")
    st.caption(NOTA_DESCRIPTIVA)


def render_explorar(analisis: AnalisisIncidentes) -> None:
    """Una variable a la vez: tipo analítico, Strategy elegida por la Factory, tabla y gráfico."""
    if analisis.datos.empty:
        st.info(AVISO_SELECCION_VACIA)
        return
    variable = st.selectbox("Variable", VARIABLES_EXPLORAR, format_func=etiqueta, key="variable_explorar",
                            persist_state="session")
    st.info(f"Tipo analítico: {TIPOS.get(variable, 'sin clasificar')}. Estrategia elegida por "
            f"AnalizadorFactory: {analisis.estrategia(variable)}.")
    tabla = analisis.explorar(variable)
    if "estadistico" in tabla.columns:
        # Paso 1: los conteos (válidos y nulos) se muestran sin decimales; el resto, con dos.
        filas = [{"Estadístico": ESTADISTICOS.get(f["estadistico"], f["estadistico"]),
                  "Valor": texto_es(f["valor"], 0 if f["estadistico"] in ESTADISTICOS_CONTEO else 2)}
                 for f in tabla.to_dict("records")]
    else:
        filas = filas_es(tabla, COLUMNAS_CATEGORICA_ES)
    st.dataframe(filas, hide_index=True, width="stretch")
    st.plotly_chart(grafico_variable(analisis.datos, variable), width="stretch")
    st.caption("En variables categóricas el gráfico muestra las diez categorías más frecuentes; "
               "la tabla conserva todas.")


def render_relaciones(analisis: AnalisisIncidentes) -> None:
    """Boxplot y mapa de Spearman (Seaborn) con descarga PNG, segmentos y correlaciones."""
    if analisis.datos.empty:
        st.info(AVISO_SELECCION_VACIA)
        return
    # Paso 1: distribución de horas por prioridad.
    st.subheader("Horas hasta resolución por prioridad")
    boxplot = figura_boxplot_prioridad(analisis.datos)
    st.pyplot(boxplot, width="stretch")
    st.download_button("Descargar boxplot (PNG)", data=lambda: figura_a_png(boxplot), file_name="boxplot_prioridad.png",
                       mime="image/png", key="descargar_boxplot", on_click="ignore")
    st.caption(NOTA_DESCRIPTIVA)
    # Paso 2: segmentos por categoría o grupo de asignación.
    st.subheader("Segmentos")
    izquierda, derecha = st.columns(2)
    variable = izquierda.segmented_control(
        "Segmentar por", VARIABLES_SEGMENTO, format_func=etiqueta, default="category", required=True,
        key="variable_segmento", persist_state="session") or "category"
    top = derecha.slider("Segmentos a mostrar", 5, 20, 10, key="top_segmentos", persist_state="session")
    segmentos = analisis.segmentar(variable, top)
    st.dataframe(filas_es(segmentos, COLUMNAS_SEGMENTO_ES), hide_index=True, width="stretch")
    st.plotly_chart(grafico_segmentos(segmentos, variable), width="stretch")
    st.caption(f"Con menos de {MIN_MUESTRA} duraciones válidas la mediana del segmento es solo indicativa. "
               "El resto de segmentos se agrupa en una sola fila.")
    # Paso 3: correlaciones de los contadores con las horas.
    st.subheader("Correlaciones con las horas hasta resolución")
    relaciones = [{**r, "variable": etiqueta(r["variable"])} for r in analisis.relaciones()]
    st.dataframe(filas_es(relaciones, COLUMNAS_CORRELACION_ES), hide_index=True, width="stretch")
    mapa = figura_mapa_spearman(analisis.matriz_relaciones())
    st.pyplot(mapa, width="stretch")
    st.download_button("Descargar mapa de Spearman (PNG)", data=lambda: figura_a_png(mapa),
                       file_name="mapa_spearman.png", mime="image/png", key="descargar_mapa", on_click="ignore")
    st.caption("Una correlación describe asociación, no causa. Las modificaciones del registro crecen "
               "mecánicamente mientras el incidente sigue abierto, y un incidente largo acumula reasignaciones.")


# Paso 4: agente local


def _render_verificacion(modelo: str) -> None:
    """Botón Verificar Ollama: comprueba servicio y modelo, lo precalienta y guarda estado_ollama."""
    error = None
    if st.button("Verificar Ollama", key="btn_verificar",
                 help="Comprueba que Ollama responda y que el modelo esté instalado, y lo precalienta en "
                      "memoria para que la primera consulta no espere la carga."):
        with st.status(f"Verificando Ollama y precalentando {modelo}…", expanded=False) as estado:
            try:
                cliente = ClienteOllama(modelo)
                resultado = cliente.verificar(precalentar=True)
                resultado["version"] = cliente.version()
                st.session_state["estado_ollama"] = resultado
                estado.update(label="Ollama verificado", state="complete")
            except ErrorOllama as exc:
                st.session_state.pop("estado_ollama", None)
                estado.update(label="Ollama no respondió", state="error")
                error = str(exc)
    if error is not None:
        st.error(error)
        st.caption("El análisis estadístico sigue disponible; solo el agente necesita Ollama.")
    resultado = st.session_state.get("estado_ollama")
    if not resultado:
        return
    if resultado["modelo_instalado"]:
        digest = (resultado.get("digest") or "")[:12] or SIN_DATO_UI
        st.success(f"Ollama {resultado.get('version') or SIN_DATO_UI} accesible · modelo {modelo} · digest {digest}")
        if resultado.get("precalentado"):
            st.badge("Modelo en memoria", color="green")
            st.caption(f"Precalentado en {texto_es(resultado.get('segundos_precalentamiento'), 1)} s.")
    else:
        st.warning(f"Falta el modelo seleccionado. Ejecute: ollama pull {modelo}")
    with st.expander("Modelos instalados"):
        st.write(", ".join(str(m) for m in resultado.get("modelos", [])) or SIN_DATO_UI)


def _consultar(analisis: AnalisisIncidentes, modelo: str, pregunta: str, herramienta: str | None,
               argumentos: dict[str, Any] | None) -> bool:
    """Prepara la consulta, muestra la traza, transmite la interpretación y guarda el RegistroConsulta.

    Devuelve True si el texto se transmitió en esta ejecución. Un rechazo de la guarda no llama
    al modelo. Ante ErrorOllama o ValueError muestra st.error, borra el texto parcial y no guarda
    ninguna respuesta: nunca se simula una respuesta.
    """
    # Paso 1: una consulta nueva borra la anterior para no mostrar una respuesta vieja como nueva.
    st.session_state.pop("respuesta_agente", None)
    agente = AgenteIncidentes(analisis, ClienteOllama(modelo), al_finalizar=acumular_muestra)
    zona = None
    try:
        consulta = agente.preparar(pregunta, herramienta, modo="manual" if herramienta else "automatico",
                                   argumentos=argumentos)
        if consulta.rechazada:
            # Paso 2: el texto fijo del rechazo se consume sin llamar al modelo.
            for _ in agente.interpretar(consulta):
                pass
            st.session_state["respuesta_agente"] = agente.finalizar(consulta)
            return False
        # Paso 3: traza de la selección y de la ejecución antes de interpretar.
        with st.status("Herramientas ejecutadas", expanded=True) as estado:
            st.dataframe(filas_traza(consulta.pasos), hide_index=True, width="stretch")
            estado.update(label="Herramientas ejecutadas: " + ", ".join(e.nombre for e in consulta.ejecuciones),
                          state="complete", expanded=False)
        # Paso 4: interpretación en streaming bajo su encabezado y cierre con verificación de cifras.
        zona = st.empty()
        with zona.container():
            st.subheader("Respuesta del agente")
            st.write_stream(agente.interpretar(consulta))
        st.session_state["respuesta_agente"] = agente.finalizar(consulta)
        return True
    except (ErrorOllama, ValueError) as exc:
        if zona is not None:
            zona.empty()
        st.error(str(exc))
        st.caption("El análisis estadístico sigue disponible. Esta consulta no generó una respuesta del modelo.")
        return False


def _render_resultado_agente(registro: RegistroConsulta, mostrar_texto: bool) -> None:
    """Respuesta persistente con insignias, nota de revisión humana, traza, evidencia y descarga JSON.

    Con `mostrar_texto` en False el encabezado y el texto ya se transmitieron en esta ejecución.
    """
    if mostrar_texto:
        st.subheader("Respuesta del agente")
    if registro.estado == "rechazada":
        st.warning(registro.respuesta)
        st.badge("Rechazada por la guarda: no se llamó al modelo", color="gray")
    else:
        if mostrar_texto:
            st.markdown(registro.respuesta)
        color, texto = insignia_cifras(registro.verificacion_cifras)
        st.badge(texto, color=color, help="Cada cifra de la respuesta se busca en la evidencia enviada al modelo.")
        if registro.origen_seleccion in ORIGENES:
            color_origen, texto_origen = ORIGENES[registro.origen_seleccion]
            st.badge(texto_origen, color=color_origen)
        no_verificadas = registro.verificacion_cifras.get("no_verificadas") or []
        if no_verificadas:
            with st.expander("Cifras no encontradas en la evidencia"):
                st.write(", ".join(str(c.get("texto")) for c in no_verificadas))
    st.caption(f"Modelo {registro.modelo} · {numero_es(registro.segundos, 1)} s · "
               f"modo {MODOS_UI.get(registro.modo, registro.modo)} · trace {registro.trace_id[:8]} · "
               "requiere revisión humana.")
    st.caption(registro.revision_humana)
    with st.expander("Traza"):
        st.dataframe(filas_traza(registro.traza), hide_index=True, width="stretch")
    with st.expander("Evidencia enviada al modelo"):
        st.json(registro.contexto)
    st.download_button("Descargar consulta y evidencia (JSON)", data=registro.a_json,
                       file_name=f"consulta_{registro.trace_id[:8]}.json", mime="application/json",
                       key="descargar_consulta", on_click="ignore")


def render_agente(analisis: AnalisisIncidentes, modelo: str) -> None:
    """Verificación de Ollama, modo, preguntas sugeridas, formulario de consulta y resultado."""
    st.subheader("Agente local")
    st.caption(f"{LEMA_AGENTE} Herramientas autorizadas (solo lectura): " + ", ".join(HERRAMIENTAS.values()) + ".")
    _render_verificacion(modelo)
    st.divider()
    # Paso 1: modo de selección de herramientas.
    automatico = st.toggle("Modo automático", value=True, key="modo_automatico", persist_state="session",
                           help="Activado: el modelo elige la herramienta y Python la valida. "
                                "Desactivado: la elige usted.")
    herramienta, argumentos = None, None
    if not automatico:
        herramienta = st.selectbox("Herramienta autorizada", tuple(HERRAMIENTAS), format_func=HERRAMIENTAS.get,
                                   key="herramienta_manual", persist_state="session")
        if herramienta == "segmentar":
            izquierda, derecha = st.columns(2)
            argumentos = {
                "variable": izquierda.selectbox("Segmentar por", VARIABLES_SEGMENTO, format_func=etiqueta,
                                                key="segmento_manual", persist_state="session"),
                "top": derecha.slider("Segmentos", 3, 10, 5, key="top_manual", persist_state="session"),
            }
    # Paso 2: preguntas sugeridas que rellenan el formulario por callback.
    st.write("Preguntas sugeridas")
    columnas = st.columns(2)
    for i, sugerida in enumerate(PREGUNTAS_SUGERIDAS):
        columnas[i % 2].button(sugerida, key=f"sugerida_{i}", on_click=usar_pregunta_sugerida, args=(sugerida,),
                               width="stretch")
    st.session_state.setdefault("pregunta_agente", PREGUNTAS_SUGERIDAS[0])
    with st.form("consulta_agente"):
        pregunta = st.text_area("Pregunta", key="pregunta_agente", max_chars=MAX_CARACTERES, persist_state="session")
        enviar = st.form_submit_button("Consultar al modelo local", type="primary", key="consultar_agente",
                                       disabled=analisis.datos.empty)
    # Paso 3: consulta y resultado persistente.
    transmitida = _consultar(analisis, modelo, pregunta or "", herramienta, argumentos) if enviar else False
    registro = st.session_state.get("respuesta_agente")
    if registro is not None:
        _render_resultado_agente(registro, mostrar_texto=not transmitida)


# Paso 5: observabilidad de la aplicación y arquitectura


def _filas_sli(slis: SLIs) -> list[dict[str, str]]:
    """Filas SLI y valor para mostrar, según SLIS_MOSTRADOS."""
    filas = []
    for campo, rotulo, unidad in SLIS_MOSTRADOS:
        valor = getattr(slis, campo)
        if unidad == "%":
            texto = porcentaje_es(valor)
        elif unidad == "s" and _es_finito(valor):
            texto = f"{numero_es(valor, 2)} s"
        else:
            texto = texto_es(valor, 2)
        filas.append({"SLI": rotulo, "Valor": texto})
    return filas


def _filas_slo(tabla: pd.DataFrame) -> list[dict[str, str]]:
    """Filas de tabla_slos con rótulos en español y porcentajes es-EC."""
    return [{"SLO": f["slo"], "Descripción": f["descripcion"], "Objetivo": porcentaje_es(f["objetivo"], 0),
             "Eventos válidos": texto_es(f["eventos_validos"]), "Eventos buenos": texto_es(f["eventos_buenos"]),
             "Cumplimiento": porcentaje_es(f["cumplimiento"]), "Resultado": f["resultado"],
             "Presupuesto de error restante": porcentaje_es(f["fraccion_restante"])}
            for f in tabla.to_dict("records")]


def _render_salud(medicion: dict[str, Any], modelo: str) -> None:
    """Estado de Ollama (sonda cacheada), integridad del CSV y tiempo de carga."""
    st.button("Actualizar estado", key="actualizar_salud", on_click=salud_ollama.clear,
              help="La sonda se guarda 10 s para no consultar Ollama en cada interacción.")
    salud = salud_ollama(modelo)
    ollama, integridad, carga = st.columns(3)
    with ollama:
        st.metric("Ollama", salud.get("version") or "No accesible")
        if not salud.get("accesible"):
            st.badge("Servicio no accesible", color="red")
        elif salud.get("modelo_cargado"):
            st.badge("Modelo cargado en memoria", color="green")
        else:
            st.badge("Modelo no cargado: use Verificar Ollama", color="orange")
        detalle = [f"Modelo {salud.get('modelo', modelo)}"]
        if salud.get("expira_utc"):
            detalle.append(f"expira {salud['expira_utc']}")
        if _es_finito(salud.get("latencia_ms")):
            detalle.append(f"sonda {numero_es(salud['latencia_ms'], 1)} ms")
        if salud.get("error"):
            detalle.append(str(salud["error"]))
        st.caption(" · ".join(detalle))
    with integridad:
        estado = medicion["integridad"]
        color, texto = INSIGNIAS_INTEGRIDAD[estado["estado"]]
        st.markdown("**Integridad del CSV**")
        st.badge(texto, color=color, help=f"Calculado: {estado['calculado']}")
        st.caption(f"Esperado {(estado['esperado'] or SIN_DATO_UI)[:12]} · calculado {str(estado['calculado'])[:12]}")
    with carga:
        st.metric("Carga de datos", f"{numero_es((medicion['duracion_ms'] or 0) / 1000, 1)} s")
        st.caption(f"Medida al llenar la caché · trace {medicion['trace_id'][:8]} · {medicion['momento_utc']}")


def _render_slos() -> None:
    """SLI y SLO de la sesión o del registro local, con su presupuesto de error."""
    ventana = st.segmented_control("Ventana", tuple(VENTANAS), format_func=VENTANAS.get, default="sesion",
                                   required=True, key="ventana_slo", persist_state="session") or "sesion"
    muestras = list(st.session_state.get("muestras", [])) if ventana == "sesion" else leer_muestras()[0]
    slis = calcular_slis(muestras)
    evaluaciones = evaluar_slos(muestras)
    restantes = [e.fraccion_restante for e in evaluaciones if e.fraccion_restante is not None]
    # Paso 1: un SLO sin eventos no cuenta como incumplido; solo se cuentan los evaluados.
    evaluados = [e for e in evaluaciones if e.cumple is not None]
    columnas = st.columns(4)
    columnas[0].metric("SLOs cumplidos", f"{sum(1 for e in evaluados if e.cumple)}/{len(evaluados)}"
                       if evaluados else SIN_DATO_UI)
    columnas[1].metric("Consultas registradas", numero_es(slis.consultas, 0))
    columnas[2].metric("Latencia p95", f"{numero_es(slis.latencia_p95_s, 1)} s"
                       if _es_finito(slis.latencia_p95_s) else SIN_DATO_UI)
    columnas[3].metric("Presupuesto de error mínimo", porcentaje_es(min(restantes)) if restantes else SIN_DATO_UI)
    if not muestras:
        st.info("Sin datos: aún no hay consultas en esta ventana, así que cada SLO queda «sin eventos».")
    elif any(not e.suficiente for e in evaluaciones):
        st.caption(f"Muestras insuficientes: con menos de {min(s.minimo_eventos for s in SLOS)} eventos válidos "
                   "el resultado es indicativo, no concluyente.")
    st.dataframe(_filas_slo(tabla_slos(evaluaciones)), hide_index=True, width="stretch")
    with st.expander("Todos los SLI de la ventana"):
        st.dataframe(_filas_sli(slis), hide_index=True, width="stretch")


def _render_ultimas_consultas() -> None:
    """Últimas consultas del registro local (trace_id, evento, ms) y descarga del JSONL."""
    eventos = [e for e in leer_eventos() if e.get("evento") != SPAN_CARGA]
    tabla = ultimas_consultas(eventos, N_CONSULTAS)
    if tabla.empty:
        st.info("Aún no hay consultas registradas. Use «Agente local» y vuelva aquí.")
    else:
        st.dataframe([{"Trace": str(f["trace_id"])[:8], "Momento (UTC)": momento_utc(f["ts_utc"]),
                       "Evento": ETIQUETAS_EVENTO.get(f["evento"], _texto_o_vacio(f["evento"])),
                       "ms": texto_es(f["ms"], 1) if _es_finito(f["ms"]) else "",
                       "Estado": _texto_o_vacio(f["estado"])} for f in tabla.to_dict("records")],
                     hide_index=True, width="stretch")
    ruta = ruta_log()
    existe = ruta.exists()
    st.download_button("Descargar registro (JSONL)", data=ruta.read_bytes if existe else b"", file_name=ruta.name,
                       mime="application/x-ndjson", key="descargar_logs", on_click="ignore", disabled=not existe)
    st.caption(f"Registro local: {ruta}. Guarda longitudes, huellas y métricas; nunca el texto de la pregunta "
               "ni de la respuesta.")


def render_observabilidad(medicion: dict[str, Any], modelo: str) -> None:
    """Salud, SLI frente a SLO y últimas consultas; solo métricas de la aplicación, nunca del CSV."""
    st.subheader("Observabilidad de la aplicación")
    st.warning(AVISO_METRICAS_APP)
    _render_salud(medicion, modelo)
    st.subheader("SLI y SLO")
    _render_slos()
    st.subheader("Últimas consultas")
    _render_ultimas_consultas()


def render_arquitectura() -> None:
    """Diagrama, tabla de paradigmas y patrones con el problema que resuelven y evidencia del código."""
    st.subheader("Arquitectura")
    if RUTA_DIAGRAMA.exists():
        st.image(str(RUTA_DIAGRAMA), caption="Módulos y dependencias de la aplicación", width="stretch")
    else:
        st.warning(AVISO_SIN_DIAGRAMA)
    st.subheader("Paradigmas y patrones aplicados")
    st.dataframe([{"Pieza": pieza, "Símbolos": ", ".join(simbolos), "Problema que resuelve": problema}
                  for pieza, simbolos, problema in PATRONES], hide_index=True, width="stretch")
    st.subheader("Evidencia en el código")
    st.caption("Factory: decide la Strategy según el tipo de la serie (src/eda.py).")
    st.code(textwrap.dedent(inspect.getsource(AnalizadorFactory.crear)), language="python")
    st.caption("Registro de herramientas: la lista blanca que el modelo no puede ampliar (src/herramientas.py).")
    st.code(fuente_registro(), language="python")
    st.subheader("Ciclo de eventos de la interfaz")
    st.markdown(TEXTO_CICLO)
    st.subheader("Ejercicio")
    st.write(EJERCICIO)


def main() -> None:
    """Orquesta una ejecución: logger, barra lateral, datos, filtro, firma, indicadores y la pestaña abierta."""
    st.set_page_config(page_title="Incidentes TI", layout="wide")
    # Paso 1: logger JSONL idempotente; la ruta sale de INCIDENTES_LOG_DIR en cada ejecución.
    configurar_logger()
    st.session_state.setdefault("muestras", [])
    st.title("Análisis de incidentes TI con un agente local")
    st.caption("Proyecto integrador · Kevin Alexis Barrera Llerena · Universidad Casa Grande")
    # Paso 2: datos, selección y firma que invalida la respuesta del agente.
    fuente, subido, modelo = configurar_barra_lateral()
    datos, medicion = obtener_datos(fuente, subido)
    prioridades = elegir_prioridades(datos)
    analisis = AnalisisIncidentes(datos.incidentes, datos.control, prioridades)
    invalidar_si_cambia(firma_estado(datos.control["sha256_csv"], prioridades, modelo))
    render_kpis(datos, analisis)
    # Paso 3: pestañas perezosas; solo se dibuja la abierta (sin seguimiento, .open es None y se dibujan todas).
    vistas: dict[str, Callable[[], None]] = {
        "Datos": lambda: render_datos(datos, analisis, medicion),
        "Resumen": lambda: render_resumen(analisis),
        "Explorar": lambda: render_explorar(analisis),
        "Relaciones y segmentos": lambda: render_relaciones(analisis),
        "Agente local": lambda: render_agente(analisis, modelo),
        "Observabilidad": lambda: render_observabilidad(medicion, modelo),
        "Arquitectura": render_arquitectura,
    }
    for nombre, pestana in zip(PESTANAS, st.tabs(PESTANAS, key="pestana", on_change="rerun")):
        if pestana.open is not False:
            with pestana:
                vistas[nombre]()


if __name__ == "__main__":
    main()
