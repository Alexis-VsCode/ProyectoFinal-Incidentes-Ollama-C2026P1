# Especificaciones por paquete

El proyecto se desarrolló con especificaciones previas al código (SDD) y pruebas escritas antes de
la implementación (TDD). Cada paquete tuvo una especificación congelada con su objetivo, su contrato
y criterios observables; cada criterio se demuestra con una prueba con nombre. Este índice resume
esas especificaciones.

| Paquete | Alcance | Archivos |
|---|---|---|
| W1 | Datos y análisis | `src/diccionario.py`, `src/data_loader.py`, `src/eda.py` |
| W2 | Visualizaciones | `src/visualizations.py` |
| W3 | Cliente de Ollama, verificación y observabilidad | `src/llm_client.py`, `src/verificacion.py`, `src/observabilidad.py` |
| W4 | Herramientas y agente | `src/herramientas.py`, `src/agent.py`, `src/preguntas_demo.py`, `scripts/comparar_modelos.py` |
| F2 | Interfaz Streamlit | `app.py`, `scripts/validar_interfaz.py` |

Comandos:

```powershell
.\.venv\Scripts\python.exe -m pytest -q                 # suite sin Ollama
.\.venv\Scripts\python.exe -m pytest -m vivo -v         # pruebas con Ollama real
.\.venv\Scripts\python.exe -m pytest -k <nombre> -v     # una prueba por nombre
```

Marcadores de `pytest.ini`: `vivo` (Ollama real, excluido por defecto), `csv_real` (CSV completo)
e `interfaz` (Streamlit AppTest).

---

## W1. Datos y análisis

**Objetivo.** Cargar el CSV, consolidar 119.998 eventos en 20.769 incidentes y ofrecer los cálculos
del caso por una fachada única, sin depender de Streamlit ni de Ollama.

**Contrato principal.**

```python
# src/diccionario.py
DURACION = "tiempo_resolucion_horas"; SIN_DATO = "(sin dato)"; MIN_MUESTRA = 30
VARIABLES_SEGMENTO = ("category", "assignment_group")
DICCIONARIO: tuple[Variable, ...]          # 13 variables con etiqueta y tipo analítico en español
def tabla_diccionario() -> pd.DataFrame

# src/data_loader.py
def cargar_csv(contenido: bytes) -> DatosIncidentes          # eventos, incidentes, control
def firma_archivo(ruta: Path) -> tuple[str, int, int]         # ruta, tamaño, mtime; no lee contenido
def cargar_ruta(ruta: Path) -> DatosIncidentes                # rechaza > 50 MiB antes de leer
def verificar_integridad(control: dict, esperado: str | None) -> dict   # coincide | difiere | no_aplica

# src/eda.py
def perfil_calidad(df) -> pd.DataFrame
def comparar_prioridades(df) -> pd.DataFrame
def segmentar(df, variable: str, top: int = 10) -> pd.DataFrame
def spearman(x, y) -> float | None                            # Pearson sobre rangos, sin SciPy
def correlaciones(df) -> list[dict]                           # Pearson y Spearman con la duración
def matriz_spearman(df) -> pd.DataFrame
class AnalizadorVariable(ABC); class AnalizadorNumerico; class AnalizadorCategorico   # Strategy
class AnalizadorFactory: crear(serie) -> AnalizadorVariable                           # Factory
class AnalisisIncidentes                                                              # Facade
```

**Criterios observables y pruebas.**

| Criterio | Prueba |
|---|---|
| Los eventos se consolidan sin ponderar por número de actualizaciones | `test_consolida_incidentes_sin_ponderar_por_eventos`, `test_csv_sintetico_consolida_9_eventos_en_8_incidentes` |
| Las fechas se leen como día/mes | `test_orden_de_fechas_es_dia_mes` |
| Los nulos no se convierten en cero | `test_nulos_no_se_convierten_en_cero` |
| El diccionario cubre las 13 columnas analíticas en español | `test_diccionario_cubre_las_13_columnas_analiticas`, `test_etiquetas_y_descripciones_en_espanol_no_vacias` |
| La duración se describe como variable histórica, no como SLO ni MTTR | `test_descripcion_de_duracion_niega_slo_sla_y_mttr` |
| La carga por ruta no lee el contenido para la firma y rechaza archivos grandes | `test_firma_archivo_no_lee_contenido_y_cambia_con_mtime`, `test_cargar_ruta_rechaza_archivo_grande_antes_de_leer` |
| El SHA-256 del CSV incluido coincide con el esperado | `test_sha256_csv_incluido_es_el_hash_del_taller1`, `test_integridad_del_csv_incluido_coincide` |
| La segmentación agrupa el resto, cuenta nulos como "(sin dato)" y rechaza variables no autorizadas | `test_segmentar_top_agrupa_resto_en_otros`, `test_segmentar_nulos_como_sin_dato`, `test_segmentar_rechaza_variable_no_autorizada_y_top_invalido` |
| Spearman funciona sin SciPy | `test_spearman_funciona_sin_scipy`, `test_spearman_es_pearson_sobre_rangos_y_devuelve_none_si_no_esta_definido` |
| Factory intercambia analizadores según el tipo | `test_factory_intercambia_analizadores` |
| Las funciones de análisis no mutan la entrada | `test_funciones_de_analisis_no_mutan_la_entrada` |
| Con el CSV real se reproducen 220/337/19.554/658 incidentes y medianas 78,17/48,32/24,29/22,85 h | `test_comparar_prioridades_real_reproduce_taller1`, `test_indicadores_reales_sin_cambios` |

---

## W2. Visualizaciones

**Objetivo.** Elegir cada gráfico según el tipo de variable y la pregunta analítica. El módulo solo
depende de `src.diccionario`, nunca de la interfaz.

**Contrato principal.**

```python
def grafico_prioridades(tabla) -> go.Figure          # barras por prioridad (ordinal)
def grafico_duracion(df) -> go.Figure                # ECDF, eje x logarítmico, marcas de mediana y p90
def grafico_variable(df, variable: str) -> go.Figure  # histograma log, barras por valor o top 10
def grafico_segmentos(segmentos, variable) -> go.Figure
def figura_boxplot_prioridad(df) -> Figure           # Seaborn, escala logarítmica, n visible
def figura_mapa_spearman(matriz) -> Figure           # Seaborn, triángulo inferior, escala -1 a 1
def figura_a_png(figura: Figure, dpi: int = 150) -> bytes
```

**Criterios observables y pruebas.**

| Criterio | Prueba |
|---|---|
| La ECDF usa eje logarítmico y porcentajes, y marca mediana y p90 | `test_grafico_duracion_ecdf_con_eje_log_y_porcentaje`, `test_grafico_duracion_marca_mediana_y_p90` |
| La ECDF informa los ceros y los sin dato | `test_grafico_duracion_informa_ceros_y_sin_dato` |
| Cada tipo de variable recibe su gráfico | `test_grafico_variable_duracion_usa_bins_logaritmicos_con_ticks_en_horas`, `test_grafico_variable_conteo_discreto_usa_barras_por_valor`, `test_grafico_variable_categorica_limita_a_diez_e_informa_total` |
| El boxplot respeta el orden de prioridad y muestra n | `test_figura_boxplot_prioridad_eje_log_orden_y_n` |
| El mapa de calor tiene escala fija de -1 a 1 | `test_figura_mapa_spearman_triangulo_inferior_y_rango_fijo` |
| Las figuras de Matplotlib no quedan abiertas en pyplot | `test_figuras_matplotlib_no_dejan_figuras_en_pyplot` |
| Entradas vacías devuelven un aviso sin excepción | `test_entradas_vacias_devuelven_aviso_sin_excepcion` |
| Formato numérico en español | `test_separadores_numericos_en_espanol`, `test_horas_en_formato_espanol` |

---

## W3. Cliente de Ollama, verificación y observabilidad

**Objetivo.** Aislar el protocolo HTTP de Ollama detrás de un contrato, comprobar que las cifras de
una respuesta existan en la evidencia y medir el servicio de la propia aplicación.

**Contrato principal.**

```python
# src/llm_client.py
class ClienteChat(Protocol)                          # contrato del que depende el agente
class ClienteOllama:                                 # Adapter; solo hosts locales y modelos en lista blanca
    def chat(mensajes, herramientas=None, *, opciones=None) -> RespuestaModelo
    def chat_stream(mensajes, *, opciones=None) -> FlujoRespuesta
    def info_modelo() -> InfoModelo                  # digest, cuantización, capacidades
    def version() -> str; def precalentar() -> RespuestaModelo
    def verificar(precalentar=True) -> dict        # botón «Verificar Ollama»
    def estado() -> dict                           # sonda de salud; nunca lanza
class ErrorOllama(RuntimeError)                      # tipo, operación, código HTTP

# src/verificacion.py
def verificar_cifras(texto: str, contexto, pregunta: str = "") -> VerificacionCifras

# src/observabilidad.py
def configurar_logger(ruta=None) -> logging.Logger   # JSONL idempotente
def span(nombre, trace_id, *, destino=None, **atributos)
def calcular_slis(muestras) -> SLIs
def evaluar_slos(muestras, slos=SLOS) -> list[EvaluacionSLO]
```

**Criterios observables y pruebas.**

| Criterio | Prueba |
|---|---|
| El payload sigue el formato oficial de `/api/chat` con `tools` y nunca envía `think` | `test_chat_envia_payload_oficial_con_tools`, `test_chat_sin_herramientas_omite_tools_y_nunca_envia_think` |
| Las llamadas a herramientas se leen como objeto o como texto JSON | `test_chat_parsea_tool_calls_con_argumentos_objeto`, `test_chat_decodifica_argumentos_en_texto_json` |
| El contexto del modelo no cambia entre peticiones (evita recargas) | `test_opciones_no_pueden_cambiar_num_ctx` |
| Los errores de conexión, timeout, modelo ausente y memoria tienen mensajes legibles y tipo | `test_errores_http_traducidos`, `test_error_ollama_lleva_tipo_operacion_y_codigo` |
| Solo se aceptan hosts locales y modelos autorizados | `test_url_no_local_rechazada`, `test_modelo_fuera_de_lista_blanca_rechazado` |
| El streaming concatena fragmentos y expone métricas; un error a mitad se propaga | `test_chat_stream_concatena_fragmentos_y_expone_metricas`, `test_chat_stream_error_a_mitad_lanza_error_ollama` |
| Las cifras inventadas quedan sin verificar y el redondeo correcto se acepta | `test_cifras_inventadas_quedan_no_verificadas`, `test_tolerancia_de_redondeo`, `test_truncamiento_no_se_acepta` |
| Fechas, años, listas e identificadores no se cuentan como cifras | `test_ignora_fechas_horas_y_anios`, `test_ignora_marcadores_de_lista`, `test_identificadores_no_son_cifras` |
| El log escribe una línea JSON válida por evento y es idempotente | `test_logger_escribe_una_linea_json_valida_por_evento`, `test_configurar_logger_es_idempotente` |
| Los spans miden y registran aunque haya excepción o cancelación | `test_span_mide_ms_y_registra_ok`, `test_span_registra_aunque_haya_excepcion`, `test_span_en_generador_cancelado_marca_cancelado` |
| Los SLI y SLO se calculan exactos sobre un lote sintético | `test_calcular_slis_lote_sintetico_s1`, `test_slo_latencia_presupuesto_negativo`, `test_umbral_de_latencia_es_estricto` |
| Ningún SLO menciona incidentes, resolución ni MTTR | `test_slos_no_mencionan_el_dataset` |

---

## W4. Herramientas y agente

**Objetivo.** Un agente acotado: guarda en Python, selección por *tool calling* sobre una lista
blanca de cinco herramientas de lectura, respaldo determinista, interpretación en streaming,
verificación de cifras y una traza por consulta.

**Contrato principal.**

```python
# src/herramientas.py
REGISTRO: RegistroHerramientas         # resumen, prioridades, calidad, relaciones, segmentar
class RegistroHerramientas:
    def esquemas_ollama() -> list[dict]
    def validar(nombre, argumentos) -> tuple[dict, tuple[str, ...]]   # nombre, enum y rangos
    def ejecutar(nombre, argumentos, analisis) -> ResultadoHerramienta
def enrutar_por_palabras(pregunta, registro=None) -> tuple[str, dict] | None

# src/agent.py
def detectar_fuera_de_alcance(pregunta: str) -> MotivoRechazo | None
class AgenteIncidentes:
    def __init__(analisis, cliente: ClienteChat, registro=REGISTRO, al_finalizar=None)
    def preparar(pregunta, herramienta=None, *, modo=None, argumentos=None) -> ConsultaPreparada
    def interpretar(consulta) -> Iterator[str]          # streaming sin herramientas
    def finalizar(consulta) -> RegistroConsulta         # JSON descargable con traza y verificación

# scripts/comparar_modelos.py
python -m scripts.comparar_modelos --modelos llama3.2:3b llama3.2:1b
```

**Criterios observables y pruebas.**

| Criterio | Prueba |
|---|---|
| El registro tiene exactamente cinco herramientas con esquemas en formato Ollama | `test_registro_tiene_cinco_herramientas_en_orden`, `test_esquemas_siguen_formato_ollama` |
| Herramientas inexistentes y argumentos fuera de rango se rechazan | `test_validar_herramienta_inexistente`, `test_validar_enum_fuera_de_lista`, `test_validar_rangos_y_tipos_de_top` |
| El modelo elige herramienta y argumentos válidos | `test_automatico_elige_herramienta_valida`, `test_automatico_elige_segmentar_con_argumentos` |
| Si la selección no es válida, entra el respaldo y queda registrado | `test_herramienta_inexistente_activa_respaldo`, `test_sin_tool_call_activa_respaldo`, `test_respaldo_activado_registra_motivo` |
| Como máximo dos herramientas por consulta, sin duplicados | `test_tope_de_dos_llamadas_y_sin_duplicados` |
| Las peticiones de acción se rechazan sin llamar al modelo | `test_pregunta_de_accion_se_rechaza_sin_llamar_modelo` |
| La guarda no bloquea preguntas de análisis legítimas | `test_guarda_no_bloquea_preguntas_de_analisis` |
| En modo manual no se pide selección al modelo | `test_manual_no_pide_seleccion_al_modelo` |
| La interpretación no recibe herramientas y el contexto enviado es exacto | `test_interpretacion_sin_tools_y_num_ctx_constante`, `test_contexto_enviado_es_exacto_y_redondeado` |
| Cada consulta tiene un `trace_id`, sus spans en orden y un único cierre | `test_consulta_genera_trace_id_y_spans_en_orden`, `test_una_sola_consulta_finalizada_por_trace` |
| El log no guarda la pregunta ni la respuesta | `test_log_no_contiene_pregunta_ni_respuesta` |
| Las preguntas de demostración son coherentes con el registro y el enrutador | `test_preguntas_demo_coherentes_con_registro_y_enrutador` |
| El comparador usa los mismos SLI y SLO y escribe su log aparte | `test_comparador_usa_calcular_slis_y_evaluar_slos`, `test_comparador_escribe_log_separado_de_app` |
| Con Ollama real, las llamadas llegan como objeto y las preguntas demo responden | `test_vivo_tool_calling_devuelve_argumentos_objeto`, `test_vivo_preguntas_demo`, `test_vivo_slis_calculables_desde_log` |

---

## F2. Interfaz Streamlit

**Objetivo.** Una interfaz de siete pestañas que solo compone: no calcula, no habla HTTP y no
conoce pandas más allá de mostrar tablas. Las pestañas no visibles no se calculan y el CSV se lee
una sola vez por sesión.

**Contrato principal.**

```python
PESTANAS = ("Datos", "Resumen", "Explorar", "Relaciones y segmentos",
            "Agente local", "Observabilidad", "Arquitectura")
@st.cache_resource
def datos_incluidos(ruta: str, tamano: int, mtime_ns: int) -> ...   # una carga real por firma
def render_datos(...); def render_resumen(...); def render_explorar(...)
def render_relaciones(...); def render_agente(...); def render_observabilidad(...)
def render_arquitectura(); def main()
```

Claves de widgets estables para las pruebas: `fuente`, `modelo`, `filtro_prioridades`, `pestana`,
`variable_explorar`, `variable_segmento`, `btn_verificar`, `modo_automatico`,
`herramientas_manual`, `pregunta_agente`, `consultar_agente`, `descargar_consulta`.

**Criterios observables y pruebas** (Streamlit AppTest, marcador `interfaz`).

| Criterio | Prueba |
|---|---|
| La aplicación arranca sin excepción y muestra 20.769 incidentes | `test_inicio_sin_excepcion_y_kpi_total` |
| El filtro 1 - Critical muestra 220 incidentes y 78,17 h | `test_filtro_critical_muestra_220_y_mediana` |
| Una selección vacía muestra un aviso sin romper la aplicación | `test_seleccion_vacia_muestra_aviso_sin_excepcion` |
| Se recorren las siete pestañas sin excepción | `test_recorre_todas_las_pestanas_sin_excepcion` |
| Relaciones muestra las figuras de Seaborn y Resumen los gráficos Plotly | `test_relaciones_muestra_figuras_seaborn`, `test_resumen_muestra_graficos_plotly` |
| Un rerun no vuelve a leer el CSV | `test_rerun_no_relee_ni_reprocesa_csv` |
| El modo manual muestra el selector de herramientas | `test_agente_modo_manual_muestra_selector_de_herramientas` |
| Sin Ollama, «Verificar Ollama» muestra el error y el análisis sigue disponible | `test_verificar_ollama_sin_servicio_muestra_error` |
| Cambiar el filtro invalida la respuesta anterior del agente | `test_cambio_de_filtro_invalida_respuesta_del_agente` |
| La pestaña Observabilidad funciona sin consultas registradas | `test_observabilidad_sin_consultas_renderiza_salud_slos_y_aviso` |
| Resumen rotula sus percentiles como descriptivos, no como SLO | `test_resumen_rotula_percentiles_como_descriptivos` |

`scripts/validar_interfaz.py` recorre la interfaz sin navegador y guarda los tiempos medidos en
`resultados_locales/`.
