# Decisiones de diseño

Registro breve de las decisiones que dan forma al proyecto. Cada una indica el contexto, lo que se
decidió, las alternativas que se descartaron y sus consecuencias.

## ADR-01. Dataset y unidad de análisis: el incidente, no el evento

**Contexto.** La guía pide un dataset público de Kaggle con un problema analítico claro. El CSV
Incident Response Log (versión 2) tiene 119.998 filas, pero cada fila es una actualización de un
incidente: el identificador `number` se repite.

**Decisión.** Usar ese CSV, verificado por SHA-256
(`5d77ea4853e443cf8447fca0f3089b7689e5469225cb68f72b2b9feea5f2d0d2`), y consolidar los eventos en
una observación por incidente, la última actualización de cada `number`. De 119.998 eventos salen
20.769 incidentes. Los `?` quedan como desconocidos, sin imputación; la duración se deriva como
`resolved_at - opened_at` y los 1.418 casos sin duración válida se excluyen solo de las medidas de
horas.

**Alternativas.** Analizar por evento (infla conteos y promedios); descargar la versión de UCI
(otro tamaño, 141.712 eventos, no comparable con el trabajo previo); imputar duraciones faltantes
como cero (sesgaría la mediana hacia abajo).

**Consecuencias.** Todas las cifras de la interfaz, el agente y las pruebas salen de la misma
tabla. El control del archivo conserva los conteos de eventos para que la diferencia sea visible.

## ADR-02. Python 3.13 y dependencias fijadas

**Contexto.** En el equipo de desarrollo, `py` abre Python 3.14 por defecto, y `pandas==2.2.3` no
publica paquetes para 3.14.

**Decisión.** Crear el entorno con `py -3.13 -m venv .venv` y fijar las dependencias directas en
`requirements.txt`. Las pruebas van aparte en `requirements-dev.txt`.

**Alternativas.** Subir pandas a una versión con soporte para 3.14 (cambia el comportamiento ya
validado en los talleres); no fijar versiones (instalaciones distintas en cada equipo).

**Consecuencias.** El README advierte del problema y el runbook explica cómo recrear el entorno.

## ADR-03. Agente híbrido con lista blanca de herramientas

**Contexto.** La guía pide un agente que use herramientas autorizadas y no reemplace los cálculos
de Python. Un modelo de 3B parámetros puede elegir mal o inventar argumentos, y la plantilla de
llama3.2 empuja al modelo a llamar siempre a alguna herramienta.

**Decisión.** Un flujo acotado en Python:

1. guarda que rechaza acciones, código y manipulación antes de llamar al modelo;
2. selección por *tool calling* sobre un registro de cinco herramientas de solo lectura
   (`resumen`, `prioridades`, `calidad`, `relaciones`, `segmentar`), con validación de nombre,
   enumerados y rangos;
3. respaldo determinista por palabras clave cuando no hay una llamada válida;
4. modo manual en el que el usuario elige la herramienta;
5. interpretación en streaming sin herramientas disponibles;
6. verificación de las cifras citadas contra la evidencia enviada.

Como máximo hay dos llamadas al modelo por consulta.

**Alternativas.** Selección manual únicamente (no demuestra uso de herramientas por el agente);
bucle libre de herramientas con varias rondas (más lento y propenso a repetir llamadas);
bibliotecas de agentes como LangChain (dependencia grande para cinco funciones).

**Consecuencias.** El modelo nunca ejecuta nada que no esté en el registro, y el registro solo
tiene lecturas. La traza deja ver si eligió el modelo o el respaldo.

## ADR-04. Modelo por defecto: llama3.2:3b, comparado con llama3.2:1b

**Contexto.** La guía recomienda `llama3.2:3b` y admite `llama3.2:1b` en equipos con menos
recursos. El equipo no tiene GPU: con CPU, `llama3.2:3b` (Q4_K_M, digest `a80c4f17acd5`) genera
unos 7 tokens por segundo en Ollama 0.40.6.

**Decisión.** `llama3.2:3b` es el modelo por defecto. `scripts/comparar_modelos.py` lo compara con
`llama3.2:1b` sobre las mismas preguntas y mide acierto de herramienta, cifras verificadas,
respuestas truncadas, errores y latencia.

**Resultado medido (18/10/2026, siete preguntas, dos repeticiones por modelo, 28 consultas, 0 errores).**

| Indicador | llama3.2:3b | llama3.2:1b |
|---|---|---|
| Acierto de herramienta | 91,7 % | 41,7 % |
| Cifras citadas presentes en la evidencia | 91 de 91 | 101 de 102 |
| Mediana de latencia por consulta | 45,5 s | 25,3 s |
| Velocidad de generación (mediana) | 7,2 tokens/s | 12,6 tokens/s |
| Rechazo de la pregunta de acción | 100 % | 100 % |

Se mantiene `llama3.2:3b`. El modelo de 1B es 1,8 veces más rápido, pero elige mal la herramienta en
más de la mitad de las preguntas, y el agente depende de esa elección. Las únicas respuestas
truncadas del 3B ocurrieron en la pregunta que combina dos herramientas, que solo usa el comparador.
La comparación corrió en paralelo con otras tareas del equipo; en una corrida aislada de las seis
preguntas de demostración, el 3B tuvo una mediana de 32,7 s y un p95 de 35,4 s.

**Alternativas.** `qwen2.5:3b` (también admite herramientas, pero su licencia en el tamaño 3B no
es Apache 2.0 y no aporta a lo que pide la guía).

**Consecuencias.** El modelo y su digest quedan registrados en cada consulta; cambiar de modelo
es una opción de la barra lateral, no un cambio de código.

## ADR-05. Spearman por rangos, sin SciPy

**Contexto.** Las horas de resolución son muy asimétricas (mediana 24,60 h frente a media
187,68 h) y los contadores tienen muchos empates en cero. En pandas, `Series.corr(method="spearman")`
importa SciPy, que no está entre las dependencias.

**Decisión.** Calcular Spearman como Pearson sobre rangos promedio:
`x.rank().corr(y.rank())`. Se informa junto a Pearson.

**Alternativas.** Agregar SciPy (dependencia pesada para una sola función); usar solo Pearson
(sensible a la cola larga).

**Consecuencias.** Una prueba verifica que el cálculo funciona aunque SciPy no esté instalado.

## ADR-06. Seaborn para boxplot y mapa de calor; Plotly para el resto

**Contexto.** La guía pide Plotly y, cuando sea útil, Matplotlib o Seaborn, eligiendo cada gráfico
según el tipo de variable.

**Decisión.** Plotly para los gráficos interactivos (barras por prioridad, ECDF de la duración en
escala logarítmica, distribución de variables, segmentos). Seaborn para dos figuras estadísticas:
el boxplot de horas por prioridad en escala logarítmica y el mapa de calor de Spearman con escala
fija de -1 a 1. Las figuras de Matplotlib se crean con `Figure()` y el backend `Agg`, sin pyplot.

**Alternativas.** Todo en Plotly (el mapa de calor anotado y el boxplot logarítmico salen más
claros en Seaborn); histograma lineal de la duración (aplasta la mayoría de los casos contra cero).

**Consecuencias.** Las figuras de Seaborn también se descargan como PNG para el informe.

## ADR-07. Observabilidad con la biblioteca estándar, sin OpenTelemetry

**Contexto.** Interesa saber dónde se va el tiempo de una consulta y si el agente cumple lo que
promete, sin montar infraestructura para una aplicación local.

**Decisión.** Registro JSONL con `logging`, un `trace_id` por consulta, tres spans (selección,
ejecución, interpretación), un evento final por consulta del que se derivan todos los SLI y tres
SLO con presupuesto de error calculado por lote. Detalle en [OBSERVABILIDAD.md](OBSERVABILIDAD.md).

**Alternativas.** OpenTelemetry, Prometheus o Grafana (exportadores y servicios que nadie atiende
en un equipo local); alertas por tasa de consumo (no hay guardia).

**Consecuencias.** Cero dependencias nuevas. El comparador de modelos reutiliza las mismas
funciones de SLI y SLO.

## ADR-08. Recortes deliberados

| Recorte | Razón |
|---|---|
| Tendencia mensual | `opened_at` solo cubre del 29/02/2016 al 13/05/2016: tres meses con censura al final no sostienen una tendencia |
| `incident_state` como segmento | 20.768 incidentes están en Closed y 1 en New; no discrimina nada |
| `qwen2.5:3b` | Licencia distinta y no lo pide la guía; la comparación se limita a los dos modelos de la guía |
| scikit-learn y predicción | El alcance es descriptivo; predecir tiempos exigiría variables disponibles al abrir el caso y una partición temporal |

## ADR-09. Patrones de diseño, solo donde hay una variación real

| Patrón | Dónde | Problema que resuelve |
|---|---|---|
| Strategy | `AnalizadorVariable` con `AnalizadorNumerico` y `AnalizadorCategorico` | Resumir variables numéricas y categóricas con la misma llamada, sin condicionales por tipo en la interfaz |
| Factory | `AnalizadorFactory.crear(serie)` | Decidir en un solo punto qué estrategia corresponde a cada variable |
| Facade | `AnalisisIncidentes` | Dar a la interfaz y al agente una sola puerta al análisis filtrado, sin exponer pandas |
| Adapter | `ClienteOllama` y el protocolo `ClienteChat` | Aislar el HTTP y los errores de Ollama; el agente depende del contrato y se prueba con un cliente falso |
| Registro | `RegistroHerramientas` y `REGISTRO` | Una lista blanca única: lo que el modelo puede pedir, los esquemas que recibe y lo que la interfaz ofrece en modo manual salen del mismo lugar |

No se agregaron más patrones porque no había otra variación que los justificara: la carga de datos
es un único formato y la interfaz es una sola aplicación.
