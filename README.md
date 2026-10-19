# Análisis de incidentes TI con un agente local

[![Pruebas](https://github.com/Alexis-VsCode/ProyectoFinal-Incidentes-Ollama-C2026P1/actions/workflows/pruebas.yml/badge.svg)](https://github.com/Alexis-VsCode/ProyectoFinal-Incidentes-Ollama-C2026P1/actions/workflows/pruebas.yml)

[![Versión](https://img.shields.io/github/v/release/Alexis-VsCode/ProyectoFinal-Incidentes-Ollama-C2026P1?label=versi%C3%B3n)](https://github.com/Alexis-VsCode/ProyectoFinal-Incidentes-Ollama-C2026P1/releases)
![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB)
![Licencia MIT](https://img.shields.io/badge/licencia-MIT-green)

- Página del proyecto: https://alexis-vscode.github.io/ProyectoFinal-Incidentes-Ollama-C2026P1/
- Video demostrativo: https://youtu.be/luE5Ukz4PAE
- Versión entregada: [v1.0.0](https://github.com/Alexis-VsCode/ProyectoFinal-Incidentes-Ollama-C2026P1/releases/tag/v1.0.0) (cambios en [CHANGELOG.md](CHANGELOG.md))

Proyecto Final Integrador de *Paradigmas de Programación para Inteligencia Artificial y Análisis de
Datos* (Universidad Casa Grande, Maestría en Inteligencia Artificial y Ciencia de Datos, C2026 P1).

Aplicación modular en Python y Streamlit que analiza un historial público de incidentes de TI y
consulta a un agente que se ejecuta en el propio equipo con Ollama. Python calcula todas las cifras;
el modelo elige entre herramientas de lectura autorizadas y redacta la interpretación.

## Resultados medidos

Medidos el 18/10/2026 en el equipo de referencia: Windows 11, Intel Core i9-13900, 16 GB de RAM, sin
GPU dedicada, Ollama 0.40.6.

| Indicador | Valor |
|---|---|
| Incidentes únicos analizados | 20.769 (de 119.998 eventos) |
| Mediana hasta resolución | 24,6 h (media 187,68 h: cola larga) |
| Herramienta elegida correctamente por `llama3.2:3b` | 91,7 % (frente a 41,7 % de `llama3.2:1b`) |
| Cifras de las respuestas presentes en la evidencia | 36 de 36 en la corrida de demostración |
| Latencia por consulta (mediana / p95) | 32,7 s / 35,4 s |
| Pruebas automatizadas | 519 con pytest, en Windows y Linux mediante GitHub Actions |

La comparación completa de modelos y su criterio de decisión están en
[docs/DECISIONES.md](docs/DECISIONES.md).

## Qué problema resuelve y para quién

Un registro de incidentes guarda muchas actualizaciones por caso. Si cada fila se cuenta como un
incidente, las frecuencias y los promedios salen mal. La aplicación consolida los eventos en una
observación por incidente y permite revisar el volumen, la calidad de los datos y la duración
registrada por prioridad, categoría y grupo de asignación.

Está pensada para analistas y responsables de soporte que quieren explorar ese historial y pedir
una explicación en lenguaje natural sin perder la trazabilidad de los números. El agente no ejecuta
acciones ni comandos: solo consulta cálculos ya hechos.

El alcance es descriptivo. Los datos son públicos y anonimizados, de otra organización. No hay
modelo predictivo ni análisis de causa raíz.

## Dataset

| Campo | Valor |
|---|---|
| Nombre | Incident Response Log, versión 2 (Vipul Shinde, Kaggle) |
| Enlace | https://www.kaggle.com/datasets/vipulshinde/incident-response-log/versions/2 |
| Fuente original | Amaral, C., Fantinato, M. y Peres, S. (2018). *Incident management process enriched event log*. UCI Machine Learning Repository. https://doi.org/10.24432/C57S4H |
| Licencias | Los metadatos de Kaggle declaran CC0; la ficha de UCI declara CC BY 4.0. Se cita a ambas fuentes |
| Archivo | `data/incident_event_log.csv`, 39.147.503 bytes |
| SHA-256 | `5d77ea4853e443cf8447fca0f3089b7689e5469225cb68f72b2b9feea5f2d0d2` |
| Contenido | 119.998 eventos, 36 columnas, 20.769 incidentes únicos |

**Evento no es incidente.** Cada fila del CSV es una actualización de un incidente (cambio de estado,
reasignación, modificación). El identificador `number` agrupa todas las filas de un mismo caso. La
aplicación conserva los eventos y construye una tabla analítica con la última actualización de cada
incidente: de 119.998 eventos salen 20.769 incidentes. Todas las cifras de la interfaz y del agente
se calculan sobre esa tabla.

Los valores `?` se tratan como desconocidos y no se imputan. La duración se deriva como
`resolved_at - opened_at` en horas; 1.418 incidentes no tienen duración válida y quedan fuera de las
medidas de horas, pero sí cuentan como incidentes.

El CSV se versiona byte a byte (`.gitattributes` con `-text`) para que su SHA-256 no cambie al
clonar en Windows. La procedencia completa está en [data/README.md](data/README.md).

## Requisitos

- Windows 11 de 64 bits (hay una variante breve para macOS o Linux más abajo).
- Python 3.13 x64. Se probó con Python 3.13.3.
- Ollama para Windows y el modelo `llama3.2:3b` (unos 2 GB). Alternativa más liviana:
  `llama3.2:1b`.
- 16 GB de RAM recomendados. Con CPU y sin GPU, `llama3.2:3b` genera unos 7 tokens por segundo,
  así que una respuesta completa tarda varias decenas de segundos.
- Git para clonar el repositorio.

Equipo de referencia: Intel Core i9-13900, 16 GB de RAM, sin GPU dedicada, Windows 11, Ollama
0.40.6.

## Instalación en Windows (PowerShell)

### 1. Clonar el repositorio

```powershell
git clone https://github.com/Alexis-VsCode/ProyectoFinal-Incidentes-Ollama-C2026P1.git
cd ProyectoFinal-Incidentes-Ollama-C2026P1
```

### 2. Crear el entorno virtual con Python 3.13

```powershell
py -3.13 -m venv .venv
```

> **Importante:** usa `py -3.13` y no `py` a secas. Si tienes Python 3.14 instalado, `py` lo elige
> por defecto, y `pandas==2.2.3` no publica paquetes para 3.14: la instalación intentaría compilar
> pandas y fallaría. Comprueba la versión con `.\.venv\Scripts\python.exe --version`.

Los comandos siguientes llaman directamente al intérprete del entorno, así que no hace falta
activarlo (y funcionan aunque PowerShell bloquee `Activate.ps1`). Si prefieres activarlo:
`.\.venv\Scripts\Activate.ps1`.

### 3. Instalar las dependencias

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Las dependencias directas están fijadas en `requirements.txt` (Streamlit, pandas, NumPy, Plotly,
requests, Matplotlib y seaborn). No se usa SciPy: la correlación de Spearman se calcula por rangos.

### 4. Instalar Ollama y descargar el modelo

1. Descarga e instala Ollama desde https://ollama.com/download.
2. Ábrelo desde el menú Inicio (queda escuchando en `http://127.0.0.1:11434`).
3. Abre una terminal nueva y descarga el modelo:

```powershell
ollama --version
ollama pull llama3.2:3b
ollama list
```

Modelo alternativo para equipos con menos memoria:

```powershell
ollama pull llama3.2:1b
```

Si Ollama ya está corriendo en segundo plano no ejecutes `ollama serve`: ese comando solo sirve
para arrancar el servidor a mano cuando no está activo. Los modelos se guardan fuera del
repositorio y nunca se versionan.

### 5. Ejecutar la aplicación

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abre http://127.0.0.1:8501. Para detenerla, `Ctrl+C` en la terminal.

La primera vez, Streamlit puede pedir un correo electrónico en la terminal («Email:»). No es
obligatorio: basta con presionar Enter.

En las siguientes sesiones puedes hacer doble clic en `iniciar.bat`, que arranca la aplicación con
el intérprete del entorno.

### macOS o Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m streamlit run app.py
```

Instala Ollama siguiendo https://ollama.com/download para tu sistema.

## Estructura del proyecto

```text
ProyectoFinal-Incidentes-Ollama-C2026P1/
    app.py                    interfaz Streamlit: solo compone pestañas y estado
    iniciar.bat               arranque rápido en Windows
    requirements.txt          dependencias de ejecución (fijadas)
    requirements-dev.txt      dependencias de pruebas (pytest)
    pytest.ini                configuración y marcadores de pruebas
    .gitignore  .gitattributes  LICENSE
    data/
        incident_event_log.csv            dataset (SHA-256 verificado)
        incident-response-log-metadata.json
        README.md                         procedencia y licencia
    assets/
        arquitectura.png  arquitectura.svg
    src/
        diccionario.py        diccionario de variables en español
        data_loader.py        carga, validación, consolidación y SHA-256
        eda.py                Strategy, Factory, Facade, segmentación y Spearman
        visualizations.py     gráficos Plotly y figuras Seaborn
        llm_client.py         adaptador HTTP de Ollama
        herramientas.py       registro único de herramientas autorizadas
        agent.py              orquestación del agente
        verificacion.py       verificación de cifras de la respuesta
        observabilidad.py     logs JSONL, trazas, SLI y SLO
        preguntas_demo.py     preguntas de demostración
    scripts/
        validar_datos.py      comprobación de conteos del CSV
        validar_interfaz.py   recorrido de la interfaz sin navegador
        comparar_modelos.py   comparación medida entre modelos
        diagnostico_windows.ps1
    tests/                    pruebas pytest
    docs/
        DECISIONES.md  OBSERVABILIDAD.md  RUNBOOK.md
        specs/README.md       especificaciones por paquete
```

`resultados_locales/` se crea al ejecutar (logs, evidencias, comparaciones) y está excluida de Git.

## Pestañas de la aplicación

| Pestaña | Qué muestra |
|---|---|
| Datos | Control del archivo (eventos, columnas, incidentes, SHA-256), reglas de preparación, diccionario de variables con nulos y descarga de la selección |
| Resumen | Media, mediana y percentil 90 de horas; incidentes por prioridad; ECDF de la duración en escala logarítmica |
| Explorar | Cualquier variable con el analizador que elige `AnalizadorFactory` (numérico o categórico) y su gráfico |
| Relaciones y segmentos | Boxplot de horas por prioridad y mapa de calor de Spearman (Seaborn); segmentos por categoría o grupo de asignación |
| Agente local | Verificar Ollama, modo automático o manual, preguntas sugeridas, respuesta en streaming, traza de herramientas e insignia de cifras verificadas |
| Observabilidad | Salud de Ollama, integridad del CSV, SLI frente a SLO y últimas consultas con su `trace_id` |
| Arquitectura | Diagrama, patrones y paradigmas con su justificación |

La barra lateral elige la fuente de datos (CSV incluido o uno cargado con el mismo esquema), el
modelo y el filtro de prioridades. Todas las pestañas usan la misma selección.

## Cómo funciona el agente

Python calcula y el modelo interpreta. El modelo nunca recibe el CSV ni calcula ninguna cifra.

1. **Guarda previa.** Antes de llamar al modelo, Python rechaza las peticiones de acción
   («reinicia el servidor», «cierra los incidentes»), de código o de manipulación de reglas. El
   rechazo es inmediato y no consume inferencia.
2. **Lista blanca de cinco herramientas**, todas de solo lectura y definidas en un único registro
   (`src/herramientas.py`):

   | Herramienta | Qué entrega |
   |---|---|
   | `resumen` | Totales de la selección: incidentes, duraciones válidas, media, mediana y p90 de horas |
   | `prioridades` | Incidentes y horas por prioridad |
   | `calidad` | Nulos, porcentaje de nulos y valores únicos por columna |
   | `relaciones` | Correlaciones de Pearson y Spearman de reasignaciones y reaperturas con las horas |
   | `segmentar` | Ranking de categorías o grupos de asignación (`variable` y `top` de 3 a 10) |

3. **Tool calling.** En modo automático el modelo recibe los esquemas de esas herramientas y
   responde con el nombre y los argumentos que quiere usar. Python valida el nombre, los valores
   permitidos y los rangos antes de ejecutar nada.
4. **Respaldo determinista.** Si el modelo no propone una llamada válida (herramienta inexistente,
   argumento fuera de rango o solo texto), un enrutador por palabras clave elige la herramienta. La
   traza registra el origen de cada selección (`modelo`, `respaldo_reglas`, `usuario`).
5. **Modo manual.** El usuario elige las herramientas y el modelo solo redacta.
6. **Interpretación en streaming.** Con la evidencia calculada, el modelo responde en español con
   Hechos observados, Interpretación y Limitaciones, sin herramientas disponibles en ese paso.
7. **Verificación de cifras.** `verificar_cifras` extrae los números de la respuesta y comprueba
   que existan en la evidencia enviada. La interfaz muestra cuántas cifras se verificaron. Comprueba
   presencia, no atribución: la revisión humana sigue siendo necesaria.

Cada consulta se puede descargar como JSON con la pregunta, las herramientas, el contexto enviado,
el modelo y su digest, las métricas de Ollama, la verificación y la traza.

## Patrones y paradigmas

| Pieza | Símbolo | Problema real que resuelve |
|---|---|---|
| Strategy | `AnalizadorVariable`, `AnalizadorNumerico`, `AnalizadorCategorico` | Las variables numéricas y categóricas se resumen distinto, sin `if` por tipo en la interfaz |
| Factory | `AnalizadorFactory.crear(serie)` | Concentra en un solo punto la decisión de qué estrategia usar |
| Facade | `AnalisisIncidentes` | La interfaz y el agente acceden al análisis por una sola puerta, sin conocer pandas |
| Adapter | `ClienteOllama` y el protocolo `ClienteChat` | Aísla HTTP y errores de Ollama; el agente depende de un contrato y se prueba con un doble |
| Registro (lista blanca) | `RegistroHerramientas` y `REGISTRO` | El modelo solo puede pedir lo que está ahí; fuente única para el agente y la interfaz |
| Funcional | `segmentar`, `spearman`, `correlaciones`, `verificar_cifras`, `calcular_slis` | Funciones puras que se reutilizan en la interfaz, el agente y las pruebas |
| Imperativo | `cargar_csv`, `main()` | Validaciones paso a paso con control explícito de errores |
| POO | `DatosIncidentes`, `RegistroConsulta`, `MuestraConsulta` | Contratos de datos con estado |
| Eventos | reruns y callbacks de Streamlit, `session_state`, invalidación por firma | La interfaz reacciona a cada cambio y descarta respuestas de otra selección |

El porqué de cada decisión está en [docs/DECISIONES.md](docs/DECISIONES.md).

## Observabilidad

La aplicación mide su propio servicio, no el de los incidentes del CSV.

- **Logs JSONL** en `resultados_locales/logs/app.jsonl`: una línea por evento con `ts_utc`,
  `nivel`, `evento`, `trace_id` y `atributos`.
- **Traza por consulta.** Cada consulta tiene un `trace_id` y tres spans: selección de herramienta,
  ejecución de herramienta e interpretación. Así se ve en qué paso se consume el tiempo.
- **SLI** calculados de los eventos: latencia p50 y p95, tiempo hasta el primer token, tokens por
  segundo, tasa de error, tasa de respuestas truncadas, tasa de respaldo y proporción de cifras
  verificadas.
- **SLO** con presupuesto de error: 95 % de consultas en menos de 90 s, 95 % sin error de Ollama y
  90 % de cifras presentes en la evidencia.
- **Privacidad:** el log no guarda el texto de las preguntas ni de las respuestas, solo longitudes y
  un hash corto.

> Las horas entre apertura y resolución del dataset son una variable histórica de otra
> organización. No son SLO, SLA ni MTTR, y no aparecen en la pestaña Observabilidad.

Detalle en [docs/OBSERVABILIDAD.md](docs/OBSERVABILIDAD.md).

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Por defecto se excluyen las pruebas marcadas `vivo`, que necesitan Ollama real. Para correrlas, con
Ollama abierto y el modelo descargado:

```powershell
.\.venv\Scripts\python.exe -m pytest -m vivo -v
```

Comprobaciones adicionales:

```powershell
.\.venv\Scripts\python.exe -m scripts.validar_datos      # 20.769 incidentes, 220 Critical
.\.venv\Scripts\python.exe -m scripts.validar_interfaz   # recorre la interfaz sin navegador
```

Las especificaciones de cada paquete, con sus criterios y los nombres de las pruebas que los
demuestran, están en [docs/specs/README.md](docs/specs/README.md).

## Solución de problemas

| Síntoma | Primer paso |
|---|---|
| `pip install` intenta compilar pandas | El entorno se creó con Python 3.14: bórralo y vuelve a crearlo con `py -3.13 -m venv .venv` |
| «No se pudo conectar con Ollama» | Abre Ollama desde Inicio y pulsa «Verificar Ollama» |
| «Modelo no disponible» | `ollama pull llama3.2:3b` |
| La respuesta tarda mucho | Pulsa «Verificar Ollama» para precalentar el modelo, cierra aplicaciones pesadas o usa `llama3.2:1b` |
| El puerto 8501 está ocupado | `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502` |

Procedimientos completos (síntoma, diagnóstico, comando y verificación) en
[docs/RUNBOOK.md](docs/RUNBOOK.md).

## Licencia y atribución

El código se publica con licencia MIT (ver [LICENSE](LICENSE)).

El dataset conserva su propia licencia y atribución: Shinde, V. (2020), *Incident Response Log*
(versión 2), Kaggle, y Amaral, C., Fantinato, M. y Peres, S. (2018), *Incident management process
enriched event log*, UCI Machine Learning Repository, https://doi.org/10.24432/C57S4H.

## Autor

Kevin Alexis Barrera Llerena, 2026.
