# Observabilidad de la aplicación

La aplicación aplica prácticas de SRE a sí misma: registra eventos estructurados, traza cada
consulta al agente y calcula indicadores (SLI) y objetivos (SLO) de su propio servicio.

> **Regla de dominio.** Los SLI y SLO miden la aplicación y su agente: latencia de la consulta,
> errores de Ollama y cifras verificadas. La duración entre apertura y resolución de los incidentes
> del CSV es una variable histórica y descriptiva de otra organización. **No es SLO, ni SLA, ni
> MTTR**, y no aparece en la pestaña Observabilidad.

Todo se implementa con la biblioteca estándar (`logging`, `json`, `uuid`, `time`, `datetime`) en
`src/observabilidad.py`.

## Qué se registra y dónde

- **Archivo:** `resultados_locales/logs/app.jsonl`, una línea JSON por evento. La carpeta está
  excluida de Git.
- **Comparador de modelos:** escribe en un archivo aparte,
  `resultados_locales/logs/comparacion_<fecha>.jsonl`, para que el tráfico de prueba no se mezcle
  con el uso real.
- Sin configurar el logger no se escribe nada (handler nulo): las pruebas no ensucian la carpeta.

## Campos de cada línea

| Campo | Contenido |
|---|---|
| `ts_utc` | Fecha y hora UTC en ISO 8601 con milisegundos |
| `nivel` | `INFO`, `WARNING` o `ERROR` |
| `evento` | Nombre del evento o del span |
| `trace_id` | 32 caracteres hexadecimales; identifica la consulta |
| `atributos` | Datos del evento (ver el catálogo) |

## Catálogo de eventos

| Evento | Nivel | Atributos principales |
|---|---|---|
| `consulta_iniciada` | INFO | modo, modelo, longitud de la pregunta, hash corto de la pregunta |
| `guarda_rechazo` | WARNING | categoría (acción, código, manipulación) y coincidencia |
| `seleccion_herramienta` (span) | INFO o ERROR | llamadas propuestas y válidas, `done_reason`, tokens de prompt y de salida |
| `respaldo_activado` | WARNING | motivo (sin llamada, herramienta no autorizada, argumentos inválidos, modelo sin herramientas, selección truncada), herramienta y origen |
| `ejecucion_herramienta` (span) | INFO o ERROR | herramienta, argumentos, origen |
| `interpretacion` (span) | INFO o ERROR | `done_reason`, truncada, tokens, tokens por segundo, tiempo hasta el primer token |
| `ollama_error` | ERROR | tipo (conexión, timeout, no instalado, memoria, http, stream...), operación y código HTTP |
| `consulta_finalizada` | INFO o WARNING | estado, segundos, primer token, tokens por segundo, origen de la selección, cifras evaluadas y verificadas, tipo de error |

## Trazas y spans

Cada consulta recibe un `trace_id` al empezar. Dentro de ella hay tres spans secuenciales:

1. `seleccion_herramienta`: el modelo propone qué herramienta usar (solo en modo automático).
2. `ejecucion_herramienta`: Python ejecuta cada herramienta válida (uno por herramienta).
3. `interpretacion`: el modelo redacta la respuesta en streaming.

Cada span mide su duración en milisegundos y su estado (`ok`, `error` o `cancelado`). Con
`llama3.2:3b` en CPU, casi todo el tiempo se va en la interpretación, porque el modelo genera unos
7 tokens por segundo.

**Invariante:** cada `trace_id` emite exactamente un evento `consulta_finalizada`, con estado
`respondida`, `rechazada`, `error` o `cancelada`. Los errores de entrada (pregunta vacía,
herramienta no autorizada en modo manual) se rechazan antes de crear la traza y no cuentan como
eventos del servicio.

## SLI

Los eventos válidos para los SLI son las consultas respondidas más las que terminaron en error. Las
rechazadas no llaman al modelo y las canceladas las cortó el usuario.

| SLI | Cálculo |
|---|---|
| Latencia p50 y p95 | Percentil de los segundos de punta a punta de las consultas respondidas |
| Primer token p50 y p95 | Percentil del tiempo hasta el primer fragmento: la espera que percibe el usuario |
| Tokens por segundo p50 | Métrica de Ollama en la interpretación (`eval_count / eval_duration`) |
| Tasa de error | Errores de Ollama / consultas válidas, con desglose por tipo |
| Tasa de truncadas | Respuestas con `done_reason = length` / respondidas |
| Selección por el modelo y respaldo | En modo automático: proporción elegida por el modelo frente al respaldo determinista |
| Cifras verificadas | Suma de cifras verificadas / suma de cifras evaluadas |

Los percentiles usan el método de rango más cercano. Si no hay eventos, el SLI vale `None`.

## SLO y presupuesto de error

| SLO | Objetivo | Evento bueno |
|---|---|---|
| `latencia_consulta` | 95 % | Consulta respondida en menos de 90 s (90,0 s cuenta como mala) |
| `disponibilidad_agente` | 95 % | Consulta válida terminada sin error de Ollama |
| `cifras_verificadas` | 90 % | Cifra citada en la respuesta que existe en la evidencia enviada |

Para cada SLO se calcula:

- **cumplimiento** = buenos / válidos;
- **presupuesto total** = (1 - objetivo) x válidos;
- **presupuesto restante** = total - (válidos - buenos), que puede ser negativo;
- **suficiente**: hacen falta al menos 20 eventos válidos; con menos, el resultado es indicativo.

La ventana es el lote: el archivo de log (filtrable por fecha y modelo) o una corrida del
comparador. Una aplicación local sin tráfico continuo no justifica ventanas móviles de 28 días.

El umbral de latencia es un supuesto inicial que se calibra con las mediciones del comparador. El
objetivo de cifras no se baja para que se cumpla: si no se alcanza, se revisa el prompt o el modelo.

Qué hacer si se agota un presupuesto:

- **Latencia:** usar el modo manual (ahorra la selección), precalentar el modelo o reducir la
  longitud de la respuesta.
- **Disponibilidad:** revisar que Ollama esté abierto, la RAM libre y el precalentamiento.
- **Cifras:** revisar el prompt de interpretación o comparar modelos.

## Cómo leerlo

En la aplicación, la pestaña **Observabilidad** muestra la salud de Ollama, la integridad del CSV,
la tabla de SLI frente a SLO con el presupuesto restante y las últimas consultas con su
`trace_id`. Desde ahí se puede descargar el log.

Desde PowerShell:

```powershell
# Últimas 20 líneas
Get-Content resultados_locales\logs\app.jsonl -Tail 20

# Solo los cierres de consulta
Get-Content resultados_locales\logs\app.jsonl | ConvertFrom-Json |
    Where-Object evento -eq 'consulta_finalizada' |
    Select-Object ts_utc, trace_id, @{n='estado';e={$_.atributos.estado}}, @{n='segundos';e={$_.atributos.segundos}}

# Todos los eventos de una consulta
Get-Content resultados_locales\logs\app.jsonl | ConvertFrom-Json | Where-Object trace_id -eq '<trace_id>'
```

La diferencia entre la duración de un span y la `total_duration` que informa Ollama es el costo de
HTTP y de Python.

## Privacidad

El log **no guarda** el texto de las preguntas, el contexto enviado al modelo ni las respuestas.
Solo guarda longitudes y un hash corto de la pregunta (`pregunta_sha256_12`). La evidencia
completa de una consulta está en el JSON que el usuario descarga desde la pestaña del agente, y
solo si decide descargarlo.

Para empezar de cero basta con borrar `resultados_locales/logs/app.jsonl` con la aplicación
detenida.

## Lo que se dejó fuera

OpenTelemetry, Prometheus, Grafana y cualquier exportador; alertas y tasas de consumo; ventanas
móviles; rotación de logs; spans anidados; propagar el `trace_id` a Ollama; y SLO sobre tokens por
segundo o sobre la tasa de respaldo, que son señales de diagnóstico y no promesas al usuario.
