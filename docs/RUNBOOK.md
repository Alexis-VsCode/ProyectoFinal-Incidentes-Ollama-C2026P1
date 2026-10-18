# Runbook

Procedimientos para los fallos más probables al ejecutar la aplicación en Windows. Todos los
comandos se ejecutan en PowerShell desde la carpeta del repositorio.

## Ollama no responde

**Síntoma.** El botón «Verificar Ollama» o una consulta muestran "No se pudo conectar con Ollama en
127.0.0.1:11434". El análisis sigue disponible; solo falla el agente.

**Diagnóstico.**

```powershell
ollama --version
Invoke-RestMethod -Uri http://127.0.0.1:11434/api/version
```

Si el segundo comando falla, el servidor de Ollama no está corriendo.

**Comando.** Abre Ollama desde el menú Inicio. Si no tienes la aplicación de bandeja, arráncalo a
mano en otra terminal:

```powershell
ollama serve
```

No lances `ollama serve` si Ollama ya está abierto: el puerto quedaría ocupado.

**Verificación.** `Invoke-RestMethod -Uri http://127.0.0.1:11434/api/version` devuelve la versión
y «Verificar Ollama» termina en verde.

## Modelo no descargado

**Síntoma.** "Modelo no disponible. Ejecute: ollama pull llama3.2:3b".

**Diagnóstico.**

```powershell
ollama list
```

**Comando.**

```powershell
ollama pull llama3.2:3b
# o, en equipos con poca memoria:
ollama pull llama3.2:1b
```

**Verificación.** `ollama list` muestra el modelo y «Verificar Ollama» informa su digest.

## Latencia alta o timeout

**Síntoma.** La respuesta tarda más de un minuto, o aparece "Ollama superó N s sin responder".

**Diagnóstico.** En la pestaña Observabilidad, revisa el span `interpretacion` de la consulta y los
tokens por segundo. En CPU, `llama3.2:3b` genera unos 7 tokens por segundo; la primera consulta
suma la carga del modelo en memoria.

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:11434/api/ps
```

Si la lista está vacía, el modelo no está cargado.

**Comando.**

1. Pulsa «Verificar Ollama» antes de consultar: además de comprobar la conexión, precalienta el
   modelo.
2. Cierra aplicaciones que consuman RAM (navegadores con muchas pestañas, Teams, máquinas
   virtuales).
3. Usa el modo manual del agente, que ahorra la llamada de selección.
4. Si persiste, elige `llama3.2:1b` en la barra lateral.

**Verificación.** `/api/ps` lista el modelo y la siguiente consulta muestra el primer token en
pocos segundos.

## Puerto 8501 ocupado

**Síntoma.** Streamlit informa "Port 8501 is already in use" o abre otra aplicación.

**Diagnóstico.**

```powershell
Get-NetTCPConnection -LocalPort 8501 -State Listen | Select-Object OwningProcess
Get-Process -Id (Get-NetTCPConnection -LocalPort 8501 -State Listen).OwningProcess
```

**Comando.** Si es una instancia anterior de esta aplicación, ciérrala con `Ctrl+C` en su
terminal. Si no, usa otro puerto:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

**Verificación.** La aplicación abre en http://127.0.0.1:8502 (o en 8501 si se liberó).

## CSV inválido o SHA-256 distinto

**Síntoma.** Al cargar un CSV propio aparece un error de columnas, fechas o tamaño; o la pestaña
Observabilidad muestra «SHA-256 difiere» con el CSV incluido.

**Diagnóstico.**

```powershell
Get-FileHash data\incident_event_log.csv -Algorithm SHA256
git status --short data
```

El hash esperado es `5D77EA4853E443CF8447FCA0F3089B7689E5469225CB68F72B2B9FEEA5F2D0D2`. Un hash
distinto suele deberse a una conversión de fin de línea al clonar o a un archivo editado.

**Comando.** Restaura el archivo versionado byte a byte:

```powershell
git checkout -- data/incident_event_log.csv
```

Para un CSV propio: debe tener el mismo esquema de 36 columnas, fechas en formato día/mes/año
hora:minuto y pesar menos de 50 MiB.

**Verificación.**

```powershell
.\.venv\Scripts\python.exe -m scripts.validar_datos
```

Debe informar 119.998 eventos, 36 columnas y 20.769 incidentes, y la pestaña Observabilidad debe
mostrar «SHA-256 coincide».

## Entorno creado con Python 3.14

**Síntoma.** `pip install -r requirements.txt` intenta compilar pandas y falla, o pide un
compilador de C.

**Diagnóstico.**

```powershell
.\.venv\Scripts\python.exe --version
py --list
```

**Comando.** Borra solo el entorno de este proyecto y créalo de nuevo con 3.13:

```powershell
Remove-Item -Recurse -Force .venv
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si `py --list` no muestra 3.13, instálalo desde https://www.python.org/downloads/windows/.

**Verificación.** `.\.venv\Scripts\python.exe --version` muestra `Python 3.13.x` y la instalación
termina sin compilar.
