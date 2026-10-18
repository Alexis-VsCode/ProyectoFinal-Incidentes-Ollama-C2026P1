# Procedencia del CSV

- Dataset: *Incident Response Log*, Vipul Shinde, versión 2.
- Fuente: https://www.kaggle.com/datasets/vipulshinde/incident-response-log/versions/2
- Archivo: `incident_event_log.csv`, sin modificaciones respecto de la descarga.
- Tamaño: 39.147.503 bytes.
- SHA-256: `5d77ea4853e443cf8447fca0f3089b7689e5469225cb68f72b2b9feea5f2d0d2`.
- Contenido: 119.998 eventos, 36 columnas y 20.769 incidentes únicos.

## Fuente original y licencia

La fuente original es Amaral, C., Fantinato, M. y Peres, S. (2018), *Incident management process
enriched event log*, UCI Machine Learning Repository, https://doi.org/10.24432/C57S4H. La ficha de
UCI describe 141.712 eventos y 24.918 incidentes: no son los conteos de este CSV. Se desconoce el
procedimiento que produjo la diferencia de tamaño, por lo que no se atribuye a ninguna limpieza
propia.

Los metadatos de Kaggle, conservados en `incident-response-log-metadata.json`, declaran la
licencia CC0; la ficha de UCI declara CC BY 4.0. Se mantiene la atribución a ambas fuentes y no se
modifica ni se amplía la licencia del conjunto original.

## Cómo obtenerlo de nuevo

1. Abre el enlace de Kaggle y selecciona la versión 2.
2. Descarga y extrae `incident_event_log.csv` en esta carpeta.
3. Comprueba el hash:

```powershell
Get-FileHash data\incident_event_log.csv -Algorithm SHA256
```

No sustituyas este CSV por el de UCI: son versiones distintas y deben identificarse por separado.
El archivo se versiona byte a byte (`.gitattributes` con `-text`) para que el hash no cambie al
clonar.

## Tratamiento en la aplicación

- Las transformaciones ocurren en memoria; el archivo no se modifica.
- Los valores `?` se tratan como desconocidos, sin imputación.
- Las filas de eventos no se eliminan por parecer similares.
- Para la tabla analítica se conserva la última observación de cada `number`.
- El dataset está anonimizado y corresponde a otra organización.
