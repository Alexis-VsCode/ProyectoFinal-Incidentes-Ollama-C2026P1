# Procedencia del CSV

- Dataset de los Talleres 1 y 2: **Incident Response Log**, Vipul Shinde, versión 2.
- Fuente: https://www.kaggle.com/datasets/vipulshinde/incident-response-log/versions/2
- Archivo: `incident_event_log.csv`, recuperado sin modificar del paquete del Taller 1.
- Tamaño: 39.147.503 bytes.
- SHA-256: `5d77ea4853e443cf8447fca0f3089b7689e5469225cb68f72b2b9feea5f2d0d2`.
- Contenido verificado: 119.998 eventos, 36 columnas y 20.769 incidentes únicos.

La fuente original es Amaral, C., Fantinato, M., y Peres, S. (2018), *Incident management process enriched event log*, UCI Machine Learning Repository, https://doi.org/10.24432/C57S4H. La ficha original contiene 141.712 eventos y 24.918 incidentes: no son los conteos del CSV de esta entrega. No se conoce aquí el procedimiento que produjo la diferencia de tamaño; no se atribuye a una limpieza realizada por el estudiante.

La metadata de Kaggle conservada con el taller declara CC0; la ficha de la fuente original UCI declara CC BY 4.0. Se mantiene la atribución a ambas fuentes y la referencia al origen. No se modifica ni se amplía la licencia del conjunto original. El archivo `incident-response-log-metadata.json` preserva la metadata obtenida con el trabajo anterior.

Se reutiliza el archivo original; las transformaciones ocurren en memoria. Si lo descargas de nuevo, selecciona la versión 2 en Kaggle, extrae el CSV y comprueba el hash. No sustituyas silenciosamente este CSV por el de UCI: ambas versiones deben identificarse de forma distinta.

Las columnas `?` se tratan como información desconocida, sin imputación. Las filas de eventos no se eliminan por parecer similares. Se selecciona una última observación por `number` únicamente para la tabla analítica. El dataset está anonimizado y no representa datos de TEUNO ni del ISSPOL.
