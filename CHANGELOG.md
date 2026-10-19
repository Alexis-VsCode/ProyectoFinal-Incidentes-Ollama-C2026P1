# Cambios

## v1.0.0 (18/10/2026)

Versión entregada del Proyecto Final Integrador C2026P1.

- Aplicación Streamlit de siete pestañas: Datos, Resumen, Explorar, Relaciones y segmentos, Agente
  local, Observabilidad y Arquitectura.
- Carga del Incident Response Log con control de esquema, consolidación por incidente y verificación
  SHA-256 del CSV.
- Análisis con funciones propias: perfil de calidad, indicadores, segmentación por categoría y grupo,
  correlaciones de Pearson y Spearman.
- Gráficos con Plotly (ECDF logarítmica, barras, segmentos) y Seaborn (diagrama de caja y mapa de
  calor).
- Agente con `llama3.2:3b` en Ollama: elección de herramientas autorizadas por tool calling,
  respaldo determinista, lecturas calculadas por Python, verificación de cifras y rechazo de
  acciones.
- Observabilidad del agente: registro JSONL con trace_id, tres SLO y presupuesto de error.
- Comparación medida de `llama3.2:3b` frente a `llama3.2:1b`.
- 519 pruebas con pytest y ejecución automática en Windows y Linux con GitHub Actions.
- Página del proyecto en GitHub Pages y documentación de decisiones, observabilidad y solución de
  problemas.
