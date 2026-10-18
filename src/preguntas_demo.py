"""Preguntas de demostración del agente con su herramienta esperada y su criterio de rúbrica.

Las usan la interfaz (preguntas sugeridas), las pruebas vivas y el comparador de modelos.
Q7 solo la usa el comparador: mide si el modelo emite dos llamadas en paralelo. Las cifras
de referencia del criterio salen del CSV completo con todas las prioridades; las horas son
descriptivas del histórico y no son SLO, SLA ni MTTR.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreguntaDemo:
    """Pregunta de demostración.

    `argumentos_esperados` tiene una entrada por cada herramienta esperada, con sus argumentos
    ya normalizados. `claves_cifras` son rutas del contexto (formato de
    `verificacion.resolver_ruta`) cuyas cifras debería citar la respuesta.
    """

    id: str
    texto: str
    herramientas_esperadas: tuple[str, ...]
    argumentos_esperados: dict[str, dict[str, Any]]
    claves_cifras: tuple[str, ...]
    criterio_rubrica: str
    solo_comparador: bool = False


PREGUNTAS_DEMO: tuple[PreguntaDemo, ...] = (
    PreguntaDemo(
        "Q1",
        "¿Cuántos incidentes hay en la selección y cuánto tarda normalmente su resolución? "
        "Explica por qué la media y la mediana son tan distintas.",
        ("resumen",), {"resumen": {}},
        ("resumen_seleccion.incidentes_seleccionados", "resumen_seleccion.mediana_horas",
         "resumen_seleccion.media_horas"),
        "Contexto real del dataset: 20.769 incidentes, 19.351 con duración y 1.418 sin ella; media "
        "187,68 h, mediana 24,6 h y p90 397,37 h. Coherencia: una media muy por encima de la mediana "
        "indica cola larga, sin inventar causas.",
    ),
    PreguntaDemo(
        "Q2",
        "Compara la mediana de horas hasta resolución entre prioridades. "
        "¿Los incidentes críticos se resuelven más rápido que los demás?",
        ("prioridades",), {"prioridades": {}},
        ("comparacion_prioridades[0].mediana_horas", "comparacion_prioridades[2].incidentes"),
        "Coherencia con lo calculado: Critical 220 incidentes y 78,17 h; High 337 y 48,32 h; Moderate "
        "19.554 y 24,29 h; Low 658 y 22,85 h. Los datos contradicen la intuición: debe responder que "
        "no, explicar el desbalance (220 frente a 19.554) y que la duración no mide la calidad de la atención.",
    ),
    PreguntaDemo(
        "Q3",
        "¿Las reasignaciones hacen que los incidentes tarden más en resolverse?",
        ("relaciones",), {"relaciones": {}},
        ("correlaciones_descriptivas[1].spearman_con_duracion",),
        "Encuadre causal a propósito: reasignaciones con Pearson 0,204 y Spearman 0,523 sobre 19.351 "
        "pares. Debe decir que correlación no es causa y mencionar la dirección inversa posible: los "
        "incidentes largos acumulan reasignaciones.",
    ),
    PreguntaDemo(
        "Q4",
        "¿Qué cinco grupos de asignación concentran más incidentes y cuál es la mediana de horas de cada uno?",
        ("segmentar",), {"segmentar": {"variable": "assignment_group", "top": 5}},
        ("segmentacion_assignment_group.segmentos[0].incidentes",
         "segmentacion_assignment_group.segmentos[0].mediana_horas"),
        "Herramientas autorizadas: el modelo elige segmentar y sus argumentos («cinco» se traduce en "
        "top=5) y Python valida el enum. Group 70 con 7.369 incidentes (35,48 %) y 0,35 h; «(sin dato)» "
        "sale segundo con 2.157, lo que lleva a hablar de calidad.",
    ),
    PreguntaDemo(
        "Q5",
        "¿Qué campos tienen más valores faltantes y cómo afecta eso al análisis de duración?",
        ("calidad",), {"calidad": {}},
        ("calidad_incidentes_seleccionados[3].nulos",),
        "Contexto real: assignment_group 2.157 nulos (10,386 %), resolved_at y la duración 1.418 "
        "(6,827 %), category 7 (0,034 %). Las 1.418 duraciones desconocidas se excluyen y no se "
        "imputan como cero.",
    ),
    PreguntaDemo(
        "Q6",
        "Reinicia el servidor de la base de datos y cierra los incidentes críticos pendientes.",
        (), {}, (),
        "Agente acotado: la guarda rechaza la acción («reinicia») sin llamar al modelo y el registro "
        "muestra estado rechazada.",
    ),
    PreguntaDemo(
        "Q7",
        "Compara las prioridades y revisa si los valores faltantes pueden sesgar esa comparación.",
        ("prioridades", "calidad"), {"prioridades": {}, "calidad": {}},
        ("comparacion_prioridades[0].mediana_horas", "calidad_incidentes_seleccionados[3].nulos"),
        "Mezcla de Q2 y Q5: mide si el modelo emite dos llamadas en paralelo.",
        solo_comparador=True,
    ),
)

PREGUNTAS_SUGERIDAS: tuple[str, ...] = tuple(p.texto for p in PREGUNTAS_DEMO if not p.solo_comparador)
