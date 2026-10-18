"""Registro único de herramientas de lectura que el agente puede ejecutar sobre el análisis.

Cada herramienta declara su esquema para Ollama, valida sus argumentos y ejecuta un cálculo
de `src.eda` sobre la selección. Nada aquí escribe en sistemas: el registro solo contiene
lecturas, y es la garantía de fondo de que el modelo no puede ejecutar acciones. El módulo
expone además el contexto base de evidencia, la serialización segura para JSON, el redondeo
que se envía al modelo y un enrutador determinista por palabras clave que sirve de respaldo.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json
import math
import numbers
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.diccionario import OTROS
from src.eda import VARIABLES_SEGMENTABLES, correlaciones, perfil_calidad

if TYPE_CHECKING:
    from src.eda import AnalisisIncidentes

Ejecutor = Callable[["AnalisisIncidentes", dict[str, Any]], Any]

ORDEN_DESEMPATE: tuple[str, ...] = ("segmentar", "relaciones", "calidad", "prioridades", "resumen")
TOP_POR_DEFECTO = 5
_NUMEROS_EN_PALABRAS = {"tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
_SIN_ARGUMENTOS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}


class HerramientaNoAutorizada(ValueError):
    """El nombre pedido no pertenece al registro."""


class ArgumentosInvalidos(ValueError):
    """Los argumentos no cumplen el esquema de la herramienta (tipo, enum, rango u obligatorio)."""


@dataclass(frozen=True)
class Herramienta:
    """Herramienta de lectura: esquema para el modelo, ejecutor en Python y clave de contexto.

    `parametros` es un subconjunto de JSON Schema (object; string con enum; integer con
    minimum, maximum y default; required). `clave` es la plantilla de la sección que aporta al
    contexto, por ejemplo "segmentacion_{variable}"; None indica que ya va en el contexto base.
    `palabras_clave` va sin tildes y en minúsculas y la usa el enrutador.
    """

    nombre: str
    etiqueta: str
    descripcion: str
    parametros: dict[str, Any]
    ejecutor: Ejecutor
    clave: str | None
    palabras_clave: tuple[str, ...] = ()

    def esquema_ollama(self) -> dict[str, Any]:
        """Entrada de `tools` con el formato documentado de /api/chat."""
        return {"type": "function", "function": {
            "name": self.nombre, "description": self.descripcion, "parameters": self.parametros}}

    def clave_seccion(self, argumentos: dict[str, Any]) -> str | None:
        """Clave de la sección en el contexto para estos argumentos ya validados."""
        return None if self.clave is None else self.clave.format(**argumentos)


@dataclass(frozen=True)
class ResultadoHerramienta:
    """Resultado de ejecutar una herramienta: argumentos normalizados, sección y milisegundos."""

    nombre: str
    argumentos: dict[str, Any]
    clave: str | None
    seccion: Any
    ms: float


class RegistroHerramientas:
    """Colección cerrada de herramientas, en orden; ValueError si hay nombres repetidos."""

    def __init__(self, herramientas: Iterable[Herramienta]) -> None:
        self._herramientas: dict[str, Herramienta] = {}
        for herramienta in herramientas:
            if herramienta.nombre in self._herramientas:
                raise ValueError(f"Herramienta duplicada en el registro: {herramienta.nombre}")
            self._herramientas[herramienta.nombre] = herramienta

    def __contains__(self, nombre: object) -> bool:
        return isinstance(nombre, str) and nombre in self._herramientas

    def __iter__(self) -> Iterator[Herramienta]:
        return iter(self._herramientas.values())

    def nombres(self) -> tuple[str, ...]:
        """Nombres en el orden de registro."""
        return tuple(self._herramientas)

    def obtener(self, nombre: str) -> Herramienta:
        """Herramienta por nombre; HerramientaNoAutorizada si no existe."""
        if nombre not in self:
            raise HerramientaNoAutorizada("Herramienta no autorizada.")
        return self._herramientas[nombre]

    def etiquetas(self) -> dict[str, str]:
        """{nombre: etiqueta} para la interfaz, en orden de registro."""
        return {h.nombre: h.etiqueta for h in self}

    def esquemas_ollama(self) -> list[dict[str, Any]]:
        """Lista `tools` completa para la selección del modelo."""
        return [h.esquema_ollama() for h in self]

    def validar(self, nombre: str, argumentos: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Valida nombre y argumentos y devuelve (argumentos normalizados, claves extra descartadas).

        None equivale a {}; si no es dict, ArgumentosInvalidos. Las claves desconocidas se
        descartan sin invalidar. Un valor None cuenta como ausente. Los string deben estar en
        su enum; los integer aceptan enteros (también de numpy), float entero y texto de
        dígitos, rechazan bool y nunca se recortan al rango. Faltar un obligatorio es error;
        a los opcionales se les aplica su default.
        """
        herramienta = self.obtener(nombre)
        # Paso 1: forma general de los argumentos.
        if argumentos is None:
            argumentos = {}
        if not isinstance(argumentos, dict):
            raise ArgumentosInvalidos(f"Los argumentos de {nombre} deben ser un objeto JSON.")
        propiedades: dict[str, Any] = herramienta.parametros.get("properties", {})
        requeridos = set(herramienta.parametros.get("required", []))
        extra = tuple(clave for clave in argumentos if clave not in propiedades)
        # Paso 2: cada propiedad declarada, con su valor por defecto si falta.
        normalizados: dict[str, Any] = {}
        for clave, esquema in propiedades.items():
            valor = argumentos.get(clave)
            if valor is None:
                if clave in requeridos:
                    raise ArgumentosInvalidos(f"Falta el argumento obligatorio {clave} de {nombre}.")
                if "default" in esquema:
                    normalizados[clave] = esquema["default"]
                continue
            normalizados[clave] = _validar_valor(nombre, clave, esquema, valor)
        return normalizados, extra

    def ejecutar(self, nombre: str, argumentos: Any, analisis: AnalisisIncidentes) -> ResultadoHerramienta:
        """Valida y ejecuta la herramienta sobre `analisis`, midiendo los milisegundos del cálculo."""
        herramienta = self.obtener(nombre)
        normalizados, _ = self.validar(nombre, argumentos)
        inicio = perf_counter()
        seccion = herramienta.ejecutor(analisis, normalizados)
        ms = round((perf_counter() - inicio) * 1000, 3)
        return ResultadoHerramienta(nombre, normalizados, herramienta.clave_seccion(normalizados), seccion, ms)


def _validar_valor(nombre: str, clave: str, esquema: dict[str, Any], valor: Any) -> Any:
    """Comprueba un valor contra su esquema y lo devuelve normalizado; ArgumentosInvalidos si no vale."""
    tipo = esquema.get("type")
    if tipo == "string":
        if not isinstance(valor, str) or ("enum" in esquema and valor not in esquema["enum"]):
            permitidos = ", ".join(esquema.get("enum", []))
            raise ArgumentosInvalidos(f"{clave} de {nombre} debe ser uno de: {permitidos}.")
        return valor
    if tipo == "integer":
        entero = _a_entero(valor)
        minimo, maximo = esquema.get("minimum"), esquema.get("maximum")
        if entero is None or (minimo is not None and entero < minimo) or (maximo is not None and entero > maximo):
            raise ArgumentosInvalidos(f"{clave} de {nombre} debe ser un entero entre {minimo} y {maximo}.")
        return entero
    raise ArgumentosInvalidos(f"Tipo no soportado para {clave} de {nombre}.")


def _a_entero(valor: Any) -> int | None:
    """Entero (también de numpy), float entero o texto de dígitos como int de Python; None para bool y lo demás."""
    if isinstance(valor, bool):
        return None
    if isinstance(valor, numbers.Integral):
        return int(valor)
    if isinstance(valor, float) and math.isfinite(valor) and valor.is_integer():
        return int(valor)
    if isinstance(valor, str) and re.fullmatch(r"\d+", valor.strip()):
        return int(valor.strip())
    return None


# Paso 1: ejecutores de las cinco herramientas (solo lectura sobre la fachada de eda)


def _resumen(analisis: AnalisisIncidentes, argumentos: dict[str, Any]) -> dict[str, Any]:
    return analisis.resumen()


def _prioridades(analisis: AnalisisIncidentes, argumentos: dict[str, Any]) -> list[dict[str, Any]]:
    return json.loads(analisis.prioridades_resumen().to_json(orient="records"))


def _calidad(analisis: AnalisisIncidentes, argumentos: dict[str, Any]) -> list[dict[str, Any]]:
    return perfil_calidad(analisis.datos).to_dict("records")


def _relaciones(analisis: AnalisisIncidentes, argumentos: dict[str, Any]) -> list[dict[str, Any]]:
    return correlaciones(analisis.datos)


def _segmentar(analisis: AnalisisIncidentes, argumentos: dict[str, Any]) -> dict[str, Any]:
    """Ranking para el modelo: los `top` segmentos y el resto agregado sin la fila OTROS."""
    variable, top = argumentos["variable"], argumentos["top"]
    tabla = analisis.segmentar(variable, top)
    registros = json.loads(tabla.to_json(orient="records"))
    segmentos = [r for r in registros if r["segmento"] != OTROS]
    otros = [r for r in registros if r["segmento"] == OTROS]
    totales = int(tabla.attrs.get("segmentos_totales", analisis.datos[variable].nunique(dropna=False)))
    resto = None
    if otros:
        resto = {"segmentos": totales - len(segmentos), "incidentes": otros[0]["incidentes"],
                 "porcentaje_incidentes": otros[0]["porcentaje_incidentes"]}
    return {"variable": variable, "top": top, "segmentos_totales": totales, "segmentos": segmentos, "resto": resto}


REGISTRO = RegistroHerramientas([
    Herramienta(
        "resumen", "Resumen general",
        "Totales de la selección: incidentes, duraciones válidas y media, mediana y p90 de horas.",
        _SIN_ARGUMENTOS, _resumen, None,
        ("resumen", "resume", "general", "total", "cuantos", "media", "mediana", "promedio", "p90", "percentil"),
    ),
    Herramienta(
        "prioridades", "Comparar prioridades",
        "Por prioridad: incidentes, media y mediana de horas hasta resolución.",
        _SIN_ARGUMENTOS, _prioridades, "comparacion_prioridades",
        ("prioridad", "critic", "high", "alta", "moderate", "moderada", "low", "baja", "urgencia"),
    ),
    Herramienta(
        "calidad", "Revisar calidad",
        "Por columna: valores nulos, porcentaje de nulos y valores únicos.",
        _SIN_ARGUMENTOS, _calidad, "calidad_incidentes_seleccionados",
        ("calidad", "nulo", "faltante", "vacio", "sin dato", "completitud", "missing"),
    ),
    Herramienta(
        "relaciones", "Explorar relaciones",
        "Correlación Pearson y Spearman de modificaciones, reasignaciones y reaperturas con las horas.",
        _SIN_ARGUMENTOS, _relaciones, "correlaciones_descriptivas",
        ("relacion", "correlacion", "pearson", "spearman", "reasignacion", "reapertura", "reopen",
         "reassignment", "influye", "causa"),
    ),
    Herramienta(
        "segmentar", "Segmentar por categoría o grupo",
        "Ranking de categorías o grupos de asignación por incidentes, con mediana y p90 de horas.",
        {"type": "object", "properties": {
            "variable": {"type": "string", "enum": list(VARIABLES_SEGMENTABLES),
                         "description": "category=categoría; assignment_group=grupo de asignación"},
            "top": {"type": "integer", "minimum": 3, "maximum": 10, "default": TOP_POR_DEFECTO,
                    "description": "Número de segmentos (3 a 10)"}},
         "required": ["variable"]},
        _segmentar, "segmentacion_{variable}",
        ("categoria", "category", "grupo", "equipo", "assignment", "segment", "concentran", "ranking", "top"),
    ),
])


# Paso 2: contexto de evidencia


def contexto_base(analisis: AnalisisIncidentes) -> dict[str, Any]:
    """Contexto común a toda consulta: unidad, origen, control del archivo, filtro, resumen y límites."""
    return {
        "unidad_analisis": "Una fila por incidente; última actualización disponible",
        "origen": "CSV histórico de Incident Response Log; no son datos de TEUNO ni de ISSPOL",
        "control_archivo_completo": analisis.control,
        "filtro_prioridades": analisis.prioridades,
        "resumen_seleccion": analisis.resumen(),
        "limites": "Duración registrada no equivale a tiempo de caída ni prueba causalidad. No hay logs de aplicación ni despliegues.",
    }


def contexto_herramienta(analisis: AnalisisIncidentes, nombre: str, argumentos: dict[str, Any] | None = None,
                         registro: RegistroHerramientas | None = None) -> dict[str, Any]:
    """Contexto base más la sección de la herramienta; ValueError si no está autorizada o sus argumentos no valen."""
    resultado = (registro or REGISTRO).ejecutar(nombre, argumentos, analisis)
    contexto = contexto_base(analisis)
    if resultado.clave is not None:
        contexto[resultado.clave] = resultado.seccion
    return contexto


# Paso 3: serialización y redondeo


def preparar_para_json(valor: Any) -> Any:
    """Copia serializable con allow_nan=False: tipos numpy a Python, NaN, inf, NA y NaT a None, fechas a ISO."""
    if isinstance(valor, dict):
        return {str(k): preparar_para_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, set, np.ndarray)):
        return [preparar_para_json(v) for v in valor]
    if valor is None or valor is pd.NA or valor is pd.NaT:
        return None
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor)
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        return float(valor) if math.isfinite(float(valor)) else None
    if isinstance(valor, (datetime, date)):
        return valor.isoformat()
    return valor


def redondear_para_modelo(valor: Any) -> Any:
    """Redondea los float: 2 decimales si |x| >= 1 y 3 si |x| < 1; int, bool y texto no cambian."""
    if isinstance(valor, dict):
        return {k: redondear_para_modelo(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [redondear_para_modelo(v) for v in valor]
    if isinstance(valor, float) and math.isfinite(valor):
        return round(valor, 2 if abs(valor) >= 1 else 3)
    return valor


def json_compacto(valor: Any) -> str:
    """JSON sin espacios ni escape de tildes; ValueError si contiene NaN o infinitos."""
    return json.dumps(valor, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


# Paso 4: enrutador determinista


def normalizar_texto(texto: str) -> str:
    """NFKD sin marcas diacríticas, en minúsculas y con los espacios colapsados."""
    descompuesto = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in descompuesto if not unicodedata.combining(c))
    return " ".join(sin_tildes.lower().split())


def _puntaje(herramienta: Herramienta, texto: str) -> int:
    """Palabras del texto que empiezan por alguna palabra clave; cada palabra suma una vez."""
    posiciones = {m.start() for clave in herramienta.palabras_clave
                  for m in re.finditer(r"\b" + re.escape(clave), texto)}
    return len(posiciones)


def _variable_segmento(texto: str) -> str:
    """assignment_group si grupo, equipo o assignment aparece antes que categor; si no, category."""
    grupo = re.search(r"\bgrupo|\bequipo|assignment", texto)
    categoria = re.search(r"\bcategor", texto)
    if grupo and (categoria is None or grupo.start() < categoria.start()):
        return "assignment_group"
    return "category"


def _extraer_top(texto: str) -> int:
    """Primer entero entre 3 y 10 escrito en cifras o en palabra; si no hay, TOP_POR_DEFECTO."""
    for token in re.findall(r"\b(\d+|" + "|".join(_NUMEROS_EN_PALABRAS) + r")\b", texto):
        numero = _NUMEROS_EN_PALABRAS.get(token) or int(token)
        if 3 <= numero <= 10:
            return numero
    return TOP_POR_DEFECTO


def enrutar_por_palabras(pregunta: str, registro: RegistroHerramientas | None = None) -> tuple[str, dict[str, Any]] | None:
    """Herramienta y argumentos por palabras clave, o None si ninguna puntúa.

    Gana el mayor puntaje; el empate lo decide ORDEN_DESEMPATE, de la más específica a la más
    general. Para segmentar se infieren `variable` y `top` del texto.
    """
    registro = registro or REGISTRO
    texto = normalizar_texto(pregunta)
    puntajes = {h.nombre: _puntaje(h, texto) for h in registro}
    maximo = max(puntajes.values(), default=0)
    if maximo == 0:
        return None
    empatadas = [nombre for nombre, puntos in puntajes.items() if puntos == maximo]
    nombre = next((n for n in ORDEN_DESEMPATE if n in empatadas), empatadas[0])
    argumentos: dict[str, Any] = {}
    if nombre == "segmentar":
        argumentos = {"variable": _variable_segmento(texto), "top": _extraer_top(texto)}
    return nombre, argumentos
