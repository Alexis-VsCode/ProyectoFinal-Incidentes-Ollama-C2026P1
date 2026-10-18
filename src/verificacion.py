"""Verificador de cifras: comprueba que cada número de una respuesta existe en la evidencia enviada.

Son funciones puras. Se extraen las cifras del texto (formatos español e inglés, con la
ambigüedad miles/decimal resuelta en dos candidatos), se comparan con los números del
contexto tal como se envió al modelo, exigiendo redondeo correcto, y se clasifican en
verificadas, no verificadas o ignoradas (años, enteros pequeños, números citados en la
pregunta, números de una etiqueta de segmento y formatos irregulares). Las unidades admitidas son "%" (valor o fracción) y
"h"/"horas" (sin conversión); no se convierten días, minutos ni multiplicadores.

Límite declarado: se comprueba que la cifra está en la evidencia, no que se atribuya al
concepto correcto. Por eso la revisión humana sigue siendo obligatoria.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import math
import numbers
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

EstadoCifra = Literal["verificada", "no_verificada", "ignorada"]
MotivoIgnorada = Literal["anio", "entero_pequeno", "en_pregunta", "formato", "etiqueta"]
CLAVES_EXCLUIDAS = ("sha256", "digest")
DECIMALES_PROPORCION = 3
EQUIVALENCIAS_ETIQUETA = {"grupo": "group", "categoria": "category", "subcategoria": "subcategory"}

_ETIQUETA_CONTEXTO = re.compile(r"\s*([^\W\d_]+) (\d+)\s*")
_PALABRA_PREVIA = re.compile(r"([^\W\d_]+) $")

_MASCARAS = (
    re.compile(r"\d{4}-\d{2}(?:-\d{2})?(?:[ T]\d{2}:\d{2}(?::\d{2})?)?"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}(?: \d{1,2}:\d{2})?"),
    re.compile(r"\d{1,2}:\d{2}(?::\d{2})?"),
    re.compile(r"^\s*(?:[-*]\s*)?\d{1,2}[.)](?=\s)", re.MULTILINE),
)
_TOKEN = re.compile(
    r"(?<![\w.,/−-])(?P<signo>[-−])?(?P<num>\d+(?:[.,]\d+)*)(?![.,]?\d)"
    r"(?:\s?(?P<unidad>%|(?:horas?|hrs?|h)\b)|(?!\w))",
    re.IGNORECASE,
)
_RUTA_VALIDA = re.compile(r"[^.\[\]]+(?:\[\d+\])*(?:\.[^.\[\]]+(?:\[\d+\])*)*")
_SEGMENTO_RUTA = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


@dataclass(frozen=True)
class CifraExtraida:
    """Número leído del texto: texto literal, posición, candidatos con sus decimales y unidad.

    `candidatos` tiene dos valores si la lectura es ambigua (miles o decimal) y está vacío si
    el formato es irregular. `unidad` es None, "%" o "h".
    """

    texto: str
    posicion: int
    candidatos: tuple[float, ...]
    decimales: tuple[int, ...]
    unidad: str | None


@dataclass(frozen=True)
class CifraVerificada:
    """Clasificación de una cifra; `referencia` es el número del contexto más cercano si coincide."""

    cifra: CifraExtraida
    estado: EstadoCifra
    referencia: float | None
    motivo: MotivoIgnorada | None

    def a_dict(self) -> dict[str, Any]:
        datos: dict[str, Any] = {"texto": self.cifra.texto, "posicion": self.cifra.posicion}
        if self.estado == "verificada":
            datos["referencia"] = self.referencia
        elif self.estado == "no_verificada":
            datos["candidatos"] = list(self.cifra.candidatos)
            datos["unidad"] = self.cifra.unidad
        else:
            datos["motivo"] = self.motivo
        return datos


@dataclass(frozen=True)
class VerificacionCifras:
    """Resultado de verificar un texto: cada aparición cuenta por separado."""

    cifras: tuple[CifraVerificada, ...]

    def _por_estado(self, estado: EstadoCifra) -> tuple[CifraVerificada, ...]:
        return tuple(c for c in self.cifras if c.estado == estado)

    @property
    def verificadas(self) -> tuple[CifraVerificada, ...]:
        return self._por_estado("verificada")

    @property
    def no_verificadas(self) -> tuple[CifraVerificada, ...]:
        return self._por_estado("no_verificada")

    @property
    def ignoradas(self) -> tuple[CifraVerificada, ...]:
        return self._por_estado("ignorada")

    @property
    def evaluadas(self) -> int:
        """Verificadas más no verificadas; las ignoradas no cuentan."""
        return len(self.verificadas) + len(self.no_verificadas)

    @property
    def proporcion(self) -> float | None:
        """Verificadas / evaluadas con 3 decimales; None si no se evaluó ninguna."""
        if self.evaluadas == 0:
            return None
        return round(len(self.verificadas) / self.evaluadas, DECIMALES_PROPORCION)

    def a_dict(self) -> dict[str, Any]:
        """Resumen serializable: conteos, proporción y el detalle de cada grupo."""
        return {
            "evaluadas": self.evaluadas,
            "verificadas": len(self.verificadas),
            "proporcion": self.proporcion,
            "detalle_verificadas": [c.a_dict() for c in self.verificadas],
            "no_verificadas": [c.a_dict() for c in self.no_verificadas],
            "ignoradas": [c.a_dict() for c in self.ignoradas],
        }


def _enmascarar(texto: str) -> str:
    # Paso 1: fechas, horas y marcadores de lista se cambian por espacios (se conservan posiciones)
    for patron in _MASCARAS:
        texto = patron.sub(lambda m: "".join("\n" if c == "\n" else " " for c in m.group()), texto)
    return texto


def _leer_numero(num: str) -> tuple[tuple[float, ...], tuple[int, ...]] | None:
    """Candidatos y decimales de un número escrito; None si el formato es irregular."""
    separadores = [c for c in num if c in ".,"]
    if not separadores:
        return (float(int(num)),), (0,)
    if len(set(separadores)) == 1:
        partes = num.split(separadores[0])
        if len(partes) == 2:
            entero, fraccion = partes
            if len(fraccion) == 3 and 1 <= len(entero) <= 3 and entero != "0":
                return (float(entero + fraccion), float(f"{entero}.{fraccion}")), (0, 3)
            return (float(f"{entero}.{fraccion}"),), (len(fraccion),)
        if 1 <= len(partes[0]) <= 3 and all(len(p) == 3 for p in partes[1:]):
            return (float("".join(partes)),), (0,)
        return None
    ultimo = max(num.rfind("."), num.rfind(","))
    separador_decimal = num[ultimo]
    entero, fraccion = num[:ultimo], num[ultimo + 1:]
    if separador_decimal in entero:
        return None
    grupos = entero.split("," if separador_decimal == "." else ".")
    if not (1 <= len(grupos[0]) <= 3 and all(len(g) == 3 for g in grupos[1:])):
        return None
    return (float(f"{''.join(grupos)}.{fraccion}"),), (len(fraccion),)


def extraer_cifras(texto: str) -> list[CifraExtraida]:
    """Cifras del texto en orden de aparición, sin fechas, horas, marcadores ni identificadores.

    Un número se toma completo o no se toma: si va pegado a letras que no son unidad
    ("1.5x", "24,6hs", "3.2B") no se extrae ninguna parte de él.
    """
    cifras = []
    for m in _TOKEN.finditer(_enmascarar(texto)):
        # Paso 2: lectura del número y del signo; un formato irregular queda sin candidatos
        lectura = _leer_numero(m.group("num"))
        candidatos, decimales = lectura if lectura is not None else ((), ())
        if m.group("signo"):
            candidatos = tuple(-c for c in candidatos)
        unidad = m.group("unidad")
        if unidad is not None:
            unidad = "%" if unidad == "%" else "h"
        cifras.append(CifraExtraida(texto[m.start():m.end()], m.start(), candidatos, decimales, unidad))
    return cifras


def _clave_excluida(clave: str) -> bool:
    clave = clave.lower()
    return any(excluida in clave for excluida in CLAVES_EXCLUIDAS)


def numeros_de_contexto(contexto: Any) -> list[float]:
    """Números de referencia del contexto enviado, sin repetir y en orden de aparición.

    Se toman int y float finitos (no bool), y los números de los textos (claves y valores)
    con el mismo tokenizador. Se excluye todo lo que cuelga de claves con "sha256" o "digest".
    """
    vistos: dict[float, None] = {}

    def recorrer(valor: Any) -> None:
        if valor is None or isinstance(valor, bool):
            return
        if isinstance(valor, numbers.Real):
            numero = float(valor)
            if math.isfinite(numero):
                vistos.setdefault(numero, None)
        elif isinstance(valor, str):
            for cifra in extraer_cifras(valor):
                for candidato in cifra.candidatos:
                    vistos.setdefault(candidato, None)
        elif isinstance(valor, dict):
            for clave, subvalor in valor.items():
                if _clave_excluida(str(clave)):
                    continue
                recorrer(str(clave))
                recorrer(subvalor)
        elif isinstance(valor, (list, tuple, set, frozenset)):
            for subvalor in valor:
                recorrer(subvalor)

    recorrer(contexto)
    return list(vistos)


def _normalizar_palabra(palabra: str) -> str:
    """Palabra sin tildes, en minúsculas y con su equivalente en inglés si lo tiene ("grupo" pasa a "group")."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFKD", palabra) if not unicodedata.combining(c)).lower()
    return EQUIVALENCIAS_ETIQUETA.get(sin_tildes, sin_tildes)


def etiquetas_de_contexto(contexto: Any) -> frozenset[tuple[str, int]]:
    """Pares (palabra normalizada, entero) de los textos del contexto que son una etiqueta completa.

    Solo cuenta un valor de texto que entero es "Palabra N", como "Group 70" o "Category 26";
    las frases largas no aportan etiquetas. Se excluye lo que cuelga de claves con "sha256" o "digest".
    """
    pares: set[tuple[str, int]] = set()

    def recorrer(valor: Any) -> None:
        if isinstance(valor, str):
            encontrada = _ETIQUETA_CONTEXTO.fullmatch(valor)
            if encontrada:
                pares.add((_normalizar_palabra(encontrada.group(1)), int(encontrada.group(2))))
        elif isinstance(valor, dict):
            for clave, subvalor in valor.items():
                if not _clave_excluida(str(clave)):
                    recorrer(subvalor)
        elif isinstance(valor, (list, tuple, set, frozenset)):
            for subvalor in valor:
                recorrer(subvalor)

    recorrer(contexto)
    return frozenset(pares)


def _es_etiqueta(texto: str, cifra: CifraExtraida, etiquetas: frozenset[tuple[str, int]]) -> bool:
    """True si la cifra es un entero sin signo ni unidad pegado, tras un espacio, a la palabra de una etiqueta."""
    if not etiquetas or cifra.unidad is not None or cifra.decimales != (0,) or cifra.texto[:1] in "-−":
        return False
    previa = _PALABRA_PREVIA.search(texto, 0, cifra.posicion)
    return previa is not None and (_normalizar_palabra(previa.group(1)), int(cifra.candidatos[0])) in etiquetas


def _coincide(valor: float, decimales: int, objetivo: float) -> float | None:
    """Distancia si `valor` es el redondeo correcto de `objetivo` con esos decimales; si no, None."""
    distancia = abs(valor - objetivo)
    tolerancia = 0.5 * 10 ** (-decimales) + 1e-9 * max(1.0, abs(objetivo))
    return distancia if distancia <= tolerancia else None


def _clasificar(cifra: CifraExtraida, referencias: Sequence[float],
                de_pregunta: Sequence[float], es_etiqueta: bool = False) -> CifraVerificada:
    if not cifra.candidatos:
        return CifraVerificada(cifra, "ignorada", None, "formato")
    if es_etiqueta:
        return CifraVerificada(cifra, "ignorada", None, "etiqueta")
    # Paso 3: coincidencia con alguna referencia, con la unidad aplicada a la referencia
    mejor: float | None = None
    menor = math.inf
    for valor, decimales in zip(cifra.candidatos, cifra.decimales):
        for referencia in referencias:
            objetivos = (referencia, referencia * 100) if cifra.unidad == "%" else (referencia,)
            for objetivo in objetivos:
                distancia = _coincide(valor, decimales, objetivo)
                if distancia is not None and distancia < menor:
                    mejor, menor = referencia, distancia
    if mejor is not None:
        return CifraVerificada(cifra, "verificada", mejor, None)
    # Paso 4: lo que no coincide se ignora si viene de la pregunta, es un año o un entero pequeño
    if any(abs(v - p) <= 1e-9 * max(1.0, abs(p)) for v in cifra.candidatos for p in de_pregunta):
        return CifraVerificada(cifra, "ignorada", None, "en_pregunta")
    if cifra.unidad is None and cifra.decimales == (0,):
        valor = cifra.candidatos[0]
        if 1900 <= valor <= 2100:
            return CifraVerificada(cifra, "ignorada", None, "anio")
        if 0 <= valor <= 10:
            return CifraVerificada(cifra, "ignorada", None, "entero_pequeno")
    return CifraVerificada(cifra, "no_verificada", None, None)


def verificar_cifras(texto: str, contexto: Any, pregunta: str = "") -> VerificacionCifras:
    """Clasifica cada cifra de `texto` contra los números del `contexto` enviado al modelo.

    El número que forma parte de una etiqueta de segmento del contexto ("Group 70") se ignora
    con motivo "etiqueta" y no entra en la proporción.
    """
    referencias = numeros_de_contexto(contexto)
    etiquetas = etiquetas_de_contexto(contexto)
    de_pregunta = [c for cifra in extraer_cifras(pregunta) for c in cifra.candidatos]
    return VerificacionCifras(tuple(_clasificar(c, referencias, de_pregunta, _es_etiqueta(texto, c, etiquetas))
                                    for c in extraer_cifras(texto)))


def resolver_ruta(contexto: Any, ruta: str) -> Any:
    """Valor de una ruta como "segmentacion_assignment_group.segmentos[0].incidentes".

    Lanza ValueError si la ruta está mal formada y LookupError (KeyError o IndexError) si no existe.
    """
    if not _RUTA_VALIDA.fullmatch(ruta or ""):
        raise ValueError(f"Ruta mal formada: {ruta!r}")
    actual = contexto
    for clave, indice in _SEGMENTO_RUTA.findall(ruta):
        if clave:
            if not isinstance(actual, dict) or clave not in actual:
                raise KeyError(ruta)
            actual = actual[clave]
        else:
            posicion = int(indice)
            if not isinstance(actual, (list, tuple)) or posicion >= len(actual):
                raise IndexError(ruta)
            actual = actual[posicion]
    return actual


def cobertura_cifras(verificacion: VerificacionCifras, contexto: Any, rutas: Sequence[str]) -> float | None:
    """Proporción de rutas clave (resueltas y numéricas) cuyo valor aparece entre las referencias verificadas.

    Las rutas que no existen no cuentan. Devuelve None si ninguna se resolvió; 3 decimales.
    """
    referencias = [c.referencia for c in verificacion.verificadas if c.referencia is not None]
    resueltas = cubiertas = 0
    for ruta in rutas:
        try:
            valor = resolver_ruta(contexto, ruta)
        except (LookupError, ValueError):
            continue
        if isinstance(valor, bool) or not isinstance(valor, numbers.Real) or not math.isfinite(float(valor)):
            continue
        resueltas += 1
        numero = float(valor)
        if any(abs(numero - r) <= 1e-9 * max(1.0, abs(numero)) for r in referencias):
            cubiertas += 1
    if resueltas == 0:
        return None
    return round(cubiertas / resueltas, DECIMALES_PROPORCION)
