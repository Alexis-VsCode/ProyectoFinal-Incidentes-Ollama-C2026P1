"""Pruebas del verificador de cifras: extracción, formatos, tolerancia y clasificación.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

import json

import pytest

from src.verificacion import (
    cobertura_cifras,
    extraer_cifras,
    numeros_de_contexto,
    resolver_ruta,
    verificar_cifras,
)

CONTEXTO = {
    "resumen_seleccion": {"incidentes_seleccionados": 20769, "duraciones_validas": 19351,
                          "media_horas": 187.68, "mediana_horas": 24.6, "p90_horas": 397.37},
    "comparacion_prioridades": [
        {"priority": "1 - Critical", "incidentes": 220, "mediana_horas": 78.17},
        {"priority": "3 - Moderate", "incidentes": 19554, "mediana_horas": 24.29},
    ],
}


def estados(texto: str, contexto=CONTEXTO, pregunta: str = "") -> list[tuple[str, str, str | None]]:
    resultado = verificar_cifras(texto, contexto, pregunta)
    return [(c.cifra.texto, c.estado, c.motivo) for c in resultado.cifras]


def estado_unico(texto: str, contexto=CONTEXTO, pregunta: str = "") -> str:
    lista = verificar_cifras(texto, contexto, pregunta).cifras
    assert len(lista) == 1, lista
    return lista[0].estado


# Paso 1: extracción y lectura de formatos


@pytest.mark.parametrize(("texto", "candidatos", "unidad"), [
    ("24,6", (24.6,), None),
    ("78.17", (78.17,), None),
    ("187,68 h", (187.68,), "h"),
    ("24,6h", (24.6,), "h"),
    ("24,6 horas", (24.6,), "h"),
    ("19.351,5", (19351.5,), None),
    ("19,351.5", (19351.5,), None),
    ("1,234,567", (1234567.0,), None),
    ("35,48 %", (35.48,), "%"),
    ("35,48%", (35.48,), "%"),
    ("−0,12", (-0.12,), None),
    ("Group 70", (70.0,), None),
])
def test_extrae_formatos_es_y_en(texto, candidatos, unidad):
    cifras = extraer_cifras(texto)
    assert len(cifras) == 1
    assert cifras[0].candidatos == pytest.approx(candidatos)
    assert cifras[0].unidad == unidad


def test_ambiguedad_miles_o_decimal():
    for texto in ("20.769", "20,769"):
        cifra = extraer_cifras(texto)[0]
        assert set(cifra.candidatos) == {20769.0, 20.769}
        assert dict(zip(cifra.candidatos, cifra.decimales)) == {20769.0: 0, 20.769: 3}
    cifra = extraer_cifras("0,123")[0]
    assert cifra.candidatos == (0.123,)
    assert cifra.decimales == (3,)


def test_formato_irregular_se_ignora():
    resultado = verificar_cifras("valor 1.23.456", CONTEXTO)
    assert [(c.estado, c.motivo) for c in resultado.cifras] == [("ignorada", "formato")]
    assert resultado.evaluadas == 0


def test_posicion_y_texto_de_la_cifra():
    cifra = extraer_cifras("mediana de 24,6 h.")[0]
    assert cifra.texto == "24,6 h"
    assert cifra.posicion == 11


# Paso 2: lo que no es cifra


def test_ignora_marcadores_de_lista():
    resultado = verificar_cifras("1. Hechos\n2) Interpretación\n- 3. Limitaciones", CONTEXTO)
    assert resultado.evaluadas == 0
    assert resultado.cifras == ()


def test_ignora_fechas_horas_y_anios():
    assert extraer_cifras("del 2016-02-29 al 13/05/2016 a las 11:49") == []
    assert extraer_cifras("2016-05-13T10:20:30") == []
    assert estados("en 2016") == [("2016", "ignorada", "anio")]


def test_enteros_pequenos_no_penalizan():
    contexto = {"incidentes": 220, "mediana_horas": 78.17}
    assert estados("3 apartados", contexto) == [("3", "ignorada", "entero_pequeno")]
    assert estados("top 5", contexto) == [("5", "ignorada", "entero_pequeno")]
    assert estados("3 apartados", {"incidentes": 3}) == [("3", "verificada", None)]


def test_identificadores_no_son_cifras():
    for texto in ("p90", "llama3.2:3b", "INC1", "Q4_K_M", "3B"):
        assert extraer_cifras(texto) == [], texto


@pytest.mark.parametrize("texto", ["1.5x", "24,6hs", "3.2B", "12ms", "20.769abc"])
def test_numero_pegado_a_letras_no_se_trunca(texto):
    assert extraer_cifras(texto) == []
    assert verificar_cifras(texto, {"valores": [1, 24, 3, 20]}).cifras == ()


# Paso 3: tolerancia, signo y unidades


@pytest.mark.parametrize(("texto", "esperado"), [
    ("78,17", "verificada"), ("78,2", "verificada"), ("78", "verificada"),
    ("78,16", "no_verificada"), ("78,1", "no_verificada"),
])
def test_tolerancia_de_redondeo(texto, esperado):
    assert estado_unico(texto, {"mediana": 78.1667}) == esperado


def test_truncamiento_no_se_acepta():
    assert estado_unico("187", {"media": 187.678}) == "no_verificada"
    assert estado_unico("188", {"media": 187.678}) == "verificada"


def test_sin_conversion_de_dias_ni_multiplicadores():
    assert estado_unico("3,3 días", {"mediana_horas": 78.17}) == "no_verificada"
    assert estado_unico("21 minutos", {"mediana_horas": 0.35}) == "no_verificada"
    assert estado_unico("20,8 mil", {"incidentes": 20769}) == "no_verificada"


def test_horas_sin_conversion():
    assert estado_unico("24,6 horas", CONTEXTO) == "verificada"
    assert estado_unico("187,68 h", CONTEXTO) == "verificada"


def test_porcentaje_acepta_valor_o_fraccion():
    assert estado_unico("35,48 %", {"porcentaje_incidentes": 35.48}) == "verificada"
    assert estado_unico("6,83 %", {"porcentaje_nulos": 0.06827}) == "verificada"
    assert estado_unico("94 %", {"porcentaje_incidentes": 35.48}) == "no_verificada"


def test_signo_negativo_en_correlaciones():
    assert estado_unico("−0,12", {"pearson": -0.12}) == "verificada"
    assert estado_unico("-0,12", {"pearson": -0.12}) == "verificada"
    assert estado_unico("−0,12", {"pearson": 0.12}) == "no_verificada"


def test_ambigua_se_verifica_con_cualquier_candidato():
    resultado = verificar_cifras("20.769 incidentes", CONTEXTO)
    assert resultado.cifras[0].estado == "verificada"
    assert resultado.cifras[0].referencia == 20769
    assert estado_unico("0,204", {"pearson": 0.204}) == "verificada"


def test_cifras_inventadas_quedan_no_verificadas():
    assert estado_unico("31,5 h", CONTEXTO) == "no_verificada"


# Paso 4: referencias del contexto


def test_cifras_de_cadenas_del_contexto():
    contexto = {"segmentos": [{"segmento": "Category 26", "incidentes": 3000}]}
    assert estado_unico("26", contexto) == "verificada"
    assert 1.0 in numeros_de_contexto({"priority": "1 - Critical"})


def test_hash_y_digest_excluidos():
    contexto = {"control_archivo_completo": {"sha256_csv": "5d77ea4853e443cf", "eventos": 119998},
                "digest": 4853, "pregunta_SHA256_12": [7777], "modelo": {"digest_corto": "4853 a80c"}}
    assert estado_unico("4853", contexto) == "no_verificada"
    assert estado_unico("7777", contexto) == "no_verificada"
    assert estado_unico("119.998", contexto) == "verificada"


def test_booleanos_y_no_finitos_no_son_referencia():
    referencias = numeros_de_contexto({"a": True, "b": float("nan"), "c": float("inf"), "d": 2.5, "e": None})
    assert referencias == [2.5]


def test_numero_citado_en_pregunta_se_ignora():
    pregunta = "¿Qué pasa con los 1000 incidentes más lentos?"
    assert estados("Los 1000 incidentes", CONTEXTO, pregunta) == [("1000", "ignorada", "en_pregunta")]
    assert estados("Los 1000 incidentes", CONTEXTO) == [("1000", "no_verificada", None)]


# Paso 5: resultado y cobertura


def test_resultado_serializable_y_proporcion():
    vacio = verificar_cifras("Sin cifras aquí.", CONTEXTO)
    assert vacio.evaluadas == 0
    assert vacio.proporcion is None
    resultado = verificar_cifras("Hay 20.769 incidentes, mediana 24,6 h y 31,5 h inventadas en 2016.", CONTEXTO)
    assert resultado.evaluadas == 3
    assert len(resultado.verificadas) == 2
    assert len(resultado.no_verificadas) == 1
    assert len(resultado.ignoradas) == 1
    assert resultado.proporcion == 0.667
    datos = json.loads(json.dumps(resultado.a_dict(), allow_nan=False))
    assert datos["evaluadas"] == 3
    assert datos["verificadas"] == 2
    assert datos["proporcion"] == 0.667
    assert datos["no_verificadas"][0]["texto"] == "31,5 h"
    assert datos["ignoradas"][0]["motivo"] == "anio"


def test_resolver_ruta_con_indices():
    assert resolver_ruta(CONTEXTO, "resumen_seleccion.mediana_horas") == 24.6
    assert resolver_ruta(CONTEXTO, "comparacion_prioridades[1].incidentes") == 19554
    with pytest.raises(LookupError):
        resolver_ruta(CONTEXTO, "comparacion_prioridades[5].incidentes")
    with pytest.raises(LookupError):
        resolver_ruta(CONTEXTO, "no_existe.valor")


def test_cobertura_de_cifras_clave():
    resultado = verificar_cifras("La selección tiene 20.769 incidentes y una mediana de 24,6 h.", CONTEXTO)
    rutas = ["resumen_seleccion.incidentes_seleccionados", "resumen_seleccion.mediana_horas",
             "resumen_seleccion.media_horas", "ruta.inexistente"]
    assert cobertura_cifras(resultado, CONTEXTO, rutas) == 0.667
    assert cobertura_cifras(resultado, CONTEXTO, ["ruta.inexistente"]) is None
    assert cobertura_cifras(resultado, CONTEXTO, ["comparacion_prioridades[0].incidentes"]) == 0.0
