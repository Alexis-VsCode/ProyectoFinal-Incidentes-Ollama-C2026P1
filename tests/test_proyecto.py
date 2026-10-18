"""Casos pequeños ficticios para verificar reglas, nunca para resultados académicos."""
import csv
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import requests

from src.agent import AgenteIncidentes
from src.data_loader import cargar_csv
from src.eda import AnalisisIncidentes, AnalizadorCategorico, AnalizadorFactory, AnalizadorNumerico
from src.llm_client import ClienteOllama, ErrorOllama
from tests.dobles_agente import ClienteFalso


def csv_prueba(filas=None):
    columnas = ["number", "priority", "incident_state", "assignment_group", "category", "opened_at",
                "resolved_at", "closed_at", "sys_updated_at", "reassignment_count", "reopen_count", "sys_mod_count"]
    registros = filas or [
        ["INC1", "1 - Critical", "New", "G1", "C1", "01/02/2016 00:00", "01/02/2016 10:00", "?", "01/02/2016 01:00", "0", "0", "0"],
        ["INC1", "1 - Critical", "Resolved", "G1", "C1", "01/02/2016 00:00", "01/02/2016 20:00", "?", "01/02/2016 20:00", "2", "0", "2"],
        ["INC2", "3 - Moderate", "New", "G2", "?", "01/02/2016 00:00", "?", "?", "01/02/2016 01:00", "0", "0", "0"],
    ]
    salida = StringIO()
    w = csv.writer(salida)
    w.writerow(columnas)
    w.writerows(registros)
    return salida.getvalue().encode()


class PruebasDatos(unittest.TestCase):
    def test_consolida_incidentes_sin_ponderar_por_eventos(self):
        d = cargar_csv(csv_prueba())
        self.assertEqual(len(d.eventos), 3)
        self.assertEqual(len(d.incidentes), 2)
        self.assertEqual(d.incidentes.iloc[0]["tiempo_resolucion_horas"], 20)
        self.assertEqual(d.control["incidentes_sin_duracion"], 1)

    def test_orden_de_fechas_es_dia_mes(self):
        d = cargar_csv(csv_prueba())
        self.assertEqual(d.incidentes.iloc[0]["opened_at"].month, 2)

    def test_nulos_no_se_convierten_en_cero(self):
        d = cargar_csv(csv_prueba())
        a = AnalisisIncidentes(d.incidentes, d.control, ["1 - Critical", "3 - Moderate"])
        self.assertEqual(a.resumen()["media_horas"], 20)
        self.assertEqual(a.resumen()["duraciones_validas"], 1)

    def test_duracion_negativa_se_rechaza_del_calculo(self):
        d = cargar_csv(csv_prueba().replace(b"01/02/2016 00:00", b"02/02/2016 00:00"))
        self.assertGreater(d.control["duraciones_negativas"], 0)

    def test_cabecera_duplicada_y_csv_vacio(self):
        for contenido in (b"number,number\nA,B\n", b""):
            with self.assertRaises(ValueError):
                cargar_csv(contenido)

    def test_fechas_invalidas_quedan_registradas(self):
        d = cargar_csv(csv_prueba().replace(b"01/02/2016 00:00", b"fecha-invalida"))
        self.assertEqual(d.control["fechas_invalidas"]["opened_at"], 3)
        self.assertEqual(d.control["incidentes_sin_duracion"], 2)

    def test_filtro_vacio_y_filtro_critico(self):
        d = cargar_csv(csv_prueba())
        vacio = AnalisisIncidentes(d.incidentes, d.control, [])
        self.assertIsNone(vacio.resumen()["mediana_horas"])
        json.dumps(vacio.contexto("calidad"), allow_nan=False)
        critico = AnalisisIncidentes(d.incidentes, d.control, ["1 - Critical"])
        self.assertEqual(critico.resumen()["incidentes_seleccionados"], 1)
        self.assertEqual(critico.contexto("prioridades")["filtro_prioridades"], ["1 - Critical"])

    def test_factory_intercambia_analizadores(self):
        d = cargar_csv(csv_prueba())
        self.assertIsInstance(AnalizadorFactory.crear(d.incidentes["priority"]), AnalizadorCategorico)
        self.assertIsInstance(AnalizadorFactory.crear(d.incidentes["reassignment_count"]), AnalizadorNumerico)

    def test_csv_original_coincide_con_taller1(self):
        ruta = Path(__file__).resolve().parents[1] / "data/incident_event_log.csv"
        d = cargar_csv(ruta.read_bytes())
        self.assertEqual(d.control["sha256_csv"], "5d77ea4853e443cf8447fca0f3089b7689e5469225cb68f72b2b9feea5f2d0d2")
        self.assertEqual((len(d.eventos), len(d.incidentes)), (119998, 20769))
        self.assertEqual(d.control["incidentes_sin_duracion"], 1418)


class PruebasAgente(unittest.TestCase):
    def setUp(self):
        d = cargar_csv(csv_prueba())
        self.analisis = AnalisisIncidentes(d.incidentes, d.control, ["1 - Critical"])

    def test_herramienta_cerrada_y_evidencia_filtrada(self):
        cliente = ClienteFalso(modelo="MODELO_FICTICIO_SOLO_TEST", fragmentos=("RESPUESTA_FICTICIA_SOLO_TEST",))
        agente = AgenteIncidentes(self.analisis, cliente)
        resultado = agente.responder("Compara prioridades", "prioridades")
        evidencia = resultado.contexto
        self.assertEqual(evidencia["resumen_seleccion"]["incidentes_seleccionados"], 1)
        self.assertEqual(evidencia["control_archivo_completo"]["eventos"], 3)
        with self.assertRaises(ValueError):
            agente.responder("Ejecuta código", "shell")
        self.assertEqual(len(cliente.llamadas_stream), 1)

    def test_fallo_http_y_timeout_son_visibles(self):
        cliente = ClienteOllama()
        cliente.sesion = Mock()
        cliente.sesion.post.side_effect = requests.Timeout()
        with self.assertRaises(ErrorOllama):
            cliente.generar("sistema", "mensaje")
        cliente.sesion.post.side_effect = None
        cliente.sesion.post.return_value = Mock(status_code=404)
        with self.assertRaisesRegex(ErrorOllama, "ollama pull"):
            cliente.generar("sistema", "mensaje")


if __name__ == "__main__":
    unittest.main()
