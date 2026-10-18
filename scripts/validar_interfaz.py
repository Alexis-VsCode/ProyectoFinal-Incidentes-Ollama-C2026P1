"""Comprobación de UI sin navegador ni inferencia: python -m scripts.validar_interfaz."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def main():
    app = Path(__file__).resolve().parents[1] / 'app.py'
    vista = AppTest.from_file(str(app), default_timeout=40).run()
    assert not vista.exception, 'Error al cargar la interfaz'
    assert next(m.value for m in vista.metric if m.label == 'Incidentes seleccionados') == '20,769'
    vista.multiselect[0].set_value(['1 - Critical']).run()
    assert not vista.exception
    assert next(m.value for m in vista.metric if m.label == 'Incidentes seleccionados') == '220'
    vista.multiselect[0].set_value([]).run()
    assert not vista.exception
    next(s for s in vista.selectbox if s.label == 'Modelo local').set_value('llama3.2:1b').run()
    assert not vista.exception
    print('OK: carga, filtro crítico, selección vacía y cambio de modelo.')
    print('Esta prueba no evalúa una respuesta real de Ollama ni la apariencia del navegador.')


if __name__ == '__main__':
    main()
