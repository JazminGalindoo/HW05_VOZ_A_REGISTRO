"""La app arranca sin errores con los resultados de cache/results y nunca transcribe.

AppTest ejecuta app.py completo (todas las pestañas) sin abrir un navegador.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.asr

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app(monkeypatch):
    # Si la app intentara transcribir el corpus al arrancar, este test falla
    def prohibido(*args, **kwargs):
        raise AssertionError("la app no debe transcribir al arrancar")
    monkeypatch.setattr(src.asr, "transcribir", prohibido)
    monkeypatch.setattr("src.engine.transcribir", prohibido)
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_arranca_sin_errores(app):
    assert not app.exception, app.exception


def test_cinco_pestanas(app):
    assert [t.label for t in app.tabs] == ["1 · Problema y método", "2 · Corpus", "3 · Resultados",
                                            "4 · Probar un audio", "5 · Pedidos (Tarea 2)"]


def test_chequeo_atrapa_el_total_de_l02(app):
    # Con "referencia" (texto humano) el resultado no depende de qué haya en cache
    app.sidebar.selectbox[0].set_value("referencia").run()
    app.selectbox(key="ped_clip").set_value("L02").run()
    assert not app.exception
    errores = " ".join(e.value for e in app.error)
    assert "No cuadra" in errores and "80" in errores and "90" in errores


def test_corpus_muestra_registro_verificado(app):
    app.sidebar.selectbox[0].set_value("referencia").run()
    app.radio(key="corpus_tarea").set_value("entrevista").run()
    app.selectbox(key="corpus_clip").set_value("E03").run()
    tablas = [d.value for d in app.dataframe]
    registro = next(t for t in tablas if "verificación" in t.columns)
    assert set(registro["verificación"]) == {"verificado"}
