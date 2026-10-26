"""Tests mínimos del esqueleto: el motor no importa UI y config.yaml carga."""

import subprocess
import sys
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


def test_engine_sin_ui():
    # Se corre en un proceso nuevo para que sys.modules esté limpio
    comando = (
        "import sys; import src.engine; "
        "bad={'streamlit','gradio','telegram'}&set(sys.modules); "
        "assert not bad, bad"
    )
    resultado = subprocess.run([sys.executable, "-c", comando], cwd=RAIZ)
    assert resultado.returncode == 0


def test_config_tiene_secciones():
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    for seccion in ["rutas", "asr", "glosario", "normalizacion", "extraccion",
                    "validacion", "degradacion", "metricas", "regresion"]:
        assert seccion in cfg, f"falta la sección {seccion}"
