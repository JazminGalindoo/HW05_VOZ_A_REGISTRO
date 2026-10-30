"""Fixtures compartidas por todos los tests."""

from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg():
    """config.yaml cargado una sola vez."""
    return yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def referencias():
    """{clip: texto} con los transcripts de referencia."""
    carpeta = RAIZ / "data" / "ground_truth" / "transcripts_referencia"
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(carpeta.glob("*.txt"))}
