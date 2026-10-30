"""Tests de validate.py: chequeo aritmético y diagnóstico ASR vs hablante.

Sin audio todavía: las "palabras del ASR" se simulan partiendo la referencia
en palabras con una probabilidad fija (como si el ASR hubiera acertado todo).
"""

import pytest

from src.extract_rules import extraer_reglas
from src.schemas import leer_ground_truth
from src.validate import acierto_diagnostico, chequear, diagnosticar, prob_numero


def palabras_simuladas(texto: str, prob: float = 0.95) -> list[dict]:
    return [{"word": w, "probability": prob} for w in texto.split()]


@pytest.fixture
def pedidos(cfg, referencias):
    return {c: extraer_reglas(referencias[c], "pedido", cfg) for c in ["L01", "L02", "L03", "L04"]}


# ---------------- chequeo aritmético ----------------

@pytest.mark.parametrize("clip, cuadra, esperado, dicho", [
    ("L01", True, 89, 89),
    ("L02", False, 90, 80),   # dice 80, lo correcto es 90
    ("L03", False, 65, 75),   # se olvidó del 2x1
    ("L04", False, 90, 98),   # dice 98, lo correcto es 90
])
def test_chequeo_detecta_l02_l03_l04(cfg, pedidos, clip, cuadra, esperado, dicho):
    r = chequear(pedidos[clip], cfg)
    assert r["cuadra"] is cuadra
    assert r["total_esperado"] == esperado and r["total_dicho"] == dicho


def test_chequeo_coincide_con_ground_truth(cfg):
    for clip, gt in leer_ground_truth("pedido", cfg).items():
        r = chequear(gt, cfg)
        assert r["total_esperado"] == gt["_meta"]["total_correcto"], clip
        assert r["cuadra"] is (not gt["_meta"]["trampa_total_erroneo"]), clip


def test_tolerancia(cfg, pedidos):
    pedido = dict(pedidos["L01"], total_dicho=89.4)  # dentro de 0.5
    assert chequear(pedido, cfg)["cuadra"] is True


# ---------------- probabilidad de los números ----------------

def test_prob_numero_agrupa_palabras(cfg):
    palabras = palabras_simuladas("el total es ochenta y nueve soles.")
    palabras[4]["probability"] = 0.3  # la "y" de "ochenta y nueve"
    assert prob_numero(89, palabras, cfg) == 0.3
    assert prob_numero(80, palabras, cfg) is None  # 80 no es 89


def test_prob_numero_no_confunde_con_numeros_largos(cfg):
    palabras = palabras_simuladas("avenida arequipa dos mil ochocientos noventa")
    assert prob_numero(2890, palabras, cfg) == 0.95
    assert prob_numero(90, palabras, cfg) is None


def test_prob_numero_ultimo_total(cfg):
    palabras = palabras_simuladas("total es ochenta soles ... no, el total es noventa")
    palabras[2]["probability"] = 0.5
    assert prob_numero(80, palabras, cfg, ultimo=True) == 0.5


# ---------------- diagnóstico ----------------

def test_diagnostico_en_l0_con_confianza_alta(cfg, referencias, pedidos):
    """Con números claros, los totales mal dichos son error del HABLANTE."""
    diagnosticos = {c: diagnosticar(pedidos[c], palabras_simuladas(referencias[c]), cfg)["diagnostico"]
                    for c in pedidos}
    assert diagnosticos == {"L01": "ok", "L02": "error_hablante",
                            "L03": "error_hablante", "L04": "error_hablante"}
    resultado = acierto_diagnostico(diagnosticos, leer_ground_truth("pedido", cfg))
    assert resultado["aciertos"] == 4 and resultado["total"] == 4


def test_diagnostico_asr_si_duda_en_un_numero(cfg, referencias, pedidos):
    palabras = palabras_simuladas(referencias["L04"])
    for p in palabras:
        if p["word"].strip(".,") == "ocho":  # "noventa y ocho": el ASR dudó
            p["probability"] = 0.4
    d = diagnosticar(pedidos["L04"], palabras, cfg)
    assert d["diagnostico"] == "error_asr" and "total_dicho" in d["motivo"]


def test_diagnostico_asr_si_l0_cuadraba(cfg, referencias, pedidos):
    pedido = dict(pedidos["L01"], total_dicho=81)  # en L3 el ASR oyó 81
    d = diagnosticar(pedido, palabras_simuladas(referencias["L01"]), cfg, cuadra_en_l0=True)
    assert d["diagnostico"] == "error_asr"


def test_diagnostico_sin_probabilidades(cfg, pedidos):
    assert diagnosticar(pedidos["L02"], [], cfg)["diagnostico"] == "indeterminado"
