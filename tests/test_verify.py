"""Tests de verify.py: regla de verificación literal."""

import pytest

from src.extract_rules import extraer_reglas
from src.normalize import normalizar
from src.verify import etiquetas, numero_en, verificar


def test_numero_completo_no_dentro_de_otro():
    t = "avenida arequipa 2890 el total es 98 soles"
    assert numero_en(98, t)
    assert numero_en(98.0, t)          # 98.0 se escribe "98"
    assert not numero_en(90, t)        # "90" está dentro de "2890": no cuenta
    assert not numero_en(89, t)
    assert numero_en(2.5, "son 2.5 anos")
    assert not numero_en(2, "son 2.5 anos")


@pytest.mark.parametrize("clip, schema", [("E01", "entrevista"), ("E02", "entrevista"),
                                          ("E03", "entrevista"), ("E04", "entrevista"),
                                          ("L02", "pedido"), ("L03", "pedido"), ("L04", "pedido")])
def test_lo_extraido_por_reglas_esta_verificado(cfg, referencias, clip, schema):
    """Las reglas solo copian lo dicho: todo campo con valor debe quedar verificado."""
    v = verificar(extraer_reglas(referencias[clip], schema, cfg), referencias[clip], cfg)
    assert all(v.values()), [c for c, ok in v.items() if not ok]


def test_sin_promocion_no_hay_nada_que_verificar(cfg, referencias):
    v = verificar(extraer_reglas(referencias["L01"], "pedido", cfg), referencias["L01"], cfg)
    assert v["promocion"] is False and v["descuento"] is False
    assert v["items"] and v["total_dicho"]


def test_llm_no_puede_corregir_el_total(cfg, referencias):
    """L02: se dijo 80 (mal). Si el LLM 'corrige' a 90, queda sin_verificar."""
    registro = extraer_reglas(referencias["L02"], "pedido", cfg)
    registro["total_dicho"] = 90
    v = verificar(registro, referencias["L02"], cfg)
    assert v["total_dicho"] is False
    assert etiquetas(v, cfg)["total_dicho"] == cfg["verificacion"]["etiqueta_no"]


def test_llm_no_puede_inventar_tecnologias(cfg, referencias):
    registro = extraer_reglas(referencias["E02"], "entrevista", cfg)
    registro["tecnologias"] = registro["tecnologias"] + ["Python"]  # E02 nunca dice Python
    assert verificar(registro, referencias["E02"], cfg)["tecnologias"] is False


def test_categorias_y_disponibilidad(cfg, referencias):
    # moneda "PEN" se verifica porque se dijo "soles"; fecha por "primero de diciembre"
    registro = {"moneda": "PEN", "disponibilidad": "2026-12-01", "nivel_ingles": "basico"}
    assert all(verificar(registro, referencias["E03"], cfg).values())
    assert verificar({"moneda": "USD"}, referencias["E03"], cfg)["moneda"] is False


def test_se_verifica_sobre_el_transcript_normalizado(cfg):
    # El ASR escribió palabras, el LLM devolvió dígitos: igual se verifica
    assert verificar({"pretension_salarial": 6500},
                     "unos seis mil quinientos soles", cfg)["pretension_salarial"]
    assert normalizar("seis mil quinientos", cfg) == "6500"
