"""Tests de metrics.py con ejemplos verificables a mano."""

import numpy as np
import pytest

from src.extract_rules import extraer_reglas
from src.metrics import (accuracy_por_campo, bootstrap_diferencia, bootstrap_ic, campo_correcto,
                         cer, errores_wer, error_absoluto, macro_f1, wer, wer_conjunto)
from src.schemas import leer_ground_truth


# ---------------- WER / CER ----------------

def test_wer_a_mano(cfg):
    # ref 4 palabras; hip: 1 sustitución (b->x) + 1 borrado (d) -> 2/4
    assert wer("a b c d", "a x c", cfg) == 0.5


def test_wer_normaliza_numeros_y_puntuacion(cfg):
    assert wer("El total es ochenta y nueve soles.", "el total es 89 soles", cfg) == 0.0


def test_cer_a_mano(cfg):
    assert cer("hola", "hila", cfg) == 0.25  # 1 carácter de 4


def test_wer_conjunto_pesa_por_largo(cfg):
    pares = [errores_wer("a b c d", "a x c", cfg),  # 2 errores / 4 palabras
             errores_wer("e f", "e f", cfg)]        # 0 errores / 2 palabras
    assert pares == [(2, 4), (0, 2)]
    assert wer_conjunto(pares) == 2 / 6             # no es el promedio (0.5 + 0) / 2


# ---------------- campos ----------------

def test_direccion_con_abreviaturas(cfg):
    assert campo_correcto("direccion_entrega", "jiron huallaga 382, departamento 4 b, cercado",
                          "Jr. Huallaga 382, dpto 4B, Cercado", cfg)
    assert not campo_correcto("direccion_entrega", "jiron huallaga 328, cercado",
                              "Jr. Huallaga 382, Cercado", cfg)


def test_items_sin_importar_orden(cfg):
    a = [{"producto": "ensalada", "cantidad": 1, "precio_unitario": 8},
         {"producto": "medio pollo", "cantidad": 1, "precio_unitario": 36}]
    assert campo_correcto("items", list(reversed(a)), a, cfg)
    b = [dict(a[0], cantidad=2), a[1]]
    assert not campo_correcto("items", b, a, cfg)


def test_accuracy_reglas_sobre_referencias(cfg, referencias):
    """Reglas vs ground truth: todo bien salvo E01 tecnologías (lista Excel)."""
    for schema, clips in [("entrevista", ["E01", "E02", "E03", "E04"]),
                          ("pedido", ["L01", "L02", "L03", "L04"])]:
        extraidos = {c: extraer_reglas(referencias[c], schema, cfg) for c in clips}
        acc = accuracy_por_campo(extraidos, leer_ground_truth(schema, cfg), cfg)
        esperado = {campo: 1.0 for campo in acc}
        if schema == "entrevista":
            esperado["tecnologias"] = 0.75
        assert acc == esperado


def test_error_absoluto(cfg):
    extraidos = {"L02": {"total_dicho": 90}, "L04": {"total_dicho": None}}
    verdad = {"L02": {"total_dicho": 80}, "L04": {"total_dicho": 98}}
    assert error_absoluto(extraidos, verdad, "total_dicho") == {"L02": 10.0, "L04": None}


# ---------------- macro-F1 ----------------

def test_macro_f1_a_mano():
    reales = ["intermedio", "avanzado", "basico", "avanzado"]
    predichos = ["intermedio", "avanzado", "intermedio", "avanzado"]
    # intermedio: P=1/2, R=1/1 -> F1=2/3 ; avanzado: P=R=1 -> 1 ; basico: P=0, R=0 -> 0
    # nativo no aparece: no entra. Macro = (2/3 + 1 + 0) / 3
    f1 = macro_f1(reales, predichos, ["basico", "intermedio", "avanzado", "nativo"])
    assert f1 == pytest.approx((2 / 3 + 1 + 0) / 3)


def test_macro_f1_perfecto():
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b", "c"]) == 1.0


# ---------------- bootstrap ----------------

def test_bootstrap_reproducible_y_contiene_estimacion(cfg):
    aciertos = [1, 1, 0, 1]
    r1 = bootstrap_ic(aciertos, np.mean, cfg)
    r2 = bootstrap_ic(aciertos, np.mean, cfg)
    assert r1 == r2  # semilla fija
    assert r1["ic_bajo"] <= r1["estimacion"] == 0.75 <= r1["ic_alto"]
    assert r1["n_clips"] == 4


def test_bootstrap_sin_variacion_da_ic_cerrado(cfg):
    r = bootstrap_ic([1, 1, 1, 1], np.mean, cfg)
    assert r["ic_bajo"] == r["ic_alto"] == 1.0


def test_bootstrap_wer_conjunto(cfg):
    pares = [(2, 4), (0, 2), (1, 10), (3, 6)]
    r = bootstrap_ic(pares, wer_conjunto, cfg)
    assert r["estimacion"] == pytest.approx(6 / 22)
    assert 0 <= r["ic_bajo"] <= r["estimacion"] <= r["ic_alto"] <= 1


def test_bootstrap_diferencia_pareada(cfg):
    # B siempre 0.1 peor que A en cada clip -> diferencia exacta, IC cerrado
    a = [0.10, 0.30, 0.20, 0.40]
    b = [x + 0.1 for x in a]
    r = bootstrap_diferencia(a, b, np.mean, cfg)
    assert r["estimacion"] == pytest.approx(-0.1)
    assert r["ic_bajo"] == pytest.approx(-0.1) and r["ic_alto"] == pytest.approx(-0.1)
