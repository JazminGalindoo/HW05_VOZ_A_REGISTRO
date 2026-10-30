"""
metrics.py — Métricas de evaluación.

Responsabilidad:
    - WER y CER con jiwer (sobre texto normalizado con normalize.py).
    - Accuracy por campo, macro-F1 para categóricas (nivel_ingles, metodo_pago)
      y error absoluto para numéricos.
    - Intervalos de confianza por bootstrap remuestreando clips
      (n de réplicas y semilla en config.yaml).

Con 4 clips por tarea los IC son MUY anchos: sirven para mostrar cuánta
incertidumbre hay, no para afirmar diferencias pequeñas.
"""

import re

import jiwer
import numpy as np

from src.normalize import normalizar

CAMPOS_NUMERICOS = {"anios_experiencia", "pretension_salarial", "descuento", "total_dicho"}


# =====================================================================
# WER / CER
# =====================================================================

def wer(referencia: str, hipotesis: str, cfg: dict) -> float:
    """Word Error Rate = (sustituciones + borrados + inserciones) / palabras de la referencia."""
    return jiwer.wer(normalizar(referencia, cfg), normalizar(hipotesis, cfg))


def cer(referencia: str, hipotesis: str, cfg: dict) -> float:
    """Igual que WER pero contando caracteres."""
    return jiwer.cer(normalizar(referencia, cfg), normalizar(hipotesis, cfg))


def errores_wer(referencia: str, hipotesis: str, cfg: dict) -> tuple[int, int]:
    """(errores, palabras de la referencia) de un clip.

    Sirve para el WER de un CONJUNTO de clips: sum(errores) / sum(palabras),
    que pesa cada clip por su largo (no es el promedio de los WER).
    """
    r = jiwer.process_words(normalizar(referencia, cfg), normalizar(hipotesis, cfg))
    return r.substitutions + r.deletions + r.insertions, r.substitutions + r.deletions + r.hits


def wer_conjunto(pares: list[tuple[int, int]]) -> float:
    """WER de varios clips a partir de sus (errores, palabras)."""
    return sum(e for e, _ in pares) / sum(n for _, n in pares)


# =====================================================================
# Comparación de campos
# =====================================================================

def _direccion_canonica(texto: str, cfg: dict) -> set[str]:
    """'Jr. Huallaga 382, dpto 4B' y 'jiron huallaga 382 departamento 4 b' -> mismo conjunto."""
    t = normalizar(texto, cfg)
    t = re.sub(r"(\d) ([a-z])\b", r"\1\2", t)  # "4 b" -> "4b"
    abreviaturas = cfg["metricas"]["abreviaturas"]
    return {abreviaturas.get(p, p) for p in t.split() if p != "en"}


def _items_canonicos(items: list[dict]) -> list[tuple]:
    """Lista ordenada de (producto, cantidad, precio): el orden en que se dijeron no importa."""
    return sorted((i["producto"].lower(), float(i["cantidad"]),
                   None if i["precio_unitario"] is None else float(i["precio_unitario"]))
                  for i in items)


def campo_correcto(campo: str, extraido, verdad, cfg: dict) -> bool:
    """¿El valor extraído coincide con el ground truth? Regla según el tipo de campo."""
    if campo == "promocion":
        # El texto de la promo se dice de mil formas: aquí solo cuenta si se detectó.
        # Su valor en soles se evalúa en el campo "descuento".
        return (extraido is None) == (verdad is None)
    if extraido is None or verdad is None:
        return extraido is None and verdad is None
    if campo in CAMPOS_NUMERICOS:
        return float(extraido) == float(verdad)
    if campo == "tecnologias":
        return {normalizar(x, cfg) for x in extraido} == {normalizar(x, cfg) for x in verdad}
    if campo == "items":
        return _items_canonicos(extraido) == _items_canonicos(verdad)
    if campo == "direccion_entrega":
        return _direccion_canonica(extraido, cfg) == _direccion_canonica(verdad, cfg)
    # texto y categorías: iguales después de normalizar
    return normalizar(str(extraido), cfg) == normalizar(str(verdad), cfg)


def tabla_aciertos(extraidos: dict[str, dict], verdad: dict[str, dict], cfg: dict) -> dict:
    """{campo: {clip: True/False}} para los clips que están en ambos."""
    clips = sorted(set(extraidos) & set(verdad))
    campos = [c for c in verdad[clips[0]] if not c.startswith("_")]
    return {campo: {clip: campo_correcto(campo, extraidos[clip].get(campo), verdad[clip][campo], cfg)
                    for clip in clips}
            for campo in campos}


def accuracy_por_campo(extraidos: dict, verdad: dict, cfg: dict) -> dict[str, float]:
    """{campo: fracción de clips donde el campo es correcto}."""
    return {campo: float(np.mean(list(por_clip.values())))
            for campo, por_clip in tabla_aciertos(extraidos, verdad, cfg).items()}


def error_absoluto(extraidos: dict, verdad: dict, campo: str) -> dict[str, float | None]:
    """{clip: |extraído - verdad|}; None si alguno falta."""
    errores = {}
    for clip in sorted(set(extraidos) & set(verdad)):
        e, v = extraidos[clip].get(campo), verdad[clip][campo]
        errores[clip] = None if e is None or v is None else abs(float(e) - float(v))
    return errores


# =====================================================================
# Macro-F1 (escrita a mano para poder explicarla)
# =====================================================================

def macro_f1(reales: list, predichos: list, clases: list) -> float:
    """Promedio simple del F1 de cada clase.

    Para cada clase c:
        precision = aciertos de c / veces que se predijo c
        recall    = aciertos de c / veces que c era la verdad
        F1        = 2·P·R / (P + R)   (0 si P + R = 0)
    Se promedian solo las clases que aparecen en reales o predichos.
    """
    f1s = []
    for c in clases:
        vp = sum(r == c and p == c for r, p in zip(reales, predichos))
        predichos_c = sum(p == c for p in predichos)
        reales_c = sum(r == c for r in reales)
        if predichos_c == 0 and reales_c == 0:
            continue  # la clase no aparece: no entra al promedio
        precision = vp / predichos_c if predichos_c else 0.0
        recall = vp / reales_c if reales_c else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


# =====================================================================
# Bootstrap por clip
# =====================================================================

def bootstrap_ic(datos: list, estadistico, cfg: dict) -> dict:
    """IC por bootstrap: se remuestrean CLIPS con reemplazo y se recalcula el estadístico.

    datos: un elemento por clip (p. ej. 1/0 de acierto, o (errores, palabras)).
    estadistico: función que recibe una lista de elementos y devuelve un número.
    """
    m = cfg["metricas"]
    rng = np.random.default_rng(m["semilla"])  # semilla fija -> mismo IC siempre
    n = len(datos)
    replicas = [estadistico([datos[i] for i in rng.integers(0, n, n)])
                for _ in range(m["bootstrap_n"])]
    alfa = m["bootstrap_alpha"]
    return {
        "estimacion": estadistico(datos),
        "ic_bajo": float(np.quantile(replicas, alfa / 2)),
        "ic_alto": float(np.quantile(replicas, 1 - alfa / 2)),
        "n_clips": n,
    }


def bootstrap_diferencia(datos_a: list, datos_b: list, estadistico, cfg: dict) -> dict:
    """IC de estadístico(A) - estadístico(B) con datos PAREADOS (mismos clips en A y B).

    Se usan los mismos índices remuestreados en A y en B, así la variación entre
    clips (unos son más difíciles que otros) se cancela. Si el IC incluye 0,
    la muestra no permite afirmar que A y B sean distintos.
    """
    assert len(datos_a) == len(datos_b), "A y B deben tener los mismos clips"
    pares = list(zip(datos_a, datos_b))
    return bootstrap_ic(
        pares,
        lambda muestra: estadistico([a for a, _ in muestra]) - estadistico([b for _, b in muestra]),
        cfg,
    )
