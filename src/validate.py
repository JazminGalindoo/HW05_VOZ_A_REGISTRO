"""
validate.py — Validación aritmética del pedido y diagnóstico.

Responsabilidad:
    - Chequear sum(cantidad * precio_unitario) - descuento == total_dicho
      con la tolerancia de config.yaml.
    - Si no cuadra, diagnosticar si es error del ASR o del hablante usando:
        * probabilidad por palabra de los tokens numéricos implicados, y
        * consistencia entre niveles (si L0 cuadra y Lk no -> ASR).
    - Medir cuántas veces acierta el diagnóstico con las trampas marcadas
      en el ground truth.

Regla de diagnóstico (en este orden):
    1. Cuadra                                   -> "ok"
    2. En L0 cuadraba y en este nivel no        -> "error_asr"  (lo rompió la degradación)
    3. Sin probabilidades por palabra           -> "indeterminado"
    4. Algún número implicado con prob < umbral -> "error_asr"  (el ASR dudó)
    5. Todos los números con prob alta          -> "error_hablante" (se dijo así)
"""

import re

from src.normalize import normalizar


def chequear(registro: dict, cfg: dict) -> dict:
    """Compara lo que DEBERÍA costar con lo que se DIJO."""
    items = registro["items"]
    if not items or registro["total_dicho"] is None or any(
        i["precio_unitario"] is None for i in items
    ):
        return {"cuadra": None, "motivo": "faltan precios o total"}

    suma = sum(i["cantidad"] * i["precio_unitario"] for i in items)
    esperado = suma - (registro["descuento"] or 0)
    diferencia = registro["total_dicho"] - esperado
    return {
        "suma_items": suma,
        "descuento": registro["descuento"] or 0,
        "total_esperado": esperado,
        "total_dicho": registro["total_dicho"],
        "diferencia": diferencia,
        "cuadra": abs(diferencia) <= cfg["validacion"]["tolerancia_soles"],
    }


# --- Probabilidad de los números en el audio ---

def palabras_de(segments: list[dict]) -> list[dict]:
    """Aplana los segmentos de faster-whisper: [{'word': ' ochenta', 'probability': 0.97}, ...]"""
    return [p for s in segments for p in s.get("words", [])]


def prob_numero(numero, palabras: list[dict], cfg: dict, ultimo: bool = False):
    """Probabilidad mínima de las palabras que forman el número (None si no se encuentra).

    Busca grupos de 1..N palabras seguidas cuya normalización sea el número:
    ["ochenta", "y", "nueve"] -> "89". Con ultimo=True usa la última aparición
    (el total que vale es el último que se dijo).
    """
    objetivo = str(int(numero)) if float(numero).is_integer() else str(numero)
    largo_max = cfg["validacion"]["max_palabras_numero"]
    sueltas = [normalizar(p["word"], cfg) for p in palabras]

    hallados = []
    i = 0
    while i < len(palabras):
        if not sueltas[i][:1].isdigit():  # un número siempre empieza con una palabra-número
            i += 1
            continue
        # El número completo que empieza aquí: el grupo MÁS LARGO que se normaliza
        # a un solo número ("ochenta y nueve" -> 89, no 80)
        for largo in range(largo_max, 0, -1):
            grupo = palabras[i:i + largo]
            texto = normalizar(" ".join(p["word"] for p in grupo), cfg)
            if re.fullmatch(r"\d+(?:\.\d+)?", texto):
                break
        if texto == objetivo:
            hallados.append(float(min(p["probability"] for p in grupo)))
        i += largo  # saltar el número entero ("noventa" de "2890" no es un 90)
    if not hallados:
        return None
    return hallados[-1] if ultimo else hallados[0]


def numeros_implicados(registro: dict) -> list[tuple[str, float, bool]]:
    """[(nombre, valor, usar_ultima_aparicion)] de los números que entran en la cuenta.

    Las cantidades iguales a 1 se omiten: "un/una" aparece en todas partes y no informa.
    """
    numeros = [("total_dicho", registro["total_dicho"], True)]
    if registro["descuento"]:
        numeros.append(("descuento", registro["descuento"], False))
    for item in registro["items"]:
        numeros.append((f"precio {item['producto']}", item["precio_unitario"], False))
        if item["cantidad"] != 1:
            numeros.append((f"cantidad {item['producto']}", item["cantidad"], False))
    return numeros


def diagnosticar(registro: dict, palabras: list[dict], cfg: dict,
                 cuadra_en_l0: bool | None = None) -> dict:
    """Diagnóstico de un pedido en un nivel de degradación (ver reglas arriba)."""
    chequeo = chequear(registro, cfg)
    if chequeo["cuadra"] is None:
        return dict(chequeo, diagnostico="indeterminado")
    if chequeo["cuadra"]:
        return dict(chequeo, diagnostico="ok", motivo="la cuenta cuadra")
    if cuadra_en_l0:
        return dict(chequeo, diagnostico="error_asr",
                    motivo="en L0 cuadraba: la degradación cambió algún número")
    if not palabras:
        return dict(chequeo, diagnostico="indeterminado", motivo="sin probabilidades por palabra")

    umbral = cfg["validacion"]["prob_palabra_min"]
    probs = {nombre: prob_numero(valor, palabras, cfg, ultimo)
             for nombre, valor, ultimo in numeros_implicados(registro)}
    dudosos = {n: p for n, p in probs.items() if p is not None and p < umbral}
    if dudosos:
        return dict(chequeo, diagnostico="error_asr", probabilidades=probs,
                    motivo=f"el ASR dudó en: {', '.join(dudosos)}")
    return dict(chequeo, diagnostico="error_hablante", probabilidades=probs,
                motivo=f"todos los números con prob >= {umbral}: se dijo así")


def acierto_diagnostico(diagnosticos: dict[str, str], ground_truth: dict) -> dict:
    """Compara diagnósticos de L0 contra las trampas del ground truth.

    Esperado: "error_hablante" si el clip tiene trampa de total erróneo, "ok" si no.
    """
    detalle = {}
    for clip, diagnostico in diagnosticos.items():
        esperado = ("error_hablante" if ground_truth[clip]["_meta"]["trampa_total_erroneo"]
                    else "ok")
        detalle[clip] = {"esperado": esperado, "obtenido": diagnostico,
                         "acierta": diagnostico == esperado}
    aciertos = sum(d["acierta"] for d in detalle.values())
    return {"aciertos": aciertos, "total": len(detalle), "detalle": detalle}
