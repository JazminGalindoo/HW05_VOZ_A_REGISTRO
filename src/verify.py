"""
verify.py — Regla de verificación literal.

Responsabilidad:
    - Para cada campo extraído, decidir si es "verificado" (su valor aparece
      literalmente en el transcript normalizado) o "sin_verificar".
    - Es la defensa contra que el LLM "arregle" en silencio lo que el ASR oyó:
      si el LLM devuelve un total que nadie dijo, ese total queda sin_verificar.

Qué significa "literal" según el tipo de campo:
    - número: aparece como número completo ("90" NO cuenta dentro de "2890").
    - texto corto (ultimo_puesto): la frase normalizada aparece tal cual.
    - texto compuesto (direccion, promocion): cada palabra aparece en el transcript.
    - lista (tecnologias): cada elemento aparece.
    - categoría (moneda, metodo_pago): aparece alguna de sus palabras (config.yaml).
    - disponibilidad: aparece su forma dicha ("2 semanas", "1 de diciembre"...).
    - items: aparecen el producto (algún alias), la cantidad y el precio de cada item.
    - None, lista vacía o descuento 0: False (no hay nada que comprobar).
"""

import re

from src.normalize import normalizar

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _formato(numero) -> str:
    """6500.0 -> '6500', 2.5 -> '2.5'"""
    numero = float(numero)
    return str(int(numero)) if numero.is_integer() else str(numero)


def numero_en(numero, t: str) -> bool:
    """¿El número aparece completo en el texto normalizado?"""
    n = re.escape(_formato(numero))
    # Ni dígitos ni punto decimal pegados a los lados
    return re.search(rf"(?<![\d.]){n}(?!\d|\.\d)", t) is not None


def frase_en(texto: str, t: str, cfg: dict) -> bool:
    """¿La frase normalizada aparece tal cual (palabras completas)?"""
    frase = normalizar(texto, cfg)
    return bool(frase) and re.search(rf"\b{re.escape(frase)}\b", t) is not None


def palabras_en(texto: str, t: str, cfg: dict) -> bool:
    """¿Cada palabra del texto aparece en el transcript?"""
    palabras_t = set(t.split())
    palabras = normalizar(texto.replace("2x1", "2 por 1"), cfg).split()
    return bool(palabras) and all(p in palabras_t for p in palabras)


def _disponibilidad_en(valor: str, t: str, cfg: dict) -> bool:
    if valor == "inmediata":
        return re.search(r"inmediat|desde ya", t) is not None
    m = re.fullmatch(r"(\d+)_(\w+)", valor)  # "2_semanas" -> "2 semanas"
    if m:
        return frase_en(f"{m.group(1)} {m.group(2)}", t, cfg)
    m = re.fullmatch(r"\d{4}-(\d{2})-(\d{2})", valor)  # el año no se suele decir
    if m:
        dia, mes = int(m.group(2)), MESES[int(m.group(1)) - 1]
        dias = ["primero", "1"] if dia == 1 else [str(dia)]
        return any(frase_en(f"{d} de {mes}", t, cfg) for d in dias)
    return False


def _categoria_en(campo: str, valor: str, t: str, cfg: dict) -> bool:
    reglas = cfg["extraccion"]["reglas"]
    if campo == "moneda":
        palabras = [p for p, codigo in reglas["monedas"].items() if codigo == valor]
    elif campo == "metodo_pago":
        palabras = reglas["metodos_pago"].get(valor, [])
    else:  # nivel_ingles: la categoría misma es la palabra
        palabras = [valor]
    return any(frase_en(p, t, cfg) for p in palabras)


def _item_en(item: dict, t: str, cfg: dict) -> bool:
    carta = {p["producto"]: p for p in cfg["extraccion"]["reglas"]["carta"]}
    alias = carta.get(item["producto"], {}).get("alias", [item["producto"]])
    producto_ok = any(frase_en(a, t, cfg) for a in alias)
    precio_ok = item["precio_unitario"] is not None and numero_en(item["precio_unitario"], t)
    return producto_ok and numero_en(item["cantidad"], t) and precio_ok


def verificar_campo(campo: str, valor, t: str, cfg: dict) -> bool:
    """t es el transcript YA normalizado."""
    if valor is None or valor == [] or (campo == "descuento" and valor == 0):
        return False
    if campo in ("anios_experiencia", "pretension_salarial", "descuento", "total_dicho"):
        return numero_en(valor, t)
    if campo == "ultimo_puesto":
        return frase_en(valor, t, cfg)
    if campo in ("direccion_entrega", "promocion"):
        return palabras_en(valor, t, cfg)
    if campo == "tecnologias":
        return all(frase_en(v, t, cfg) for v in valor)
    if campo in ("moneda", "metodo_pago", "nivel_ingles"):
        return _categoria_en(campo, valor, t, cfg)
    if campo == "disponibilidad":
        return _disponibilidad_en(valor, t, cfg)
    if campo == "items":
        return all(_item_en(item, t, cfg) for item in valor)
    raise ValueError(f"campo sin regla de verificación: {campo}")


def verificar(registro: dict, transcript: str, cfg: dict) -> dict[str, bool]:
    """{campo: True si verificado}. Ignora claves internas como '_meta'."""
    t = normalizar(transcript, cfg)
    return {campo: verificar_campo(campo, valor, t, cfg)
            for campo, valor in registro.items() if not campo.startswith("_")}


def etiquetas(verificado: dict[str, bool], cfg: dict) -> dict[str, str]:
    """{campo: 'verificado' | 'sin_verificar'} para mostrar en la app."""
    ok, no = cfg["verificacion"]["etiqueta_ok"], cfg["verificacion"]["etiqueta_no"]
    return {campo: ok if v else no for campo, v in verificado.items()}
