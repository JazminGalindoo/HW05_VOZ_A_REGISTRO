"""
normalize.py — Normalización de texto.

Responsabilidad:
    - Normalizar texto para WER/CER, IGUAL para hipótesis y referencia:
      minúsculas, sin puntuación, espacios colapsados, (opcional) sin tildes.
    - Convertir números dichos en palabras a dígitos
      ("seis mil quinientos" -> "6500") con text2num (lang="es"),
      más reglas propias documentadas para coloquialismos
      ("seis lucas", "dos años y medio"), definidos en config.yaml.
    - El transcript normalizado es la base de la verificación literal.

Orden de los pasos (importa):
    1. minúsculas
    2. proteger palabras que text2num convierte mal (ordinales, "unos")
    3. text2num: palabras -> dígitos (necesita las tildes: "veintitrés")
    4. devolver las palabras protegidas
    5. reglas propias: "y medio", "lucas"
    6. quitar tildes, puntuación y espacios repetidos
"""

import re
import unicodedata

from text_to_num import alpha2digit


def quitar_tildes(texto: str) -> str:
    """'inglés' -> 'ingles'. Separa cada letra de su tilde (NFD) y borra la tilde."""
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def palabras_a_digitos(texto: str, cfg_norm: dict) -> str:
    """Convierte números en palabras a dígitos, protegiendo palabras problemáticas."""
    # 2. Reemplazar cada palabra protegida por una marca que text2num no entiende
    protegidas = cfg_norm["palabras_protegidas"]
    for i, palabra in enumerate(protegidas):
        texto = re.sub(rf"\b{palabra}\b", f"zzprot{i}zz", texto)

    # 3. text2num hace la conversión principal
    texto = alpha2digit(texto, "es", threshold=cfg_norm["umbral_text2num"])

    # 4. Devolver las palabras originales
    for i, palabra in enumerate(protegidas):
        texto = texto.replace(f"zzprot{i}zz", palabra)

    # 5a. "2 años y medio" -> "2.5 años"
    unidades = "|".join(cfg_norm["unidades_y_medio"])
    texto = re.sub(rf"\b(\d+) ({unidades}) y medio\b", r"\1.5 \2", texto)
    # 5b. "3 y medio años" -> "3.5 años"
    texto = re.sub(rf"\b(\d+) y medio ({unidades})\b", r"\1.5 \2", texto)

    # 5c. "6 lucas" -> "6000"
    for palabra, factor in cfg_norm["multiplicadores"].items():
        texto = re.sub(
            rf"\b(\d+) {palabra}\b",
            lambda m: str(int(m.group(1)) * factor),
            texto,
        )
    return texto


def quitar_puntuacion(texto: str) -> str:
    """Borra puntuación, pero conserva el punto decimal ('2.5')."""
    # Un punto que NO está entre dos dígitos es puntuación
    texto = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", texto)
    # Todo lo que no sea letra, dígito, punto o espacio se vuelve espacio
    return re.sub(r"[^\w\s.]", " ", texto)


def unir_miles(texto: str) -> str:
    """'6,500' y '9.000' -> '6500' y '9000'. Whisper escribe así los montos.

    Una coma o un punto seguido de EXACTAMENTE 3 dígitos es separador de miles;
    '2.5' sigue siendo decimal. Se repite para '1.234.567'.
    """
    anterior = None
    while anterior != texto:
        anterior = texto
        texto = re.sub(r"(\d)[.,](\d{3})(?!\d)", r"\1\2", texto)
    return texto


def normalizar(texto: str, cfg: dict) -> str:
    """Normalización completa según config.yaml (sección 'normalizacion')."""
    cfg_norm = cfg["normalizacion"]
    if cfg_norm["unir_miles"]:
        texto = unir_miles(texto)  # antes de todo: después la coma se volvería espacio
    if cfg_norm["minusculas"]:
        texto = texto.lower()
    if cfg_norm["numeros_a_digitos"]:
        texto = palabras_a_digitos(texto, cfg_norm)
    if cfg_norm["quitar_tildes"]:
        texto = quitar_tildes(texto)
    if cfg_norm["quitar_puntuacion"]:
        texto = quitar_puntuacion(texto)
    if cfg_norm["colapsar_espacios"]:
        texto = " ".join(texto.split())
    return texto
