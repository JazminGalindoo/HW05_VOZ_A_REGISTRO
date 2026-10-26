"""
normalize.py — Normalización de texto.

Responsabilidad:
    - Normalizar texto para WER/CER, IGUAL para hipótesis y referencia:
      minúsculas, sin puntuación, espacios colapsados, (opcional) sin tildes.
    - Convertir números dichos en palabras a dígitos
      ("seis mil quinientos" -> "6500") con text2num (lang="es"),
      más reglas propias documentadas para coloquialismos
      ("seis lucas", "tres y medio años"), definidos en config.yaml.
    - El transcript normalizado es la base de la verificación literal.
"""
