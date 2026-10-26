"""
extract_rules.py — Extracción por reglas (regex).

Responsabilidad:
    - Extraer los campos de ambos esquemas ("entrevista" y "pedido")
      desde el transcript normalizado usando expresiones regulares simples.
    - Sirve como línea base transparente para comparar contra el LLM.
    - Devuelve un dict con los campos definidos en schemas.py (null si no encuentra).
"""
