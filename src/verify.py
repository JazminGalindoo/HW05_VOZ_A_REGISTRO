"""
verify.py — Regla de verificación literal.

Responsabilidad:
    - Para cada campo extraído, decidir si es "verificado" (su valor aparece
      literalmente en el transcript normalizado) o "sin_verificar".
    - Es la defensa contra que el LLM "arregle" en silencio lo que el ASR oyó.
"""
