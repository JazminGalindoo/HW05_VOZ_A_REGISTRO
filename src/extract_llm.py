"""
extract_llm.py — Extracción con LLM.

Responsabilidad:
    - Enviar el transcript al LLM configurado (proveedor, modelo, prompts y
      temperatura 0 en config.yaml; API key leída de .env).
    - Pedir salida JSON y validarla contra schemas.py; reintentar si no valida.
    - El LLM NO puede corregir al ASR: lo que devuelva pasa luego por verify.py.
"""
