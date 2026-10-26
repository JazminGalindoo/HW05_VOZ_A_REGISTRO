"""
validate.py — Validación aritmética del pedido y diagnóstico.

Responsabilidad:
    - Chequear sum(cantidad * precio_unitario) - promocion == total_dicho
      con la tolerancia de config.yaml.
    - Si no cuadra, diagnosticar si es error del ASR o del hablante usando:
        * probabilidad por palabra de los tokens numéricos implicados, y
        * consistencia entre niveles (si L0 cuadra y Lk no -> ASR).
    - Medir cuántas veces acierta el diagnóstico con las trampas marcadas
      en el ground truth.
"""
