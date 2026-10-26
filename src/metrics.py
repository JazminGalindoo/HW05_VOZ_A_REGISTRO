"""
metrics.py — Métricas de evaluación.

Responsabilidad:
    - WER y CER con jiwer (sobre texto normalizado con normalize.py).
    - Accuracy por campo, macro-F1 para categóricas (nivel_ingles, metodo_pago)
      y error absoluto para numéricos.
    - Intervalos de confianza por bootstrap remuestreando clips
      (n de réplicas y semilla en config.yaml).
"""
