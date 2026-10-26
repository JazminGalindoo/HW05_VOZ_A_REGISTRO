"""
schemas.py — Definición de los campos de cada esquema.

Responsabilidad:
    - entrevista: anios_experiencia, ultimo_puesto, tecnologias,
      pretension_salarial (monto + moneda), disponibilidad, nivel_ingles.
    - pedido: items [{producto, tamano_o_plan, cantidad, precio_unitario}],
      promocion, direccion_entrega o plazo_contrato, metodo_pago, total_dicho.
    - Valida que el JSON del LLM (o de las reglas) tenga los campos y tipos correctos.
"""
