"""
schemas.py — Definición de los campos de cada esquema.

Responsabilidad:
    - entrevista: anios_experiencia, ultimo_puesto, tecnologias,
      pretension_salarial + moneda, disponibilidad, nivel_ingles.
    - pedido: items [{producto, tamano_o_plan, cantidad, precio_unitario}],
      promocion, descuento, direccion_entrega, metodo_pago, total_dicho.
    - Generar el JSON schema que se le pasa al LLM (structured outputs).
    - Validar que un registro (del LLM o de las reglas) tenga los campos y tipos correctos.
    - Leer el ground truth (CSV) en el mismo formato que un registro extraído.
"""

import csv
from pathlib import Path

ESQUEMAS_VALIDOS = ("entrevista", "pedido")

# Tipos JSON: "number", "string", "array"... con "null" si el dato puede faltar
NUM_O_NULL = {"type": ["number", "null"]}
TEXTO_O_NULL = {"type": ["string", "null"]}


def _categoria(valores: list) -> dict:
    """Campo categórico: uno de los valores permitidos, o null."""
    return {"type": ["string", "null"], "enum": list(valores) + [None]}


def _objeto(propiedades: dict) -> dict:
    """Objeto con TODOS los campos obligatorios y sin campos extra (lo exige structured outputs)."""
    return {
        "type": "object",
        "properties": propiedades,
        "required": list(propiedades),
        "additionalProperties": False,
    }


def esquema_json(schema: str, cfg: dict) -> dict:
    """JSON schema del registro. Las categorías vienen de config.yaml."""
    cat = cfg["extraccion"]["categorias"]
    if schema == "entrevista":
        return _objeto({
            "anios_experiencia": NUM_O_NULL,
            "ultimo_puesto": TEXTO_O_NULL,
            "tecnologias": {"type": "array", "items": {"type": "string"}},
            "pretension_salarial": NUM_O_NULL,
            "moneda": _categoria(cat["moneda"]),
            "disponibilidad": TEXTO_O_NULL,
            "nivel_ingles": _categoria(cat["nivel_ingles"]),
        })
    if schema == "pedido":
        item = _objeto({
            "producto": {"type": "string"},
            "tamano_o_plan": TEXTO_O_NULL,
            "cantidad": {"type": "number"},
            "precio_unitario": NUM_O_NULL,
        })
        return _objeto({
            "items": {"type": "array", "items": item},
            "promocion": TEXTO_O_NULL,
            "descuento": {"type": "number"},
            "direccion_entrega": TEXTO_O_NULL,
            "metodo_pago": _categoria(cat["metodo_pago"]),
            "total_dicho": NUM_O_NULL,
        })
    raise ValueError(f"schema debe ser uno de {ESQUEMAS_VALIDOS}, no {schema!r}")


# --- Validación (un validador mínimo, suficiente para estos esquemas) ---

PYTHON_DE = {"number": (int, float), "string": str, "array": list, "object": dict}


def _errores(valor, regla: dict, ruta: str) -> list[str]:
    tipos = regla["type"] if isinstance(regla["type"], list) else [regla["type"]]
    if valor is None:
        return [] if "null" in tipos else [f"{ruta}: no puede ser null"]
    # bool es subclase de int en Python: no lo aceptamos como número
    if isinstance(valor, bool) or not any(
        isinstance(valor, PYTHON_DE[t]) for t in tipos if t != "null"
    ):
        return [f"{ruta}: tipo {type(valor).__name__}, se esperaba {tipos}"]
    if "enum" in regla and valor not in regla["enum"]:
        return [f"{ruta}: {valor!r} no está en {regla['enum']}"]
    if isinstance(valor, list):
        return [e for i, v in enumerate(valor) for e in _errores(v, regla["items"], f"{ruta}[{i}]")]
    if isinstance(valor, dict):
        faltan = set(regla["required"]) - set(valor)
        sobran = set(valor) - set(regla["properties"])
        errores = [f"{ruta}: falta {c}" for c in sorted(faltan)]
        errores += [f"{ruta}: campo extra {c}" for c in sorted(sobran)]
        for campo, sub in regla["properties"].items():
            if campo in valor:
                errores += _errores(valor[campo], sub, f"{ruta}.{campo}")
        return errores
    return []


def validar(registro: dict, schema: str, cfg: dict) -> list[str]:
    """Lista de errores; vacía si el registro es válido."""
    return _errores(registro, esquema_json(schema, cfg), schema)


# --- Ground truth ---

def _numero(texto: str):
    """'4' -> 4, '2.5' -> 2.5, '' -> None."""
    if texto is None or texto.strip() == "":
        return None
    valor = float(texto)
    return int(valor) if valor.is_integer() else valor


def _items_gt(texto: str) -> list[dict]:
    """'pollo entero x1 @64; ensalada x1 @8' -> [{producto, cantidad, precio_unitario}, ...]"""
    items = []
    for parte in texto.split(";"):
        producto, resto = parte.strip().rsplit(" x", 1)
        cantidad, precio = resto.split(" @")
        items.append({
            "producto": producto.strip(),
            "tamano_o_plan": None,  # el ground truth no lo separa del producto
            "cantidad": _numero(cantidad),
            "precio_unitario": _numero(precio),
        })
    return items


def leer_ground_truth(schema: str, cfg: dict) -> dict[str, dict]:
    """{clip: registro} leído del CSV, con los mismos campos que un registro extraído.

    Las columnas extra del CSV (trampas, orden de campos...) van en registro['_meta'].
    """
    clave = "ground_truth_entrevistas" if schema == "entrevista" else "ground_truth_llamadas"
    ruta = Path(__file__).resolve().parent.parent / cfg["rutas"][clave]
    registros = {}
    with open(ruta, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            if schema == "entrevista":
                registro = {
                    "anios_experiencia": _numero(fila["anios_experiencia"]),
                    "ultimo_puesto": fila["ultimo_puesto"],
                    "tecnologias": fila["tecnologias"].split("|"),
                    "pretension_salarial": _numero(fila["pretension_salarial"]),
                    "moneda": fila["moneda"],
                    "disponibilidad": fila["disponibilidad"],
                    "nivel_ingles": fila["nivel_ingles"],
                    "_meta": {"orden_campos": fila["orden_campos"]},
                }
            else:
                registro = {
                    "items": _items_gt(fila["items"]),
                    "promocion": fila["promocion"] or None,
                    "descuento": _numero(fila["descuento"]),
                    "direccion_entrega": fila["direccion_entrega"],
                    "metodo_pago": fila["metodo_pago"],
                    "total_dicho": _numero(fila["total_dicho"]),
                    "_meta": {
                        "total_correcto": _numero(fila["total_correcto"]),
                        "trampa_autocorreccion": fila["trampa_autocorreccion"] == "1",
                        "trampa_total_erroneo": fila["trampa_total_erroneo"] == "1",
                    },
                }
            registros[fila["clip"]] = registro
    return registros
