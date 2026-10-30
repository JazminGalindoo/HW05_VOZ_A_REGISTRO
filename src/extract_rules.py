"""
extract_rules.py — Extracción por reglas (regex).

Responsabilidad:
    - Extraer los campos de ambos esquemas ("entrevista" y "pedido")
      desde el transcript normalizado usando expresiones regulares simples.
    - Sirve como línea base transparente para comparar contra el LLM.
    - Devuelve un dict con los campos definidos en schemas.py (None si no encuentra).

Idea general: cada campo se busca DESPUÉS de una "pregunta ancla"
(p. ej. "pretensión", "inglés"), así no importa en qué orden se dijeron.
Para el pedido se usa el RESUMEN FINAL de la operadora ("su pedido es..."),
que ya incluye las autocorrecciones del cliente.

Limitación conocida: no entiende negaciones ("usaba Excel, pero ya casi no"
-> igual lista Excel). Es justo lo que se compara contra el LLM.
"""

import re

from src.normalize import normalizar

NUM = r"(\d+(?:\.\d+)?)"  # un número: "4", "2.5"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _numero(texto: str):
    """'4' -> 4, '2.5' -> 2.5"""
    valor = float(texto)
    return int(valor) if valor.is_integer() else valor


def _despues_de(ancla: str, texto: str) -> int:
    """Posición donde termina la primera aparición del ancla (0 si no aparece)."""
    m = re.search(ancla, texto)
    return m.end() if m else 0


def _buscar(patron: str, texto: str, desde: int = 0):
    return re.compile(patron).search(texto, desde)


# =====================================================================
# Entrevista
# =====================================================================

def _anios_experiencia(t: str):
    inicio = _despues_de(r"cuant[oa]s? (?:tiempo de )?(?:anos|experiencia)", t)
    m = _buscar(rf"{NUM} anos", t, inicio)
    return _numero(m.group(1)) if m else None


def _ultimo_puesto(t: str):
    # "mi ultimo puesto fue de analista de datos en una distribuidora"
    # "su cargo actual cual es soy jefa de proyectos ti en una empresa"
    m = _buscar(
        r"(?:ultimo puesto|cargo actual)(?: cual es)?(?: fue)?(?: de)?(?: exactamente)?"
        # "en una empresa" queda "en 1 empresa" después de normalizar
        r" (?:fui |era |soy )?(.+?) en (?:1|el|la) ",
        t,
    )
    if not m or len(m.group(1).split()) > 8:  # más de 8 palabras: no es un cargo
        return None
    return m.group(1)


def _tecnologias(t: str, cfg: dict) -> list[str]:
    encontradas = []
    for nombre in cfg["extraccion"]["reglas"]["tecnologias_conocidas"]:
        m = re.search(rf"\b{re.escape(normalizar(nombre, cfg))}\b", t)
        if m:
            encontradas.append((m.start(), nombre))
    return [nombre for _, nombre in sorted(encontradas)]  # en el orden en que se dijeron


def _pretension(t: str, cfg: dict):
    monedas = cfg["extraccion"]["reglas"]["monedas"]
    inicio = _despues_de(r"pretension|sueldo|expectativa|salari", t)
    m = _buscar(rf"{NUM} ({'|'.join(monedas)})\b", t, inicio)
    if not m:
        return None, None
    return _numero(m.group(1)), monedas[m.group(2)]


def _disponibilidad(t: str, cfg: dict):
    inicio = _despues_de(r"empezar|incorporar|disponible", t)
    candidatos = []  # (posición, valor): gana el que aparece primero

    m = _buscar(r"inmediat|desde ya", t, inicio)
    if m:
        candidatos.append((m.start(), "inmediata"))

    m = _buscar(r"\ben (\d+) (semanas?|mes|meses)\b", t, inicio)
    if m:
        candidatos.append((m.start(), f"{m.group(1)}_{m.group(2)}"))

    m = _buscar(rf"(?:a partir del?|desde el) (primero|\d+) de ({'|'.join(MESES)})", t, inicio)
    if m:
        dia = 1 if m.group(1) == "primero" else int(m.group(1))
        mes = MESES.index(m.group(2)) + 1
        anio = cfg["extraccion"]["anio_referencia"]
        candidatos.append((m.start(), f"{anio}-{mes:02d}-{dia:02d}"))

    return min(candidatos)[1] if candidatos else None


def _nivel_ingles(t: str, cfg: dict):
    niveles = cfg["extraccion"]["categorias"]["nivel_ingles"]
    inicio = _despues_de(r"ingles", t)
    m = _buscar(rf"\b({'|'.join(niveles)})\b", t, inicio)
    return m.group(1) if m else None


def extraer_entrevista(t: str, cfg: dict) -> dict:
    monto, moneda = _pretension(t, cfg)
    return {
        "anios_experiencia": _anios_experiencia(t),
        "ultimo_puesto": _ultimo_puesto(t),
        "tecnologias": _tecnologias(t, cfg),
        "pretension_salarial": monto,
        "moneda": moneda,
        "disponibilidad": _disponibilidad(t, cfg),
        "nivel_ingles": _nivel_ingles(t, cfg),
    }


# =====================================================================
# Pedido
# =====================================================================

def _alias_normalizados(cfg: dict) -> list[tuple[str, dict]]:
    """[(alias normalizado, producto de la carta)], del alias más largo al más corto.

    Más largo primero para que "gaseosa de litro y medio" gane a "gaseosa".
    """
    pares = [
        (normalizar(alias, cfg), producto)
        for producto in cfg["extraccion"]["reglas"]["carta"]
        for alias in producto["alias"]
    ]
    return sorted(pares, key=lambda par: len(par[0]), reverse=True)


def _resumen_final(t: str) -> str:
    """Último resumen de la operadora: 'su pedido es ... el total es'. Si no hay, todo el texto."""
    resumenes = re.findall(r"(?:su pedido es|entonces seria|entonces es)(.+?)(?:el total es|$)", t)
    return resumenes[-1] if resumenes else t


def _productos(resumen: str, cfg: dict) -> list[tuple[dict, int]]:
    """[(producto, cantidad)] en el orden en que aparecen en el resumen."""
    ocupado = []  # tramos ya usados, para no contar "chaufa" dentro de "arroz chaufa"
    hallados = []
    for alias, producto in _alias_normalizados(cfg):
        for m in re.finditer(rf"(?:\b(\d+|unas|unos) )?\b{re.escape(alias)}\b", resumen):
            if any(m.start() < fin and inicio < m.end() for inicio, fin in ocupado):
                continue
            ocupado.append((m.start(), m.end()))
            # "unas papas" o sin número -> 1
            cantidad = int(m.group(1)) if m.group(1) and m.group(1).isdigit() else 1
            hallados.append((m.start(), producto, cantidad))
    return [(producto, cantidad) for _, producto, cantidad in sorted(hallados, key=lambda h: h[0])]


def _precio(producto: dict, t: str, cfg: dict):
    """Primer 'ALIAS [a] N soles' en toda la llamada (la operadora dice los precios)."""
    for alias in producto["alias"]:
        m = re.search(rf"\b{re.escape(normalizar(alias, cfg))} (?:a )?{NUM} soles", t)
        if m:
            return _numero(m.group(1))
    return None


def _promocion(t: str, items: list[dict], cfg: dict):
    """(texto de la promoción, descuento en soles)."""
    m = re.search(rf"({NUM} soles de descuento(?: en compras mayores a \d+ soles)?)", t)
    if m:
        return m.group(1), _numero(m.group(2))

    # "las gaseosas estan 2 por 1": el descuento es el precio de la unidad gratis
    m = re.search(r"((?:\S+ ){0,4})2 por 1", t)
    if m:
        for alias, producto in _alias_normalizados(cfg):
            if alias in m.group(1):
                precio = next((i["precio_unitario"] for i in items
                               if i["producto"] == producto["producto"]), None)
                return f"{producto['producto']} 2x1", precio or 0
    return None, 0


def _direccion(t: str, cfg: dict):
    m = re.search(r"\b(avenida|jiron|calle|pasaje)((?: [a-z]+){1,4}?) (\d+)\b", t)
    if not m:
        return None
    partes = [f"{m.group(1)}{m.group(2)} {m.group(3)}"]
    # Interior y distrito se buscan en las ~12 palabras siguientes
    ventana = " ".join(t[m.end():].split()[:12])
    unidad = re.search(r"\b(departamento|interior) (\d+(?: [a-z])?)\b", ventana)
    if unidad:
        partes.append(f"{unidad.group(1)} {unidad.group(2).replace(' ', '')}")  # "4 b" -> "4b"
    for distrito in cfg["extraccion"]["reglas"]["distritos"]:
        if re.search(rf"\b{normalizar(distrito, cfg)}\b", ventana):
            partes.append(normalizar(distrito, cfg))
            break
    return ", ".join(partes)


def _metodo_pago(t: str, cfg: dict):
    inicio = _despues_de(r"como va a pagar|forma de pago|como desea pagar|metodo de pago", t)
    mejor = None  # (posición, categoría): gana la palabra que aparece primero
    for categoria, palabras in cfg["extraccion"]["reglas"]["metodos_pago"].items():
        for palabra in palabras:
            m = _buscar(rf"\b{palabra}\b", t, inicio)
            if m and (mejor is None or m.start() < mejor[0]):
                mejor = (m.start(), categoria)
    return mejor[1] if mejor else None


def _total_dicho(t: str):
    totales = re.findall(rf"total es (?:de )?{NUM}", t)
    return _numero(totales[-1]) if totales else None  # el último total dicho


def extraer_pedido(t: str, cfg: dict) -> dict:
    items = [
        {
            "producto": producto["producto"],
            "tamano_o_plan": producto["tamano"],
            "cantidad": cantidad,
            "precio_unitario": _precio(producto, t, cfg),
        }
        for producto, cantidad in _productos(_resumen_final(t), cfg)
    ]
    promocion, descuento = _promocion(t, items, cfg)
    return {
        "items": items,
        "promocion": promocion,
        "descuento": descuento,
        "direccion_entrega": _direccion(t, cfg),
        "metodo_pago": _metodo_pago(t, cfg),
        "total_dicho": _total_dicho(t),
    }


def extraer_reglas(transcript: str, schema: str, cfg: dict) -> dict:
    """Punto de entrada: normaliza el transcript y aplica las reglas del esquema."""
    t = normalizar(transcript, cfg)
    if schema == "entrevista":
        return extraer_entrevista(t, cfg)
    if schema == "pedido":
        return extraer_pedido(t, cfg)
    raise ValueError(f"schema desconocido: {schema!r}")
