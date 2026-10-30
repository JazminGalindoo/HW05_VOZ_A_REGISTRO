"""Tests de schemas.py, extract_rules.py y extract_llm.py (sin red)."""

import json
from types import SimpleNamespace

import pytest

from src.extract_llm import ErrorExtraccion, armar_solicitud, extraer_llm
from src.extract_rules import extraer_reglas
from src.schemas import leer_ground_truth, validar


def _sin_meta(registro):
    return {k: v for k, v in registro.items() if k != "_meta"}


def _items_simples(items):
    return [(i["producto"], i["cantidad"], i["precio_unitario"]) for i in items]


# ---------------- schemas ----------------

@pytest.mark.parametrize("schema", ["entrevista", "pedido"])
def test_ground_truth_cumple_el_esquema(cfg, schema):
    for clip, registro in leer_ground_truth(schema, cfg).items():
        assert validar(_sin_meta(registro), schema, cfg) == [], clip


def test_validar_detecta_errores(cfg):
    malo = {"anios_experiencia": "cuatro", "nivel_ingles": "experto"}
    errores = validar(malo, "entrevista", cfg)
    assert any("anios_experiencia" in e for e in errores)  # tipo incorrecto
    assert any("experto" in e for e in errores)            # fuera de la categoría
    assert any("falta" in e for e in errores)              # faltan campos


# ---------------- reglas: entrevistas ----------------

@pytest.mark.parametrize("clip", ["E01", "E02", "E03", "E04"])
def test_reglas_entrevista(cfg, referencias, clip):
    gt = leer_ground_truth("entrevista", cfg)[clip]
    r = extraer_reglas(referencias[clip], "entrevista", cfg)
    assert validar(r, "entrevista", cfg) == []
    for campo in ["anios_experiencia", "pretension_salarial", "moneda",
                  "disponibilidad", "nivel_ingles"]:
        assert r[campo] == gt[campo], campo
    assert r["ultimo_puesto"] == gt["ultimo_puesto"].lower()
    if clip == "E01":
        # Limitación conocida: "usaba Excel, pero eso ya casi no" -> las reglas igual lo listan
        assert r["tecnologias"] == gt["tecnologias"] + ["Excel"]
    else:
        assert r["tecnologias"] == gt["tecnologias"]


# ---------------- reglas: pedidos ----------------

@pytest.mark.parametrize("clip", ["L01", "L02", "L03", "L04"])
def test_reglas_pedido(cfg, referencias, clip):
    gt = leer_ground_truth("pedido", cfg)[clip]
    r = extraer_reglas(referencias[clip], "pedido", cfg)
    assert validar(r, "pedido", cfg) == []
    # Pedido FINAL: las autocorrecciones (L01, L02, L03) ya están aplicadas
    assert _items_simples(r["items"]) == _items_simples(gt["items"])
    assert r["descuento"] == gt["descuento"]
    assert (r["promocion"] is None) == (gt["promocion"] is None)
    assert r["metodo_pago"] == gt["metodo_pago"]
    assert r["total_dicho"] == gt["total_dicho"]  # el total DICHO, aunque esté mal
    # La dirección se compara a fondo en metrics.py; aquí número y distrito
    numero = gt["direccion_entrega"].split(",")[0].split()[-1]
    distrito = gt["direccion_entrega"].split(",")[-1].strip().lower()
    assert numero in r["direccion_entrega"] and distrito in r["direccion_entrega"]


def test_reglas_no_duplican_productos_repetidos(cfg):
    # Encontrado al probar el pipeline real: el resumen repetía cada producto
    texto = ("Su pedido es un pollo entero y una gaseosa. El pollo entero a sesenta y cuatro "
             "soles y la gaseosa a diez soles. El total es setenta y cuatro soles.")
    r = extraer_reglas(texto, "pedido", cfg)
    assert _items_simples(r["items"]) == [("pollo entero", 1, 64), ("gaseosa 1.5L", 1, 10)]


def test_reglas_con_formas_de_hablar_del_audio_real(cfg):
    # Frases reales del ASR: "está a N soles", "su pedido sería", "el total sería de", montos "6,500"
    texto = ("Un pollo entero, por favor. Pollo entero está a 64 soles. Entonces, su pedido sería "
             "un pollo entero. El total sería de 64 soles.")
    r = extraer_reglas(texto, "pedido", cfg)
    assert _items_simples(r["items"]) == [("pollo entero", 1, 64)] and r["total_dicho"] == 64
    e = extraer_reglas("¿Cuál es su pretensión salarial? Unos 6,500 soles.", "entrevista", cfg)
    assert e["pretension_salarial"] == 6500


# ---------------- LLM (Gemini) con cliente falso ----------------

from google.genai import errors, types  # noqa: E402

from src.extract_llm import esquema_para_gemini  # noqa: E402


@pytest.fixture
def cfg_rapido(cfg):
    """Sin pausas del nivel gratuito: los tests no deben esperar."""
    c = json.loads(json.dumps(cfg))
    c["extraccion"]["llm"].update(pausa_entre_llamadas_s=0, espera_si_limite_s=0)
    return c


class ClienteFalso:
    """Imita genai.Client: guarda las solicitudes y devuelve respuestas preparadas.

    Cada respuesta es (texto, motivo_de_fin) o una excepción para lanzar.
    """

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.solicitudes = []
        self.models = SimpleNamespace(generate_content=self._generar)

    def _generar(self, **solicitud):
        self.solicitudes.append(solicitud)
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        texto, motivo = r
        return SimpleNamespace(text=texto, prompt_feedback=None,
                               candidates=[SimpleNamespace(finish_reason=types.FinishReason[motivo])])


def _registro_e01():
    return {"anios_experiencia": 4, "ultimo_puesto": "analista de datos",
            "tecnologias": ["Python", "SQL", "Power BI"], "pretension_salarial": 6500,
            "moneda": "PEN", "disponibilidad": "2_semanas", "nivel_ingles": "intermedio"}


def test_solicitud_gemini_json_estructurado_y_temperatura_0(cfg, referencias):
    s = armar_solicitud(referencias["E01"], "entrevista", cfg)
    assert s["model"] == cfg["extraccion"]["llm"]["modelo"]
    assert s["config"].temperature == 0
    assert s["config"].response_mime_type == "application/json"
    assert s["config"].response_json_schema["required"]  # el esquema de schemas.py
    assert referencias["E01"] in s["contents"]  # va el transcript completo


def test_esquema_para_gemini_sin_listas_en_type(cfg):
    from src.schemas import esquema_json
    g = esquema_para_gemini(esquema_json("pedido", cfg))
    assert '"type": [' not in json.dumps(g)  # Gemini no acepta ["number", "null"]
    pago = g["properties"]["metodo_pago"]["anyOf"]
    assert {"type": "null"} in pago and None not in pago[0]["enum"]


def test_llm_devuelve_registro_valido(cfg_rapido, referencias):
    cliente = ClienteFalso([(json.dumps(_registro_e01()), "STOP")])
    assert extraer_llm(referencias["E01"], "entrevista", cfg_rapido, cliente) == _registro_e01()


def test_llm_reintenta_si_el_json_no_valida(cfg_rapido, referencias):
    malo = dict(_registro_e01(), nivel_ingles="experto")
    cliente = ClienteFalso([("no es json", "STOP"), (json.dumps(malo), "STOP"),
                            (json.dumps(_registro_e01()), "STOP")])
    assert extraer_llm(referencias["E01"], "entrevista", cfg_rapido, cliente) == _registro_e01()
    assert len(cliente.solicitudes) == 3


def test_llm_espera_si_el_api_responde_429_o_503(cfg_rapido, referencias):
    limite = errors.ClientError(429, {"error": {"message": "cuota", "status": "RESOURCE_EXHAUSTED"}})
    saturado = errors.ServerError(503, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})
    cliente = ClienteFalso([limite, saturado, (json.dumps(_registro_e01()), "STOP")])
    assert extraer_llm(referencias["E01"], "entrevista", cfg_rapido, cliente) == _registro_e01()
    assert len(cliente.solicitudes) == 3


def test_llm_otros_errores_no_se_esconden(cfg_rapido, referencias):
    malo = errors.ClientError(400, {"error": {"message": "bad request", "status": "INVALID_ARGUMENT"}})
    with pytest.raises(errors.ClientError):
        extraer_llm(referencias["E01"], "entrevista", cfg_rapido, ClienteFalso([malo]))


def test_llm_rechazo_lanza_error(cfg_rapido, referencias):
    cliente = ClienteFalso([("", "SAFETY")])
    with pytest.raises(ErrorExtraccion):
        extraer_llm(referencias["E01"], "entrevista", cfg_rapido, cliente)
