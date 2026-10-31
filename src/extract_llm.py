"""
extract_llm.py — Extracción con LLM (Google Gemini).

Responsabilidad:
    - Enviar el transcript al LLM configurado (proveedor, modelo, prompts y
      temperatura 0 en config.yaml; API key leída de .env: GEMINI_API_KEY).
    - Pedir salida JSON estructurada: se pasa el JSON schema de schemas.py
      (response_json_schema) para que el modelo solo pueda responder con esa forma;
      igual se valida y se reintenta si no cumple.
    - El LLM NO puede corregir al ASR: lo que devuelva pasa luego por verify.py.
    - Nivel gratuito: pausa entre llamadas y espera si el API responde 429.

Para los tests se pasa un cliente falso en el parámetro `cliente`,
así no se llama a la red ni se gasta cuota.
"""

import copy
import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from src.schemas import esquema_json, validar

RAIZ = Path(__file__).resolve().parent.parent

# Motivos de fin que significan "el modelo no quiso responder"
RECHAZOS = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"}


class ErrorExtraccion(Exception):
    """El LLM no devolvió un registro válido."""


def crear_cliente(cfg: dict) -> genai.Client:
    """Cliente real de Gemini. La API key se lee de .env (nunca del código).

    Con timeout: sin él, una solicitud que nunca responde bloquea la corrida para siempre.
    """
    load_dotenv()
    llm = cfg["extraccion"]["llm"]
    return genai.Client(api_key=os.environ.get(llm["api_key_env"]),
                        http_options=types.HttpOptions(timeout=llm["timeout_s"] * 1000))  # en ms


def es_limite_diario(e: errors.APIError) -> bool:
    """¿El 429 es por la cuota DIARIA? Esa no se libera en segundos: esperar no sirve."""
    return "PerDay" in json.dumps(e.details, default=str)


def ruta_cache_llm(transcript: str, schema: str, cfg: dict) -> Path:
    """Archivo de cache de UNA respuesta: depende del modelo, los prompts y el transcript.

    Si cambia cualquiera de los tres, el hash cambia y se vuelve a preguntar al LLM.
    """
    llm = cfg["extraccion"]["llm"]
    clave = json.dumps([llm["modelo"], llm["temperatura"], cfg["extraccion"]["prompts"],
                        schema, transcript], ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(clave.encode("utf-8")).hexdigest()[:16]
    return RAIZ / cfg["rutas"]["cache_llm"] / f"{schema}__{llm['modelo']}__{h}.json"


def esquema_para_gemini(esquema: dict) -> dict:
    """Adapta el JSON schema de schemas.py a lo que acepta Gemini.

    Gemini no acepta una lista en "type" (["number", "null"]) ni null dentro de "enum":
    eso se escribe como anyOf: [{"type": "number"}, {"type": "null"}].
    """
    esquema = copy.deepcopy(esquema)
    if isinstance(esquema.get("type"), list):
        tipos = esquema.pop("type")
        valores = esquema.pop("enum", None)
        opciones = []
        for t in tipos:
            opcion = {"type": t}
            if valores is not None and t != "null":
                opcion["enum"] = [v for v in valores if v is not None]
            opciones.append(opcion)
        esquema["anyOf"] = opciones
    if "properties" in esquema:
        esquema["properties"] = {k: esquema_para_gemini(v) for k, v in esquema["properties"].items()}
    if "items" in esquema:
        esquema["items"] = esquema_para_gemini(esquema["items"])
    return esquema


def armar_solicitud(transcript: str, schema: str, cfg: dict) -> dict:
    """Parámetros de models.generate_content. Separado para poder revisarlo en los tests."""
    llm = cfg["extraccion"]["llm"]
    prompts = cfg["extraccion"]["prompts"]
    instruccion = prompts[schema].format(anio=cfg["extraccion"]["anio_referencia"])
    return {
        "model": llm["modelo"],
        "contents": f"{instruccion}\n\n<transcripcion>\n{transcript}\n</transcripcion>",
        "config": types.GenerateContentConfig(
            system_instruction=prompts["sistema"],
            temperature=llm["temperatura"],
            max_output_tokens=llm["max_tokens"],
            response_mime_type="application/json",
            response_json_schema=esquema_para_gemini(esquema_json(schema, cfg)),
        ),
    }


def _llamar(cliente, solicitud: dict, llm: dict):
    """Una llamada al API, esperando si responde 429 (límite del nivel gratuito)
    o 503 (modelo saturado, "high demand"): los dos son temporales."""
    for _ in range(llm["max_esperas_por_limite"] + 1):
        time.sleep(llm["pausa_entre_llamadas_s"])  # no pasar el límite por minuto
        try:
            return cliente.models.generate_content(**solicitud)
        except errors.APIError as e:
            if e.code not in (429, 503):
                raise
            if e.code == 429 and es_limite_diario(e):
                raise ErrorExtraccion("cuota DIARIA del nivel gratuito agotada "
                                      f"({llm['modelo']}): reintentar mañana") from e
            time.sleep(llm["espera_si_limite_s"])
    raise ErrorExtraccion("el API siguió respondiendo 429/503: se agotaron las esperas")


def extraer_llm(transcript: str, schema: str, cfg: dict, cliente=None) -> dict:
    """Devuelve el registro extraído por el LLM o lanza ErrorExtraccion.

    Con el cliente real y llm.usar_cache, la respuesta se lee/guarda en cache/llm/.
    Con un cliente falso (tests) no se usa el cache.
    """
    llm = cfg["extraccion"]["llm"]
    usar_cache = cliente is None and llm.get("usar_cache", False)
    if usar_cache:
        ruta = ruta_cache_llm(transcript, schema, cfg)
        if ruta.exists():
            return json.loads(ruta.read_text(encoding="utf-8"))
    registro = _extraer(transcript, schema, cfg, cliente or crear_cliente(cfg))
    if usar_cache:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return registro


def _extraer(transcript: str, schema: str, cfg: dict, cliente) -> dict:
    """Pregunta al LLM, valida y reintenta (sin cache)."""
    llm = cfg["extraccion"]["llm"]
    solicitud = armar_solicitud(transcript, schema, cfg)

    problemas = []
    for _ in range(1 + llm["reintentos"]):
        respuesta = _llamar(cliente, solicitud, llm)

        if respuesta.prompt_feedback and respuesta.prompt_feedback.block_reason:
            raise ErrorExtraccion(f"solicitud bloqueada: {respuesta.prompt_feedback.block_reason}")
        candidato = respuesta.candidates[0] if respuesta.candidates else None
        motivo = candidato.finish_reason.name if candidato and candidato.finish_reason else None
        if motivo in RECHAZOS:
            raise ErrorExtraccion(f"el modelo rechazó la solicitud: {motivo}")
        if motivo == "MAX_TOKENS":
            problemas = ["respuesta cortada por max_tokens"]
            continue

        try:
            registro = json.loads(respuesta.text or "")
        except json.JSONDecodeError as e:
            problemas = [f"JSON inválido: {e}"]
            continue

        problemas = validar(registro, schema, cfg)
        if not problemas:
            return registro

    raise ErrorExtraccion(f"sin registro válido tras los reintentos: {problemas}")
