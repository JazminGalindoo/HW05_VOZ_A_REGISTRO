"""
extract_llm.py — Extracción con LLM.

Responsabilidad:
    - Enviar el transcript al LLM configurado (proveedor, modelo, prompts y
      esfuerzo en config.yaml; API key leída de .env).
    - Pedir salida JSON con structured outputs (el API garantiza que el JSON
      cumple el esquema de schemas.py); igual se valida y se reintenta si falla.
    - El LLM NO puede corregir al ASR: lo que devuelva pasa luego por verify.py.

Sobre la "temperatura 0": desde la versión de API 2026-10-21 (SDK anthropic 2.x)
el parámetro temperature ya no existe (error 400). La estabilidad de la salida
viene del JSON schema y del effort bajo; ver config.yaml.

Para los tests se pasa un cliente falso en el parámetro `cliente`,
así no se llama a la red ni se gasta dinero.
"""

import json
import os

import anthropic
from dotenv import load_dotenv

from src.schemas import esquema_json, validar


class ErrorExtraccion(Exception):
    """El LLM no devolvió un registro válido."""


def crear_cliente(cfg: dict) -> anthropic.Anthropic:
    """Cliente real de Anthropic. La API key se lee de .env (nunca del código)."""
    load_dotenv()
    return anthropic.Anthropic(api_key=os.environ.get(cfg["extraccion"]["llm"]["api_key_env"]))


def armar_solicitud(transcript: str, schema: str, cfg: dict) -> dict:
    """Parámetros de messages.create. Separado para poder revisarlo en los tests."""
    llm = cfg["extraccion"]["llm"]
    prompts = cfg["extraccion"]["prompts"]
    instruccion = prompts[schema].format(anio=cfg["extraccion"]["anio_referencia"])
    solicitud = {
        "model": llm["modelo"],
        "max_tokens": llm["max_tokens"],
        "system": prompts["sistema"],
        "messages": [{
            "role": "user",
            "content": f"{instruccion}\n\n<transcripcion>\n{transcript}\n</transcripcion>",
        }],
        "output_config": {
            "effort": llm["effort"],
            "format": {"type": "json_schema", "schema": esquema_json(schema, cfg)},
        },
    }
    if llm.get("fallbacks"):
        # Si el modelo rechaza la solicitud, el API la reintenta en otro modelo
        solicitud["betas"] = [llm["beta_fallbacks"]]
        solicitud["fallbacks"] = llm["fallbacks"]
    return solicitud


def extraer_llm(transcript: str, schema: str, cfg: dict, cliente=None) -> dict:
    """Devuelve el registro extraído por el LLM o lanza ErrorExtraccion."""
    cliente = cliente or crear_cliente(cfg)
    solicitud = armar_solicitud(transcript, schema, cfg)
    # Los fallbacks son una función beta: van por cliente.beta.messages
    crear = cliente.beta.messages.create if "betas" in solicitud else cliente.messages.create

    problemas = []
    for _ in range(1 + cfg["extraccion"]["llm"]["reintentos"]):
        respuesta = crear(**solicitud)

        if respuesta.stop_reason == "refusal":
            raise ErrorExtraccion("el modelo rechazó la solicitud")
        if respuesta.stop_reason == "max_tokens":
            problemas = ["respuesta cortada por max_tokens"]
            continue

        texto = next((b.text for b in respuesta.content if b.type == "text"), "")
        try:
            registro = json.loads(texto)
        except json.JSONDecodeError as e:
            problemas = [f"JSON inválido: {e}"]
            continue

        problemas = validar(registro, schema, cfg)
        if not problemas:
            return registro

    raise ErrorExtraccion(f"sin registro válido tras los reintentos: {problemas}")
