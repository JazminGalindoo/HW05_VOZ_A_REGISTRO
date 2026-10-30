"""
engine.py — Única puerta de entrada del proyecto.

Responsabilidad:
    Recibir un audio y devolver un registro estructurado (entrevista o pedido)
    junto con todo lo necesario para auditarlo: transcript, segmentos con
    probabilidad por palabra, campos extraídos, qué campos están verificados
    literalmente, y métricas de ejecución (device real, compute_type, tiempo,
    VRAM pico).

Reglas:
    - La app, el bot, los notebooks y los scripts SOLO llaman a este módulo.
    - Este módulo NO importa ninguna librería de UI (streamlit, gradio, telegram).
    - Ningún parámetro está hardcodeado: todo viene de config.yaml (cfg).

Dos funciones públicas (decisión documentada en el README):
    - process_audio(audio_path, schema, cfg): audio -> ASR (o cache) -> process_transcript.
    - process_transcript(transcript, schema, cfg, segments): solo extracción + verificación.
      Existe para evaluar sin audio (con los transcripts de referencia) y la usa
      process_audio por dentro: la lógica de extracción vive en un solo lugar.

Qué combinación de modelo/device/glosario/método se usa sale de cfg["motor"].
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.asr import guardar_cache, leer_cache, ruta_cache, transcribir
from src.extract_llm import extraer_llm
from src.extract_rules import extraer_reglas
from src.schemas import ESQUEMAS_VALIDOS
from src.verify import verificar

RAIZ = Path(__file__).resolve().parent.parent


@dataclass
class EngineResult:
    """Resultado completo de procesar un audio."""

    transcript: str = ""                                # texto completo del ASR
    segments: list = field(default_factory=list)        # segmentos con palabras y probabilidades
    fields: dict = field(default_factory=dict)          # campos extraídos según el esquema
    verified: dict = field(default_factory=dict)        # campo -> True si aparece literal en el transcript
    device: str = ""                                    # leído de la librería en runtime, no del config
    compute_type: str = ""                              # float16 / int8 ...
    elapsed_s: float = 0.0                              # segundos de transcripción
    peak_vram_mb: float | None = None                   # pico de VRAM (pynvml) menos línea base; None en CPU
    error: str | None = None                            # mensaje si algo falló
    desde_cache: bool = False                           # True si el transcript se leyó de cache/


def cargar_config(ruta: str | None = None) -> dict:
    """Lee config.yaml (por defecto el de la raíz del repo)."""
    ruta = Path(ruta) if ruta else RAIZ / "config.yaml"
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def con_motor(cfg: dict, **cambios) -> dict:
    """Copia de cfg con otras opciones de motor: con_motor(cfg, modelo='large-v3', nivel='L2')."""
    nuevo = copy.deepcopy(cfg)
    nuevo["motor"].update(cambios)
    return nuevo


def process_transcript(transcript: str, schema: str, cfg: dict | None = None,
                       segments: list | None = None) -> EngineResult:
    """Extracción + verificación sobre un texto ya transcrito."""
    cfg = cfg or cargar_config()
    resultado = EngineResult(transcript=transcript, segments=segments or [])
    try:
        if schema not in ESQUEMAS_VALIDOS:
            raise ValueError(f"schema debe ser uno de {ESQUEMAS_VALIDOS}, no {schema!r}")
        metodo = cfg["motor"]["metodo_extraccion"]
        if metodo == "reglas":
            resultado.fields = extraer_reglas(transcript, schema, cfg)
        elif metodo == "llm":
            resultado.fields = extraer_llm(transcript, schema, cfg)
        else:
            raise ValueError(f"metodo_extraccion desconocido: {metodo!r}")
        resultado.verified = verificar(resultado.fields, transcript, cfg)
    except Exception as e:  # el error se reporta, no tumba la app ni el notebook
        resultado.error = f"{type(e).__name__}: {e}"
    return resultado


def process_audio(audio_path: str, schema: str, cfg: dict | None = None) -> EngineResult:
    """
    Procesa un audio y devuelve un EngineResult.

    audio_path: ruta al archivo de audio.
    schema:     "entrevista" o "pedido".
    cfg:        configuración ya cargada de config.yaml (si es None, se carga).

    Si el transcript ya está en cache/transcripts/ (y motor.usar_cache), NO se
    re-transcribe. Con motor.solo_cache = true (app y CI) nunca se transcribe.
    """
    cfg = cfg or cargar_config()
    motor = cfg["motor"]
    try:
        cache = ruta_cache(Path(audio_path).stem, cfg)
        if motor["usar_cache"] and cache.exists():
            asr, desde_cache = leer_cache(cache), True
        elif motor["solo_cache"]:
            raise FileNotFoundError(f"no está en cache y solo_cache=true: {cache.name}")
        else:
            asr, desde_cache = transcribir(audio_path, schema, cfg), False
            guardar_cache(asr, cache)
    except Exception as e:
        return EngineResult(error=f"{type(e).__name__}: {e}")

    resultado = process_transcript(asr["transcript"], schema, cfg, asr["segments"])
    resultado.device = asr["device"]
    resultado.compute_type = asr["compute_type"]
    resultado.elapsed_s = asr["elapsed_s"]
    resultado.peak_vram_mb = asr["peak_vram_mb"]
    resultado.desde_cache = desde_cache
    return resultado
