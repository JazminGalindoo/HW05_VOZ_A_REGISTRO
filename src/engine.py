"""
engine.py — Única puerta de entrada del proyecto.

Responsabilidad:
    Recibir un audio y devolver un registro estructurado (entrevista o pedido)
    junto con todo lo necesario para auditarlo: transcript, segmentos con
    probabilidad por palabra, campos extraídos, qué campos están verificados
    literalmente, y métricas de ejecución (device real, compute_type, tiempo,
    VRAM pico).

Reglas:
    - La app, el bot, los notebooks y los scripts SOLO llaman a process_audio().
    - Este módulo NO importa ninguna librería de UI (streamlit, gradio, telegram).
    - Ningún parámetro está hardcodeado: todo viene de config.yaml (cfg).

Flujo previsto (se implementa en fases siguientes):
    asr.transcribir -> normalize -> extract_rules / extract_llm -> verify
"""

from dataclasses import dataclass, field


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


def process_audio(audio_path: str, schema: str, cfg: dict | None = None) -> EngineResult:
    """
    Procesa un audio y devuelve un EngineResult.

    audio_path: ruta al archivo de audio.
    schema:     "entrevista" o "pedido".
    cfg:        configuración ya cargada de config.yaml (si es None, se carga).
    """
    raise NotImplementedError("Se implementa en la fase del motor.")
