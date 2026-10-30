"""
asr.py — Transcripción con faster-whisper.

Responsabilidad:
    - Cargar WhisperModel con el tamaño, device y compute_type pedidos en config.yaml.
    - Transcribir en español (idioma fijo), con o sin glosario
      (initial_prompt / hotwords según la versión instalada).
    - Devolver texto, segmentos y palabras con su probabilidad.
    - Medir en runtime: device REAL (del objeto del modelo / ctranslate2),
      tiempo transcurrido y VRAM pico con pynvml (hilo que muestrea cada ~50 ms;
      torch.cuda.max_memory_allocated NO sirve porque CTranslate2 no usa PyTorch).
    - Guardar y leer el cache de transcripts (cache/transcripts/*.json).
"""

import inspect
import json
import threading
import time
from pathlib import Path

import ctranslate2
import faster_whisper
from faster_whisper import WhisperModel

RAIZ = Path(__file__).resolve().parent.parent


# =====================================================================
# Device y VRAM
# =====================================================================

def resolver_device(pedido: str) -> str:
    """'auto' -> 'cuda' si CTranslate2 ve una GPU, si no 'cpu'."""
    if pedido == "auto":
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return pedido


class MedidorVRAM:
    """Mide el pico de VRAM de la GPU mientras corre un bloque `with`.

    Un hilo lee la memoria usada (pynvml) cada `intervalo_ms`. Resultado:
    pico - línea base (lo que había ocupado antes de empezar), en MB.
    Si no hay GPU NVIDIA, pico_mb queda en None.
    """

    def __init__(self, gpu_indice: int, intervalo_ms: int):
        self.gpu_indice = gpu_indice
        self.intervalo_s = intervalo_ms / 1000
        self.pico_mb = None
        self._parar = threading.Event()

    def _usada_mb(self) -> float:
        import pynvml
        return pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used / 1024 ** 2

    def _muestrear(self):
        while not self._parar.is_set():
            self._pico = max(self._pico, self._usada_mb())
            time.sleep(self.intervalo_s)

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_indice)
        except Exception:  # sin driver NVIDIA (CPU): no se mide
            self._gpu = None
            return self
        self._base = self._pico = self._usada_mb()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *errores):
        if self._gpu is None:
            return
        self._parar.set()
        self._hilo.join()
        self._pico = max(self._pico, self._usada_mb())  # última lectura
        self.pico_mb = round(self._pico - self._base, 1)
        import pynvml
        pynvml.nvmlShutdown()


# =====================================================================
# Modelo y transcripción
# =====================================================================

_MODELOS = {}  # cache en memoria: cargar un modelo tarda, se reutiliza


def cargar_modelo(nombre: str, device: str, compute_type: str, cfg: dict) -> WhisperModel:
    clave = (nombre, device, compute_type)
    if clave not in _MODELOS:
        _MODELOS[clave] = WhisperModel(
            nombre, device=device, compute_type=compute_type,
            download_root=str(RAIZ / cfg["rutas"]["modelos"]),
        )
    return _MODELOS[clave]


def soporta_hotwords() -> bool:
    """¿La versión instalada de faster-whisper tiene el parámetro hotwords?"""
    return "hotwords" in inspect.signature(WhisperModel.transcribe).parameters


def parametros_glosario(modo: str, schema: str, cfg: dict) -> dict:
    """Argumentos extra de transcribe() según el modo: 'plano' -> ninguno."""
    if modo == "plano":
        return {}
    g = cfg["glosario"][schema]
    extra = {"initial_prompt": g["initial_prompt"]}
    if cfg["glosario"]["usar_hotwords"] and soporta_hotwords():
        extra["hotwords"] = " ".join(g["terminos"])
    return extra


def transcribir(audio_path: str, schema: str, cfg: dict) -> dict:
    """Transcribe un audio con la combinación de cfg['motor']. Devuelve un dict serializable."""
    m, a = cfg["motor"], cfg["asr"]
    device = resolver_device(m["device"])
    modelo = cargar_modelo(m["modelo"], device, m["compute_type"], cfg)
    extra = parametros_glosario(m["glosario"], schema, cfg)

    with MedidorVRAM(a["gpu_indice"], a["vram_intervalo_ms"]) as vram:
        inicio = time.perf_counter()
        segmentos, info = modelo.transcribe(
            audio_path, language=a["idioma"], beam_size=a["beam_size"],
            vad_filter=a["vad_filter"], word_timestamps=a["word_timestamps"], **extra,
        )
        # transcribe() es perezoso: el trabajo real ocurre al recorrer los segmentos,
        # por eso se recorren DENTRO de la medición de tiempo y VRAM
        segments = [{
            "start": s.start, "end": s.end, "text": s.text,
            "words": [{"word": w.word, "start": w.start, "end": w.end,
                       "probability": w.probability} for w in (s.words or [])],
        } for s in segmentos]
        elapsed_s = time.perf_counter() - inicio

    return {
        "audio": Path(audio_path).name,
        "transcript": "".join(s["text"] for s in segments).strip(),
        "segments": segments,
        # Prueba de device: se lee del objeto CTranslate2, NO del config
        "device": modelo.model.device,
        "compute_type": modelo.model.compute_type,
        "elapsed_s": round(elapsed_s, 3),
        "duracion_audio_s": round(info.duration, 3),
        "seg_por_min_audio": round(elapsed_s / (info.duration / 60), 3) if info.duration else None,
        "peak_vram_mb": vram.pico_mb,
        "modelo": m["modelo"],
        "glosario": m["glosario"],
        "parametros_glosario": extra,
        "faster_whisper": faster_whisper.__version__,
        "ctranslate2": ctranslate2.__version__,
    }


# =====================================================================
# Cache de transcripts
# =====================================================================

def ruta_cache(clip: str, cfg: dict) -> Path:
    """cache/transcripts/{clip}__{modelo}__{device}__{compute}__{glosario}__{nivel}.json"""
    m = cfg["motor"]
    nombre = cfg["cache_nombre"].format(
        clip=clip, modelo=m["modelo"], device=resolver_device(m["device"]),
        compute=m["compute_type"], glosario=m["glosario"], nivel=m["nivel"],
    )
    return RAIZ / cfg["rutas"]["cache_transcripts"] / nombre


def guardar_cache(resultado: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")


def leer_cache(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))
