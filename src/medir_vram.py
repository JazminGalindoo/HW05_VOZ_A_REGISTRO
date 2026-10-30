"""
medir_vram.py — VRAM TOTAL de un modelo en la GPU (se corre en Colab, una vez por modelo).

Responsabilidad:
    - asr.transcribir mide la VRAM EXTRA durante la transcripción: empieza a medir
      con el modelo ya cargado, así que no cuenta los pesos.
    - Aquí se mide desde ANTES de cargar el modelo, en un proceso nuevo:
        base        = memoria usada en la GPU antes de empezar (otros procesos)
        con_modelo  = después de cargar el modelo        -> modelo_mb = con_modelo - base
        pico        = máximo durante la transcripción    -> total_mb  = pico - base
                                                            extra_mb  = pico - con_modelo
      total_mb es lo comparable con "lo que ocupa en la tarjeta" (contexto CUDA + pesos
      + transcripción); extra_mb debería parecerse al peak_vram_mb del cache.
    - Guarda el resultado en cache/results/vram_modelos.json (clave "modelo|compute").

Uso (en Colab, celda 11 del notebook):
    python -m src.medir_vram --modelo large-v3 --compute float16 --audio data/audio/entrevistas/E01.m4a
"""

import argparse
import json
import threading
import time
from pathlib import Path

from src.cuda_colab import precargar_cuda12

RAIZ = Path(__file__).resolve().parent.parent


def medir(modelo: str, compute: str, audio: str, cfg: dict) -> dict:
    precargar_cuda12()  # proceso nuevo: hay que volver a precargar CUDA 12
    import pynvml
    from faster_whisper import WhisperModel  # después de precargar

    a = cfg["asr"]
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(a["gpu_indice"])

    def usada_mb() -> float:
        return pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 1024 ** 2

    base = usada_mb()
    whisper = WhisperModel(modelo, device="cuda", compute_type=compute,
                           download_root=str(RAIZ / cfg["rutas"]["modelos"]))
    con_modelo = usada_mb()

    # Hilo que muestrea la memoria mientras se transcribe (igual que asr.MedidorVRAM)
    pico, parar = [con_modelo], threading.Event()

    def muestrear():
        while not parar.is_set():
            pico[0] = max(pico[0], usada_mb())
            time.sleep(a["vram_intervalo_ms"] / 1000)

    hilo = threading.Thread(target=muestrear, daemon=True)
    hilo.start()
    segmentos, _ = whisper.transcribe(audio, language=a["idioma"], beam_size=a["beam_size"],
                                      vad_filter=a["vad_filter"], word_timestamps=a["word_timestamps"])
    list(segmentos)  # transcribe() es perezoso: aquí ocurre el trabajo
    parar.set()
    hilo.join()
    pico[0] = max(pico[0], usada_mb())

    resultado = {
        "gpu": pynvml.nvmlDeviceGetName(gpu),
        "device_real": whisper.model.device, "compute_real": whisper.model.compute_type,
        "audio": Path(audio).name,
        "base_mb": round(base, 1),
        "modelo_mb": round(con_modelo - base, 1),
        "total_mb": round(pico[0] - base, 1),
        "extra_mb": round(pico[0] - con_modelo, 1),
    }
    pynvml.nvmlShutdown()
    return resultado


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description="VRAM total de un modelo (proceso nuevo)")
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--compute", required=True)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    resultado = medir(args.modelo, args.compute, args.audio, cfg)
    print(args.modelo, args.compute, resultado)

    ruta = RAIZ / cfg["rutas"]["cache_results"] / "vram_modelos.json"
    todos = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    todos[f"{args.modelo}|{args.compute}"] = resultado
    ruta.write_text(json.dumps(todos, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
