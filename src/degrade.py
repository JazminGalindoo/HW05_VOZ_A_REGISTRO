"""
degrade.py — Escalera de degradación L0–L3 (Tarea 2), reproducible.

Responsabilidad:
    - L0: audio original.
    - L1: tipo WhatsApp (mono, Opus a bajo bitrate) con ffmpeg.
    - L2: tipo teléfono (paso banda 300–3400 Hz, remuestreo a 8 kHz).
    - L3: L1 + ruido ambiente mezclado a un SNR objetivo; el SNR real se
      CALCULA con el RMS de señal y de ruido y se reporta.
    - Todos los parámetros vienen de config.yaml.

ffmpeg se llama con subprocess (sin librerías intermedias): cada comando
es el mismo que se escribiría en la terminal.

Uso:  python -m src.degrade      (degrada todas las llamadas de data/audio/llamadas)
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

RAIZ = Path(__file__).resolve().parent.parent


def _ffmpeg(cfg: dict) -> str:
    ruta = shutil.which(cfg["degradacion"]["ffmpeg"])
    if ruta is None:
        raise FileNotFoundError("No se encontró ffmpeg. Instálalo y reinicia la terminal.")
    return ruta


def _correr(cfg: dict, argumentos: list[str]) -> bytes:
    """Ejecuta ffmpeg; si falla, lanza el error con el mensaje de ffmpeg."""
    comando = [_ffmpeg(cfg), "-hide_banner", "-loglevel", "error", "-y", *argumentos]
    return subprocess.run(comando, check=True, capture_output=True).stdout


def leer_mono(ruta, sample_rate: int, cfg: dict) -> np.ndarray:
    """Decodifica cualquier audio a mono float32 a la frecuencia pedida."""
    crudo = _correr(cfg, ["-i", str(ruta), "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"])
    return np.frombuffer(crudo, dtype=np.float32).astype(np.float64)


def rms(x: np.ndarray) -> float:
    """Raíz del promedio de los cuadrados: el 'volumen' de una señal."""
    return float(np.sqrt(np.mean(x ** 2)))


def snr_db(senal: np.ndarray, ruido: np.ndarray) -> float:
    """SNR en decibeles = 20·log10(RMS señal / RMS ruido)."""
    return 20 * np.log10(rms(senal) / rms(ruido))


# =====================================================================
# Niveles
# =====================================================================

def l1_whatsapp(entrada, salida, cfg: dict) -> Path:
    c = cfg["degradacion"]["L1"]
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    _correr(cfg, ["-i", str(entrada), "-ac", str(c["canales"]),
                  "-c:a", c["codec"], "-b:a", f"{c['bitrate_kbps']}k", str(salida)])
    return Path(salida)


def l2_telefono(entrada, salida, cfg: dict) -> Path:
    c = cfg["degradacion"]["L2"]
    bajo, alto = c["paso_banda_hz"]
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    _correr(cfg, ["-i", str(entrada), "-ac", "1",
                  "-af", f"highpass=f={bajo},lowpass=f={alto}",
                  "-ar", str(c["sample_rate_hz"]), str(salida)])
    return Path(salida)


def mezclar_ruido(senal: np.ndarray, ruido: np.ndarray, snr_objetivo: float):
    """Mezcla ruido a un SNR dado. Devuelve (mezcla, snr_medido).

    1. El ruido se repite o recorta al largo de la señal.
    2. Se escala para que 20·log10(RMS señal / RMS ruido) = snr_objetivo.
    3. Si la mezcla pasa de ±1 (saturaría), se baja el volumen de TODO por igual:
       eso no cambia el SNR.
    """
    repeticiones = int(np.ceil(len(senal) / len(ruido)))
    ruido = np.tile(ruido, repeticiones)[:len(senal)]
    ganancia = rms(senal) / (rms(ruido) * 10 ** (snr_objetivo / 20))
    ruido = ruido * ganancia
    mezcla = senal + ruido
    pico = np.max(np.abs(mezcla))
    if pico > 1:
        mezcla, senal, ruido = mezcla / pico, senal / pico, ruido / pico
    return mezcla, snr_db(senal, ruido)


def l3_ruido(entrada_l1, ruta_ruido, salida, cfg: dict, snr_objetivo: float | None = None) -> dict:
    """L1 + ruido ambiente. Devuelve el SNR objetivo y el MEDIDO."""
    c = cfg["degradacion"]
    snr_objetivo = c["L3"]["snr_db"] if snr_objetivo is None else snr_objetivo
    sr = c["sample_rate_mezcla_hz"]
    mezcla, snr_medido = mezclar_ruido(leer_mono(entrada_l1, sr, cfg),
                                       leer_mono(ruta_ruido, sr, cfg), snr_objetivo)
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    sf.write(salida, mezcla, sr, subtype="FLOAT")  # FLOAT: sin pérdida al guardar
    return {"snr_objetivo_db": snr_objetivo, "snr_medido_db": round(snr_medido, 2)}


# =====================================================================
# Todas las llamadas
# =====================================================================

def degradar_todo(cfg: dict) -> dict:
    """Crea L1, L2 y L3 de cada llamada y guarda el SNR medido en cache/results."""
    r, c = cfg["rutas"], cfg["degradacion"]
    destino = RAIZ / r["audio_degradado"]
    reporte = {}
    for original in sorted((RAIZ / r["audio_llamadas"]).iterdir()):
        if original.suffix.lower() not in c["extensiones_audio"]:
            continue
        clip = original.stem
        l1 = l1_whatsapp(original, destino / "L1" / f"{clip}{c['L1']['extension']}", cfg)
        l2_telefono(original, destino / "L2" / f"{clip}{c['L2']['extension']}", cfg)
        reporte[clip] = l3_ruido(l1, RAIZ / r["audio_ruido"],
                                 destino / "L3" / f"{clip}{c['L3']['extension']}", cfg)
        print(clip, reporte[clip])

    ruta_reporte = RAIZ / c["reporte"]
    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    ruta_reporte.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    return reporte


if __name__ == "__main__":
    degradar_todo(yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8")))
