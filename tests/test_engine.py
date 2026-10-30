"""Tests de engine.py y asr.py.

Los rápidos no cargan ningún modelo: usan un cache falso en una carpeta temporal.
El test marcado `lento` descarga Whisper `tiny` y transcribe de verdad en CPU;
no corre en CI. Para correrlo:  pytest -m lento
"""

import json

import numpy as np
import pytest
import soundfile as sf

from src.asr import parametros_glosario, resolver_device, ruta_cache
from src.engine import EngineResult, con_motor, process_audio, process_transcript


@pytest.fixture
def cfg_cache(cfg, tmp_path):
    """cfg con el cache en una carpeta temporal y device fijo en cpu."""
    c = con_motor(cfg, device="cpu", compute_type="int8", solo_cache=True)
    c["rutas"]["cache_transcripts"] = str(tmp_path)  # ruta absoluta: reemplaza a la del repo
    return c


def escribir_cache_falso(cfg, clip, transcript):
    datos = {"transcript": transcript,
             "segments": [{"text": transcript, "words": [{"word": w, "probability": 0.95}
                                                         for w in transcript.split()]}],
             "device": "cpu", "compute_type": "int8_float32", "elapsed_s": 1.5, "peak_vram_mb": None}
    ruta = ruta_cache(clip, cfg)
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


# ---------------- process_transcript ----------------

def test_process_transcript_pedido(cfg, referencias):
    r = process_transcript(referencias["L02"], "pedido", cfg)
    assert r.error is None
    assert r.fields["total_dicho"] == 80 and r.verified["total_dicho"] is True


def test_process_transcript_schema_invalido(cfg):
    r = process_transcript("hola", "factura", cfg)
    assert r.fields == {} and "schema" in r.error


# ---------------- process_audio con cache ----------------

def test_process_audio_lee_cache_sin_transcribir(cfg_cache, referencias):
    escribir_cache_falso(cfg_cache, "L04", referencias["L04"])
    r = process_audio("data/audio/llamadas/L04.wav", "pedido", cfg_cache)  # el audio ni existe
    assert isinstance(r, EngineResult) and r.error is None
    assert r.desde_cache is True
    assert r.device == "cpu" and r.compute_type == "int8_float32"  # lo que dijo el ASR, no el config
    assert r.fields["total_dicho"] == 98
    assert r.segments[0]["words"][0]["probability"] == 0.95


def test_solo_cache_nunca_transcribe(cfg_cache):
    r = process_audio("data/audio/llamadas/L01.wav", "pedido", cfg_cache)
    assert r.error and "solo_cache" in r.error and r.transcript == ""


def test_nombre_del_cache(cfg):
    c = con_motor(cfg, modelo="large-v3", device="cuda", compute_type="float16",
                  glosario="glosario", nivel="L2")
    assert ruta_cache("L03", c).name == "L03__large-v3__cuda__float16__glosario__L2.json"


def test_con_motor_no_modifica_el_original(cfg):
    con_motor(cfg, modelo="large-v3")
    assert cfg["motor"]["modelo"] == "small"


# ---------------- asr: piezas sin modelo ----------------

def test_resolver_device():
    assert resolver_device("cpu") == "cpu" and resolver_device("cuda") == "cuda"
    assert resolver_device("auto") in ("cpu", "cuda")


def test_parametros_glosario(cfg):
    assert parametros_glosario("plano", "pedido", cfg) == {}
    g = parametros_glosario("glosario", "pedido", cfg)
    assert g["initial_prompt"] == cfg["glosario"]["pedido"]["initial_prompt"]
    assert "Yape" in g["hotwords"]  # faster-whisper 1.2.1 tiene hotwords


# ---------------- ASR real (lento) ----------------

@pytest.mark.lento
def test_asr_real_tiny_en_cpu(cfg, tmp_path):
    sr = 16000
    audio = tmp_path / "tono.wav"
    sf.write(audio, 0.1 * np.sin(2 * np.pi * 440 * np.arange(2 * sr) / sr), sr)
    c = con_motor(cfg, modelo="tiny", device="cpu", compute_type="int8", usar_cache=False)
    c["rutas"]["cache_transcripts"] = str(tmp_path / "cache")
    r = process_audio(str(audio), "pedido", c)
    assert r.error is None
    assert r.device == "cpu"                    # leído del objeto CTranslate2
    assert r.compute_type.startswith("int8")
    assert r.elapsed_s > 0 and r.peak_vram_mb is None
    assert (tmp_path / "cache" / "tono__tiny__cpu__int8__plano__L0.json").exists()
