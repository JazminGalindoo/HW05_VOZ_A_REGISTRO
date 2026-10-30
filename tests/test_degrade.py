"""Tests de degrade.py con audio sintético (tonos y ruido generados con numpy)."""

import shutil

import numpy as np
import pytest
import soundfile as sf

from src.degrade import l1_whatsapp, l2_telefono, l3_ruido, leer_mono, mezclar_ruido, rms, snr_db

SR = 16000
necesita_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no instalado")


def tono(frecuencia: float, segundos: float = 2.0, amplitud: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * segundos)) / SR
    return amplitud * np.sin(2 * np.pi * frecuencia * t)


def guardar(ruta, x):
    sf.write(ruta, x, SR, subtype="FLOAT")
    return ruta


# ---------------- SNR (sin ffmpeg) ----------------

def test_snr_a_mano():
    senal = np.ones(100)        # RMS 1
    ruido = np.full(100, 0.1)   # RMS 0.1 -> 20·log10(10) = 20 dB
    assert snr_db(senal, ruido) == pytest.approx(20.0)


@pytest.mark.parametrize("objetivo", [10, 5, 0])
def test_mezcla_alcanza_el_snr_objetivo(objetivo):
    rng = np.random.default_rng(0)
    senal = tono(440)
    ruido = rng.normal(0, 1, SR // 2)  # más corto que la señal: se repite
    mezcla, medido = mezclar_ruido(senal, ruido, objetivo)
    assert len(mezcla) == len(senal)
    assert medido == pytest.approx(objetivo, abs=0.01)


def test_mezcla_no_satura_y_conserva_snr():
    rng = np.random.default_rng(1)
    senal = tono(440, amplitud=0.95)  # casi al máximo: la suma pasaría de 1
    mezcla, medido = mezclar_ruido(senal, rng.normal(0, 1, SR), 0)
    assert np.max(np.abs(mezcla)) <= 1.0
    assert medido == pytest.approx(0, abs=0.01)


# ---------------- niveles (con ffmpeg) ----------------

@necesita_ffmpeg
def test_l1_whatsapp_mono_opus(cfg, tmp_path):
    entrada = guardar(tmp_path / "x.wav", np.stack([tono(440), tono(440)], axis=1))  # estéreo
    salida = l1_whatsapp(entrada, tmp_path / "L1" / "x.ogg", cfg)
    assert sf.info(salida).channels == 1
    assert len(leer_mono(salida, SR, cfg)) == pytest.approx(2 * SR, rel=0.05)


@necesita_ffmpeg
def test_l2_telefono_corta_graves_y_baja_a_8k(cfg, tmp_path):
    def ganancia(frecuencia):
        entrada = guardar(tmp_path / f"t{frecuencia}.wav", tono(frecuencia))
        salida = l2_telefono(entrada, tmp_path / "L2" / f"t{frecuencia}.wav", cfg)
        assert sf.info(salida).samplerate == 8000
        return rms(leer_mono(salida, 8000, cfg)[800:]) / rms(tono(frecuencia))  # sin el arranque

    assert ganancia(1000) > 0.7   # voz: pasa
    assert ganancia(100) < 0.3    # por debajo de 300 Hz: se atenúa


@necesita_ffmpeg
def test_l3_snr_medido_y_comprobado_en_el_archivo(cfg, tmp_path):
    rng = np.random.default_rng(2)
    l1 = l1_whatsapp(guardar(tmp_path / "x.wav", tono(440)), tmp_path / "x.ogg", cfg)
    ruido = guardar(tmp_path / "ruido.wav", 0.1 * rng.normal(0, 1, SR))
    info = l3_ruido(l1, ruido, tmp_path / "L3" / "x.wav", cfg, snr_objetivo=5)
    assert info["snr_medido_db"] == pytest.approx(5, abs=0.05)

    # Comprobación independiente: archivo L3 - señal L1 = ruido agregado
    senal = leer_mono(l1, SR, cfg)
    mezcla, sr = sf.read(tmp_path / "L3" / "x.wav")
    assert sr == SR
    assert snr_db(senal, mezcla - senal) == pytest.approx(5, abs=0.05)
