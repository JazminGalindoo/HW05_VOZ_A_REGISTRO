"""Tests de evaluar.py con un cache FALSO armado desde las referencias.

Combinación A = transcripción perfecta (la referencia tal cual).
Combinación B = a la referencia le falta 1 de cada 10 palabras (WER > 0).
L1 de A: en L01 el ASR "oye" 81 en vez de 89, con baja confianza.
"""

import json

import pytest

from src.asr import ruta_cache
from src.engine import con_motor
from src.evaluar import evaluar, guardar, verificar_umbrales

A = {"modelo": "small", "device": "cuda", "compute_type": "float16", "glosario": "plano"}
B = {"modelo": "large-v3", "device": "cuda", "compute_type": "float16", "glosario": "plano"}


def escribir(cfg, clip, combo, nivel, texto, prob_baja=()):
    palabras = [{"word": w, "probability": 0.4 if w.strip(".,") in prob_baja else 0.95}
                for w in texto.split()]
    datos = {"transcript": texto, "segments": [{"text": texto, "words": palabras}],
             "device": combo["device"], "compute_type": combo["compute_type"], "elapsed_s": 2.0,
             "duracion_audio_s": 120.0, "seg_por_min_audio": 1.0, "peak_vram_mb": 900.0}
    ruta_cache(clip, con_motor(cfg, **combo, nivel=nivel)).write_text(json.dumps(datos), encoding="utf-8")


@pytest.fixture(scope="module")
def resultados(cfg, referencias, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("evaluar")
    c = json.loads(json.dumps(cfg))  # copia
    c["rutas"]["cache_transcripts"] = str(tmp / "transcripts")
    c["rutas"]["cache_results"] = str(tmp / "results")
    c["regresion"]["combinacion"] = A
    (tmp / "transcripts").mkdir()
    for clip, texto in referencias.items():
        escribir(c, clip, A, "L0", texto)
        palabras = texto.split()
        escribir(c, clip, B, "L0", " ".join(p for i, p in enumerate(palabras) if i % 10 != 5))
    l01 = referencias["L01"].replace("ochenta y nueve soles", "ochenta y uno soles")
    escribir(c, "L01", A, "L1", l01, prob_baja={"uno"})
    escribir(c, "Z99", A, "L0", "clip sintético sin ground truth")
    return c, evaluar(c, ["reglas"])


def test_wer_por_combinacion(resultados):
    _, r = resultados
    res = r["asr_resumen"].set_index(["modelo", "nivel", "tarea"])
    assert res.loc[("small", "L0", "entrevista"), "wer"] == 0
    assert res.loc[("large-v3", "L0", "pedido"), "wer"] > 0
    assert res.loc[("small", "L0", "pedido"), "n_clips"] == 4


def test_comparacion_tamano(resultados):
    _, r = resultados
    fila = r["comparaciones"].set_index("comparacion").loc["tamaño (small -> large-v3)"]
    assert fila["n_clips"] == 8 and fila["wer_a"] == 0
    assert fila["dif_wer_b_menos_a"] == pytest.approx(fila["wer_b"]) and fila["dif_wer_b_menos_a"] > 0
    assert fila["clips_texto_identico"] == 0
    # Las comparaciones sin datos en cache no aparecen
    assert len(r["comparaciones"]) == 1


def test_extraccion_perfecta_igual_a_referencia(resultados):
    _, r = resultados
    ext = r["extraccion_por_campo"]
    ref = ext[ext["grupo"] == "referencia"].set_index(["tarea", "campo"])["accuracy"]
    a = ext[(ext["grupo"] == "small|cuda|float16|plano") & (ext["nivel"] == "L0")]
    assert (a.set_index(["tarea", "campo"])["accuracy"] == ref).all()


def test_validacion_y_diagnostico(resultados):
    _, r = resultados
    v = r["pedidos_validacion"]
    l0 = v[(v["grupo"] == "small|cuda|float16|plano") & (v["nivel"] == "L0")].set_index("clip")
    assert l0["diagnostico"].to_dict() == {"L01": "ok", "L02": "error_hablante",
                                           "L03": "error_hablante", "L04": "error_hablante"}
    assert l0["acierta"].all()
    l1 = v[(v["nivel"] == "L1") & (v["clip"] == "L01")].iloc[0]
    assert l1["diagnostico"] == "error_asr" and l1["acierta"]  # en L0 cuadraba
    ref = v[v["grupo"] == "referencia"]
    assert ref.loc[ref["diagnostico"] == "indeterminado", "acierta"].isna().all()


def test_sinteticos_quedan_fuera(resultados):
    _, r = resultados
    assert r["resumen"]["clips_ignorados_sin_ground_truth"] == ["Z99"]
    assert "Z99" not in set(r["asr_por_clip"]["clip"])


def test_umbrales(resultados):
    c, r = resultados
    assert verificar_umbrales(r, c) == []
    estricto = json.loads(json.dumps(c))
    estricto["regresion"]["wer_max"] = -1
    assert any("WER" in f for f in verificar_umbrales(r, estricto))


def test_guardar(resultados, tmp_path):
    c, r = resultados
    c = json.loads(json.dumps(c))
    c["rutas"]["cache_results"] = str(tmp_path)
    carpeta = guardar(r, c)
    for nombre in ["asr_por_clip.csv", "asr_resumen.csv", "comparaciones.csv",
                   "extraccion_por_campo.csv", "pedidos_validacion.csv", "resumen.json"]:
        assert (carpeta / nombre).exists(), nombre
