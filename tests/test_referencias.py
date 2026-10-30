"""Los transcripts de referencia existen, uno por clip, y sin nombres de quien habla."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "data" / "ground_truth" / "transcripts_referencia"
CLIPS = ["E01", "E02", "E03", "E04", "L01", "L02", "L03", "L04"]


def test_un_txt_por_clip():
    for clip in CLIPS:
        assert (CARPETA / f"{clip}.txt").exists(), f"falta {clip}.txt"


def test_sin_nombres_de_hablante():
    # En guiones.md cada turno es "**HABLANTE:** texto"; en la referencia solo queda el texto
    for clip in CLIPS:
        texto = (CARPETA / f"{clip}.txt").read_text(encoding="utf-8")
        assert "**" not in texto
        assert "PERSONA 1" not in texto and "PERSONA 2" not in texto


def test_revisar_referencias_encuentra_diferencias():
    import sys
    sys.path.insert(0, str(RAIZ / "scripts_grabacion"))
    from revisar_referencias import fragmentos, mmss

    ref = "el total es 89 soles gracias".split()
    asr = [(p, t) for t, p in enumerate("el total seria de 89 soles gracias".split(), start=60)]
    frags = fragmentos(ref, asr)
    assert len(frags) == 1
    assert "**es**" in frags[0]["guion"] and "**seria de**" in frags[0]["asr"]
    assert mmss(frags[0]["tiempo"]) == "1:02"


def test_solo_campos_filtra_por_valor_que_cambia(cfg):
    import sys
    sys.path.insert(0, str(RAIZ / "scripts_grabacion"))
    from revisar_referencias import toca_un_campo, vocabulario_de_campos

    voc = vocabulario_de_campos("L04", cfg)
    assert "tarjeta" in voc and "arequipa" in voc
    assert toca_un_campo(["28"], ["25"], voc)                                 # número distinto
    assert toca_un_campo(["en", "lince"], ["inficio"], voc)                  # se perdió el distrito
    assert not toca_un_campo(["con", "tarjeta"], ["el", "pedido", "con", "tarjeta"], voc)  # mismo valor
    assert not toca_un_campo(["1", "ensalada"], ["ensalada", "ok"], voc)     # "1" suelto = artículo
