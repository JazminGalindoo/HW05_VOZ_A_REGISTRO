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
