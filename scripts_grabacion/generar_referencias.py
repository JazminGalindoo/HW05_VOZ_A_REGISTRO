"""
generar_referencias.py — Crea los transcripts de referencia a partir de guiones.md.

Para cada clip (E01..E04, L01..L04) escribe un .txt en
data/ground_truth/transcripts_referencia/ con SOLO el texto hablado,
sin el nombre de quien habla (una intervención por línea).

Es ground truth escrito por humano (regla 6): sale del guion, no del ASR.
Si al grabar alguien dijo algo distinto, se corrige guiones.md y se
vuelve a correr este script.

Uso:  python scripts_grabacion/generar_referencias.py
"""

import re
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent

# "## E01 — Entrevistadora: ..." -> captura "E01"
PATRON_CLIP = re.compile(r"^##\s+([EL]\d{2})\b")
# "**JAZMÍN:** Buenos días..." -> captura "Buenos días..."
PATRON_TURNO = re.compile(r"^\*\*[^*]+:\*\*\s*(.+)$")


def leer_guiones(ruta: Path) -> dict[str, list[str]]:
    """Devuelve {clip: [intervención, intervención, ...]}."""
    clips: dict[str, list[str]] = {}
    clip_actual = None
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        encabezado = PATRON_CLIP.match(linea)
        if encabezado:
            clip_actual = encabezado.group(1)
            clips[clip_actual] = []
        elif linea.startswith("# "):
            # Un título de sección (# LLAMADAS...) cierra el clip anterior
            clip_actual = None
        elif clip_actual:
            turno = PATRON_TURNO.match(linea)
            if turno:  # se ignoran notas como *(trampa: ...)*
                clips[clip_actual].append(turno.group(1))
    return clips


def main():
    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    guiones = RAIZ / cfg["rutas"]["guiones"]
    salida = RAIZ / cfg["rutas"]["transcripts_referencia"]
    salida.mkdir(parents=True, exist_ok=True)

    for clip, turnos in leer_guiones(guiones).items():
        (salida / f"{clip}.txt").write_text("\n".join(turnos) + "\n", encoding="utf-8")
        print(f"{clip}: {len(turnos)} intervenciones")


if __name__ == "__main__":
    main()
