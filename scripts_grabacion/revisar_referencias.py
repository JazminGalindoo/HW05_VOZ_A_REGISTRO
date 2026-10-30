"""
revisar_referencias.py — Lista los fragmentos donde el guion y el ASR no coinciden.

Para qué: los transcripts de referencia salen del guion, pero al grabar se
improvisó. Antes de medir WER hay que corregir la referencia con lo que se dijo
DE VERDAD, escuchando el audio (una persona, no el ASR: regla 6).
Este script solo dice DÓNDE escuchar:

    - Alinea el guion normalizado con la salida de un ASR (por defecto large-v3 int8).
    - Junta las diferencias cercanas en fragmentos y les pone el minuto:segundo
      aproximado (del segmento del ASR donde caen).
    - Pista: si otros modelos escribieron lo mismo que el ASR, probablemente se dijo
      así (improvisación); si solo uno, probablemente es error del ASR. Es una pista,
      la decisión es de quien escucha.

Uso:  python scripts_grabacion/revisar_referencias.py   -> docs/revisar_referencias.md
"""

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.engine import cargar_config  # noqa: E402
from src.normalize import normalizar  # noqa: E402

PRINCIPAL = "large-v3__cuda__int8__plano__L0"
OTROS = ["large-v3__cuda__float16__plano__L0", "small__cuda__float16__plano__L0"]
CONTEXTO = 4       # palabras de contexto a cada lado
JUNTAR_SI_HAY = 4  # diferencias separadas por menos palabras iguales se juntan


def tokens_con_tiempo(datos: dict, cfg: dict) -> list[tuple[str, float]]:
    """[(palabra normalizada, segundo aproximado)].

    Se normaliza por SEGMENTO (así "sesenta y cuatro" queda junto) y el tiempo de
    cada palabra se interpola dentro del segmento.
    """
    salida = []
    for s in datos["segments"]:
        palabras = normalizar(s["text"], cfg).split()
        for i, p in enumerate(palabras):
            salida.append((p, s["start"] + (s["end"] - s["start"]) * i / max(len(palabras), 1)))
    return salida


def mmss(segundos: float) -> str:
    return f"{int(segundos // 60)}:{int(segundos % 60):02d}"


def fragmentos(ref: list[str], asr: list[tuple[str, float]]) -> list[dict]:
    """Bloques donde difieren, ya juntados, con contexto."""
    palabras_asr = [p for p, _ in asr]
    ops = [op for op in SequenceMatcher(None, ref, palabras_asr, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    # Juntar diferencias cercanas: (i1, i2, j1, j2)
    grupos = []
    for _, i1, i2, j1, j2 in ops:
        if grupos and i1 - grupos[-1][1] < JUNTAR_SI_HAY and j1 - grupos[-1][3] < JUNTAR_SI_HAY:
            grupos[-1] = (grupos[-1][0], i2, grupos[-1][2], j2)
        else:
            grupos.append((i1, i2, j1, j2))

    salida = []
    for i1, i2, j1, j2 in grupos:
        t = asr[min(j1, len(asr) - 1)][1] if asr else 0.0

        def con_contexto(lista, a, b):
            antes, despues = " ".join(lista[max(0, a - CONTEXTO):a]), " ".join(lista[b:b + CONTEXTO])
            medio = " ".join(lista[a:b]) or "∅"
            return f"{antes} **{medio}** {despues}".strip()

        # Lo que escribió el ASR con 2 palabras de contexto, para buscarlo en otros modelos
        clave = " ".join(palabras_asr[max(0, j1 - 2):j2 + 2])
        salida.append({"tiempo": t, "guion": con_contexto(ref, i1, i2),
                       "asr": con_contexto(palabras_asr, j1, j2), "clave": clave})
    return salida


def main():
    cfg = cargar_config()
    cache = RAIZ / cfg["rutas"]["cache_transcripts"]
    refs = RAIZ / cfg["rutas"]["transcripts_referencia"]
    lineas = [
        "# Revisar referencias contra el audio",
        "",
        "Generado por `python scripts_grabacion/revisar_referencias.py`. **No editar a mano.**",
        "",
        "Los transcripts de referencia salen del guion, pero al grabar se improvisó. "
        "Para cada fragmento: escuchar el audio en el minuto indicado y anotar qué se dijo de verdad. "
        "La corrección se hace en `scripts_grabacion/guiones.md` (una persona, nunca copiando al ASR) "
        "y luego se corre `python scripts_grabacion/generar_referencias.py`.",
        "",
        f"- **Guion**: lo que dice la referencia hoy. **ASR**: lo que escribió `{PRINCIPAL}`. "
        "Todo normalizado (minúsculas, sin tildes, números en dígitos); en **negrita** lo que difiere; ∅ = nada.",
        "- **¿Otros modelos igual?** sí = large-v3 float16 y small también lo escribieron así → "
        "probablemente se **dijo así** (improvisación). no = probablemente **error del ASR**. Es una pista: decide el oído.",
        "- Tiempos aproximados (±3 s).",
        "",
    ]
    total = 0
    for ruta_ref in sorted(refs.glob("*.txt")):
        clip = ruta_ref.stem
        principal = cache / f"{clip}__{PRINCIPAL}.json"
        if not principal.exists():
            continue
        ref = normalizar(ruta_ref.read_text(encoding="utf-8"), cfg).split()
        asr = tokens_con_tiempo(json.loads(principal.read_text(encoding="utf-8")), cfg)
        otros = [" " + normalizar(json.loads((cache / f"{clip}__{o}.json").read_text(encoding="utf-8"))
                                  ["transcript"], cfg) + " "
                 for o in OTROS if (cache / f"{clip}__{o}.json").exists()]
        frags = fragmentos(ref, asr)
        total += len(frags)
        lineas += [f"## {clip} — {len(frags)} fragmentos", "",
                   "| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |",
                   "|---|---|---|---|---|---|"]
        for n, f in enumerate(frags, 1):
            igual = "sí" if otros and all(f" {f['clave']} " in o for o in otros) else "no"
            lineas.append(f"| {n} | {mmss(f['tiempo'])} | {f['guion']} | {f['asr']} | {igual} | |")
        lineas.append("")

    salida = RAIZ / "docs" / "revisar_referencias.md"
    salida.write_text("\n".join(lineas), encoding="utf-8")
    print(f"{total} fragmentos -> {salida}")


if __name__ == "__main__":
    main()
