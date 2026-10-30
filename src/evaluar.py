"""
evaluar.py — Métricas finales desde cache/ (NUNCA transcribe).

Responsabilidad:
    - Leer cada transcript de cache/transcripts/ y compararlo con su referencia
      humana: WER, CER, segundos por minuto de audio, VRAM pico, device real.
    - Comparaciones pareadas de config (tamaño, device, precisión, glosario):
      diferencia de WER con IC bootstrap y si el texto cambió.
    - Extracción por campo (accuracy con IC, macro-F1, error absoluto, % verificado)
      pasando cada transcript por engine.process_audio con solo_cache=true.
    - Pedidos: chequeo aritmético y diagnóstico ASR/hablante por nivel, contra las trampas.
    - Escribir todo en cache/results/ (lo lee la app) y, con --verificar,
      fallar si la combinación vigilada queda bajo los umbrales (lo usa el CI).

Grupos que NUNCA se mezclan:
    - "referencia": los transcripts humanos como si fueran el ASR (cota superior
      de la extracción; mide las reglas/LLM, no el ASR).
    - cada combinación de ASR por separado.
    - clips que no están en el ground truth (p. ej. sintéticos): se ignoran y se avisa.

Uso:
    python -m src.evaluar              # reglas
    python -m src.evaluar --llm        # además extracción con LLM (necesita API key)
    python -m src.evaluar --verificar  # sale con error si no se cumplen los umbrales
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.asr import leer_cache
from src.engine import cargar_config, con_motor, process_audio, process_transcript
from src.metrics import (bootstrap_diferencia, bootstrap_ic, cer, error_absoluto, errores_wer,
                         macro_f1, tabla_aciertos, wer, wer_conjunto)
from src.schemas import leer_ground_truth
from src.validate import chequear, diagnosticar, palabras_de

RAIZ = Path(__file__).resolve().parent.parent
COMBO = ["modelo", "device", "compute_type", "glosario"]  # qué define una configuración de ASR
CAMPOS_NUM = ["anios_experiencia", "pretension_salarial", "descuento", "total_dicho"]


# =====================================================================
# Lectura
# =====================================================================

def leer_corridas(cfg: dict) -> list[dict]:
    """Un dict por archivo de cache: claves del nombre + contenido del JSON."""
    # La plantilla del nombre ({clip}__{modelo}__...) se convierte en una regex
    claves = re.findall(r"\{(\w+)\}", cfg["cache_nombre"])
    patron = re.escape(cfg["cache_nombre"])
    for clave in claves:
        patron = patron.replace(re.escape(f"{{{clave}}}"), f"(?P<{clave}>.+?)")
    corridas = []
    for ruta in sorted((RAIZ / cfg["rutas"]["cache_transcripts"]).glob("*.json")):
        m = re.fullmatch(patron, ruta.name)
        if not m:
            continue
        corrida = m.groupdict()
        corrida["compute_type"] = corrida.pop("compute")  # mismo nombre que en "motor"
        corrida["datos"] = leer_cache(ruta)
        corridas.append(corrida)
    return corridas


def leer_referencias(cfg: dict) -> dict[str, str]:
    carpeta = RAIZ / cfg["rutas"]["transcripts_referencia"]
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(carpeta.glob("*.txt"))}


# =====================================================================
# ASR
# =====================================================================

def tabla_asr(corridas: list[dict], referencias: dict, esquema_de: dict, cfg: dict) -> pd.DataFrame:
    """Una fila por transcript cacheado."""
    filas = []
    for c in corridas:
        d, clip = c["datos"], c["clip"]
        errores, palabras = errores_wer(referencias[clip], d["transcript"], cfg)
        filas.append({
            "clip": clip, "tarea": esquema_de[clip], **{k: c[k] for k in COMBO}, "nivel": c["nivel"],
            "device_real": d["device"], "compute_real": d["compute_type"],
            "wer": wer(referencias[clip], d["transcript"], cfg),
            "cer": cer(referencias[clip], d["transcript"], cfg),
            "errores": errores, "palabras_ref": palabras,
            "elapsed_s": d["elapsed_s"], "duracion_audio_s": d.get("duracion_audio_s"),
            "seg_por_min_audio": d.get("seg_por_min_audio"), "peak_vram_mb": d["peak_vram_mb"],
            "transcript": d["transcript"],
        })
    return pd.DataFrame(filas)


def resumen_asr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Por tarea y combinación: WER del conjunto con IC, CER, velocidad y VRAM."""
    filas = []
    for clave, g in df.groupby(["tarea", *COMBO, "nivel"]):
        ic = bootstrap_ic(list(zip(g["errores"], g["palabras_ref"])), wer_conjunto, cfg)
        filas.append({
            **dict(zip(["tarea", *COMBO, "nivel"], clave)),
            "n_clips": len(g), "device_real": "/".join(sorted(set(g["device_real"]))),
            "compute_real": "/".join(sorted(set(g["compute_real"]))),
            "wer": ic["estimacion"], "wer_ic_bajo": ic["ic_bajo"], "wer_ic_alto": ic["ic_alto"],
            "cer_promedio": g["cer"].mean(),
            "seg_por_min_audio": g["seg_por_min_audio"].mean(),
            "peak_vram_mb_max": g["peak_vram_mb"].max(),
        })
    return pd.DataFrame(filas)


def _filtrar(df: pd.DataFrame, combo: dict, nivel: str = "L0") -> pd.DataFrame:
    mascara = df["nivel"] == nivel
    for k in COMBO:
        mascara &= df[k] == combo[k]
    return df[mascara].set_index("clip")


def tabla_comparaciones(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Comparaciones pareadas de config: mismos clips en A y en B."""
    filas = []
    for comp in cfg["evaluacion"]["comparaciones"]:
        a, b = _filtrar(df, comp["a"]), _filtrar(df, comp["b"])
        clips = sorted(set(a.index) & set(b.index))
        if not clips:
            continue  # todavía no está en cache
        pares_a = [(a.loc[c, "errores"], a.loc[c, "palabras_ref"]) for c in clips]
        pares_b = [(b.loc[c, "errores"], b.loc[c, "palabras_ref"]) for c in clips]
        dif = bootstrap_diferencia(pares_b, pares_a, wer_conjunto, cfg)  # WER(b) - WER(a)
        filas.append({
            "comparacion": comp["nombre"], "n_clips": len(clips),
            "wer_a": wer_conjunto(pares_a), "wer_b": wer_conjunto(pares_b),
            "dif_wer_b_menos_a": dif["estimacion"],
            "dif_ic_bajo": dif["ic_bajo"], "dif_ic_alto": dif["ic_alto"],
            "ic_incluye_0": dif["ic_bajo"] <= 0 <= dif["ic_alto"],
            "seg_por_min_a": a.loc[clips, "seg_por_min_audio"].mean(),
            "seg_por_min_b": b.loc[clips, "seg_por_min_audio"].mean(),
            "vram_max_a": a.loc[clips, "peak_vram_mb"].max(),
            "vram_max_b": b.loc[clips, "peak_vram_mb"].max(),
            # "¿Cambió el texto?": idéntico carácter a carácter, y WER de B contra A
            "clips_texto_identico": sum(a.loc[c, "transcript"] == b.loc[c, "transcript"] for c in clips),
            "wer_entre_salidas": float(np.mean([wer(a.loc[c, "transcript"], b.loc[c, "transcript"], cfg)
                                                for c in clips])),
        })
    return pd.DataFrame(filas)


# =====================================================================
# Extracción y validación
# =====================================================================

def extraer_grupos(corridas: list[dict], referencias: dict, esquema_de: dict,
                   metodos: list[str], cfg: dict) -> list[dict]:
    """Un registro por (grupo, nivel, método, clip): campos, verificación y segmentos."""
    registros = []
    for metodo in metodos:
        base = con_motor(cfg, metodo_extraccion=metodo, solo_cache=True)
        for clip, texto in referencias.items():  # grupo "referencia": sin ASR
            r = process_transcript(texto, esquema_de[clip], base)
            registros.append({"grupo": "referencia", "nivel": "-", "metodo": metodo, "clip": clip,
                              "tarea": esquema_de[clip], "resultado": r})
        for c in corridas:  # cada combinación de ASR: SIEMPRE por process_audio
            motor = {k: c[k] for k in COMBO}
            r = process_audio(f"{c['clip']}.wav", esquema_de[c["clip"]],
                              con_motor(base, **motor, nivel=c["nivel"]))
            registros.append({"grupo": "|".join(motor.values()), "nivel": c["nivel"], "metodo": metodo,
                              "clip": c["clip"], "tarea": esquema_de[c["clip"]], "resultado": r})
    return registros


def tabla_extraccion(registros: list[dict], cfg: dict) -> pd.DataFrame:
    """Por grupo/nivel/método/tarea y campo: accuracy con IC y % verificado."""
    filas = []
    verdad = {t: leer_ground_truth(t, cfg) for t in ("entrevista", "pedido")}
    claves = sorted({(r["grupo"], r["nivel"], r["metodo"], r["tarea"]) for r in registros})
    for grupo, nivel, metodo, tarea in claves:
        sub = [r for r in registros if (r["grupo"], r["nivel"], r["metodo"], r["tarea"])
               == (grupo, nivel, metodo, tarea)]
        extraidos = {r["clip"]: r["resultado"].fields for r in sub}
        verificados = {r["clip"]: r["resultado"].verified for r in sub}
        for campo, por_clip in tabla_aciertos(extraidos, verdad[tarea], cfg).items():
            ic = bootstrap_ic([int(v) for v in por_clip.values()], np.mean, cfg)
            fila = {"grupo": grupo, "nivel": nivel, "metodo": metodo, "tarea": tarea, "campo": campo,
                    "n_clips": ic["n_clips"], "accuracy": ic["estimacion"],
                    "ic_bajo": ic["ic_bajo"], "ic_alto": ic["ic_alto"],
                    "pct_verificado": np.mean([verificados[c].get(campo, False) for c in por_clip]),
                    "errores_de_ejecucion": sum(r["resultado"].error is not None for r in sub)}
            if campo in CAMPOS_NUM:
                errores = [e for e in error_absoluto(extraidos, verdad[tarea], campo).values()
                           if e is not None]
                fila["error_absoluto_medio"] = np.mean(errores) if errores else None
            if campo == cfg["evaluacion"]["macro_f1"][tarea]:
                clips = sorted(por_clip)
                fila["macro_f1"] = macro_f1([verdad[tarea][c][campo] for c in clips],
                                            [extraidos[c].get(campo) for c in clips],
                                            cfg["extraccion"]["categorias"][campo])
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_extraccion_por_clip(registros: list[dict], cfg: dict) -> pd.DataFrame:
    """Una fila por clip: cuántos campos salieron bien. Se cruza con el WER del clip
    para responder: ¿el WER predice si el registro sirve?"""
    verdad = {t: leer_ground_truth(t, cfg) for t in ("entrevista", "pedido")}
    filas = []
    for r in registros:
        res = r["resultado"]
        aciertos = tabla_aciertos({r["clip"]: res.fields}, verdad[r["tarea"]], cfg)
        correctos = [por_clip[r["clip"]] for por_clip in aciertos.values()]
        filas.append({"grupo": r["grupo"], "nivel": r["nivel"], "metodo": r["metodo"],
                      "tarea": r["tarea"], "clip": r["clip"],
                      "campos_correctos": sum(correctos), "n_campos": len(correctos),
                      "accuracy_registro": sum(correctos) / len(correctos),
                      "campos_verificados": sum(res.verified.values()),
                      "error": res.error})
    return pd.DataFrame(filas)


def tabla_capturas(validacion: pd.DataFrame) -> pd.DataFrame:
    """Por grupo y nivel: de los totales mal dichos a propósito, cuántos atrapó el chequeo."""
    if validacion.empty:
        return pd.DataFrame()
    filas = []
    for (grupo, nivel, metodo), g in validacion.groupby(["grupo", "nivel", "metodo"]):
        trampa = g["trampa_total_erroneo"].astype(bool)
        no_cuadra = g["cuadra"] == False  # noqa: E712 (None = no se pudo chequear)
        filas.append({
            "grupo": grupo, "nivel": nivel, "metodo": metodo, "n_llamadas": len(g),
            "trampas": int(trampa.sum()),
            "capturadas": int((trampa & no_cuadra).sum()),             # no cuadra y era trampa
            "perdidas": int((trampa & (g["cuadra"] == True)).sum()),  # noqa: E712 cuadra aunque era trampa
            "falsas_alarmas": int((~trampa & no_cuadra).sum()),        # no cuadra sin trampa
            "sin_chequeo": int(g["cuadra"].isna().sum()),
            "diagnostico_aciertos": int((g["acierta"] == True).sum()),  # noqa: E712
            "diagnostico_evaluados": int(g["acierta"].notna().sum()),
        })
    return pd.DataFrame(filas)


def tabla_validacion(registros: list[dict], cfg: dict) -> pd.DataFrame:
    """Pedidos: ¿cuadra la cuenta?, diagnóstico y si acierta contra las trampas."""
    verdad = leer_ground_truth("pedido", cfg)
    pedidos = [r for r in registros if r["tarea"] == "pedido" and r["resultado"].error is None]
    # ¿Cuadraba en L0 la misma combinación? (para la regla "L0 cuadra y Lk no -> ASR")
    cuadra_l0 = {(r["grupo"], r["metodo"], r["clip"]): chequear(r["resultado"].fields, cfg)["cuadra"]
                 for r in pedidos if r["nivel"] == "L0"}
    filas = []
    for r in pedidos:
        res = r["resultado"]
        previo = cuadra_l0.get((r["grupo"], r["metodo"], r["clip"])) if r["nivel"] not in ("L0", "-") else None
        d = diagnosticar(res.fields, palabras_de(res.segments), cfg, cuadra_en_l0=previo)
        trampa = verdad[r["clip"]]["_meta"]["trampa_total_erroneo"]
        # Lo correcto: si el hablante se equivocó, "error_hablante"; si no, "ok" si cuadra
        # y, si no cuadra, la culpa es del pipeline (ASR o extracción)
        esperado = "error_hablante" if trampa else ("ok" if d["cuadra"] else "error_asr")
        filas.append({"grupo": r["grupo"], "nivel": r["nivel"], "metodo": r["metodo"], "clip": r["clip"],
                      "total_dicho": d.get("total_dicho"), "total_esperado": d.get("total_esperado"),
                      "total_correcto_gt": verdad[r["clip"]]["_meta"]["total_correcto"],
                      "cuadra": d["cuadra"], "diagnostico": d["diagnostico"], "motivo": d.get("motivo"),
                      "trampa_total_erroneo": trampa, "esperado": esperado,
                      # sin probabilidades (grupo referencia) no hay diagnóstico que evaluar
                      "acierta": None if d["diagnostico"] == "indeterminado"
                      else d["diagnostico"] == esperado})
    return pd.DataFrame(filas)


# =====================================================================
# Todo junto
# =====================================================================

def evaluar(cfg: dict, metodos: list[str]) -> dict:
    esquema_de = {clip: tarea for tarea in ("entrevista", "pedido")
                  for clip in leer_ground_truth(tarea, cfg)}
    referencias = {c: t for c, t in leer_referencias(cfg).items() if c in esquema_de}
    todas = leer_corridas(cfg)
    corridas = [c for c in todas if c["clip"] in esquema_de]
    ignorados = sorted({c["clip"] for c in todas if c["clip"] not in esquema_de})

    asr = tabla_asr(corridas, referencias, esquema_de, cfg)
    registros = extraer_grupos(corridas, referencias, esquema_de, metodos, cfg)
    resultados = {
        "asr_por_clip": asr,
        "asr_resumen": resumen_asr(asr, cfg) if len(asr) else pd.DataFrame(),
        "comparaciones": tabla_comparaciones(asr, cfg) if len(asr) else pd.DataFrame(),
        "extraccion_por_campo": tabla_extraccion(registros, cfg),
        "extraccion_por_clip": tabla_extraccion_por_clip(registros, cfg),
        "pedidos_validacion": tabla_validacion(registros, cfg),
    }
    resultados["capturas_por_nivel"] = tabla_capturas(resultados["pedidos_validacion"])
    resultados["resumen"] = {
        "metodos": metodos,
        "n_clips_por_tarea": pd.Series(esquema_de).value_counts().to_dict(),
        "transcripts_en_cache": len(corridas),
        "combinaciones_en_cache": sorted({"|".join(c[k] for k in COMBO) + f"|{c['nivel']}" for c in corridas}),
        "clips_ignorados_sin_ground_truth": ignorados,
        "advertencias": [
            f"Muestra chica: {len(referencias)} clips en total. Los IC bootstrap son anchos; "
            "una diferencia cuyo IC incluye 0 NO es evidencia de mejora.",
            "El grupo 'referencia' mide solo la extracción (sin errores de ASR) y las reglas se "
            "escribieron mirando los guiones: es una cota optimista.",
        ],
    }
    return resultados


def guardar(resultados: dict, cfg: dict) -> Path:
    carpeta = RAIZ / cfg["rutas"]["cache_results"]
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in resultados.items():
        if nombre == "resumen":
            (carpeta / "resumen.json").write_text(
                json.dumps(tabla, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        else:
            tabla.drop(columns=["transcript"], errors="ignore").to_csv(carpeta / f"{nombre}.csv", index=False)
    return carpeta


def verificar_umbrales(resultados: dict, cfg: dict) -> list[str]:
    """Lista de fallas (vacía si todo cumple). Lo usa el CI."""
    umbral, combo = cfg["regresion"], cfg["regresion"]["combinacion"]
    grupo = "|".join(combo[k] for k in COMBO)
    fallas = []

    resumen = resultados["asr_resumen"]
    if len(resumen):
        fila = resumen[(resumen[COMBO].astype(str).agg("|".join, axis=1) == grupo) & (resumen["nivel"] == "L0")]
        for _, f in fila.iterrows():
            if f["wer"] > umbral["wer_max"]:
                fallas.append(f"WER {f['tarea']} {grupo} = {f['wer']:.3f} > {umbral['wer_max']}")

    ext = resultados["extraccion_por_campo"]
    vigilado = grupo if (ext["grupo"] == grupo).any() else "referencia"
    sub = ext[(ext["grupo"] == vigilado) & (ext["nivel"].isin(["L0", "-"]))]
    for (metodo, tarea), g in sub.groupby(["metodo", "tarea"]):
        if g["accuracy"].mean() < umbral["accuracy_campos_min"]:
            fallas.append(f"accuracy {vigilado} {metodo} {tarea} = {g['accuracy'].mean():.3f} "
                          f"< {umbral['accuracy_campos_min']}")
    print(f"Umbrales revisados sobre: {grupo if vigilado == grupo else 'referencia (sin cache aún)'}")
    return fallas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Métricas finales desde cache/")
    parser.add_argument("--llm", action="store_true", help="incluir extracción con LLM")
    parser.add_argument("--verificar", action="store_true", help="fallar si no se cumplen umbrales")
    args = parser.parse_args(argv)

    cfg = cargar_config()
    metodos = cfg["evaluacion"]["metodos"] + (["llm"] if args.llm else [])
    resultados = evaluar(cfg, metodos)
    print("Resultados en", guardar(resultados, cfg))
    print(json.dumps(resultados["resumen"], indent=2, ensure_ascii=False))

    if args.verificar:
        fallas = verificar_umbrales(resultados, cfg)
        for f in fallas:
            print("FALLA:", f)
        return 1 if fallas else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
