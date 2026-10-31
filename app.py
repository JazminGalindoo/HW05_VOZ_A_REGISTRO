"""
app.py — Interfaz Streamlit.   Uso:  streamlit run app.py

Responsabilidad:
    - Mostrar resultados leyendo SOLO de cache/ (nunca re-transcribe el corpus).
    - Cualquier procesamiento pasa por src.engine (process_audio / process_transcript).

Pestañas:
    Tarea 1: 1 Problema y método · 2 Corpus · 3 Resultados · 4 Probar un audio
    Tarea 2: 5 Pedidos (escalera, validación aritmética, capturadas vs perdidas)

Se actualiza sola: las tablas de cache/results se leen usando la fecha de
modificación del archivo como clave del cache de Streamlit; cuando
`python -m src.evaluar` las reescribe, la próxima interacción ya muestra lo nuevo.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.engine import EngineResult, cargar_config, con_motor, process_audio, process_transcript
from src.schemas import leer_ground_truth
from src.validate import chequear, diagnosticar, palabras_de
from src.verify import etiquetas

RAIZ = Path(__file__).resolve().parent
cfg = cargar_config()
CARPETA_RESULTADOS = RAIZ / cfg["rutas"]["cache_results"]
COMBO = ["modelo", "device", "compute_type", "glosario"]
TAREAS = {"entrevista": "Tarea 1 — Entrevistas", "pedido": "Tarea 2 — Pedidos"}


# =====================================================================
# Lectura de resultados (se recarga sola cuando cambia el archivo)
# =====================================================================

@st.cache_data
def _leer_csv(ruta: str, fecha_modificacion: float) -> pd.DataFrame:
    return pd.read_csv(ruta)


def resultado(nombre: str) -> pd.DataFrame:
    """Tabla de cache/results; vacía si todavía no existe."""
    ruta = CARPETA_RESULTADOS / f"{nombre}.csv"
    if not ruta.exists():
        return pd.DataFrame()
    try:
        return _leer_csv(str(ruta), ruta.stat().st_mtime)
    except pd.errors.EmptyDataError:  # tabla sin filas (p. ej. sin transcripts en cache)
        return pd.DataFrame()


def resultado_json(nombre: str) -> dict:
    ruta = CARPETA_RESULTADOS / f"{nombre}.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


@st.cache_data
def ground_truth(schema: str) -> dict:
    return leer_ground_truth(schema, cfg)


def referencia(clip: str) -> str:
    return (RAIZ / cfg["rutas"]["transcripts_referencia"] / f"{clip}.txt").read_text(encoding="utf-8")


# =====================================================================
# Pasar un clip del corpus por el motor (sin transcribir)
# =====================================================================

@st.cache_data(show_spinner=False)
def procesar_clip(clip: str, schema: str, grupo: str, nivel: str, metodo: str) -> EngineResult:
    """grupo 'referencia' -> el texto humano; si no, el transcript cacheado de esa combinación."""
    base = con_motor(cfg, metodo_extraccion=metodo, solo_cache=True)
    if grupo == "referencia":
        return process_transcript(referencia(clip), schema, base)
    combo = dict(zip(COMBO, grupo.split("|")))
    return process_audio(f"{clip}.wav", schema, con_motor(base, **combo, nivel=nivel))


def archivo_de_audio(clip: str, schema: str, nivel: str = "L0") -> Path | None:
    """El audio está solo en la PC de quien grabó (no se sube al repo)."""
    r = cfg["rutas"]
    if nivel == "L0":
        carpeta = RAIZ / (r["audio_entrevistas"] if schema == "entrevista" else r["audio_llamadas"])
    else:
        carpeta = RAIZ / r["audio_degradado"] / nivel
    for extension in cfg["degradacion"]["extensiones_audio"]:
        if (carpeta / f"{clip}{extension}").exists():
            return carpeta / f"{clip}{extension}"
    return None


def formato(valor) -> str:
    if valor is None:
        return "—"
    if isinstance(valor, list) and valor and isinstance(valor[0], dict):  # items del pedido
        return "; ".join(f"{i['producto']} ×{i['cantidad']} @ {i['precio_unitario']}" for i in valor)
    if isinstance(valor, list):
        return ", ".join(map(str, valor)) or "—"
    return str(valor)


def tabla_registro(res: EngineResult, verdad: dict | None = None) -> pd.DataFrame:
    """Campo · valor extraído · verificado/sin_verificar (· valor correcto)."""
    estado = etiquetas(res.verified, cfg)
    filas = [{"campo": c, "extraído": formato(v), "verificación": estado.get(c, "—")}
             for c, v in res.fields.items()]
    if verdad:
        for f in filas:
            f["correcto (ground truth)"] = formato(verdad.get(f["campo"]))
    return pd.DataFrame(filas)


def mostrar_chequeo(res: EngineResult) -> None:
    """La cuenta del pedido, escrita para que se lea en voz alta en el video."""
    if res.error or not res.fields.get("items"):
        st.info("No hay items para chequear.")
        return
    d = diagnosticar(res.fields, palabras_de(res.segments), cfg)
    if d["cuadra"] is None:
        st.info(f"No se pudo chequear: {d['motivo']}")
        return
    suma = " + ".join(f"{i['cantidad']}×{i['precio_unitario']}" for i in res.fields["items"])
    cuenta = f"{suma} − {d['descuento']} = **{d['total_esperado']:g}**  ·  total dicho: **{d['total_dicho']:g}**"
    if d["cuadra"]:
        st.success(f"Cuadra. {cuenta}")
    else:
        st.error(f"No cuadra (diferencia {d['diferencia']:+g}). {cuenta}")
        st.write(f"Diagnóstico: **{d['diagnostico']}** — {d.get('motivo', '')}")


# =====================================================================
# Página
# =====================================================================

st.set_page_config(page_title=cfg["app"]["titulo"], layout="wide")
st.title(cfg["app"]["titulo"])

extraccion = resultado("extraccion_por_campo")
resumen = resultado_json("resumen")
if extraccion.empty:
    st.warning("No hay resultados en cache/results. Corre:  python -m src.evaluar")
    st.stop()

# Barra lateral: qué combinación de ASR y qué método de extracción mirar
with st.sidebar:
    st.header("Qué mirar")
    grupos = sorted(extraccion["grupo"].unique(), key=lambda g: (g != "referencia", g))
    inicial = cfg["app"]["combinacion_inicial"]
    grupo = st.selectbox("Transcripción", grupos, index=grupos.index(inicial) if inicial in grupos else 0,
                         help="'referencia' = el texto humano como si fuera el ASR (mide solo la extracción)")
    metodo = st.radio("Extracción", sorted(extraccion["metodo"].unique()), horizontal=True)
    st.caption(f"Transcripts en cache: {resumen.get('transcripts_en_cache', 0)}")
    if grupo == "referencia":
        st.info("Sin ASR: todavía no hay transcripts en cache o elegiste 'referencia'.")

t1, t2, t3, t4, t5 = st.tabs(["1 · Problema y método", "2 · Corpus", "3 · Resultados",
                              "4 · Probar un audio", "5 · Pedidos (Tarea 2)"])

# ---------------------------------------------------------------- 1
with t1:
    st.subheader("El audio que nadie procesa")
    st.markdown("""
El celular ya transcribe notas de voz, pero lo que una organización necesita no es texto:
son **campos**, verificados, en una tabla, con una medida de cuánto confiar en ellos.

- **Tarea 1:** entrevista grabada → registro del candidato (6 campos).
- **Tarea 2:** llamada de venta → pedido que **se chequea solo**: Σ cantidad × precio − descuento = total dicho.

Las dos tareas usan **el mismo motor**: `process_audio(audio, schema)` en `src/engine.py`.
""")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f2f2f2", fontname="Helvetica"];
  subgraph cluster_una { label="Una vez"; style=dashed;
    guion [label="Guiones +\\nground truth"]; grabar [label="Grabar\\n(WhatsApp)"];
    degradar [label="Escalera\\nL1–L3"]; colab [label="Colab GPU:\\ngrilla de ASR"]; cache [label="cache/\\ntranscripts"]; }
  subgraph cluster_cada { label="Por cada audio"; style=dashed;
    asr [label="ASR\\nfaster-whisper"]; norm [label="Normalizar\\n(números)"]; ext [label="Extraer\\nreglas | LLM"];
    ver [label="Verificar\\nliteral"]; val [label="Validar\\naritmética"]; }
  guion -> grabar -> degradar -> colab -> cache;
  cache -> norm [style=dotted, label="app / CI"]; asr -> norm -> ext -> ver -> val;
  metr [label="src.evaluar\\n→ cache/results"]; ver -> metr; val -> metr;
}
""")
    st.markdown("""
**Método, en corto**
- **Ground truth escrito antes de grabar** (guiones): por eso no costó nada; nunca se usa la salida de otro ASR como referencia.
- **GPU probada con medición**: el device se lee del modelo en runtime y la VRAM con `pynvml`; no del config.
- **Normalización** igual para ASR y referencia: "seis mil quinientos" → `6500`.
- **Verificación literal**: un valor solo es `verificado` si aparece en el transcript. El LLM no puede corregir al ASR en silencio.
- **Dos capas de métricas**: transcripción (WER, CER) y registro (accuracy por campo, macro-F1, error absoluto), con IC bootstrap.
- **Muestra chica** (pocos clips por tarea): los IC son anchos y se dice qué no se puede afirmar.
""")
    for aviso in resumen.get("advertencias", []):
        st.caption("⚠ " + aviso)

# ---------------------------------------------------------------- 2
with t2:
    c_izq, c_der = st.columns([1, 3])
    with c_izq:
        schema = st.radio("Tarea", list(TAREAS), format_func=TAREAS.get, key="corpus_tarea")
        gt = ground_truth(schema)
        clip = st.selectbox("Clip", sorted(gt), key="corpus_clip")
        audio = archivo_de_audio(clip, schema)
        if audio:
            st.audio(str(audio))
        else:
            st.caption("El audio no está en esta PC (los audios no se suben al repo).")
    with c_der:
        res = procesar_clip(clip, schema, grupo, "L0", metodo)
        col_ref, col_asr = st.columns(2)
        with col_ref:
            st.markdown("**Transcript de referencia** (escrito por humano)")
            st.text_area("referencia", referencia(clip), height=320, label_visibility="collapsed")
        with col_asr:
            st.markdown(f"**Transcript usado** · `{grupo}`")
            if res.error:
                st.warning(f"No disponible: {res.error}")
            st.text_area("asr", res.transcript, height=320, label_visibility="collapsed")
    if not res.error:
        st.markdown("**Registro extraído**")
        st.dataframe(tabla_registro(res, gt[clip]), hide_index=True, width="stretch")

    with st.expander("Ground truth y distribución de cada campo"):
        for s in TAREAS:
            tabla = pd.DataFrame({c: {k: formato(v) for k, v in r.items() if k != "_meta"}
                                  for c, r in ground_truth(s).items()}).T
            st.markdown(f"**{TAREAS[s]}**")
            st.dataframe(tabla, width="stretch")
            for campo in tabla.columns:
                st.caption(f"{campo}: " + ", ".join(f"{v} ({n})" for v, n in tabla[campo].value_counts().items()))

# ---------------------------------------------------------------- 3
with t3:
    st.subheader("Capa 1 — Transcripción: ¿escuchó las palabras?")
    asr = resultado("asr_resumen")
    if asr.empty:
        st.info("Todavía no hay transcripts en cache: esta capa se llena al correr el notebook de Colab "
                "y `python -m src.evaluar`.")
    else:
        l0 = asr[asr["nivel"] == "L0"].copy()
        l0["combinación"] = l0[COMBO].astype(str).agg("|".join, axis=1)
        st.dataframe(l0[["tarea", "combinación", "device_real", "compute_real", "n_clips", "wer",
                         "wer_ic_bajo", "wer_ic_alto", "cer_promedio", "seg_por_min_audio",
                         "vram_extra_mb_max", "vram_modelo_mb", "vram_total_mb"]].round(3),
                     hide_index=True, width="stretch")
        st.caption("VRAM extra = lo que agrega transcribir con el modelo ya cargado. "
                   "VRAM total = contexto CUDA + pesos + transcripción, medida aparte con "
                   "`src/medir_vram.py` en un proceso nuevo (vacía si todavía no se midió).")
        l0["error_mas"], l0["error_menos"] = l0["wer_ic_alto"] - l0["wer"], l0["wer"] - l0["wer_ic_bajo"]
        st.plotly_chart(px.bar(l0, x="combinación", y="wer", color="tarea", barmode="group",
                               error_y="error_mas", error_y_minus="error_menos",
                               title="WER en L0 (IC 95 % bootstrap por clip)"), width="stretch")
        comp = resultado("comparaciones")
        if not comp.empty:
            st.markdown("**Comparaciones pareadas** — diferencia = WER(b) − WER(a). "
                        "Si el IC incluye 0, la muestra no permite afirmar una diferencia.")
            st.dataframe(comp.round(3), hide_index=True, width="stretch")

    st.subheader("Capa 2 — Registro: ¿el dato sirve?")
    sub = extraccion[(extraccion["grupo"] == grupo) & (extraccion["metodo"] == metodo)
                     & (extraccion["nivel"].isin(["L0", "-"]))]
    columnas = [c for c in ["tarea", "campo", "n_clips", "accuracy", "ic_bajo", "ic_alto", "pct_verificado",
                            "macro_f1", "error_absoluto_medio"] if c in sub.columns]
    st.dataframe(sub[columnas].round(3), hide_index=True, width="stretch")
    sub = sub.assign(error_mas=sub["ic_alto"] - sub["accuracy"], error_menos=sub["accuracy"] - sub["ic_bajo"])
    st.plotly_chart(px.bar(sub, x="campo", y="accuracy", color="tarea", error_y="error_mas",
                           error_y_minus="error_menos", range_y=[0, 1.05],
                           title=f"Accuracy por campo · {grupo} · {metodo}"), width="stretch")

    st.subheader("¿El WER dice si el registro sirve?")
    por_clip, asr_clip = resultado("extraccion_por_clip"), resultado("asr_por_clip")
    if asr_clip.empty or por_clip.empty:
        st.info("Se responde con los transcripts reales: cada clip con su WER y su % de campos correctos.")
    else:
        asr_clip["grupo"] = asr_clip[COMBO].astype(str).agg("|".join, axis=1)
        cruce = asr_clip.merge(por_clip[por_clip["metodo"] == metodo], on=["grupo", "nivel", "clip", "tarea"])
        st.plotly_chart(px.scatter(cruce, x="wer", y="accuracy_registro", color="tarea", symbol="nivel",
                                   hover_data=["clip", "grupo"], range_y=[-0.05, 1.05],
                                   title="WER del clip vs accuracy de su registro"), width="stretch")
        perfecto, roto = cruce[cruce["accuracy_registro"] == 1], cruce[cruce["accuracy_registro"] < 1]
        if not perfecto.empty:
            f = perfecto.loc[perfecto["wer"].idxmax()]
            st.markdown(f"- **WER más alto con registro perfecto:** {f['clip']} ({f['grupo']}, {f['nivel']}): "
                        f"WER {f['wer']:.1%}, todos los campos bien.")
        if not roto.empty:
            f = roto.loc[roto["wer"].idxmin()]
            st.markdown(f"- **WER más bajo con registro roto:** {f['clip']} ({f['grupo']}, {f['nivel']}): "
                        f"WER {f['wer']:.1%}, {int(f['campos_correctos'])}/{int(f['n_campos'])} campos bien.")

# ---------------------------------------------------------------- 4
with t4:
    st.subheader("Probar un audio nuevo")
    p = cfg["app"]["probador"]
    st.caption(f"Este es el único lugar donde la app transcribe: {p['modelo']} · {p['compute_type']} · "
               f"device {p['device']} (el real se lee del modelo). Un clip de 3 min tarda ~1 min en CPU.")
    subido = st.file_uploader("Audio", type=cfg["app"]["probador_tipos"])
    schema_p = st.radio("Esquema", list(TAREAS), format_func=TAREAS.get, horizontal=True, key="prob_schema")
    metodo_p = st.radio("Extracción", ["reglas", "llm"], horizontal=True, key="prob_metodo",
                        help="llm necesita GEMINI_API_KEY en .env")
    if subido and st.button("Procesar", type="primary"):
        ruta = Path(tempfile.mkdtemp()) / subido.name
        ruta.write_bytes(subido.getvalue())
        c = con_motor(cfg, **p, metodo_extraccion=metodo_p, solo_cache=False)
        c["rutas"]["cache_transcripts"] = str(RAIZ / cfg["app"]["carpeta_probador"])  # fuera del corpus
        with st.spinner("Transcribiendo y extrayendo..."):
            res = process_audio(str(ruta), schema_p, c)
        if res.error:
            st.error(res.error)
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Device (runtime)", res.device)
            m2.metric("Compute", res.compute_type)
            m3.metric("Segundos", f"{res.elapsed_s:.1f}" + (" (cache)" if res.desde_cache else ""))
            m4.metric("VRAM extra (MB)", "—" if res.peak_vram_mb is None else f"{res.peak_vram_mb:.0f}")
            st.text_area("Transcript", res.transcript, height=200)
            st.dataframe(tabla_registro(res), hide_index=True, width="stretch")
            if schema_p == "pedido":
                mostrar_chequeo(res)

# ---------------------------------------------------------------- 5
with t5:
    st.subheader("Escalera de degradación")
    snr = resultado_json("degradacion")
    d = cfg["degradacion"]  # los parámetros salen del config, no se escriben aquí
    banda = d["L2"]["paso_banda_hz"]
    st.markdown(f"L0 original · L1 WhatsApp (mono, Opus {d['L1']['bitrate_kbps']} kbps) · "
                f"L2 teléfono ({banda[0]}–{banda[1]} Hz, {d['L2']['sample_rate_hz'] // 1000} kHz) · "
                f"L3 = L1 + ruido ambiente a {d['L3']['snr_db']} dB (SNR medido por RMS).")
    if snr:
        st.dataframe(pd.DataFrame(snr).T, width="stretch")

    asr = resultado("asr_resumen")
    curva_acc = (extraccion[(extraccion["tarea"] == "pedido") & (extraccion["metodo"] == metodo)
                            & (extraccion["grupo"] != "referencia")]
                 .groupby(["grupo", "nivel"], as_index=False)["accuracy"].mean())
    if asr.empty or curva_acc.empty:
        st.info("La curva WER / accuracy por nivel aparece cuando estén los transcripts de L0–L3 en cache.")
    else:
        curva = asr[asr["tarea"] == "pedido"].copy()
        curva["grupo"] = curva[COMBO].astype(str).agg("|".join, axis=1)
        curva = curva.merge(curva_acc, on=["grupo", "nivel"], how="left")
        largo = curva.melt(id_vars=["grupo", "nivel"], value_vars=["wer", "accuracy"], var_name="métrica")
        st.plotly_chart(px.line(largo.sort_values("nivel"), x="nivel", y="value", color="grupo",
                                facet_col="métrica", markers=True, title="Curva de degradación (pedidos)"),
                        width="stretch")

    st.subheader("Capturadas vs perdidas")
    st.caption("Trampas = llamadas donde el total se dijo mal a propósito. Capturada = el chequeo no cuadra; "
               "perdida = cuadra igual. Falsa alarma = no cuadra sin haber trampa (error de ASR o extracción).")
    capt = resultado("capturas_por_nivel")
    if not capt.empty:
        st.dataframe(capt[capt["metodo"] == metodo], hide_index=True, width="stretch")

    st.subheader("Chequeo aritmético en vivo")
    gt_p = ground_truth("pedido")
    validacion = resultado("pedidos_validacion")
    niveles = sorted(validacion.loc[validacion["grupo"] == grupo, "nivel"].unique()) if not validacion.empty else []
    a, b = st.columns(2)
    clip_p = a.selectbox("Llamada", sorted(gt_p), key="ped_clip")
    nivel_p = b.selectbox("Nivel", niveles or ["-"], key="ped_nivel")
    meta = gt_p[clip_p]["_meta"]
    st.caption(f"Ground truth: total correcto {meta['total_correcto']} · "
               f"trampa de total erróneo: {'sí' if meta['trampa_total_erroneo'] else 'no'} · "
               f"autocorrección: {'sí' if meta['trampa_autocorreccion'] else 'no'}")
    res_p = procesar_clip(clip_p, "pedido", grupo, nivel_p if nivel_p != "-" else "L0", metodo)
    if res_p.error:
        st.warning(res_p.error)
    else:
        st.dataframe(pd.DataFrame(res_p.fields["items"]), hide_index=True, width="stretch")
        mostrar_chequeo(res_p)

    if not validacion.empty:
        with st.expander("Detalle de validación y diagnóstico (todas las llamadas y niveles)"):
            st.dataframe(validacion[(validacion["grupo"] == grupo) & (validacion["metodo"] == metodo)],
                         hide_index=True, width="stretch")
