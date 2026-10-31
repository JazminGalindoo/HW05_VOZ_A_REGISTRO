# HW05_VOZ_A_REGISTRO
ASR con faster-whisper: de entrevistas y llamadas de venta grabadas a registros estructurados y verificados. HW_05 Data Science Python.

**Video:** (pendiente)

> **Estado:** pipeline completo, corrido sobre los audios reales (Colab T4). Todos los resultados salen de `python -m src.evaluar` sobre `cache/`; ningún número está escrito a mano.

**Documento del pipeline:** [Notion — Pipeline HW05: De la voz al registro](https://gamy-maple-386.notion.site/Pipeline-HW05-De-la-voz-al-registro-7d94285ae77e830a953f81a8d1c95d86) · versión en el repo: [docs/pipeline.md](docs/pipeline.md) (qué corre una vez, qué corre por cada audio, decisiones y lo descartado).

## 1. Descripción
- **Tarea 1 — Entrevistas:** audio → registro del candidato con 6 campos (`anios_experiencia`, `ultimo_puesto`, `tecnologias`, `pretension_salarial` + `moneda`, `disponibilidad`, `nivel_ingles`), cada uno marcado `verificado` / `sin_verificar`.
- **Tarea 2 — Llamadas de venta:** audio → pedido **final** (tras las autocorrecciones del cliente) que se valida aritméticamente: `Σ cantidad × precio − descuento = total dicho`.

Ambas tareas usan **un solo motor**, `src/engine.py`:

```python
process_audio(audio_path: str, schema: str, cfg: dict | None = None) -> EngineResult
```

`EngineResult` trae `transcript`, `segments` (palabras con probabilidad), `fields`, `verified`, `device` y `compute_type` **leídos del modelo en runtime**, `elapsed_s`, `peak_vram_mb`, `error` y `desde_cache`.

**Segunda función pública (decisión):** `process_transcript(transcript, schema, cfg, segments)` hace solo extracción + verificación. Existe para evaluar sin audio (con los transcripts de referencia) y `process_audio` la usa por dentro, así la lógica vive en un único lugar.

## 2. Instalación
Probado en Windows 11 con Python 3.12.10 (en CI: Ubuntu, Python 3.12).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # y completar GEMINI_API_KEY (solo para la extracción con LLM; gratis en aistudio.google.com)
```
También se necesita **ffmpeg** en el sistema (`winget install Gyan.FFmpeg` en Windows; ya viene en Colab).

Versiones relevantes (fijadas en `requirements.txt`):

| Paquete | Versión | Nota |
|---|---|---|
| faster-whisper | 1.2.1 | `WhisperModel.transcribe` **sí** tiene `hotwords` (ver §6) |
| ctranslate2 | 4.8.4 | motor de inferencia de faster-whisper |
| av (PyAV) | 18.1.0 | **fijada a propósito**: la 19.x quitó `metadata_errors`, que faster-whisper 1.2.1 usa → `TypeError` al decodificar |
| google-genai | 2.34.0 | SDK de Gemini para la extracción con LLM (ver §8.5) |
| jiwer | 4.0.0 | WER/CER |
| text2num | 3.1.0 | números en palabras → dígitos |

## 3. Verificación: el motor no importa UI
```bash
python -c "import sys; import src.engine; bad={'streamlit','gradio','telegram'}&set(sys.modules); assert not bad, bad; print('OK: engine sin UI')"
```
También corre en cada push en GitHub Actions (`.github/workflows/regression.yml`), junto con los tests y `python -m src.evaluar --verificar`.

## 4. Estructura del repo
```
config.yaml              # TODOS los parámetros (rutas, modelos, umbrales, prompts, SNR, grilla...)
.env.example             # solo nombres de variables; .env nunca se sube
app.py                   # interfaz Streamlit; solo lee cache/
scripts_grabacion/
  guiones.md             # guiones escritos ANTES de grabar (ground truth)
  generar_referencias.py # guiones.md -> transcripts de referencia
data/
  audio/                 # NO se sube (entrevistas/, llamadas/, ruido/, degradado/L1..L3)
  ground_truth/          # entrevistas.csv, llamadas.csv, transcripts_referencia/*.txt
cache/
  transcripts/           # {clip}__{modelo}__{device}__{compute}__{glosario}__{nivel}.json
  results/               # métricas que escribe src/evaluar.py y lee la app
src/
  engine.py              # process_audio / process_transcript: única puerta de entrada
  asr.py                 # faster-whisper, device real, VRAM con pynvml, cache
  normalize.py           # normalización + números en palabras -> dígitos
  extract_rules.py       # extracción por regex
  extract_llm.py         # extracción con LLM (structured outputs)
  schemas.py             # campos, JSON schema, validación, lectura del ground truth
  verify.py              # regla de verificación literal
  validate.py            # chequeo aritmético + diagnóstico ASR vs hablante
  metrics.py             # WER/CER, accuracy, macro-F1, error absoluto, bootstrap
  degrade.py             # escalera L0–L3 con ffmpeg y SNR medido
  evaluar.py             # métricas finales desde cache/ -> cache/results/
notebooks/
  colab_gpu_benchmark.ipynb  # ÚNICO lugar donde se transcribe (Colab GPU)
tests/                   # pytest
```

## 5. Datos y ground truth
> **Datos inventados:** todo lo que se dice en los audios es inventado: nombres, empresas, cargos, sueldos, direcciones, pedidos y precios. No hay datos personales reales de nadie.
>
> **Consentimiento:** todas las personas que aparecen en las grabaciones aceptaron participar y que sus voces se usen en este trabajo. No se nombra a ninguna persona real en el repositorio.

- **Todo el audio se grabó para este trabajo** (llamada de WhatsApp en altavoz, grabada con el celular). Quien entrega este repositorio es una de las dos voces en todos los clips.
- 4 entrevistas (`E01`–`E04`, 8,9 min en total) y 4 llamadas a una pollería ficticia (`L01`–`L04`, dominio **restaurante**, 9,5 min en total). Cada clip dura entre 2,06 y 2,53 min (medido por faster-whisper, `duracion_audio_s` en el cache).
- **Ground truth escrito por humanos, antes de grabar:**
  - `scripts_grabacion/guiones.md`: el texto de cada clip. Si al grabar se dijo algo distinto, se corrige el guion.
  - `data/ground_truth/entrevistas.csv` y `llamadas.csv`: los valores correctos de cada campo.
  - `data/ground_truth/transcripts_referencia/*.txt`: se **generan** del guion con `python scripts_grabacion/generar_referencias.py` (solo el texto hablado, sin nombres de hablante). Nunca se usa salida de ASR ni subtítulos automáticos como referencia.
- **Trampas** marcadas en `llamadas.csv` para medir la validación y el diagnóstico:

| Clip | Autocorrección del cliente | Total dicho | Total correcto |
|---|---|---|---|
| L01 | sí (2 gaseosas → 1) | 89 | 89 |
| L02 | sí (3 chaufas → 2) | **80** | 90 |
| L03 | sí (pollo entero → 2 cuartos) | **75** (olvida el 2×1) | 65 |
| L04 | no | **98** | 87 (el chaufa se dijo a 25; el guion decía 28 y se corrigió tras escuchar el audio) |

- **Cómo se varió la posición y la forma de cada campo** (columna `orden_campos` de `entrevistas.csv`):

  | Clip | Orden en que aparecen los campos |
  |---|---|
  | E01 | puesto → tecnologías → experiencia → inglés → disponibilidad → salario |
  | E02 | salario → inglés → tecnologías → puesto → disponibilidad → experiencia |
  | E03 | inglés → disponibilidad → experiencia → salario → puesto → tecnologías |
  | E04 | disponibilidad → tecnologías → salario → experiencia → inglés → puesto |

  - Cada campo cae en una posición distinta en cada clip. E02 empieza directamente por el sueldo, y E04 por la disponibilidad.
  - Además, **ningún valor se dice limpio:** "si sumo todo, son cuatro años"; "dos años y medio… no llegan a tres"; "una década" junto a "diez años"; "unos seis mil quinientos soles **mensuales**"; "a partir del **primero de diciembre**" (fecha relativa, sin año); "**de inmediato**, estoy libre desde ya"; el cargo como "mi último puesto fue…", "fui…" o "soy jefa de…".
  - Las preguntas también cambian de palabras ("¿cuándo podría empezar?", "¿desde cuándo estaría disponible?", "¿en cuánto tiempo podría incorporarse?"), y en E02/E04 se invierten los roles de las voces.
- **Distribución de cada campo** (ground truth; también en la app, pestaña Corpus):

  | Campo | Valores en los 4 clips |
  |---|---|
  | `anios_experiencia` | 2.5 · 4 · 7 · 10 (rango 2.5–10, con un decimal) |
  | `ultimo_puesto` | 4 distintos (datos, backend, marketing, gestión TI) |
  | `tecnologias` | 3, 4, 3 y 3 elementos; 13 tecnologías distintas, ninguna repetida |
  | `pretension_salarial` | 3800 · 6500 · 9000 PEN y 4000 **USD** (2 monedas) |
  | `disponibilidad` | inmediata · 2 semanas · 1 mes · fecha (2026-12-01) |
  | `nivel_ingles` | básico 1 · intermedio 1 · avanzado 2 · nativo 0 (clase sin ejemplos) |
  | `items` (pedido) | 3 productos por llamada; cantidades 1–2; precios 8–64 soles |
  | `promocion` / `descuento` | sin promoción 1 · descuento de 10 soles 2 · 2×1 en gaseosas 1 |
  | `direccion_entrega` | 4 distritos (Surco, Cercado, San Isidro, Lince), **todas con dígitos** |
  | `metodo_pago` | tarjeta 2 · efectivo 1 · yape 1 · plin/transferencia 0 |
  | `total_dicho` | 89 · 80 · 75 · 98 (3 de 4 mal dichos a propósito) |

  Con 4 clips, varias clases no aparecen nunca (`nativo`, `plin`, `transferencia`). El macro-F1 se calcula sobre las clases presentes, y la muestra no dice nada de las ausentes.
- **Audios y pesos de modelo (no se suben al repo):**
  - **Audios:** son grabaciones con voces reales, así que no se publican. El equipo docente puede pedírselos a la autora por el canal del curso: se comparten por Drive con la estructura `audio/{entrevistas,llamadas,ruido}/`. Para **regenerar** un corpus propio: grabar con los guiones de `scripts_grabacion/guiones.md`, ponerlos en `data/audio/`, correr `python -m src.degrade` (escalera L1–L3) y el notebook de Colab (§10).
  - **Pesos:** faster-whisper los descarga solo la primera vez (Hugging Face) en `rutas.modelos`.
  - **No hace falta ninguno de los dos** para reproducir las métricas o abrir la app: todo sale de `cache/` (transcripts, respuestas del LLM y resultados).
- **Datos sintéticos:** ninguno entra en las métricas. Durante el desarrollo se usó una frase con voz sintética de Windows solo para probar que el pipeline corre de punta a punta; ese audio no está en el repo, y `src/evaluar.py` ignora cualquier clip que no esté en el ground truth (y lo lista en `cache/results/resumen.json`).

## 6. Versión de faster-whisper y glosario
- faster-whisper **1.2.1**, ctranslate2 **4.8.4**. Comprobado con:
  ```python
  import inspect; from faster_whisper import WhisperModel
  "hotwords" in inspect.signature(WhisperModel.transcribe).parameters   # True
  ```
- **Modo `plano`:** sin ayuda. **Modo `glosario`:** `initial_prompt` con una frase de contexto del dominio + `hotwords` con los términos clave (Python, Power BI, Yape, Plin…). Ambos por esquema, en `config.yaml → glosario`.
- **Prueba de device:** el device y el compute type se leen del objeto CTranslate2 (`model.model.device`, `model.model.compute_type`), nunca del config. Ejemplo real: pedir `int8` en CPU corre como `int8_float32`.
- **VRAM:** CTranslate2 no usa el allocator de PyTorch, así que `torch.cuda.max_memory_allocated()` no sirve. Se lee la memoria usada de la GPU con `pynvml` desde un hilo que muestrea cada 50 ms. Se reportan **dos medidas distintas**:
  - **VRAM extra durante la transcripción** (`peak_vram_mb` en el cache, `vram_extra_mb_max` en los resultados): pico − línea base, con la línea base tomada **con el modelo ya cargado**. No incluye los pesos: es lo que agrega transcribir (activaciones, búfer del beam search).
  - **VRAM total** (`vram_total_mb`): contexto CUDA + pesos + transcripción. La mide `src/medir_vram.py` en un **proceso nuevo** por modelo y precisión, desde antes de cargar el modelo (celda 11 del notebook). También separa cuánto ocupan solo los pesos (`vram_modelo_mb`).
  - **Por qué importa la diferencia:** la primera medición solo restaba la línea base con el modelo cargado, así que salía ~1 GB para large-v3 y casi igual en float16 e int8. La diferencia entre precisiones está en los **pesos**, no en la transcripción: para decir si un modelo "cabe" en una tarjeta hay que mirar la VRAM total.
  - **Medido en Tesla T4** (`cache/results/vram_modelos.json`, celda 11; audio E01; un proceso por fila):

    | Modelo · precisión | Compute real | Solo pesos (MB) | Total (MB) | Extra al transcribir (MB) |
    |---|---|---|---|---|
    | small · float16 | float16 | 582 | 884 | 302 |
    | small · int8 | int8_float16 | 358 | 596 | 238 |
    | large-v3 · float16 | float16 | 3078 | 4052 | 974 |
    | large-v3 · int8 | int8_float16 | 1606 | 2580 | 974 |

    int8 casi **divide a la mitad los pesos** de large-v3 (3078 → 1606 MB) y la VRAM total baja 36 % (4052 → 2580 MB). Lo que agrega transcribir es igual en las dos precisiones (974 MB). La línea base (contexto CUDA, 556 MB) entra en el total.
  - En CPU la VRAM no aplica (aunque Colab tenga GPU, lo que se mida ahí no es de ese modelo).
  - Antes de medir con un modelo recién cargado se descarta una corrida de calentamiento (incluye inicializar CUDA).
- **Hallazgo: Colab trae CUDA 13 y CTranslate2 necesita CUDA 12.** CTranslate2 4.8.4 está compilado para CUDA 12 + cuDNN 9; en el Colab actual (CUDA 13) falla al usar la GPU con `Library libcublas.so.12 is not found`.
  - **Arreglo** (celda 3b del notebook): instalar `nvidia-cublas-cu12` y `nvidia-cudnn-cu12==9.*` con pip y **precargar** sus `.so` con `ctypes.CDLL(..., mode=RTLD_GLOBAL)` antes de importar faster-whisper. La misma celda hace una prueba real en GPU (`tiny` sobre 1 s de silencio), así el problema aparece en segundos y no a mitad de la grilla. Si se reinicia la sesión, hay que volver a correrla.
  - **Lección:** que haya GPU en la máquina no garantiza que la librería pueda usarla; la incompatibilidad aparece recién al transcribir. Por eso el notebook prueba una transcripción real antes de la grilla, y el device de cada corrida se lee del modelo y se respalda con la VRAM medida.
  - **Descartado:** bajar a una versión de CTranslate2 para CUDA 11, o cambiar la imagen de Colab. Lo primero obliga a otra versión de faster-whisper; lo segundo no es posible en el plan gratuito.

## 7. Normalización y números en palabras
La misma normalización se aplica a hipótesis y referencia (`src/normalize.py`), en este orden:
1. Minúsculas.
2. Números en palabras → dígitos con `text2num` (`alpha2digit`, `lang="es"`), **con `threshold=0`**: con el valor por defecto, "dos chaufas" quedaba sin convertir.
3. Reglas propias (en `config.yaml → normalizacion`), porque `text2num` falla en:
   - **ordinales y "unos/unas"**: "dos cuartos de pollo" → `2 4ᵒˢ`, "unos seis mil" → `1 6000`. Se protegen esas palabras antes de convertir.
   - **"y medio"**: "dos años y medio" → `2.5 años`, "tres y medio años" → `3.5 años`. Solo con unidades de tiempo, para no tocar "gaseosa de litro y medio" (nombre de producto).
   - **"lucas"**: "seis lucas" → `6000`.
4. **Se quitan las tildes** (decisión): evita penalizar "inglés"/"ingles" entre ASR y referencia; la pérdida de información no afecta a ningún campo.
5. Se quita la puntuación (conservando el punto decimal: `2.5`) y se colapsan espacios.
6. **Separador de miles (hallazgo con el audio real):** antes que todo, `6,500` y `9.000` → `6500` y `9000`. Una coma o un punto seguido de exactamente 3 dígitos es separador de miles; `2.5` sigue siendo decimal.
   - Con las referencias (números en palabras) este caso no existía. Whisper, en cambio, escribe los montos con dígitos y separador: "seis mil quinientos" sale `6,500`.
   - Sin esta regla la coma se volvía espacio (`6 500`) y el punto se leía como decimal (`9.000` = 9). Por eso `pretension_salarial` daba 0 % de acierto con todos los modelos.

## 8. Tarea 1 — Metodología y resultados
Grilla de experimentos (`config.yaml → experimentos`), reducida para que entre en menos de 40 min en una T4 de Colab gratis:

| Combinación | Clips | Compara |
|---|---|---|
| small · cuda · float16 / int8 · plano | todos (L0) | precisión en small |
| large-v3 · cuda · float16 / int8 · plano | todos (L0) | tamaño (vs small float16) y precisión en large |
| large-v3 · cuda · int8 · glosario | todos (L0) | glosario (vs large-v3 int8 plano) |
| small · cpu · int8 · plano | **solo E01 y L01** | CPU vs GPU (vs small cuda int8, mismos clips) |
| large-v3 · cuda · int8 · plano | llamadas en L1–L3 | escalera (Tarea 2) |

La combinación principal (la que vigila el CI y la que abre la app) es **large-v3 · cuda · int8 · plano**.

**Métricas** (`src/metrics.py`, `src/evaluar.py`):
- **WER/CER** con `jiwer` sobre texto normalizado. El WER de un conjunto de clips es Σ errores / Σ palabras (pesa por largo), no el promedio de WER.
- **Intervalos de confianza:** bootstrap remuestreando **clips**, 1000 réplicas, semilla 42, IC 95 %. Las comparaciones usan bootstrap **pareado** (mismos clips en A y B): si el IC de la diferencia incluye 0, la muestra no permite afirmar que haya diferencia.
- **¿Cambió el texto?** Para cada par: cuántos clips salen idénticos carácter a carácter y el WER de una salida contra la otra.
- **Velocidad:** segundos de cómputo por minuto de audio. **VRAM:** pico − base (§6).

**Capa 1 — transcripción por configuración** (`cache/results/asr_resumen.csv`, L0, 4 clips por tarea; device y compute **leídos del modelo**):

| Modelo · device · precisión · glosario | Compute real | WER entrevista (IC 95 %) | WER pedido (IC 95 %) | CER entr. / ped. | s por min de audio (entr.) | VRAM total (MB) |
|---|---|---|---|---|---|---|
| small · cuda · float16 · plano | float16 | 0.203 (0.131–0.295) | 0.310 (0.206–0.426) | 0.153 / 0.226 | 1.89 | 884 |
| small · cuda · int8 · plano | int8_float16 | 0.205 (0.133–0.297) | 0.317 (0.210–0.437) | 0.154 / 0.227 | 2.03 | 596 |
| large-v3 · cuda · float16 · plano | float16 | 0.196 (0.157–0.245) | 0.289 (0.183–0.407) | 0.173 / 0.221 | 6.59 | 4052 |
| **large-v3 · cuda · int8 · plano** | int8_float16 | **0.176** (0.119–0.249) | **0.276** (0.168–0.397) | 0.152 / 0.212 | 7.36 | 2580 |
| large-v3 · cuda · int8 · glosario | int8_float16 | 0.212 (0.103–0.296) | 0.280 (0.173–0.399) | 0.183 / 0.213 | 6.15 | 2580 |
| small · **cpu** · int8 · plano (solo E01/L01) | int8_float32 | 0.100 (1 clip) | 0.215 (1 clip) | 0.048 / 0.136 | **32.5** | — |

**Comparaciones pareadas** (`cache/results/comparaciones.csv`; diferencia = WER(b) − WER(a) sobre los mismos clips, bootstrap pareado):

| Comparación (a → b) | Clips | WER a → b | Diferencia (IC 95 %) | ¿IC incluye 0? | s/min a → b | Clips con texto idéntico | WER entre salidas |
|---|---|---|---|---|---|---|---|
| 8.1 tamaño: small → large-v3 (float16) | 8 | 0.254 → 0.240 | −0.014 (−0.039 a 0.024) | sí | 1.88 → 6.21 | 0 / 8 | 0.104 |
| 8.2 device: cpu → cuda (small int8) | 2 | 0.153 → 0.153 | 0.000 (−0.003 a 0.004) | sí | **31.5 → 2.06** | 0 / 2 | 0.039 |
| 8.3 precisión small: float16 → int8 | 8 | 0.254 → 0.259 | +0.004 (0.001 a 0.008) | **no** | 1.88 → 2.01 | 0 / 8 | 0.035 |
| 8.3 precisión large-v3: float16 → int8 | 8 | 0.240 → 0.224 | −0.016 (−0.044 a 0.002) | sí | 6.21 → 6.54 | 1 / 8 | 0.037 |
| 8.4 glosario: plano → glosario (large-v3 int8) | 8 | 0.224 → 0.244 | +0.020 (−0.011 a 0.072) | sí | 6.54 → 6.05 | 0 / 8 | 0.046 |

Lectura:
- **8.1 Tamaño:** large-v3 baja el WER solo 1,4 puntos (el IC incluye 0) y es 3,3× más lento, con 4,6× más VRAM (884 → 4052 MB en float16). Con esta muestra, el modelo grande **no** demuestra ser mejor en WER.
- **8.2 CPU vs GPU:** la GPU es **15× más rápida** (31,5 → 2,06 s por minuto de audio) con el mismo WER. Pero el **texto cambió en los 2 clips** (WER entre salidas 3,9 %): con el mismo `int8`, la CPU corre `int8_float32` y la GPU `int8_float16`, y esa diferencia numérica cambia algunas palabras.
- **8.3 Precisión:** en small, int8 empeora un poco pero **de forma consistente** (+0,4 puntos; es el único IC que excluye 0). En large-v3, int8 no empeora (−1,6, el IC incluye 0) y ahorra 1472 MB. **En los dos casos cambia el texto** (0/8 y 1/8 clips idénticos): int8 no es "lo mismo pero más rápido". Por eso la combinación principal es large-v3 int8: mismo WER que float16, 36 % menos VRAM.
- **8.4 Glosario:** no hay evidencia de mejora. La diferencia es +2,0 puntos (peor) con un IC de −1,1 a +7,2. Con 8 clips no se puede afirmar ni que ayude ni que dañe.
- **Prueba de GPU:** en cada corrida, `device_real` = `cuda` y `compute_real` = `int8_float16`/`float16` salen del objeto CTranslate2, y cada una tiene VRAM medida > 0 (la tabla de §6). En las corridas CPU la VRAM es nula y el compute es `int8_float32`.

### 8.5 Extracción: reglas vs LLM, verificación literal
- **Reglas** (`extract_rules.py`): cada campo se busca **después de una pregunta ancla** ("pretensión", "inglés"…), así no depende del orden. Para el pedido se usa el **resumen final de la operadora** ("su pedido es… el total es…"), que ya incluye las autocorrecciones; los precios salen de lo que dice la operadora (la carta del config no tiene precios).
- **LLM** (`extract_llm.py`): Google **Gemini `gemini-3-flash-preview`** (nivel gratuito, SDK `google-genai`; se empezó con `gemini-3.8-flash`, pero agotó su cuota diaria antes de terminar), con **temperatura 0** y **salida JSON estructurada**: se le pasa el JSON schema de `schemas.py` (`response_json_schema`), así que solo puede responder con esa forma. Igual se valida y se reintenta.
  - Gemini no acepta `"type": ["number", "null"]` ni `null` dentro de `enum`, así que `esquema_para_gemini` reescribe esos campos como `anyOf`. El esquema de `schemas.py` no cambia.
  - **Nivel gratuito:** pocas solicitudes por minuto. Hay una pausa entre llamadas, y si el API responde 429 se espera y se reintenta (valores en `config.yaml → extraccion.llm`).
  - **Hallazgo: la cuota real es de 20 solicitudes por día y por modelo.** La primera corrida de `--llm` (62 llamadas: todos los grupos y niveles) la agotó. Además cada 429 esperaba 30 s × 6 intentos, así que la corrida no terminaba nunca. Arreglos:
    - Un 429 de cuota **diaria** (`...PerDay...` en el error) falla al toque. Solo se espera el límite por minuto.
    - **Timeout HTTP** (`timeout_s`): sin él, una solicitud sin respuesta bloquea para siempre.
    - **Cache de respuestas** en `cache/llm/` (se commitea). La clave es un hash de modelo + prompts + transcript: cada llamada se paga una vez, una corrida cortada retoma donde quedó y los resultados se reproducen sin API.
    - El LLM corre solo en `evaluacion.grupos_llm`: `referencia` + `large-v3|cuda|int8|plano` en L0 = 16 llamadas, que caben en un día. Las demás combinaciones y la escalera L1–L3 se comparan solo con reglas: **limitación declarada**.
  - **Descartado:** Anthropic (requiere pago). Con su API actual (versión `2026-10-21`) tampoco se podía fijar la temperatura en 0.
- **Verificación literal** (`verify.py`): un campo es `verificado` solo si su valor aparece en el transcript normalizado. Números como número completo (`90` no cuenta dentro de `2890`); frases tal cual; dirección y promoción palabra por palabra; listas elemento por elemento; categorías por sus palabras del config (`PEN` ← "soles"). Si el LLM devolviera 90 como total de L02 (se dijo 80), queda `sin_verificar`.
- **Métricas por campo:** accuracy con IC, % verificado, error absoluto medio en numéricos y macro-F1 en `nivel_ingles` (escrita a mano en `metrics.py`).

**Grupo "referencia"** (transcripts humanos como si fueran el ASR; mide solo la extracción). Sale de `python -m src.evaluar` → `cache/results/extraccion_por_campo.csv`, método reglas, 4 clips por tarea:

| Tarea | Campo | Accuracy (IC 95 %) |
|---|---|---|
| entrevista | tecnologías | 0.75 (0.25–1.00) |
| entrevista | los otros 6 campos | 1.00 |
| pedido | los 6 campos | 1.00 |

El único error es **E01 tecnologías**: la candidata dice "usaba bastante Excel, pero eso ya casi no" y las reglas listan Excel (no entienden negaciones). Ojo: las reglas se escribieron **mirando los guiones**, así que este 100 % es una cota optimista, no evidencia de que generalicen. La prueba real es con la salida del ASR.

**Con ASR** (large-v3 · cuda · int8 · plano, L0; `extraccion_por_campo.csv`; accuracy con IC 95 %, EA = error absoluto medio en numéricos):

| Tarea | Campo | Reglas | LLM | % verificado reglas / LLM |
|---|---|---|---|---|
| entrevista | anios_experiencia | 0.75 (0.25–1) · EA 0 | **1.00** · EA 0 | 75 % / 75 % |
| entrevista | ultimo_puesto | 0.50 (0–1) | 0.50 (0–1) | 50 % / 100 % |
| entrevista | tecnologias | 0.50 (0–1) | **0.75** (0.25–1) | 100 % / 100 % |
| entrevista | pretension_salarial | 1.00 · EA 0 | 1.00 · EA 0 | 100 % / 100 % |
| entrevista | moneda | 1.00 | 1.00 | 100 % / 100 % |
| entrevista | disponibilidad | **1.00** | 0.75 (0.25–1) | 100 % / 100 % |
| entrevista | nivel_ingles (macro-F1) | 1.0 | 1.0 | 100 % / 100 % |
| pedido | items | **1.00** | 0.50 (0–1) | 100 % / 100 % |
| pedido | promocion | 0.75 (0.25–1) | **1.00** | 50 % / 50 % |
| pedido | descuento | 0.75 (0.25–1) · EA 2.5 | **1.00** · EA 0 | 50 % / 75 % |
| pedido | direccion_entrega | 0.75 (0.25–1) | 0.75 (0.25–1) | 75 % / 75 % |
| pedido | metodo_pago (macro-F1) | 1.0 | 1.0 | 100 % / 100 % |
| pedido | total_dicho | 0.75 (0.25–1) · EA 0 | **1.00** · EA 0 | 75 % / 100 % |
| | **registro completo, entrevista** | 0.821 | 0.857 | |
| | **registro completo, pedido** | 0.833 | 0.875 | |

En referencia (sin ASR): reglas 0.964 / 1.000 y LLM 1.000 / 0.833 (entrevista / pedido).

- **Ningún método gana con esta muestra:** todas las diferencias caen dentro de los IC.
  - El **LLM** aguanta mejor los errores del ASR en números y montos. Cuando el ASR cambia la frase que buscan las reglas ("en totales 80"), el LLM igual entiende.
  - Las **reglas** ganan en `items`. El LLM acierta todas las cantidades y precios, pero escribe el producto a su manera ("chaufas", "gaseosa" + "litro y medio") en vez del nombre de la carta.
- **La verificación literal atrapó al LLM corrigiendo al ASR (E03).** En E03 el ASR **se saltó** "dos años y medio" (solo quedó "empecé a mitad de año, así que no llegan a tres"). El LLM igual devolvió `2.5`: lo dedujo de esa frase, y esta vez acertó. Como `2.5` no está en el transcript, el campo quedó **`sin_verificar`**. Por eso `anios_experiencia` del LLM tiene accuracy 1.00 y solo 75 % verificado. Es justo el caso que la regla existe para marcar: la próxima deducción puede salir mal.

**¿El WER dice si el registro sirve? No** (large-v3 int8, L0, `asr_por_clip.csv` + `extraccion_por_clip.csv`):

| Clip | WER | Campos correctos (reglas) | Lectura |
|---|---|---|---|
| **L04** | **44,7 %** (el más alto del corpus) | **6 / 6** | WER alto, registro perfecto |
| **L02** | **14,8 %** | **4 / 6** | WER bajo, registro roto |
| E01 | 9,3 % (el más bajo) | 6 / 7 | el error es de extracción ("Excel, pero ya casi no"), no del ASR |

- **L04:** de sus 102 errores, 59 son **inserciones**, es decir, palabras que el ASR escribió y la referencia no tiene (frases improvisadas al grabar y no corregidas, §12; más alguna repetición del ASR). Aun así los números, los productos, la dirección y el pago salen bien.
- **L02** tiene pocos errores, pero caen justo en dos campos: el total ("en **totales** 80" → sin total no hay pedido verificable) y la dirección ("**Girón Guayaga**" por "Jr. Huallaga").
- **El WER cuenta todas las palabras por igual. El registro depende de unas pocas** (un número, un nombre). Quien reporta solo WER puede llamar "bueno" a un sistema que rompe el dato que importa, y "malo" a uno que lo saca perfecto.
- **Asimetría entre tipos de campo:**
  - Un campo **categórico** sobrevive a mucho error: `nivel_ingles` y `metodo_pago` tienen macro-F1 1.0 con large-v3 en L0 (con small, `metodo_pago` baja a 0.67), porque basta con oír "avanzado" o "tarjeta" una vez.
  - Un **monto** no sobrevive a nada: `pretension_salarial` dio **0 %** con todos los modelos hasta agregar la regla del separador de miles (§7), porque Whisper escribe `6,500` y un solo carácter rompía el número.
  - El **texto libre** es el más difícil de verificar: `ultimo_puesto` 0.50 con los dos métodos.

## 9. Tarea 2 — Metodología y resultados
### 9.1 Escalera de degradación L0–L3 (SNR medido)
`python -m src.degrade` (parámetros en `config.yaml → degradacion`):

| Nivel | Qué simula | Cómo |
|---|---|---|
| L0 | original | — |
| L1 | WhatsApp | mono, Opus 12 kbps |
| L2 | teléfono | paso banda 300–3400 Hz, 8 kHz |
| L3 | L1 + ruido ambiente | ruido grabado mezclado a SNR objetivo 10 dB |

El SNR de L3 **se calcula**: el ruido se escala para que `20·log10(RMS señal / RMS ruido)` sea el objetivo, y el SNR medido de cada clip queda en `cache/results/degradacion.json`. Si L3 sigue sin errores, se baja a 5 o 0 dB (`snr_db_alternativos`) y se documenta. La escalera se transcribe con large-v3 · cuda · int8 · plano (la combinación principal).

**Por qué 10 dB, y cómo se midió:**
- **Por qué:** a 10 dB la voz tiene 10 veces más potencia que el ruido. Es el caso realista de alguien que pide por teléfono en un local o en la calle: el ruido se oye claro, pero la voz domina. A 0 dB (ruido igual de fuerte que la voz) ya no es una llamada que una persona aceptaría, y a 20 dB casi no degrada.
- **Cómo se mide:** RMS de la señal (L1) y RMS del ruido grabado, sobre las muestras a 16 kHz. El ruido se escala para que `20·log10(RMS_señal / RMS_ruido)` dé 10 dB, y luego se **re-mide** sobre lo que se mezcló. SNR medido: **10,0 dB en los 4 clips** (`degradacion.json`).
- **¿Hubo que endurecer?** No: a 10 dB el registro ya se rompe (accuracy 0.54 en L3), así que no se usaron 5 ni 0 dB.

**Curva de la escalera** (`asr_resumen.csv` y `extraccion_por_campo.csv`; extracción por reglas; 4 llamadas):

| Nivel | WER (IC 95 %) | Accuracy del registro | items | total_dicho | metodo_pago | direccion_entrega |
|---|---|---|---|---|---|---|
| L0 original | 0.276 (0.168–0.397) | **0.833** | 1.00 | 0.75 | 1.00 | 0.75 |
| L1 WhatsApp | 0.408 (0.288–0.563) | 0.625 | 0.50 | 0.50 | 0.75 | 0.50 |
| L2 teléfono | 0.463 (0.430–0.509) | 0.625 | 0.50 | 0.25 | 0.75 | 0.75 |
| L3 L1 + ruido 10 dB | 0.469 (0.371–0.571) | **0.542** | 0.50 | 0.25 | 0.50 | 0.50 |

El WER sube 19 puntos entre L0 y L3 y el registro pierde 29. `total_dicho` es el campo que más cae (0.75 → 0.25): justo el que necesita la validación. El LLM solo corrió en L0 (cuota, §8.5).

### 9.2 Validación aritmética
`validate.chequear`: `Σ cantidad × precio_unitario − descuento` contra `total_dicho`, con tolerancia de 0.5 soles. Sobre las referencias detecta **L02, L03 y L04** y L01 cuadra.

**Capturadas vs perdidas por nivel** (`cache/results/capturas_por_nivel.csv`, large-v3 int8; 3 trampas de total mal dicho: L02, L03, L04):

| Grupo · método | Capturadas | Perdidas | Falsas alarmas | Sin chequeo (faltó un precio o el total) |
|---|---|---|---|---|
| referencia · reglas | 3 | 0 | 0 | 0 |
| L0 · reglas | 1 | 1 | 0 | 1 |
| L0 · **LLM** | **3** | 0 | 0 | 0 |
| L1 · reglas | 2 | 0 | 0 | 2 |
| L2 · reglas | 1 | 0 | 0 | 3 |
| L3 · reglas | 1 | 0 | 0 | 3 |

- **Lo que atrapa:** cualquier total que no cuadre con lo que se dijo antes (L04 se atrapa en **todos** los niveles).
- **Lo que pierde:**
  - **L03 en L0:** el ASR borró la promoción "2×1", y sin ella la cuenta de lo transcrito da justo 75, el total mal dicho. El error del hablante y el del ASR se cancelan.
  - **Las llamadas sin chequeo:** si el ASR rompe un precio o el total, no hay nada que comparar. Al degradar el audio, el problema principal no es dejar pasar errores, es quedarse **sin poder chequear** (de 1 a 3 llamadas sin chequeo).
- **Cero falsas alarmas** en todos los niveles: L01, que suma bien, nunca se marcó como error.

### 9.3 Diagnóstico: error de ASR vs error del hablante
Si la cuenta no cuadra, en este orden:
1. Si la misma combinación cuadraba en L0 y en este nivel no → **error de ASR** (lo rompió la degradación).
2. Se buscan las palabras del ASR que forman cada número implicado (total, precios, descuento, cantidades > 1) y se toma su probabilidad mínima. Si alguna está bajo 0.80 → **error de ASR**.
3. Si todos los números tienen alta confianza → **error del hablante** (se dijo así).

Se mide contra las trampas: lo esperado es "error del hablante" en L02–L04 y "ok" en L01 (o "error de ASR" si deja de cuadrar por la degradación).

**Cuántas veces acierta** (`pedidos_validacion.csv`; solo se evalúan las llamadas que tienen un chequeo; "indeterminado" = sin datos para decidir):

| Grupo · método | Aciertos / evaluados |
|---|---|
| L0 · reglas | 2 / 3 |
| L0 · LLM | 2 / 4 |
| L1 · reglas | 1 / 2 |
| L2 · reglas | 1 / 1 |
| L3 · reglas | 1 / 1 |
| **Total escalera (reglas)** | **5 / 7** |

- **Acierta siempre en L04** (el total mal dicho, con números claros: "todos los números con prob ≥ 0.80, se dijo así") y en L01 (cuadra).
- **Falla de dos maneras:**
  - **Error del hablante tomado como error del ASR:** Whisper dudó en un precio que sí se dijo bien (L02 "precio ensalada"; L03 con el LLM). Basta una palabra con baja probabilidad para culpar al ASR.
  - **Errores que se cancelan:** L03 en L0, donde la cuenta cuadra por accidente (ver arriba).
- **Lección:** la regla favorece culpar al ASR. Con 4 llamadas no se puede ajustar el umbral de 0.80 sin sobreajustarlo a este corpus.

### 9.4 Un motor, dos esquemas
Las dos tareas llaman a **la misma función**, `process_audio(audio_path, schema, cfg)`. Solo cambia el argumento `schema`:

```python
from src.engine import process_audio
entrevista = process_audio("data/audio/entrevistas/E01.m4a", "entrevista")
pedido     = process_audio("data/audio/llamadas/L01.m4a", "pedido")
```

- El **ASR, la normalización, la verificación literal y el cache son idénticos** para las dos.
- Lo que depende del esquema sale de **datos**, no de código copiado: los campos y el JSON schema (`schemas.py`), los prompts y el vocabulario de las reglas (`config.yaml`), y el glosario por esquema.
- La validación aritmética (`validate.py`) es un paso **posterior** que solo aplica a `pedido`.
- No se copió ni se editó el motor para la Tarea 2. **Ventaja:** cambiar el modelo (`motor.modelo` en el config) cambia las dos tareas a la vez, con una sola grilla y un solo cache.

## 10. Cómo reproducir
1. **Grabar** y subir a Drive: `MyDrive/HW05/audio/{entrevistas,llamadas,ruido}/`.
2. **Transcribir en Colab GPU:** abrir `notebooks/colab_gpu_benchmark.ipynb` con entorno T4. Instala las dependencias y el arreglo de CUDA 12 (celda 3b, ver §6), degrada las llamadas, recorre la grilla, guarda el cache en Drive (si se desconecta, retoma sin re-transcribir) y descarga `cache_hw05.zip`.
3. **En local:** descomprimir el zip en la raíz del repo y correr
   ```bash
   python -m src.evaluar            # + --llm para incluir la extracción con LLM
   pytest
   ```
4. Commit de `cache/` (los audios no se suben).

Tests: `pytest` (rápidos, los corre el CI) y `pytest -m lento` (descarga Whisper `tiny` y transcribe de verdad en CPU).

## 11. App
```bash
streamlit run app.py
```
Carga los resultados precalculados de `cache/results/` y **no transcribe el corpus al arrancar** (el test `tests/test_app.py` lo comprueba: falla si la app intenta transcribir). Si `python -m src.evaluar` reescribe los resultados, la app los toma sola en la siguiente interacción (lee cada tabla con su fecha de modificación como clave del cache).

| Pestaña | Contenido |
|---|---|
| 1 · Problema y método | por qué no basta el celular, diagrama del pipeline, método |
| 2 · Corpus | reproductor de audio (si el audio está en la PC) con el transcript de referencia y el del ASR al lado; registro extraído contra el ground truth; distribución de cada campo |
| 3 · Resultados | capa 1 (WER/CER con IC, comparaciones pareadas) y capa 2 (accuracy por campo con IC, macro-F1, error absoluto, % verificado); WER del clip contra accuracy de su registro |
| 4 · Probar un audio | subir un clip (wav, mp3, m4a, mp4, aac, ogg, opus, flac: `app.probador_tipos`) y ver transcript, registro, campos verificados, device real y tiempo. Es lo único que transcribe (`app.probador` en el config); su cache queda fuera del repo |
| 5 · Pedidos (Tarea 2) | escalera con SNR medido, curva WER/accuracy por nivel, capturadas vs perdidas por nivel y chequeo aritmético en vivo |

En la barra lateral se elige qué transcripción mirar (cada combinación de ASR en cache, o `referencia`) y el método de extracción.

**App publicada:** _(enlace de Streamlit Community Cloud: pendiente)_. Funciona sin audios ni GPU: lee `cache/` del repo; el reproductor de audio solo aparece donde el audio existe localmente, y el probador transcribe en CPU.

## Innovación: regresión de calidad en GitHub Actions
Cada push y cada PR corre `.github/workflows/regression.yml`, que:
1. instala dependencias y ffmpeg, verifica que el motor no importe UI y corre `pytest`;
2. corre `python -m src.evaluar --verificar`: **recalcula todas las métricas desde los transcripts cacheados** (nunca transcribe: los runners no tienen GPU) y **falla** si la combinación vigilada (`regresion.combinacion`, hoy large-v3 · int8) queda bajo los umbrales de `config.yaml`: WER > `wer_max` (0.30) o accuracy media de campos < `accuracy_campos_min` (0.70).

**Comprobado que de verdad falla:**
- **En GitHub:** el PR de demostración [#24](https://github.com/JazminGalindoo/HW05_VOZ_A_REGISTRO/pull/24) solo bajó `wer_max` a 0.10. La [corrida](https://github.com/JazminGalindoo/HW05_VOZ_A_REGISTRO/actions/runs/41125632566) quedó **en rojo** con `FALLA: WER entrevista … = 0.177 > 0.1` y `FALLA: WER pedido … = 0.277 > 0.1`. El PR se cerró sin mergear.
- **En los tests:** `tests/test_evaluar.py` corre `--verificar` sobre el cache real con umbrales imposibles (WER 0.01, accuracy 0.99) y exige que salga con código 1; con los umbrales del repo, código 0.

Así, un cambio que empeore la normalización, las reglas o el cache (por ejemplo, volver a romper el separador de miles) no se puede mergear sin que se vea.

## 12. Limitaciones
- **Muestra muy chica:** 4 clips por tarea. Los IC son anchos y la muestra **no** soporta afirmar diferencias pequeñas entre modelos o configuraciones; solo diferencias grandes y consistentes.
- **Dos voces, un acento, un dominio por tarea:** nada de esto dice cómo rinde con otros hablantes o ruidos.
- **Reglas ajustadas a los guiones** (§8.5): su acierto sobre referencias es optimista.
- **Reglas ajustadas después de ver la salida del ASR:** se agregaron 3 formas de decir que aparecieron al grabar ("pollo entero **está a** 64 soles", "el total **sería de** 98", "su pedido **sería**…"), además del separador de miles (§7). Son frases comunes del español y no errores del ASR, pero se detectaron con este mismo corpus: los resultados con ASR también son algo optimistas.
  - **No** se adaptaron las reglas a los errores del ASR ("en **totales** 80", "**Girón Guayaga**… encercado", "desarrolladora de **bacnet**", "las **doceras** están a" por "las gaseosas están dos por uno"). Esos errores son el hallazgo: rompen el registro aunque el WER sea moderado.
- **Referencias = guiones, corregidos solo en números y valores de campos.** Al grabar se improvisaron frases ("¿algo más que desee?", "el total sería de…").
  - **Lo que se corrigió escuchando el audio:** solo las diferencias que tocan un número o el valor de un campo. Se listan en [docs/revisar_numeros_y_campos.md](docs/revisar_numeros_y_campos.md), con clip y minuto:segundo; ahí también se anota qué se dijo.
    - **Revisados (4 de 16):**
      - L04: chaufa a **25** soles (no 28) → corregido.
      - E03 0:42: "dos años y medio, empecé a mitad de año", sin el "mmm" del guion → corregido.
      - L04 1:34: "¿el pollo lo puede **trozar** en piezas? Es para compartir" → corregido.
      - E03 1:20 ("HubSpot") y L03 0:27 ("dos gaseosas de litro y medio… a diez soles cada una"): se dijo como en el guion. Es **error del ASR**, la referencia no cambia.
    - **No revisados (12 de 16):** los demás fragmentos de esa lista no se escucharon. Si alguno fue improvisación, la referencia tiene ahí un valor que no se dijo y el acierto de ese campo está **subestimado**. No se puede descartar.
  - **Lo que NO se corrigió:** las demás diferencias entre guion y audio (muletillas, frases de cortesía, preguntas reformuladas). Están en [docs/revisar_referencias.md](docs/revisar_referencias.md) (115 fragmentos).
  - **Consecuencia: el WER está inflado.** Mide el error del ASR **más** lo improvisado que no se corrigió. Las métricas del **registro** se afectan mucho menos, porque las diferencias en valores de campos se revisaron (salvo las 12 no escuchadas, ver arriba). Las comparaciones **pareadas** entre configuraciones (tamaño, precisión, glosario, CPU/GPU) siguen siendo válidas: todas se miden contra la misma referencia.
  - **Además, large-v3 se salta frases completas**, sobre todo al final de algunos clips (E02, E04, L01). Son borrados reales del ASR, no improvisaciones.
- **Ground truth anotado por una sola persona** y guiones leídos (habla menos espontánea que una llamada real).
- **El diagnóstico usa probabilidades de Whisper**, que no están calibradas; el umbral de 0.80 es una elección, no un valor ajustado.
- **LLM solo en L0 y en un modelo de ASR** por la cuota gratuita (§8.5). Las comparaciones en la escalera son solo con reglas.
- **Historial de commits concentrado:** el repositorio se creó el 26-oct y la mayoría de los commits son del 30-oct.

### Para qué NO usar este sistema
- **Para decidir sobre personas.** No sirve para filtrar, puntuar ni rechazar candidatos automáticamente: un `ultimo_puesto` acierta la mitad de las veces y un error del ASR cambia un sueldo.
- **Para cobrar o despachar sin que una persona confirme.** El pedido trae marcas de `sin_verificar` y un chequeo aritmético precisamente para que alguien revise. Un pedido que cuadra no está garantizado (ver L03: dos errores que se cancelan).
- **Como prueba de lo que alguien dijo** (reclamos, disputas, cosas legales). Es una transcripción automática con errores medidos de 18–47 % de WER.
- **Con datos reales de personas** sin su consentimiento y sin cumplir la ley de protección de datos personales. Además, la extracción con LLM manda el texto a un servicio externo (Gemini).
- **Fuera de lo que se midió:** otros acentos, otros idiomas, más de dos voces, otros dominios o audio peor que L3. Nada de eso está evaluado.

## 13. Uso de IA
- **Claude Code en VS Code (agente):** escribió el código, los tests y los PRs siguiendo `CLAUDE.md`, con un PR por bloque. Solo mergeaba con el CI en verde.
- **Claude (chat):** lo usé para planificar el proyecto, escribir los guiones de las entrevistas y llamadas, y preparar el guion del video.
- **Gemini (`gemini-3-flash-preview`):** es el LLM de extracción del pipeline (§8), no un asistente de desarrollo.
- **Servidores MCP:** no conecté ninguno en Claude Code. Para GitHub usé la CLI `gh`.
- **Lo que hice yo:** grabé todos los audios, corrí el notebook en Colab, escuché los fragmentos y corregí las referencias (§12), y tomé las decisiones de diseño y de alcance.
