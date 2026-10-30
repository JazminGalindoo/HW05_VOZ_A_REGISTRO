# HW05_VOZ_A_REGISTRO
ASR con faster-whisper: de entrevistas y llamadas de venta grabadas a registros estructurados y verificados. HW_05 Data Science Python.

> **Estado:** el pipeline completo está implementado y probado (tests + ensayo en CPU). Faltan los audios reales: las secciones de resultados (8 y 9) se llenan con `python -m src.evaluar` cuando el cache de Colab esté en el repo. Ningún número de resultados de este README está escrito a mano.

**Documento del pipeline:** [Notion — Pipeline HW05: De la voz al registro](https://gamy-maple-386.notion.site/Pipeline-HW05-De-la-voz-al-registro-6754285ae77e83c2873f010292a6b284) · versión en el repo: [docs/pipeline.md](docs/pipeline.md) (qué corre una vez, qué corre por cada audio, decisiones y lo descartado).

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
cp .env.example .env   # y completar ANTHROPIC_API_KEY (solo para la extracción con LLM)
```
También se necesita **ffmpeg** en el sistema (`winget install Gyan.FFmpeg` en Windows; ya viene en Colab).

Versiones relevantes (fijadas en `requirements.txt`):

| Paquete | Versión | Nota |
|---|---|---|
| faster-whisper | 1.2.1 | `WhisperModel.transcribe` **sí** tiene `hotwords` (ver §6) |
| ctranslate2 | 4.8.4 | motor de inferencia de faster-whisper |
| av (PyAV) | 18.1.0 | **fijada a propósito**: la 19.x quitó `metadata_errors`, que faster-whisper 1.2.1 usa → `TypeError` al decodificar |
| anthropic | 2.3.0 | usa la versión de API `2026-10-21` (ver §8.5) |
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
app.py                   # interfaz (pendiente); solo lee cache/
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
| L04 | no | **98** | 90 |

- En las entrevistas los campos se preguntan **en distinto orden** en cada clip (columna `orden_campos`), para que la extracción no dependa del orden.
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

### 8.1 Modelos (small vs large-v3)
_(pendiente: `cache/results/comparaciones.csv`, fila "tamaño")_
### 8.2 CPU vs GPU
_(pendiente: fila "device"; device real en `asr_resumen.csv`)_
### 8.3 Precisión (float16 vs int8) y diff de texto
_(pendiente: filas "precisión")_
### 8.4 Glosario vs plano (diferencia de WER con IC bootstrap)
_(pendiente: filas "glosario")_

### 8.5 Extracción: reglas vs LLM, verificación literal
- **Reglas** (`extract_rules.py`): cada campo se busca **después de una pregunta ancla** ("pretensión", "inglés"…), así no depende del orden. Para el pedido se usa el **resumen final de la operadora** ("su pedido es… el total es…"), que ya incluye las autocorrecciones; los precios salen de lo que dice la operadora (la carta del config no tiene precios).
- **LLM** (`extract_llm.py`): Anthropic `claude-haiku-5-5` con **structured outputs** (el API garantiza un JSON que cumple el esquema de `schemas.py`), validación propia y reintentos.
  - **Desviación del plan: no hay "temperatura 0".** Desde la versión de API `2026-10-21` (SDK `anthropic` 2.x) enviar `temperature` da error 400 y los modelos actuales solo aceptan el valor por defecto. La estabilidad de la salida viene del JSON schema y de `effort: low`; y la verificación literal impide que una salida distinta "corrija" al ASR.
- **Verificación literal** (`verify.py`): un campo es `verificado` solo si su valor aparece en el transcript normalizado. Números como número completo (`90` no cuenta dentro de `2890`); frases tal cual; dirección y promoción palabra por palabra; listas elemento por elemento; categorías por sus palabras del config (`PEN` ← "soles"). Si el LLM devolviera 90 como total de L02 (se dijo 80), queda `sin_verificar`.
- **Métricas por campo:** accuracy con IC, % verificado, error absoluto medio en numéricos y macro-F1 en `nivel_ingles` (escrita a mano en `metrics.py`).

**Grupo "referencia"** (transcripts humanos como si fueran el ASR; mide solo la extracción). Sale de `python -m src.evaluar` → `cache/results/extraccion_por_campo.csv`, método reglas, 4 clips por tarea:

| Tarea | Campo | Accuracy (IC 95 %) |
|---|---|---|
| entrevista | tecnologías | 0.75 (0.25–1.00) |
| entrevista | los otros 6 campos | 1.00 |
| pedido | los 6 campos | 1.00 |

El único error es **E01 tecnologías**: la candidata dice "usaba bastante Excel, pero eso ya casi no" y las reglas listan Excel (no entienden negaciones). Ojo: las reglas se escribieron **mirando los guiones**, así que este 100 % es una cota optimista, no evidencia de que generalicen. La prueba real es con la salida del ASR.

_(pendiente: resultados con ASR y con LLM)_

## 9. Tarea 2 — Metodología y resultados
### 9.1 Escalera de degradación L0–L3 (SNR medido)
`python -m src.degrade` (parámetros en `config.yaml → degradacion`):

| Nivel | Qué simula | Cómo |
|---|---|---|
| L0 | original | — |
| L1 | WhatsApp | mono, Opus 12 kbps |
| L2 | teléfono | paso banda 300–3400 Hz, 8 kHz |
| L3 | L1 + ruido ambiente | ruido grabado mezclado a SNR objetivo 10 dB |

El SNR de L3 **se calcula**: el ruido se escala para que `20·log10(RMS señal / RMS ruido)` sea el objetivo, y el SNR medido de cada clip queda en `cache/results/degradacion.json`. Si L3 sigue sin errores, se baja a 5 o 0 dB (`snr_db_alternativos`) y se documenta. Cada nivel se transcribe con small y large-v3 (cuda, float16, plano).

_(pendiente: WER y accuracy por nivel)_

### 9.2 Validación aritmética
`validate.chequear`: `Σ cantidad × precio_unitario − descuento` contra `total_dicho`, con tolerancia de 0.5 soles. Sobre las referencias detecta **L02, L03 y L04** y L01 cuadra.

_(pendiente: con ASR, por nivel — `cache/results/pedidos_validacion.csv`)_

### 9.3 Diagnóstico: error de ASR vs error del hablante
Si la cuenta no cuadra, en este orden:
1. Si la misma combinación cuadraba en L0 y en este nivel no → **error de ASR** (lo rompió la degradación).
2. Se buscan las palabras del ASR que forman cada número implicado (total, precios, descuento, cantidades > 1) y se toma su probabilidad mínima. Si alguna está bajo 0.80 → **error de ASR**.
3. Si todos los números tienen alta confianza → **error del hablante** (se dijo así).

Se mide contra las trampas: lo esperado es "error del hablante" en L02–L04 y "ok" en L01 (o "error de ASR" si deja de cuadrar por la degradación).

_(pendiente: aciertos del diagnóstico)_

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
| 4 · Probar un audio | subir un clip y ver transcript, registro, campos verificados, device real y tiempo. Es lo único que transcribe (`app.probador` en el config); su cache queda fuera del repo |
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
- **Referencias = guiones, no lo dicho palabra por palabra:** al grabar se improvisaron frases ("¿algo más que desee?", "el total sería de…"). Si no se corrigen los transcripts de referencia escuchando el audio, el WER mide el error del ASR **más** la diferencia entre el guion y lo dicho, y queda inflado.
- **Ground truth anotado por una sola persona** y guiones leídos (habla menos espontánea que una llamada real).
- **El diagnóstico usa probabilidades de Whisper**, que no están calibradas; el umbral de 0.80 es una elección, no un valor ajustado.

## 13. Uso de IA
_(pendiente de completar)_
El código se desarrolló con Claude Code (Anthropic) como asistente, módulo por módulo, con un PR por bloque, tests y revisión de cada cambio. Las decisiones de diseño que se tomaron en el camino están documentadas en este README (§1, §2, §7, §8.5).
