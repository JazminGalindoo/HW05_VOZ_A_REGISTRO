# HW05_VOZ_A_REGISTRO
ASR con faster-whisper: de entrevistas y llamadas de venta grabadas a registros estructurados y verificados. HW_05 Data Science Python.

> Esqueleto: las secciones marcadas con _(pendiente)_ se completan en cada fase con resultados que salen de scripts re-ejecutables.

## 1. Descripción
- **Tarea 1 — Entrevistas:** audio → registro de candidato con 6 campos verificados.
- **Tarea 2 — Llamadas de venta:** audio → registro de pedido que se valida aritméticamente.

Ambas tareas comparten un solo motor: `src.engine.process_audio(audio_path, schema, cfg)`.

## 2. Instalación
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # y completar la API key
```
También se necesita `ffmpeg` instalado en el sistema (para `src/degrade.py`).

## 3. Verificación: el motor no importa UI
```bash
python -c "import sys; import src.engine; bad={'streamlit','gradio','telegram'}&set(sys.modules); assert not bad, bad; print('OK: engine sin UI')"
```

## 4. Estructura del repo
_(pendiente)_

## 5. Datos y ground truth
_(pendiente: cómo se grabó, guiones en `scripts_grabacion/`, tablas en `data/ground_truth/`, grupo sintético separado)_

## 6. Versión de faster-whisper y glosario
_(pendiente: versión exacta instalada y salida de `inspect.signature(WhisperModel.transcribe)` — ¿existe `hotwords`?)_

## 7. Normalización y números en palabras
_(pendiente: reglas de normalización, decisión sobre tildes, casos que `text2num` no resuelve)_

## 8. Tarea 1 — Resultados
### 8.1 Modelos (small vs large-v3)
### 8.2 CPU vs GPU (prueba de device en runtime, VRAM con pynvml)
### 8.3 Precisión (float16 vs int8) y diff de texto
### 8.4 Glosario vs plano (diferencia de WER con IC bootstrap)
### 8.5 Extracción: reglas vs LLM, verificación literal

## 9. Tarea 2 — Resultados
### 9.1 Escalera de degradación L0–L3 (SNR medido)
### 9.2 Validación aritmética
### 9.3 Diagnóstico: error de ASR vs error del hablante

## 10. Cómo reproducir
_(pendiente: comandos para regenerar `cache/` y métricas; notebook de Colab GPU)_

## 11. App
_(pendiente: `streamlit run app.py`, lee solo de `cache/`)_

## 12. Limitaciones
_(pendiente: tamaño de muestra, qué no soportan los IC)_

## 13. Uso de IA
_(pendiente)_
