"""
asr.py — Transcripción con faster-whisper.

Responsabilidad:
    - Cargar WhisperModel con el tamaño, device y compute_type pedidos en config.yaml.
    - Transcribir en español (idioma fijo), con o sin glosario
      (initial_prompt / hotwords según la versión instalada).
    - Devolver texto, segmentos y palabras con su probabilidad.
    - Medir en runtime: device REAL (del objeto del modelo / ctranslate2),
      tiempo transcurrido y VRAM pico con pynvml (hilo que muestrea cada ~50 ms;
      torch.cuda.max_memory_allocated NO sirve porque CTranslate2 no usa PyTorch).
"""
