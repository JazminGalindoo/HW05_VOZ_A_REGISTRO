"""
degrade.py — Escalera de degradación L0–L3 (Tarea 2), reproducible.

Responsabilidad:
    - L0: audio original.
    - L1: tipo WhatsApp (mono, Opus a bajo bitrate) con ffmpeg.
    - L2: tipo teléfono (paso banda 300–3400 Hz, remuestreo a 8 kHz).
    - L3: L1 + ruido ambiente mezclado a un SNR objetivo; el SNR real se
      CALCULA con el RMS de señal y de ruido y se reporta.
    - Todos los parámetros vienen de config.yaml.
"""
