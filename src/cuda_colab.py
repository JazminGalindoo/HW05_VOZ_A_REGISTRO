"""
cuda_colab.py — Arreglo de CUDA 12 para CTranslate2 en Colab (que trae CUDA 13).

Responsabilidad:
    - CTranslate2 4.8.4 (motor de faster-whisper) está compilado para CUDA 12 + cuDNN 9.
      En Colab con CUDA 13 falla al usar la GPU: "Library libcublas.so.12 is not found".
    - Las librerías de CUDA 12 se instalan como paquetes de pip
      (nvidia-cublas-cu12, nvidia-cudnn-cu12==9.*) y aquí se PRECARGAN con ctypes,
      para que CTranslate2 las encuentre. Hay que llamarlo ANTES de importar faster_whisper,
      y en cada proceso nuevo (el kernel del notebook y cada `!python -m ...`).
    - En Windows o si los paquetes no están instalados, no hace nada.
"""

import ctypes
import glob
import os
import site


def precargar_cuda12() -> int:
    """Precarga libcublas/libcudnn de los paquetes nvidia-*-cu12. Devuelve cuántas cargó."""
    libs = []
    for carpeta in site.getsitepackages():
        for paquete in ["cublas", "cudnn"]:
            libs += sorted(glob.glob(os.path.join(carpeta, "nvidia", paquete, "lib", "lib*.so*")))

    # RTLD_GLOBAL: los símbolos quedan visibles para CTranslate2. Algunas librerías
    # dependen de otras, así que se reintenta hasta que carguen todas.
    pendientes = libs
    for _ in range(3):
        fallan = []
        for so in pendientes:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                fallan.append(so)
        pendientes = fallan
    return len(libs) - len(pendientes)
