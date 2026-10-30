"""Tests de normalize.py con frases reales de los guiones."""

import pytest

from src.normalize import normalizar, quitar_tildes


@pytest.mark.parametrize("entrada, esperado", [
    # Números compuestos
    ("unos seis mil quinientos soles mensuales", "unos 6500 soles mensuales"),
    ("Avenida Los Álamos mil doscientos cuarenta y cinco, en Surco.", "avenida los alamos 1245 en surco"),
    ("El total es ochenta y nueve soles.", "el total es 89 soles"),
    ("Cuatro mil dólares mensuales.", "4000 dolares mensuales"),
    # Números chicos (con el umbral por defecto de text2num no se convertían)
    ("que sean dos chaufas nomás", "que sean 2 chaufas nomas"),
    ("Un pollo entero, por favor.", "1 pollo entero por favor"),
    ("dos por uno", "2 por 1"),
    # Ordinales protegidos
    ("dos cuartos de pollo", "2 cuartos de pollo"),
    ("a partir del primero de diciembre", "a partir del primero de diciembre"),
    ("el primer mes", "el primer mes"),
    # Coloquialismos (reglas propias)
    ("Mmm, dos años y medio.", "mmm 2.5 años"),
    ("tres y medio años", "3.5 años"),
    ("seis lucas", "6000"),
    # "litro y medio" es nombre de producto: no se toca
    ("dos gaseosas de litro y medio", "2 gaseosas de litro y medio"),
])
def test_normalizar(cfg, entrada, esperado):
    # quitar_tildes también se aplica al esperado ("años" -> "anos")
    assert normalizar(entrada, cfg) == quitar_tildes(esperado)


@pytest.mark.parametrize("entrada, esperado", [
    # Así escribe Whisper los montos (encontrado con el audio real)
    ("unos 6,500 soles mensuales", "unos 6500 soles mensuales"),
    ("mi pretensión es de 9.000 soles", "mi pretension es de 9000 soles"),
    ("1.234.567", "1234567"),
    ("2.5 años", "2.5 anos"),        # decimal: no se toca
    ("12,5 soles", "12 5 soles"),    # no son 3 dígitos: no es separador de miles
])
def test_separador_de_miles(cfg, entrada, esperado):
    assert normalizar(entrada, cfg) == esperado


def test_puntuacion_y_espacios(cfg):
    assert normalizar("¿Me  escucha bien?… ¡Sí!", cfg) == "me escucha bien si"


def test_misma_salida_para_hipotesis_y_referencia(cfg):
    # Si el ASR escribe dígitos y la referencia palabras, deben quedar iguales
    assert normalizar("Son 4 años", cfg) == normalizar("son cuatro años", cfg)
