# Revisar SOLO números y valores de campos contra el audio

Subconjunto de `revisar_referencias.md`: solo los fragmentos donde lo que difiere es un número o una palabra que forma el valor de un campo (ground truth del clip o catálogo del config). El resto de diferencias no se corrige: ver README §12.

Generado por `python scripts_grabacion/revisar_referencias.py`. **No editar a mano.**

Los transcripts de referencia salen del guion, pero al grabar se improvisó. Para cada fragmento: escuchar el audio en el minuto indicado y anotar qué se dijo de verdad. La corrección se hace en `scripts_grabacion/guiones.md` (una persona, nunca copiando al ASR) y luego se corre `python scripts_grabacion/generar_referencias.py`.

- **Guion**: lo que dice la referencia hoy. **ASR**: lo que escribió `large-v3__cuda__int8__plano__L0`. Todo normalizado (minúsculas, sin tildes, números en dígitos); en **negrita** lo que difiere; ∅ = nada.
- **¿Otros modelos igual?** sí = large-v3 float16 y small también lo escribieron así → probablemente se **dijo así** (improvisación). no = probablemente **error del ASR**. Es una pista: decide el oído.
- Tiempos aproximados (±3 s).

## E01 — 0 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|

## E02 — 2 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 1:02 | todo corria en kubernetes **y cual fue su ultimo puesto exactamente desarrolladora backend** en 1 fintech me | todo corria en kubernetes **desarrolladora de bacnet** en 1 fintech me | no | |
| 2 | 1:56 | son en el equipo **son 6 desarrolladores y 1 lider tecnica perfecto gracias gracias a usted daniela le avisamos el lunes** | son en el equipo **desarrolladores desarrolladores perfecto muchas gracias** | no | |

## E03 — 2 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:42 | experiencia tiene en marketing **2.5 anos** empece a mitad de | experiencia tiene en marketing **∅** empece a mitad de | sí | |
| 2 | 1:20 | correos los mandabamos por **hubspot** y las imagenes las | correos los mandabamos por **∅** y las imagenes las | sí | |

## E04 — 1 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 1:52 | o tendria que contratar **hay 4 personas y tendria que contratar 2 mas perfecto gracias gracias andrea le escribimos en estos dias** | o tendria que contratar **ok perfecto perfecto gracias ok estare atenta a cualquier coordinacion proxima** | no | |

## L01 — 2 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 1:00 | costado de 1 farmacia **listo como** va a pagar con | costado de 1 farmacia **ok listo con que medio** va a pagar con | no | |
| 2 | 1:56 | por telefono antes de **entregar perfecto cuanto demora mas o menos 40 minutos porque hoy hay bastantes pedidos** esta bien no hay | por telefono antes de **entrar para poder comunicarnos perfecto cuanto demora** esta bien no hay | no | |

## L02 — 2 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:09 | hacer 1 pedido para **llevar a mi casa claro digame** 1 medio pollo y | hacer 1 pedido para **akai claro digame necesito 1 pollo** 1 medio pollo y | no | |
| 2 | 0:50 | la direccion de entrega **jiron huallaga 382 departamento 4 b en el cercado jiron huallaga 382 departamento 4 b** tiene alguna referencia es | la direccion de entrega **giron guayaga 382 departamento 4b encercado ok giron guayaga 382 departamento 4b** tiene alguna referencia es | sí | |

## L03 — 5 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:27 | de litro y medio **2 gaseosas** a 10 soles cada | de litro y medio **∅** a 10 soles cada | sí | |
| 2 | 0:33 | de suerte hoy las **gaseosas estan 2 por 1 asi que 1 le sale gratis ah que bueno** a que direccion lo | de suerte hoy las **doceras estan a asi que 1 les sale gratis ah buenisimo entonces deme la oferta esta bien** a que direccion lo | no | |
| 3 | 1:03 | de 1 bodega perfecto **como** va a pagar por | de 1 bodega perfecto **pero por que medio** va a pagar por | no | |
| 4 | 1:34 | otra cosa los cuartos **pueden ser de pierna si le pongo 1 nota para que sean de pierna los 2** genial y las gaseosas | otra cosa los cuartos **podrian ser pierna ok** genial y las gaseosas | no | |
| 5 | 2:06 | es 1 poco confusa **no se preocupe tenemos su numero y le timbra antes de llegar gracias cuanto demora unos 45 minutos** listo gracias gracias a | es 1 poco confusa **ok cuanto estaria demorando ok** listo gracias gracias a | sí | |

## L04 — 2 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:36 | entrega avenida arequipa 2890 **en lince es 1 edificio interior 302** avenida arequipa 2890 interior | entrega avenida arequipa 2890 **inficio interior 302 ok** avenida arequipa 2890 interior | no | |
| 2 | 1:34 | mayonesa ketchup y aji **el pollo lo puede trozar en piezas es para compartir** si claro se lo | mayonesa ketchup y aji **∅** si claro se lo | sí | |
