# Revisar referencias contra el audio

Generado por `python scripts_grabacion/revisar_referencias.py`. **No editar a mano.**

Los transcripts de referencia salen del guion, pero al grabar se improvisó. Para cada fragmento: escuchar el audio en el minuto indicado y anotar qué se dijo de verdad. La corrección se hace en `scripts_grabacion/guiones.md` (una persona, nunca copiando al ASR) y luego se corre `python scripts_grabacion/generar_referencias.py`.

- **Guion**: lo que dice la referencia hoy. **ASR**: lo que escribió `large-v3__cuda__int8__plano__L0`. Todo normalizado (minúsculas, sin tildes, números en dígitos); en **negrita** lo que difiere; ∅ = nada.
- **¿Otros modelos igual?** sí = large-v3 float16 y small también lo escribieron así → probablemente se **dijo así** (improvisación). no = probablemente **error del ASR**. Es una pista: decide el oído.
- Tiempos aproximados (±3 s).

## E01 — 11 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:07 | bien si buenos dias **la** escucho perfecto mi nombre | bien si buenos dias **le** escucho perfecto mi nombre | no | |
| 2 | 0:13 | perfecto lucia para empezar **cuenteme** en que trabajo ultimo | perfecto lucia para empezar **cuentame** en que trabajo ultimo | sí | |
| 3 | 0:17 | en que trabajo ultimo **∅** mi ultimo puesto fue | en que trabajo ultimo **en** mi ultimo puesto fue | sí | |
| 4 | 0:23 | veia los reportes de **ventas** de todas las tiendas | veia los reportes de **venta** de todas las tiendas | sí | |
| 5 | 0:52 | tablero que mostraba que **productos se vencian** antes de venderse con | tablero que mostraba que **producto se vencia** antes de venderse con | sí | |
| 6 | 0:57 | eso redujimos bastante las **∅** perdidas en las tiendas | eso redujimos bastante las **tasas asi que no hubo tantas** perdidas en las tiendas | no | |
| 7 | 1:13 | 4 anos de experiencia **∅** el puesto requiere leer | 4 anos de experiencia **ok** el puesto requiere leer | sí | |
| 8 | 1:20 | afuera como esta su **∅** ingles diria que intermedio | afuera como esta su **nivel de** ingles diria que intermedio | sí | |
| 9 | 1:26 | escribo correos pero hablando **todavia me pongo 1 poco nerviosa** es normal no se | escribo correos pero hablando **en ingles no tengo ningun problema cuanto todavia me pongo** es normal no se | no | |
| 10 | 1:32 | no se preocupe y **cuando** podria empezar si avanzamos | no se preocupe y **cuanto** podria empezar si avanzamos | sí | |
| 11 | 1:58 | es remoto o presencial **∅** es hibrido 3 dias | es remoto o presencial **de hecho** es hibrido 3 dias | sí | |

## E02 — 13 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:00 | **hola** buenas tardes daniela salas | **∅** buenas tardes daniela salas | sí | |
| 2 | 0:24 | quiza 1 poquito mas **esta dentro del rango perfecto** el equipo es internacional | quiza 1 poquito mas **∅** el equipo es internacional | sí | |
| 3 | 0:33 | que tal su nivel **∅** avanzado trabaje 2 anos | que tal su nivel **bien** avanzado trabaje 2 anos | no | |
| 4 | 0:39 | reuniones eran en ingles **excelente cuenteme** 1 poco de la | reuniones eran en ingles **∅** 1 poco de la | no | |
| 5 | 0:51 | para los servicios para **desplegar** uso docker y en | para los servicios para **expregar** uso docker y en | sí | |
| 6 | 1:02 | todo corria en kubernetes **y cual fue su ultimo puesto exactamente desarrolladora backend** en 1 fintech me | todo corria en kubernetes **desarrolladora de bacnet** en 1 fintech me | no | |
| 7 | 1:10 | porque no podian caerse **alguna vez se cayeron** 1 vez 1 viernes | porque no podian caerse **si** 1 vez 1 viernes | sí | |
| 8 | 1:13 | viernes en la noche **∅** lo resolvimos en 1 | viernes en la noche **sin embargo** lo resolvimos en 1 | sí | |
| 9 | 1:21 | que no volviera a **pasar buena respuesta y cuando podria incorporarse** de inmediato mi contrato | que no volviera a **suceder** de inmediato mi contrato | sí | |
| 10 | 1:33 | estoy libre desde ya **perfecto y en total** cuantos anos lleva programando | estoy libre desde ya **∅** cuantos anos lleva programando | no | |
| 11 | 1:40 | cuantos anos lleva programando **de forma profesional 7 anos** si cuento la universidad | cuantos anos lleva programando **7 anos de manera profesional** si cuento la universidad | no | |
| 12 | 1:50 | pero profesionalmente son 7 **muy bien tiene alguna pregunta para mi** si cuantas personas son | pero profesionalmente son 7 **∅** si cuantas personas son | sí | |
| 13 | 1:56 | son en el equipo **son 6 desarrolladores y 1 lider tecnica perfecto gracias gracias a usted daniela le avisamos el lunes** | son en el equipo **desarrolladores desarrolladores perfecto muchas gracias** | no | |

## E03 — 16 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:12 | extranjeras que tal se **defiende** en ingles ay la | extranjeras que tal se **define** en ingles ay la | sí | |
| 2 | 0:24 | verdad es basico entiendo **lo principal si lo leo despacio pero no podria tener 1 reunion en ingles entiendo** gracias por la sinceridad | verdad es basico entiendo **∅** gracias por la sinceridad | sí | |
| 3 | 0:26 | la sinceridad desde cuando **estaria disponible ahorita** estoy terminando 1 curso | la sinceridad desde cuando **podria estar disponible ahora** estoy terminando 1 curso | sí | |
| 4 | 0:42 | experiencia tiene en marketing **mmm 2.5 anos** empece a mitad de | experiencia tiene en marketing **∅** empece a mitad de | sí | |
| 5 | 0:48 | muy bien hablemos de **sueldo** cuanto estaria buscando estaba | muy bien hablemos de **suelo** cuanto estaria buscando estaba | sí | |
| 6 | 0:56 | aunque se podria conversar **∅** lo tengo en cuenta | aunque se podria conversar **ok** lo tengo en cuenta | sí | |
| 7 | 1:15 | usaba para medir usaba **∅** google analytics los correos | usaba para medir usaba **el** google analytics los correos | sí | |
| 8 | 1:20 | correos los mandabamos por **hubspot** y las imagenes las | correos los mandabamos por **∅** y las imagenes las | sí | |
| 9 | 1:23 | las hacia en canva **∅** alguna campana que le | las hacia en canva **ok** alguna campana que le | sí | |
| 10 | 1:51 | poco y quiero especializarme **∅** como se organiza cuando | poco y quiero especializarme **en algo especifico ok** como se organiza cuando | sí | |
| 11 | 1:58 | campanas al mismo tiempo **∅** hago 1 lista cada | campanas al mismo tiempo **bueno** hago 1 lista cada | no | |
| 12 | 2:03 | 1 lista cada lunes **con lo urgente y lo importante y lo** voy tachando si algo | 1 lista cada lunes **o** voy tachando si algo | no | |
| 13 | 2:06 | mi jefa que va **primero muy bien tiene** alguna pregunta para mi | mi jefa que va **a primero muy bien tienes** alguna pregunta para mi | no | |
| 14 | 2:11 | alguna pregunta para mi **∅** si hay capacitaciones al | alguna pregunta para mi **si** si hay capacitaciones al | no | |
| 15 | 2:16 | si el primer mes **tiene** acompanamiento con la jefa | si el primer mes **tienes** acompanamiento con la jefa | no | |
| 16 | 2:23 | a usted mariela le **escribimos** pronto | a usted mariela le **escribiremos** pronto | no | |

## E04 — 11 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:17 | tengo que dar 1 **preaviso** en mi empresa asi | tengo que dar 1 **previo aviso** en mi empresa asi | sí | |
| 2 | 0:34 | infraestructura la manejamos con **terraform muy bien** este puesto se paga | infraestructura la manejamos con **el terraform** este puesto se paga | sí | |
| 3 | 0:52 | a gestion en total **ya** son 10 anos 1 | a gestion en total **se podria decir que** son 10 anos 1 | sí | |
| 4 | 0:58 | 1 decada impresionante con **clientes de afuera que tal su ingles avanzado hago presentaciones a clientes de estados unidos** casi todas las semanas | 1 decada impresionante con **que clientes de afuera la verdad que he avanzado hago presentaciones internacionales hago presentaciones internacionales a clientes de ee uu** casi todas las semanas | no | |
| 5 | 1:14 | casi todas las semanas **perfecto y su cargo actual cual es** soy jefa de proyectos | casi todas las semanas **∅** soy jefa de proyectos | no | |
| 6 | 1:16 | soy jefa de proyectos **ti** en 1 empresa de | soy jefa de proyectos **tee** en 1 empresa de | no | |
| 7 | 1:19 | logistica tengo a cargo **∅** 2 equipos en total | logistica tengo a cargo **de** 2 equipos en total | sí | |
| 8 | 1:26 | el proyecto mas dificil **∅** mudar todo el sistema | el proyecto mas dificil **la verdad que** mudar todo el sistema | no | |
| 9 | 1:32 | que el negocio parara **∅** lo hicimos en etapas | que el negocio parara **sin embargo** lo hicimos en etapas | sí | |
| 10 | 1:42 | varios fines de semana **muy interesante alguna pregunta para mi** si el equipo ya | varios fines de semana **∅** si el equipo ya | no | |
| 11 | 1:52 | o tendria que contratar **hay 4 personas y tendria que contratar 2 mas perfecto gracias gracias andrea le escribimos en estos dias** | o tendria que contratar **ok perfecto perfecto gracias ok estare atenta a cualquier coordinacion proxima** | no | |

## L01 — 14 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:06 | pedir hola buenas noches **queria** 1 pollo entero por | pedir hola buenas noches **quiero** 1 pollo entero por | sí | |
| 2 | 0:19 | y medio papas familiares **∅** a 15 soles y | y medio papas familiares **esta** a 15 soles y | no | |
| 3 | 0:25 | no espere mejor que **sea solo** 1 gaseosa porque ya | no espere mejor que **solo sea** 1 gaseosa porque ya | sí | |
| 4 | 0:35 | le pongo cremas si **mayonesa y aji por favor** perfecto a que direccion | le pongo cremas si **por favor mayonesa y aji** perfecto a que direccion | no | |
| 5 | 0:40 | perfecto a que direccion **se lo** enviamos avenida los alamos | perfecto a que direccion **le** enviamos avenida los alamos | sí | |
| 6 | 1:00 | costado de 1 farmacia **listo como** va a pagar con | costado de 1 farmacia **ok listo con que medio** va a pagar con | no | |
| 7 | 1:17 | de litro y medio **el** total es 89 soles | de litro y medio **en** total es 89 soles | sí | |
| 8 | 1:23 | bien doradito la ultima **vez** llego 1 poco palido | bien doradito la ultima **es** llego 1 poco palido | sí | |
| 9 | 1:26 | llego 1 poco palido **ay disculpe le voy a dejar 1 nota a** cocina para que lo | llego 1 poco palido **la** cocina para que lo | sí | |
| 10 | 1:30 | para que lo saquen **bien dorado** gracias y las papas | para que lo saquen **tal como lo requiere dorado ok** gracias y las papas | sí | |
| 11 | 1:37 | papas llegan calientes porque **vivo** 1 poco lejos si | papas llegan calientes porque **viven** 1 poco lejos si | no | |
| 12 | 1:43 | en 1 bolsa termica **si llegan frias nos llama** y lo solucionamos ya | en 1 bolsa termica **y llegan frias nos llaman** y lo solucionamos ya | no | |
| 13 | 1:52 | dejar a mi vecina **si claro** el motorizado le avisa | dejar a mi vecina **∅** el motorizado le avisa | sí | |
| 14 | 1:56 | por telefono antes de **entregar perfecto cuanto demora mas o menos 40 minutos porque hoy hay bastantes pedidos** esta bien no hay | por telefono antes de **entrar para poder comunicarnos perfecto cuanto demora** esta bien no hay | no | |

## L02 — 15 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:09 | hacer 1 pedido para **llevar a mi casa claro digame** 1 medio pollo y | hacer 1 pedido para **akai claro digame necesito 1 pollo** 1 medio pollo y | no | |
| 2 | 0:31 | cada 1 algo mas **1 ensalada tambien 1 ensalada** 8 soles le cuento | cada 1 algo mas **tambien quiere ensalada si 1 ensalada a** 8 soles le cuento | sí | |
| 3 | 0:46 | que bien me sirve **∅** la direccion de entrega | que bien me sirve **ok** la direccion de entrega | sí | |
| 4 | 0:50 | la direccion de entrega **jiron huallaga 382 departamento 4 b en el cercado jiron huallaga 382 departamento 4 b** tiene alguna referencia es | la direccion de entrega **giron guayaga 382 departamento 4b encercado ok giron guayaga 382 departamento 4b** tiene alguna referencia es | sí | |
| 5 | 1:05 | es 1 edificio celeste **a** media cuadra del mercado | es 1 edificio celeste **∅** media cuadra del mercado | no | |
| 6 | 1:07 | media cuadra del mercado **toquen** el timbre de arriba | media cuadra del mercado **tocan** el timbre de arriba | sí | |
| 7 | 1:15 | perfecto forma de pago **en** efectivo voy a pagar | perfecto forma de pago **∅** efectivo voy a pagar | sí | |
| 8 | 1:16 | efectivo voy a pagar **con** 1 billete de 100 | efectivo voy a pagar **∅** 1 billete de 100 | sí | |
| 9 | 1:23 | medio pollo 2 chaufas **y** 1 ensalada con el | medio pollo 2 chaufas **∅** 1 ensalada con el | sí | |
| 10 | 1:26 | descuento de la promocion **el total es 80 soles perfecto viene** con cubiertos si le | descuento de la promocion **en totales 80 soles tiene** con cubiertos si le | no | |
| 11 | 1:39 | el chaufa lleva mucho **kion** a mi hijo no | el chaufa lleva mucho **guion** a mi hijo no | sí | |
| 12 | 1:46 | que vaya con poco **kion** si por favor pongale | que vaya con poco **guion** si por favor pongale | sí | |
| 13 | 1:52 | viene con limon aparte **∅** si el limon y | viene con limon aparte **verdad** si el limon y | sí | |
| 14 | 1:59 | ah que bien y **atienden** tambien los domingos si | ah que bien y **atiende** tambien los domingos si | no | |
| 15 | 2:09 | noche gracias en cuanto **llega en unos 35 minutos** listo muchas gracias a | noche gracias en cuanto **tiempo lleva unos 35 minutos aproximadamente** listo muchas gracias a | no | |

## L03 — 15 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:03 | noches hola buenas noches **queria 1 pollo entero mmm no mejor cambielo** por 2 cuartos de | noches hola buenas noches **quiero 1 pollo entero no mejor cambialo** por 2 cuartos de | no | |
| 2 | 0:17 | cada 1 que mas **unas papas familiares papas familiares** 15 soles algo de | cada 1 que mas **desearia unas papas familiares las papas familiares a** 15 soles algo de | no | |
| 3 | 0:27 | de litro y medio **2 gaseosas** a 10 soles cada | de litro y medio **∅** a 10 soles cada | sí | |
| 4 | 0:33 | de suerte hoy las **gaseosas estan 2 por 1 asi que 1 le sale gratis ah que bueno** a que direccion lo | de suerte hoy las **doceras estan a asi que 1 les sale gratis ah buenisimo entonces deme la oferta esta bien** a que direccion lo | no | |
| 5 | 1:03 | de 1 bodega perfecto **como** va a pagar por | de 1 bodega perfecto **pero por que medio** va a pagar por | no | |
| 6 | 1:06 | pagar por yape si **se puede** si claro le mandamos | pagar por yape si **es que se podria** si claro le mandamos | no | |
| 7 | 1:11 | por mensaje su pedido **es** 2 cuartos de pollo | por mensaje su pedido **seria entonces** 2 cuartos de pollo | sí | |
| 8 | 1:17 | familiares y 2 gaseosas **con la promocion el** total es 75 soles | familiares y 2 gaseosas **que es como la promocion en** total es 75 soles | no | |
| 9 | 1:23 | es 75 soles perfecto **pueden agregar bastante aji** si le ponemos aji | es 75 soles perfecto **me podria agregar bastante aji porfa** si le ponemos aji | no | |
| 10 | 1:29 | aji extra sin costo **∅** otra cosa los cuartos | aji extra sin costo **y tambien algunas otras cremas ok** otra cosa los cuartos | no | |
| 11 | 1:34 | otra cosa los cuartos **pueden ser de pierna si le pongo 1 nota para que sean de pierna los 2** genial y las gaseosas | otra cosa los cuartos **podrian ser pierna ok** genial y las gaseosas | no | |
| 12 | 1:46 | genial y las gaseosas **vienen heladas si salen directo de la refrigeradora** perfecto ah y si | genial y las gaseosas **que vengan heladas porfa directo de la refrigera que estan heladas** perfecto ah y si | no | |
| 13 | 1:57 | casa que me llame **∅** porque la numeracion de | casa que me llame **normal** porque la numeracion de | sí | |
| 14 | 2:06 | es 1 poco confusa **no se preocupe tenemos su numero y le timbra antes de llegar gracias cuanto demora unos 45 minutos** listo gracias gracias a | es 1 poco confusa **ok cuanto estaria demorando ok** listo gracias gracias a | sí | |
| 15 | 2:21 | gracias gracias a usted **∅** buenas noches | gracias gracias a usted **por su pedido** buenas noches | sí | |

## L04 — 19 fragmentos

| # | Tiempo | Guion | ASR | ¿Otros modelos igual? | Lo que se dijo |
|---|---|---|---|---|---|
| 1 | 0:07 | pollo entero por favor **pollo entero** 64 soles algo mas | pollo entero por favor **ok pollo entero esta a** 64 soles algo mas | no | |
| 2 | 0:10 | 64 soles algo mas **∅** 1 arroz chaufa y | 64 soles algo mas **que desee** 1 arroz chaufa y | no | |
| 3 | 0:14 | 1 arroz chaufa y **1 ensalada chaufa a 28 soles y** ensalada a 8 soles | 1 arroz chaufa y **ensalada ok chaufa a 25 soles y de** ensalada a 8 soles | no | |
| 4 | 0:27 | la promocion del dia **ah perfecto** direccion de entrega avenida | la promocion del dia **excelente perfecto ok la** direccion de entrega avenida | sí | |
| 5 | 0:36 | entrega avenida arequipa 2890 **en lince es 1 edificio interior 302** avenida arequipa 2890 interior | entrega avenida arequipa 2890 **inficio interior 302 ok** avenida arequipa 2890 interior | no | |
| 6 | 0:48 | 302 lince alguna referencia **∅** esta cerca de 1 | 302 lince alguna referencia **que me podria dar** esta cerca de 1 | no | |
| 7 | 0:55 | cerca de 1 banco **en la esquina el portero recibe el pedido perfecto** como va a pagar | cerca de 1 banco **el banco es el nivel pedido ok** como va a pagar | no | |
| 8 | 1:02 | como va a pagar **con tarjeta** muy bien el motorizado | como va a pagar **el pedido con tarjeta por favor** muy bien el motorizado | sí | |
| 9 | 1:08 | bien el motorizado lleva **pos entonces es** 1 pollo entero 1 | bien el motorizado lleva **el pos entonces seria** 1 pollo entero 1 | sí | |
| 10 | 1:16 | su descuento el total **es** 98 soles ya perfecto | su descuento el total **seria de** 98 soles ya perfecto | sí | |
| 11 | 1:20 | 98 soles ya perfecto **mandan cremas si** mayonesa ketchup y aji | 98 soles ya perfecto **me mandan cremas si mandamos todas** mayonesa ketchup y aji | no | |
| 12 | 1:34 | mayonesa ketchup y aji **oiga el pollo lo pueden cortar en piezas es para compartir** si claro se lo | mayonesa ketchup y aji **∅** si claro se lo | sí | |
| 13 | 1:42 | perfecto y el chaufa **es para cuantas personas** alcanza bien para 2 | perfecto y el chaufa **para cuantas personas alcanza de acuerdo a lo que ustedes venden** alcanza bien para 2 | no | |
| 14 | 1:50 | bien para 2 personas **∅** ya esta bien y | bien para 2 personas **en verdad** ya esta bien y | no | |
| 15 | 1:58 | motorizado sube si si **el edificio lo** permite sube hasta la | motorizado sube si si **si el edificio le** permite sube hasta la | no | |
| 16 | 2:00 | sube hasta la puerta **si no le llama para que baje listo me parece bien es que** estoy con mi mama | sube hasta la puerta **y si no le van a llamar para que usted pueda bajar listo** estoy con mi mama | sí | |
| 17 | 2:08 | que no camina mucho **∅** no se preocupe le | que no camina mucho **y por eso le indico en caso que no este el portero para que pueda salir ok** no se preocupe le | no | |
| 18 | 2:16 | le avisamos con tiempo **y cuanto se demora** unos 30 minutos aproximadamente | le avisamos con tiempo **cuanto tiempo sirve para traer el pedido yo creo que** unos 30 minutos aproximadamente | no | |
| 19 | 2:28 | unos 30 minutos aproximadamente **listo gracias** gracias por su pedido | unos 30 minutos aproximadamente **∅** gracias por su pedido | sí | |
