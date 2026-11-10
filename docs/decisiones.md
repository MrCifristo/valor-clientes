# Decisiones del proyecto

Cada decisión con su evidencia. Las cifras salen de `kedro run` sobre el dataset completo (`data/08_reporting/`).

## 1. La pregunta: valor futuro y no segmentación clásica

El reto pide analizar la interacción de los clientes, segmentarlos, describir sus características por región y proponer una estrategia de marketing. Antes de elegir la pregunta probamos si los rasgos típicos de una segmentación **describen al cliente**: si un rasgo es real, el mismo cliente debe mostrarlo en dos periodos distintos.

| Rasgo del cliente | Correlación ene–feb vs. mar–abr | Lectura |
|---|---|---|
| Unidades por compra | 1.00 | El 99.98% compra siempre la misma cantidad |
| Gasto por compra | 0.96 | Estable |
| Gasto total | 0.85 | Estable |
| Días desde la última compra | 0.00 | Azar |
| Precio medio comprado | 0.00 | Azar |
| Categorías preferidas (promedio de 11) | 0.00 | Azar |
| Número de compras | 0.00 | Azar (índice de dispersión 0.99 = Poisson) |
| Días con compra | 0.00 | Azar |
| % de compras con descuento | 0.00 | Azar |

**Prueba de segmentación.** K-Means con k = 3, 4 y 5, entrenado por separado en cada periodo con los mismos 98,759 clientes. Coincidencia entre periodos (ARI) y separación (silueta en ene–feb), con k = 4:

| Conjunto de variables | ARI | Silueta |
|---|---|---|
| RFM clásico (recencia, frecuencia, gasto) | 0.10 | 0.29 |
| Hábitos (recencia, frecuencia, días activos, descuento, 11 categorías) | 0.00 | 0.06 |
| Valor (unidades y gasto por compra) | 0.78 | 0.53 |

**Decisión:** no basamos la estrategia en una segmentación clásica. Los segmentos por hábitos no existen (no se separan ni se repiten) y los RFM cambian de un mes a otro porque la recencia y la frecuencia son azar. El único eje que se sostiene es el valor, así que la pregunta es **predecir el valor futuro de cada cliente** para dirigir la campaña, y segmentar por ese valor predicho.

**Región:** gasto por cliente igual en las 96 ciudades (variación 1.7%, prueba F p = 0.83; ninguna categoría se desvía más de 0.4 puntos de la mezcla nacional). Recomendación: campaña nacional.

**Tiempo y descuento:** 24 horas con la misma participación (4.15%–4.18%), ingreso diario con 0.55% de variación, 13.0 unidades por línea con o sin descuento. No hay ventanas horarias ni efecto del descuento que aprovechar.

## 2. Limpieza

| Problema | Magnitud | Decisión |
|---|---|---|
| `TotalPrice` en cero | 100% de las ventas | Se descarta y se recalcula `ingreso = precio × cantidad × (1 − descuento)` |
| Ventas sin fecha | 67,526 (1.0%) | Se eliminan (parámetro `eliminar_ventas_sin_fecha`): no se pueden asignar a un periodo |
| Duplicados | 0 por `SalesID` y por fila | Se mantiene `drop_duplicates` como salvaguarda |
| `TransactionNumber` | Único por fila | No hay canastas: no se puede hacer análisis de canasta |
| Clientes con el mismo nombre | 4,897 | Todos con dirección distinta: personas distintas, se conservan |
| Código de país | Estados Unidos aparece como "AR" | Hay un solo país; no afecta el análisis |
| Precio mínimo 0.04 | Precios repartidos parejo de 0 a 100 | No es un error de captura; se conserva |
| Cantidades | 1 a 25, sin atípicos | Se valida `cantidad >= 1` y descuento en [0, 1] |
| `employees` | — | Fuera del alcance del reto 3 (vendedores) |

## 3. Ventanas de tiempo

- Observación (variables): 1-ene al 28-feb-2018. Objetivo: 1-mar al 30-abr-2018.
- Mayo solo tiene 9 días y no se usa como objetivo.
- Las variables solo usan la ventana de observación; una prueba (`test_tabla_modelo_no_usa_el_futuro_en_las_variables`) garantiza que no hay fuga de información.
- Configurables en `conf/base/parameters_transformacion.yml`.

## 4. Muestra para CI

El dataset completo pesa ~520 MB y GitHub no acepta archivos de más de 100 MB. CI corre con una muestra de 2,000 clientes elegidos al azar (semilla 42) **con todas sus compras**, para conservar el comportamiento por cliente. El dataset completo está en el Release `datos-v1`.

## 5. Datos sintéticos (limitación)

Precios y cantidades uniformes, ventas parejas las 24 horas, frecuencia Poisson y cantidad fija por cliente indican que el dataset fue **generado artificialmente**. Las recomendaciones valen para estos datos; con datos reales habría que repetir la prueba de estabilidad antes de descartar la segmentación por hábitos.
