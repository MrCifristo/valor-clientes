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

## 6. Modelo

**Datos.** `tabla_modelo`: un cliente por fila, variables de ene–feb y objetivo `ingreso_objetivo` (gasto de mar–abr). División 80/20 por cliente (semilla 42): 79,007 clientes de entrenamiento y 19,752 de prueba. El conjunto de prueba solo se usa al final.

**Variables excluidas.**

| Variable | Por qué |
|---|---|
| `cliente_id` | Es un identificador. En estos datos tiene correlación 0.999 con `unidades_por_compra` porque el generador asignó la cantidad según el ID. Un modelo que lo use no serviría con clientes nuevos. |
| `ciudad` | Sin efecto en el gasto (prueba F, p = 0.83). Se usa solo para auditar sesgo regional. |
| `compras_objetivo` | Es del futuro (fuga). |

**Línea base.** «El cliente gastará en mar–abr lo mismo que en ene–feb», ajustado por días (61/59). Es la regla que usaría Marketing sin un modelo.

**Candidatos y Optuna.** Dos familias con el mismo protocolo: mismos 5 folds (`KFold` con semilla), mismo objetivo (RMSE medio de validación cruzada) y 30 intentos cada una con `TPESampler(seed=42)`. El escalado va dentro del `Pipeline`, así que se ajusta en cada fold y no hay fuga hacia la validación.

| Familia | Hiperparámetros buscados | Mejor RMSE CV |
|---|---|---|
| ElasticNet (variables estandarizadas) | `alpha` (1e-4 a 100, log), `l1_ratio` (0.01 a 1) | 4,761 |
| HistGradientBoosting | `learning_rate`, `max_iter`, `max_depth`, `min_samples_leaf`, `l2_regularization` | 4,763 |

Se optimiza RMSE y no la captura del top 20% porque el RMSE apunta al gasto esperado, que es lo que se quiere ordenar, mientras que la captura es discontinua y ruidosa como objetivo. La captura se reporta al final. Los espacios de búsqueda y el número de intentos están en `conf/base/parameters_entrenamiento.yml`. En CI (`conf/muestra/`) se usan 5 intentos y 3 folds.

**Selección.** El gradient boosting solo reemplaza al lineal si baja el RMSE de CV más de 1% (`tolerancia_rmse`). Empataron (−0.03%), así que el modelo final es el **lineal**. Optuna llevó `alpha` casi al límite inferior (0.0003, `l1_ratio` 0.98): la regularización no aporta y el ElasticNet queda prácticamente como una regresión sin regularizar, algo coherente con el techo de ruido. El modelo final se explica en una frase («el gasto esperado sube unos 11,700 por cada desviación estándar de unidades por compra»), es fácil de implementar y no pierde precisión.

**Techo de ruido.** Para saber si quedaba señal por capturar se calcula una referencia casi óptima: el gasto medio de entrenamiento para cada valor de unidades por compra (1 a 25). Obtiene el mismo R² (0.853) que los dos modelos. No hay más señal que extraer. El error restante viene de la frecuencia (Poisson) y del precio de lo que se compra, que son azar.

**Resultados en prueba.**

| Enfoque | MAE | RMSE | R² | Top 10% | Top 20% | Top 30% |
|---|---|---|---|---|---|---|
| Al azar | — | — | — | 10.0% | 20.0% | 30.0% |
| Línea base | 4,691 | 6,738 | 0.696 | 17.4% | 33.4% | 48.1% |
| Modelo lineal | 3,273 | 4,688 | 0.853 | 18.6% | 35.3% | 50.0% |
| Oráculo | 0 | 0 | 1 | 20.9% | 37.7% | 52.1% |

- La línea base pierde porque arrastra el azar de ene–feb: quien compró muchas veces por suerte no repite (regresión a la media).
- Mejora del top 20% sobre la línea base: +1.94 puntos (+5.8% relativo). IC 95% por bootstrap pareado (1,000 remuestreos): +1.76 a +2.11 puntos; el 100% de los remuestreos muestra mejora.
- **Impacto:** con el mismo tamaño de campaña, el modelo pone dentro de ella ~39.3 millones más (extrapolado a toda la base desde el conjunto de prueba) de gasto futuro (1.94% de los 2,029 millones de mar–abr). Es gasto *cubierto*, no ingreso incremental: no hay datos de respuesta a campañas para estimar cuánto cambia la compra por recibir la campaña.

**Interpretación.** Permutation importance en prueba: barajar `unidades_por_compra` sube el RMSE en ~12,300; ninguna otra variable pasa de 5. Coeficientes estandarizados: `unidades_por_compra` +11,662 y el resto < 200 en valor absoluto. Las variables derivadas (`unidades_total`, `ticket_medio`, `ingreso`) son colineales con ella, así que sus coeficientes individuales no deben leerse por separado. No se usó SHAP: con una sola variable dominante no agrega información a la importancia por permutación.

**Auditoría por ciudad.** Con el top 20% elegido por el modelo, la proporción de clientes elegidos por ciudad va de 12.7% a 27.0% (media 20.0%, ~206 clientes de prueba por ciudad) y el sesgo medio no se aleja de cero de forma sistemática. Las diferencias son del tamaño esperado por el azar con muestras de ese tamaño. El modelo no favorece regiones porque no usa la ciudad. Detalle en `data/08_reporting/auditoria_ciudades.csv`.

**Lista de campaña.** `puntuar_clientes` aplica el modelo al comportamiento de mar–abr (ventana de duración similar: 61 vs. 59 días) para predecir el gasto de may–jun de los 98,759 clientes. Asigna a cada uno un segmento según su valor predicho (Top 20%, Alto 30%, Medio 30%, Bajo 20%) y marca `en_campana` para el Top.

**Limitaciones y riesgos.**

- **Datos sintéticos:** la frecuencia no tiene memoria entre periodos, lo cual es irreal. Con datos reales habría más señal en frecuencia y recencia, y el gradient boosting podría ganarle al lineal.
- **Sin validación fuera de periodo:** solo hay 4 meses completos, sin estacionalidad.
- **Sin abandono:** ningún cliente tiene gasto futuro cero. Con datos reales haría falta un modelo de dos etapas (¿compra? y ¿cuánto?).
- **Predecir gasto no es predecir respuesta:** la campaña ideal se dirige a quien cambia su compra por recibirla (modelo de *uplift*), y para eso hace falta un experimento A/B.
- **Riesgo de concentración:** contactar siempre al mismo top puede saturar a esos clientes y dejar sin atención al resto. Se recomienda reservar una parte de la campaña al azar como grupo de control.
- **Heterocedasticidad:** el error crece con el gasto (ver `prediccion_vs_real.png`); las predicciones individuales de clientes grandes son menos precisas.
