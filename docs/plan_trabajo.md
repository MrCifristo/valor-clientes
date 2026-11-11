# Plan de trabajo

**Avance:** miércoles 11 de noviembre de 2026 · **Entrega final:** lunes 16 de noviembre de 2026, antes de la 1:00 PM (MiU), con la presentación ese mismo día.

## Reparto

| Bloque | Responsable | Estado |
|---|---|---|
| 1. Repo, Kedro, carga, limpieza, transformación, CI y README | Milton | Listo |
| 2. Exploración y evidencia para la pregunta | Milton | Listo (`notebooks/01_exploracion.ipynb`) |
| 3. Modelo: línea base, modelo, Optuna y evaluación | George | En revisión (`feat/modelo`) |
| 4. Presentación del avance y póster final | *Por asignar (George / Gian Paolo)* | Pendiente |

## Para el bloque 3 (modelo)

**Entrada:** `tabla_modelo` (`data/05_model_input/tabla_modelo.parquet`), con 98,759 clientes y una fila por cliente.

- Variables (ene–feb): `n_compras`, `dias_activos`, `recencia_dias`, `unidades_total`, `unidades_por_compra`, `ingreso`, `ticket_medio`, `precio_medio`, `pct_descuento`, las 11 columnas `cat_*` y `ciudad`.
- Objetivo (mar–abr): `ingreso_objetivo`. `compras_objetivo` **no** es variable: también es del futuro.

**Sugerencia de trabajo**

1. Crear los pipelines `entrenamiento` y `evaluacion` en `src/valor_clientes/pipelines/` (Kedro los registra solos) con sus `parameters_*.yml`.
2. Línea base: «el cliente gastará en mar–abr lo mismo que en ene–feb» (`ingreso`), ajustada por la diferencia de días entre periodos.
3. Separar entrenamiento y prueba por cliente (por ejemplo 80/20 con semilla) y comparar un modelo lineal con un gradient boosting o Random Forest ajustado con **Optuna**.
4. Métricas: MAE y RMSE del gasto, y la métrica de negocio: % del gasto futuro que capta el 20% de clientes con mayor predicción, contra la línea base y contra elegir al azar.
5. Un dato de partida: `unidades_por_compra` tiene correlación 0.92 con el objetivo, más que el gasto pasado (0.85). Lo que se repite es la cantidad, no la frecuencia, y el modelo debería aprenderlo.
6. Agregar las pruebas en `tests/pipelines/` y las salidas nuevas al paso «Verificar outputs» de `.github/workflows/ci.yml`.

**Trabajo en Git:** crear una rama por bloque (`feat/modelo`, `feat/presentacion`) y abrir un PR hacia `main`; el CI debe quedar en verde antes de mergear.

## Para el bloque 4 (presentación del avance)

Figuras listas en `docs/figuras/`. Guion sugerido de 8 diapositivas:

1. Portada y equipo.
2. Reto 3, stakeholder, pregunta, objetivo analítico y métrica.
3. Datos: 7 tablas, 6.7 millones de ventas y su calidad (`TotalPrice` en cero, 1% sin fecha, sin canastas).
4. Hallazgo 1: del cliente solo se repite cuánto compra (`estabilidad_clientes.png`).
5. Hallazgo 2: por qué no aplicamos una segmentación clásica (`segmentacion_kmeans.png` y `frecuencia_poisson.png`).
6. Hallazgo 3: sin diferencias por región, hora ni descuento (`perfil_regional.png`).
7. Arquitectura: Kedro (3 pipelines), CI con Ruff, pytest y la corrida del pipeline.
8. Próximos pasos: modelo, Optuna, interpretación, reporte y póster hasta el 16.

## Calendario hasta la entrega

| Día | Qué |
|---|---|
| Lun 10 | Bloques 1 y 2 terminados |
| Mié 11 | Presentación del avance; anotar los comentarios del catedrático |
| Jue 12 – Vie 13 | Modelo con Optuna, evaluación con la métrica de negocio, interpretación (importancia de variables / SHAP) |
| Sáb 14 – Dom 15 | Reporte PDF (10 secciones) y póster de una diapositiva |
| Lun 16 | Zip a MiU antes de la 1:00 PM y presentación |
