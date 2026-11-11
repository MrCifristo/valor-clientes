"""Pipeline de evaluación: métricas, interpretación, auditoría y lista de campaña."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    auditar_ciudades,
    calcular_metricas,
    explicar_modelo,
    graficar_captura,
    graficar_prediccion,
    predecir_prueba,
    puntuar_clientes,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``evaluacion``."""
    return Pipeline(
        [
            Node(
                func=predecir_prueba,
                inputs=[
                    "clientes_prueba",
                    "clientes_entrenamiento",
                    "modelo_lineal",
                    "modelo_gb",
                    "seleccion_modelo",
                    "params:modelo",
                    "params:ventanas",
                ],
                outputs="predicciones_prueba",
                name="predecir_prueba",
            ),
            Node(
                func=calcular_metricas,
                inputs=[
                    "predicciones_prueba",
                    "tabla_modelo",
                    "seleccion_modelo",
                    "params:evaluacion",
                ],
                outputs="metricas_modelo",
                name="calcular_metricas",
            ),
            Node(
                func=explicar_modelo,
                inputs=[
                    "clientes_prueba",
                    "modelo_final",
                    "seleccion_modelo",
                    "params:modelo",
                    "params:evaluacion",
                ],
                outputs="importancia_variables",
                name="explicar_modelo",
            ),
            Node(
                func=auditar_ciudades,
                inputs=["predicciones_prueba", "params:evaluacion"],
                outputs="auditoria_ciudades",
                name="auditar_ciudades",
            ),
            Node(
                func=puntuar_clientes,
                inputs=[
                    "ventas_primarias",
                    "modelo_final",
                    "seleccion_modelo",
                    "params:ventanas",
                    "params:evaluacion",
                ],
                outputs="lista_campana",
                name="puntuar_clientes",
            ),
            Node(
                func=graficar_captura,
                inputs="metricas_modelo",
                outputs="grafica_captura",
                name="graficar_captura",
            ),
            Node(
                func=graficar_prediccion,
                inputs="predicciones_prueba",
                outputs="grafica_prediccion",
                name="graficar_prediccion",
            ),
        ]
    )
