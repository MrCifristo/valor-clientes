"""Pipeline de exploración: evidencia de qué comportamiento del cliente es real."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    evaluar_segmentacion,
    graficar_estabilidad,
    graficar_regiones,
    graficar_segmentacion,
    medir_estabilidad,
    perfilar_regiones,
    perfilar_tiempo,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``exploracion``."""
    return Pipeline(
        [
            Node(
                func=medir_estabilidad,
                inputs=["ventas_primarias", "params:ventanas"],
                outputs="estabilidad_clientes",
                name="medir_estabilidad",
            ),
            Node(
                func=evaluar_segmentacion,
                inputs=["ventas_primarias", "params:ventanas", "params:exploracion"],
                outputs="segmentacion_kmeans",
                name="evaluar_segmentacion",
            ),
            Node(
                func=perfilar_regiones,
                inputs="ventas_primarias",
                outputs=["perfil_regional", "tabla_regional"],
                name="perfilar_regiones",
            ),
            Node(
                func=perfilar_tiempo,
                inputs="ventas_primarias",
                outputs="perfil_temporal",
                name="perfilar_tiempo",
            ),
            Node(
                func=graficar_estabilidad,
                inputs="estabilidad_clientes",
                outputs="grafica_estabilidad",
                name="graficar_estabilidad",
            ),
            Node(
                func=graficar_segmentacion,
                inputs=["segmentacion_kmeans", "params:exploracion.k_grafica"],
                outputs="grafica_segmentacion",
                name="graficar_segmentacion",
            ),
            Node(
                func=graficar_regiones,
                inputs="tabla_regional",
                outputs="grafica_regiones",
                name="graficar_regiones",
            ),
        ]
    )
