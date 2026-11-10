"""Pipeline de carga, diagnóstico de calidad y limpieza."""

from kedro.pipeline import Node, Pipeline

from .nodes import construir_ventas_primarias, generar_reporte_calidad, limpiar_ventas

TABLAS_DIMENSION = [
    "productos_raw",
    "categorias_raw",
    "clientes_raw",
    "ciudades_raw",
]


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``carga_limpieza``."""
    return Pipeline(
        [
            Node(
                func=generar_reporte_calidad,
                inputs=["ventas_raw", *TABLAS_DIMENSION, "paises_raw"],
                outputs="reporte_calidad_datos",
                name="generar_reporte_calidad",
            ),
            Node(
                func=limpiar_ventas,
                inputs=["ventas_raw", "params:carga_limpieza"],
                outputs="ventas_limpias",
                name="limpiar_ventas",
            ),
            Node(
                func=construir_ventas_primarias,
                inputs=["ventas_limpias", *TABLAS_DIMENSION],
                outputs="ventas_primarias",
                name="construir_ventas_primarias",
            ),
        ]
    )
