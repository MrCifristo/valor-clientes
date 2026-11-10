"""Pipeline de transformación: de ventas a la tabla cliente-objetivo."""

from kedro.pipeline import Node, Pipeline

from .nodes import construir_tabla_modelo


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``transformacion``."""
    return Pipeline(
        [
            Node(
                func=construir_tabla_modelo,
                inputs=["ventas_primarias", "params:ventanas"],
                outputs="tabla_modelo",
                name="construir_tabla_modelo",
            ),
        ]
    )
