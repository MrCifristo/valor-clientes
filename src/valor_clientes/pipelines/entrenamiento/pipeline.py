"""Pipeline de entrenamiento: división, Optuna por familia y selección."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    dividir_clientes,
    optimizar_gradient_boosting,
    optimizar_lineal,
    seleccionar_modelo,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline ``entrenamiento``."""
    return Pipeline(
        [
            Node(
                func=dividir_clientes,
                inputs=["tabla_modelo", "params:modelo"],
                outputs=["clientes_entrenamiento", "clientes_prueba"],
                name="dividir_clientes",
            ),
            Node(
                func=optimizar_lineal,
                inputs=["clientes_entrenamiento", "params:modelo", "params:optuna"],
                outputs=["modelo_lineal", "resumen_lineal", "intentos_lineal"],
                name="optimizar_lineal",
            ),
            Node(
                func=optimizar_gradient_boosting,
                inputs=["clientes_entrenamiento", "params:modelo", "params:optuna"],
                outputs=["modelo_gb", "resumen_gb", "intentos_gb"],
                name="optimizar_gradient_boosting",
            ),
            Node(
                func=seleccionar_modelo,
                inputs=[
                    "modelo_lineal",
                    "resumen_lineal",
                    "modelo_gb",
                    "resumen_gb",
                    "intentos_lineal",
                    "intentos_gb",
                    "params:modelo",
                ],
                outputs=["modelo_final", "seleccion_modelo", "intentos_optuna"],
                name="seleccionar_modelo",
            ),
        ]
    )
