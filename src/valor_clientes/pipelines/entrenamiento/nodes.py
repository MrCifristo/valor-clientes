"""Nodos de entrenamiento: separar clientes, optimizar con Optuna y elegir el modelo.

Se comparan dos familias con el mismo protocolo (mismos folds, mismo objetivo):
un modelo lineal regularizado (ElasticNet) y un gradient boosting
(HistGradientBoosting). Cada una tiene su propio estudio de Optuna y el modelo
final se elige con una regla explícita que premia la simplicidad.
"""

from __future__ import annotations

import logging
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from valor_clientes.modelo import construir_modelo, seleccionar_variables

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def dividir_clientes(
    tabla: pd.DataFrame, parametros: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa a los clientes en entrenamiento y prueba.

    Cada cliente es una fila, así que la división por filas ya es por cliente:
    ninguno aparece en ambos conjuntos.

    Args:
        tabla: Tabla del modelo (un cliente por fila).
        parametros: ``proporcion_prueba`` y ``semilla``.

    Returns:
        Las tablas de entrenamiento y de prueba.
    """
    entrenamiento, prueba = train_test_split(
        tabla,
        test_size=parametros["proporcion_prueba"],
        random_state=parametros["semilla"],
    )
    logger.info(
        "Entrenamiento: %d clientes; prueba: %d", len(entrenamiento), len(prueba)
    )
    return entrenamiento.reset_index(drop=True), prueba.reset_index(drop=True)


def _sugerir(trial: optuna.Trial, espacio: dict[str, dict]) -> dict[str, Any]:
    """Traduce el espacio de búsqueda del YAML a sugerencias de Optuna."""
    valores: dict[str, Any] = {}
    for nombre, rango in espacio.items():
        sugerir = trial.suggest_int if rango["tipo"] == "int" else trial.suggest_float
        valores[nombre] = sugerir(
            nombre, rango["bajo"], rango["alto"], log=rango.get("log", False)
        )
    return valores


def _optimizar(
    familia: str,
    entrenamiento: pd.DataFrame,
    parametros: dict[str, Any],
    optuna_cfg: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Busca hiperparámetros con Optuna (RMSE medio en validación cruzada)."""
    variables = seleccionar_variables(entrenamiento.columns, parametros["variables"])
    x = entrenamiento[variables]
    y = entrenamiento[parametros["objetivo"]]
    semilla = parametros["semilla"]
    folds = KFold(n_splits=optuna_cfg["n_splits"], shuffle=True, random_state=semilla)

    def objetivo(trial: optuna.Trial) -> float:
        modelo = construir_modelo(
            familia, _sugerir(trial, optuna_cfg["espacios"][familia]), semilla
        )
        puntajes = cross_val_score(
            modelo, x, y, cv=folds, scoring="neg_root_mean_squared_error"
        )
        return float(-puntajes.mean())

    estudio = optuna.create_study(
        study_name=familia,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=semilla),
    )
    estudio.optimize(objetivo, n_trials=optuna_cfg["n_trials"])

    modelo = construir_modelo(familia, estudio.best_params, semilla).fit(x, y)
    resumen = {
        "familia": familia,
        "variables": variables,
        "mejores_hiperparametros": estudio.best_params,
        "rmse_cv": float(estudio.best_value),
        "n_trials": len(estudio.trials),
        "n_splits": optuna_cfg["n_splits"],
    }
    intentos = estudio.trials_dataframe(attrs=("number", "value", "params", "state"))
    intentos = intentos.rename(columns={"value": "rmse_cv"})
    intentos.insert(0, "familia", familia)
    logger.info(
        "%s: RMSE CV %.1f con %s", familia, resumen["rmse_cv"], estudio.best_params
    )
    return modelo, resumen, intentos


def optimizar_lineal(
    entrenamiento: pd.DataFrame,
    parametros: dict[str, Any],
    optuna_cfg: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Optimiza ``alpha`` y ``l1_ratio`` de un ElasticNet estandarizado.

    Args:
        entrenamiento: Clientes de entrenamiento.
        parametros: Parámetros ``modelo`` (variables, objetivo, semilla).
        optuna_cfg: ``n_trials``, ``n_splits`` y ``espacios`` de búsqueda.

    Returns:
        El modelo reentrenado con todo el entrenamiento, un resumen del estudio
        y la tabla de intentos de Optuna.
    """
    return _optimizar("lineal", entrenamiento, parametros, optuna_cfg)


def optimizar_gradient_boosting(
    entrenamiento: pd.DataFrame,
    parametros: dict[str, Any],
    optuna_cfg: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Optimiza tasa de aprendizaje, árboles, profundidad y regularización.

    Args:
        entrenamiento: Clientes de entrenamiento.
        parametros: Parámetros ``modelo`` (variables, objetivo, semilla).
        optuna_cfg: ``n_trials``, ``n_splits`` y ``espacios`` de búsqueda.

    Returns:
        El modelo reentrenado con todo el entrenamiento, un resumen del estudio
        y la tabla de intentos de Optuna.
    """
    return _optimizar("gradient_boosting", entrenamiento, parametros, optuna_cfg)


def seleccionar_modelo(  # noqa: PLR0913 (un nodo de Kedro recibe sus entradas)
    modelo_lineal: Pipeline,
    resumen_lineal: dict[str, Any],
    modelo_gb: Pipeline,
    resumen_gb: dict[str, Any],
    intentos_lineal: pd.DataFrame,
    intentos_gb: pd.DataFrame,
    parametros: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    """Elige el modelo final con una regla que premia la simplicidad.

    El gradient boosting solo reemplaza al lineal si reduce el RMSE de
    validación cruzada en más de ``tolerancia_rmse`` (relativo). Si no, se
    prefiere el lineal: es más fácil de explicar a Marketing y de implementar.

    Args:
        modelo_lineal: ElasticNet optimizado.
        resumen_lineal: Resumen de su estudio.
        modelo_gb: Gradient boosting optimizado.
        resumen_gb: Resumen de su estudio.
        intentos_lineal: Intentos de Optuna del lineal.
        intentos_gb: Intentos de Optuna del gradient boosting.
        parametros: ``tolerancia_rmse``.

    Returns:
        El modelo final, la explicación de la elección y todos los intentos.
    """
    mejora = 1 - resumen_gb["rmse_cv"] / resumen_lineal["rmse_cv"]
    elegir_gb = mejora > parametros["tolerancia_rmse"]
    final, resumen = (
        (modelo_gb, resumen_gb) if elegir_gb else (modelo_lineal, resumen_lineal)
    )
    seleccion = {
        "modelo_final": resumen["familia"],
        "regla": (
            "Se elige gradient boosting solo si mejora el RMSE de validación "
            f"cruzada del lineal en más de {parametros['tolerancia_rmse']:.1%}."
        ),
        "mejora_relativa_rmse_gb_vs_lineal": float(mejora),
        "candidatos": {"lineal": resumen_lineal, "gradient_boosting": resumen_gb},
    }
    logger.info(
        "Modelo final: %s (mejora del GB sobre el lineal: %.2f%%)",
        resumen["familia"],
        100 * mejora,
    )
    intentos = pd.concat([intentos_lineal, intentos_gb], ignore_index=True)
    return final, seleccion, intentos
