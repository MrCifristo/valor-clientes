"""Piezas comunes del modelo de valor futuro.

Las usan los pipelines ``entrenamiento`` y ``evaluacion``: qué variables entran,
cómo se arma cada familia de modelo y cómo se mide la métrica de negocio.
Tenerlas en un solo lugar garantiza que ambos pipelines midan lo mismo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FAMILIAS = ("lineal", "gradient_boosting")


def seleccionar_variables(columnas: pd.Index, patrones: list[str]) -> list[str]:
    """Expande la lista de variables; ``"cat_*"`` equivale a todas las ``cat_``.

    Args:
        columnas: Columnas disponibles en la tabla.
        patrones: Nombres exactos o prefijos terminados en ``*``.

    Returns:
        Columnas a usar como variables, en el orden de ``patrones``.
    """
    variables: list[str] = []
    for patron in patrones:
        if patron.endswith("*"):
            variables += sorted(c for c in columnas if c.startswith(patron[:-1]))
        elif patron in columnas:
            variables.append(patron)
    return variables


def construir_modelo(
    familia: str, hiperparametros: dict[str, Any], semilla: int
) -> Pipeline:
    """Arma el estimador de una familia con sus hiperparámetros.

    El escalado vive dentro del ``Pipeline``, así se ajusta solo con los datos
    de entrenamiento de cada fold y no hay fuga hacia la validación.

    Args:
        familia: ``"lineal"`` (ElasticNet con variables estandarizadas) o
            ``"gradient_boosting"`` (HistGradientBoostingRegressor).
        hiperparametros: Argumentos del estimador.
        semilla: Semilla del estimador.

    Returns:
        Un ``Pipeline`` de scikit-learn sin entrenar.

    Raises:
        ValueError: Si la familia no existe.
    """
    if familia == "lineal":
        estimador = ElasticNet(**hiperparametros, max_iter=10_000, random_state=semilla)
        return Pipeline([("escalar", StandardScaler()), ("modelo", estimador)])
    if familia == "gradient_boosting":
        estimador = HistGradientBoostingRegressor(
            **hiperparametros, early_stopping=False, random_state=semilla
        )
        return Pipeline([("modelo", estimador)])
    raise ValueError(f"Familia desconocida: {familia}. Opciones: {FAMILIAS}")


def captura_top(real: np.ndarray, puntaje: np.ndarray, fraccion: float) -> float:
    """Fracción del gasto futuro total que concentra el top de clientes.

    Es la métrica de negocio: si Marketing solo puede contactar a una
    ``fraccion`` de los clientes y los elige por ``puntaje``, ¿qué parte del
    gasto futuro queda dentro de la campaña?

    Args:
        real: Gasto futuro observado de cada cliente.
        puntaje: Valor predicho con el que se ordena a los clientes.
        fraccion: Proporción de clientes contactados (p. ej. 0.2).

    Returns:
        Fracción entre 0 y 1 del gasto total que capta el top elegido.
    """
    real = np.asarray(real, dtype=float)
    k = max(1, int(round(len(real) * fraccion)))
    top = np.argsort(-np.asarray(puntaje, dtype=float), kind="stable")[:k]
    return float(real[top].sum() / real.sum())
