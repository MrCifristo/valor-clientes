"""Nodos que convierten las ventas en una tabla de un cliente por fila."""

from __future__ import annotations

import logging

import pandas as pd

from valor_clientes.clientes import filtrar_ventana, resumir_clientes

logger = logging.getLogger(__name__)


def construir_tabla_modelo(
    ventas: pd.DataFrame, ventanas: dict[str, str]
) -> pd.DataFrame:
    """Arma la tabla de entrada del modelo de valor futuro.

    Las variables se calculan solo con la ventana de observación y el objetivo
    con la ventana siguiente, de modo que el modelo nunca ve información del
    futuro que debe predecir.

    Args:
        ventas: Ventas primarias (una fila por línea de venta).
        ventanas: ``inicio_observacion``, ``corte`` y ``fin_objetivo``.

    Returns:
        Una fila por cliente activo en la observación con sus variables, su
        ``ciudad`` y el objetivo ``ingreso_objetivo`` (0 si no compró en la
        ventana objetivo), además de ``compras_objetivo``.
    """
    variables = resumir_clientes(
        ventas, ventanas["inicio_observacion"], ventanas["corte"]
    )
    futuro = filtrar_ventana(ventas, ventanas["corte"], ventanas["fin_objetivo"])
    objetivo = futuro.groupby("cliente_id").agg(
        ingreso_objetivo=("ingreso", "sum"),
        compras_objetivo=("venta_id", "size"),
    )
    ciudad = ventas.drop_duplicates("cliente_id").set_index("cliente_id")["ciudad"]

    tabla = variables.join(ciudad).join(objetivo)
    tabla[["ingreso_objetivo", "compras_objetivo"]] = tabla[
        ["ingreso_objetivo", "compras_objetivo"]
    ].fillna(0)
    tabla["compras_objetivo"] = tabla["compras_objetivo"].astype(int)
    logger.info("Tabla del modelo: %d clientes, %d columnas", *tabla.shape)
    return tabla.reset_index()
