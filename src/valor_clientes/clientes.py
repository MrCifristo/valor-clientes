"""Resumen del comportamiento de cada cliente en una ventana de tiempo.

Lo usan dos pipelines: ``transformacion`` (variables del modelo) y ``exploracion``
(comparar el mismo cliente en dos periodos). Mantenerlo en un solo lugar garantiza
que ambos midan exactamente lo mismo.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def nombre_columna_categoria(categoria: str) -> str:
    """Convierte un nombre de categoría en un nombre de columna seguro.

    Args:
        categoria: Nombre original, por ejemplo ``"Shell fish"``.

    Returns:
        Nombre con prefijo ``cat_`` en minúsculas y sin espacios, p. ej. ``cat_shell_fish``.
    """
    texto = unicodedata.normalize("NFKD", categoria).encode("ascii", "ignore").decode()
    return "cat_" + re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")


def filtrar_ventana(ventas: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Devuelve las ventas con fecha en ``[inicio, fin)``.

    Args:
        ventas: Ventas con la columna ``fecha`` de tipo datetime.
        inicio: Fecha inicial incluida (``YYYY-MM-DD``).
        fin: Fecha final excluida (``YYYY-MM-DD``).

    Returns:
        Subconjunto de las ventas dentro de la ventana.
    """
    mascara = (ventas["fecha"] >= pd.Timestamp(inicio)) & (
        ventas["fecha"] < pd.Timestamp(fin)
    )
    return ventas.loc[mascara]


def resumir_clientes(ventas: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Resume, por cliente, cuánto, cada cuánto y qué compró en una ventana.

    Args:
        ventas: Ventas primarias (una fila por línea de venta) con ``cliente_id``,
            ``fecha``, ``cantidad``, ``precio``, ``descuento``, ``ingreso`` y
            ``categoria``.
        inicio: Fecha inicial incluida.
        fin: Fecha final excluida; también es la fecha de referencia de la recencia.

    Returns:
        Una fila por cliente con frecuencia (``n_compras``, ``dias_activos``),
        recencia (``recencia_dias``), valor (``ingreso``, ``ticket_medio``,
        ``unidades_por_compra``, ``precio_medio``), uso de descuentos
        (``pct_descuento``) y la participación de cada categoría (``cat_*``).
    """
    v = filtrar_ventana(ventas, inicio, fin)
    grupos = v.groupby("cliente_id")
    resumen = grupos.agg(
        n_compras=("venta_id", "size"),
        unidades_total=("cantidad", "sum"),
        ingreso=("ingreso", "sum"),
        precio_medio=("precio", "mean"),
        ultima_compra=("fecha", "max"),
    )
    resumen["dias_activos"] = (
        v["fecha"].dt.normalize().groupby(v["cliente_id"]).nunique()
    )
    resumen["pct_descuento"] = (v["descuento"] > 0).groupby(v["cliente_id"]).mean()
    resumen["unidades_por_compra"] = resumen["unidades_total"] / resumen["n_compras"]
    resumen["ticket_medio"] = resumen["ingreso"] / resumen["n_compras"]
    resumen["recencia_dias"] = (pd.Timestamp(fin) - resumen["ultima_compra"]).dt.days

    mezcla = pd.crosstab(v["cliente_id"], v["categoria"], normalize="index")
    mezcla.columns = [nombre_columna_categoria(c) for c in mezcla.columns]

    columnas = [
        "n_compras",
        "dias_activos",
        "recencia_dias",
        "unidades_total",
        "unidades_por_compra",
        "ingreso",
        "ticket_medio",
        "precio_medio",
        "pct_descuento",
    ]
    return resumen[columnas].join(mezcla).fillna(0.0)
