"""Nodos de carga, diagnóstico de calidad y limpieza de las siete tablas."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

COLUMNAS_VENTAS = {
    "SalesID": "venta_id",
    "SalesPersonID": "vendedor_id",
    "CustomerID": "cliente_id",
    "ProductID": "producto_id",
    "Quantity": "cantidad",
    "Discount": "descuento",
    "SalesDate": "fecha",
}


def generar_reporte_calidad(  # noqa: PLR0913
    ventas: pd.DataFrame,
    productos: pd.DataFrame,
    categorias: pd.DataFrame,
    clientes: pd.DataFrame,
    ciudades: pd.DataFrame,
    paises: pd.DataFrame,
) -> dict[str, Any]:
    """Documenta los problemas de calidad de las tablas crudas, sin modificarlas.

    Args:
        ventas: Tabla ``sales`` cruda.
        productos: Tabla ``products`` cruda.
        categorias: Tabla ``categories`` cruda.
        clientes: Tabla ``customers`` cruda.
        ciudades: Tabla ``cities`` cruda.
        paises: Tabla ``countries`` cruda.

    Returns:
        Diccionario serializable con conteos de filas, faltantes, duplicados,
        integridad referencial y hallazgos puntuales (``TotalPrice`` vacío,
        códigos de país, nombres de clientes repetidos).
    """
    tablas = {
        "ventas": ventas,
        "productos": productos,
        "categorias": categorias,
        "clientes": clientes,
        "ciudades": ciudades,
        "paises": paises,
    }
    fechas = pd.to_datetime(ventas["SalesDate"], errors="coerce")
    paises_usados = paises[paises["CountryID"].isin(ciudades["CountryID"].unique())]

    reporte: dict[str, Any] = {
        "filas": {k: int(len(t)) for k, t in tablas.items()},
        "faltantes": {
            k: {c: int(n) for c, n in t.isna().sum().items() if n > 0}
            for k, t in tablas.items()
        },
        "duplicados": {
            "ventas_filas_completas": int(ventas.duplicated().sum()),
            "ventas_sales_id": int(ventas["SalesID"].duplicated().sum()),
            "ventas_transaction_number": int(
                ventas["TransactionNumber"].duplicated().sum()
            ),
            "clientes_id": int(clientes["CustomerID"].duplicated().sum()),
            "clientes_mismo_nombre": int(
                clientes.duplicated(["FirstName", "LastName"]).sum()
            ),
            "clientes_mismo_nombre_y_direccion": int(
                clientes.duplicated(["FirstName", "LastName", "Address"]).sum()
            ),
        },
        "integridad": {
            "ventas_con_producto_inexistente": int(
                (~ventas["ProductID"].isin(productos["ProductID"])).sum()
            ),
            "ventas_con_cliente_inexistente": int(
                (~ventas["CustomerID"].isin(clientes["CustomerID"])).sum()
            ),
            "clientes_con_ciudad_inexistente": int(
                (~clientes["CityID"].isin(ciudades["CityID"])).sum()
            ),
            "productos_con_categoria_inexistente": int(
                (~productos["CategoryID"].isin(categorias["CategoryID"])).sum()
            ),
        },
        "hallazgos": {
            "pct_total_price_en_cero": round(
                float((ventas["TotalPrice"] == 0).mean()) * 100, 2
            ),
            "pct_ventas_sin_fecha": round(float(fechas.isna().mean()) * 100, 2),
            "fecha_minima": str(fechas.min()),
            "fecha_maxima": str(fechas.max()),
            "paises_con_clientes": paises_usados[
                ["CountryName", "CountryCode"]
            ].to_dict("records"),
            "rango_precio": [
                float(productos["Price"].min()),
                float(productos["Price"].max()),
            ],
            "rango_cantidad": [
                int(ventas["Quantity"].min()),
                int(ventas["Quantity"].max()),
            ],
            "descuentos_distintos": sorted(
                float(d) for d in ventas["Discount"].unique()
            ),
        },
    }
    logger.info(
        "Calidad: %.1f%% de TotalPrice en cero, %.2f%% de ventas sin fecha",
        reporte["hallazgos"]["pct_total_price_en_cero"],
        reporte["hallazgos"]["pct_ventas_sin_fecha"],
    )
    return reporte


def limpiar_ventas(ventas: pd.DataFrame, parametros: dict[str, Any]) -> pd.DataFrame:
    """Renombra, tipa y filtra las ventas crudas.

    ``TotalPrice`` se descarta porque viene en cero en el 100% de las filas; el
    ingreso se recalcula después con el precio del producto. ``TransactionNumber``
    es único por fila (no agrupa canastas), así que no aporta información.

    Args:
        ventas: Tabla ``sales`` cruda.
        parametros: ``carga_limpieza`` de ``parameters_carga_limpieza.yml``.

    Returns:
        Ventas con nombres en español, fecha como datetime y solo filas válidas.
    """
    v = ventas[list(COLUMNAS_VENTAS)].rename(columns=COLUMNAS_VENTAS)
    v["fecha"] = pd.to_datetime(v["fecha"], errors="coerce")
    n_inicial = len(v)

    v = v.drop_duplicates(subset="venta_id")
    if parametros["eliminar_ventas_sin_fecha"]:
        v = v.dropna(subset=["fecha"])
    validas = (v["cantidad"] >= parametros["cantidad_minima"]) & v["descuento"].between(
        0, parametros["descuento_maximo"]
    )
    v = v.loc[validas].reset_index(drop=True)
    logger.info("Ventas: %d -> %d filas tras limpiar", n_inicial, len(v))
    return v


def construir_ventas_primarias(
    ventas: pd.DataFrame,
    productos: pd.DataFrame,
    categorias: pd.DataFrame,
    clientes: pd.DataFrame,
    ciudades: pd.DataFrame,
) -> pd.DataFrame:
    """Une las dimensiones a cada venta y calcula el ingreso de la línea.

    Args:
        ventas: Salida de :func:`limpiar_ventas`.
        productos: Tabla ``products`` cruda.
        categorias: Tabla ``categories`` cruda.
        clientes: Tabla ``customers`` cruda.
        ciudades: Tabla ``cities`` cruda.

    Returns:
        Una fila por línea de venta con producto, categoría, clase, ciudad del
        cliente e ``ingreso = precio * cantidad * (1 - descuento)``.
    """
    prod = productos[["ProductID", "Price", "CategoryID", "Class"]].merge(
        categorias, on="CategoryID", how="left"
    )
    prod = prod.rename(
        columns={
            "ProductID": "producto_id",
            "Price": "precio",
            "CategoryName": "categoria",
            "Class": "clase",
        }
    ).drop(columns="CategoryID")
    cli = clientes[["CustomerID", "CityID"]].merge(
        ciudades[["CityID", "CityName"]], on="CityID", how="left"
    )
    cli = cli.rename(columns={"CustomerID": "cliente_id", "CityName": "ciudad"}).drop(
        columns="CityID"
    )

    v = ventas.merge(prod, on="producto_id", how="inner").merge(
        cli, on="cliente_id", how="inner"
    )
    v["ingreso"] = v["precio"] * v["cantidad"] * (1 - v["descuento"])
    return v.sort_values("venta_id").reset_index(drop=True)
