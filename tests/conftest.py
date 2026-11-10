import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tablas_raw() -> dict[str, pd.DataFrame]:
    """Siete tablas mínimas con los problemas de calidad del dataset real."""
    ventas = pd.DataFrame(
        {
            "SalesID": [1, 2, 3, 4, 5, 6, 7, 7],
            "SalesPersonID": [1, 2, 1, 2, 1, 2, 1, 1],
            "CustomerID": [10, 10, 20, 20, 10, 20, 30, 30],
            "ProductID": [100, 200, 100, 200, 200, 100, 100, 100],
            "Quantity": [5, 5, 2, 2, 5, 2, 0, 0],
            "Discount": [0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            "TotalPrice": [0.0] * 8,
            "SalesDate": [
                "2018-01-05 10:00:00.000",
                "2018-02-10 11:00:00.000",
                "2018-01-20 12:00:00.000",
                None,
                "2018-03-15 09:00:00.000",
                "2018-04-02 08:00:00.000",
                "2018-01-03 08:00:00.000",
                "2018-01-03 08:00:00.000",
            ],
            "TransactionNumber": list("ABCDEFGG"),
        }
    )
    productos = pd.DataFrame(
        {
            "ProductID": [100, 200],
            "ProductName": ["Pan", "Leche"],
            "Price": [10.0, 50.0],
            "CategoryID": [1, 2],
            "Class": ["Low", "High"],
            "ModifyDate": ["2018-01-01", "2018-01-01"],
            "Resistant": ["Durable", "Weak"],
            "IsAllergic": ["False", "Unknown"],
            "VitalityDays": [0.0, 30.0],
        }
    )
    categorias = pd.DataFrame(
        {"CategoryID": [1, 2], "CategoryName": ["Grain", "Dairy"]}
    )
    clientes = pd.DataFrame(
        {
            "CustomerID": [10, 20, 30],
            "FirstName": ["Ana", "Ana", "Luis"],
            "MiddleInitial": ["T", None, "R"],
            "LastName": ["Paz", "Paz", "Gil"],
            "CityID": [1, 2, 1],
            "Address": ["Calle 1", "Calle 2", "Calle 3"],
        }
    )
    ciudades = pd.DataFrame(
        {
            "CityID": [1, 2],
            "CityName": ["Dayton", "Buffalo"],
            "Zipcode": [1, 2],
            "CountryID": [32, 32],
        }
    )
    paises = pd.DataFrame(
        {
            "CountryID": [1, 32],
            "CountryName": ["Armenia", "United States"],
            "CountryCode": ["AN", "AR"],
        }
    )
    return {
        "ventas": ventas,
        "productos": productos,
        "categorias": categorias,
        "clientes": clientes,
        "ciudades": ciudades,
        "paises": paises,
    }


@pytest.fixture
def parametros_limpieza() -> dict:
    return {
        "eliminar_ventas_sin_fecha": True,
        "cantidad_minima": 1,
        "descuento_maximo": 1.0,
    }


@pytest.fixture
def ventanas() -> dict[str, str]:
    return {
        "inicio_observacion": "2018-01-01",
        "corte": "2018-03-01",
        "fin_objetivo": "2018-05-01",
    }


@pytest.fixture
def ventas_sinteticas() -> pd.DataFrame:
    """Ventas primarias simuladas: cantidad fija por cliente, todo lo demás al azar."""
    rng = np.random.default_rng(0)
    n_clientes, n_filas = 300, 12_000
    cliente = rng.integers(1, n_clientes + 1, n_filas)
    cantidad_cliente = rng.integers(1, 26, n_clientes + 1)
    precio = rng.uniform(1, 100, n_filas)
    descuento = rng.choice([0.0, 0.1, 0.2], n_filas, p=[0.8, 0.1, 0.1])
    fechas = pd.Timestamp("2018-01-01") + pd.to_timedelta(
        rng.uniform(0, 120, n_filas), unit="D"
    )
    ventas = pd.DataFrame(
        {
            "venta_id": np.arange(n_filas),
            "vendedor_id": rng.integers(1, 24, n_filas),
            "cliente_id": cliente,
            "producto_id": rng.integers(1, 50, n_filas),
            "cantidad": cantidad_cliente[cliente],
            "descuento": descuento,
            "fecha": fechas,
            "precio": precio,
            "categoria": rng.choice(["Dairy", "Grain", "Shell fish"], n_filas),
            "clase": rng.choice(["Low", "High"], n_filas),
            "ciudad": rng.choice(["Dayton", "Buffalo", "Chicago"], n_clientes + 1)[
                cliente
            ],
        }
    )
    ventas["ingreso"] = (
        ventas["precio"] * ventas["cantidad"] * (1 - ventas["descuento"])
    )
    return ventas
