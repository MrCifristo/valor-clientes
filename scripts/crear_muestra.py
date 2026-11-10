"""Crea la muestra versionada en data/01_raw/muestra/ a partir del dataset completo.

Elige clientes al azar (con semilla fija) y conserva todas sus compras, para que la
muestra mantenga el comportamiento por cliente que analiza el proyecto.

Uso: python scripts/crear_muestra.py [--clientes 2000] [--semilla 42]
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ORIGEN = RAIZ / "data" / "01_raw"
DESTINO = ORIGEN / "muestra"


def main() -> None:
    """Escribe la muestra de clientes y copia las tablas de dimensión."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clientes", type=int, default=2000)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    DESTINO.mkdir(parents=True, exist_ok=True)
    clientes = pd.read_csv(ORIGEN / "customers.csv")
    elegidos = clientes.sample(n=args.clientes, random_state=args.semilla)
    elegidos.sort_values("CustomerID").to_csv(DESTINO / "customers.csv", index=False)

    ids = set(elegidos["CustomerID"])
    partes = [
        parte[parte["CustomerID"].isin(ids)]
        for parte in pd.read_csv(ORIGEN / "sales.csv", chunksize=1_000_000)
    ]
    pd.concat(partes).to_csv(DESTINO / "sales.csv", index=False)

    for tabla in ("products", "categories", "cities", "countries", "employees"):
        shutil.copy(ORIGEN / f"{tabla}.csv", DESTINO / f"{tabla}.csv")


if __name__ == "__main__":
    main()
