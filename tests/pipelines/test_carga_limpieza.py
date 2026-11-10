import pandas as pd
import pytest

from valor_clientes.pipelines.carga_limpieza.nodes import (
    construir_ventas_primarias,
    generar_reporte_calidad,
    limpiar_ventas,
)


def test_reporte_detecta_problemas_de_calidad(tablas_raw):
    reporte = generar_reporte_calidad(*tablas_raw.values())
    assert reporte["filas"]["ventas"] == 8
    assert reporte["hallazgos"]["pct_total_price_en_cero"] == 100.0
    assert reporte["hallazgos"]["pct_ventas_sin_fecha"] == 12.5
    assert reporte["duplicados"]["ventas_sales_id"] == 1
    assert reporte["duplicados"]["clientes_mismo_nombre"] == 1
    assert reporte["duplicados"]["clientes_mismo_nombre_y_direccion"] == 0
    assert reporte["faltantes"]["clientes"] == {"MiddleInitial": 1}
    assert reporte["hallazgos"]["paises_con_clientes"] == [
        {"CountryName": "United States", "CountryCode": "AR"}
    ]
    assert all(v == 0 for v in reporte["integridad"].values())


def test_reporte_no_modifica_las_tablas(tablas_raw):
    copia = tablas_raw["ventas"].copy()
    generar_reporte_calidad(*tablas_raw.values())
    pd.testing.assert_frame_equal(tablas_raw["ventas"], copia)


def test_limpiar_ventas_quita_sin_fecha_duplicados_e_invalidos(
    tablas_raw, parametros_limpieza
):
    limpias = limpiar_ventas(tablas_raw["ventas"], parametros_limpieza)
    # Sale: SalesID 4 (sin fecha), 7 (cantidad 0) y su duplicado.
    assert sorted(limpias["venta_id"]) == [1, 2, 3, 5, 6]
    assert "TotalPrice" not in limpias.columns
    assert pd.api.types.is_datetime64_any_dtype(limpias["fecha"])
    assert limpias["venta_id"].is_unique


def test_limpiar_ventas_conserva_sin_fecha_si_se_configura(
    tablas_raw, parametros_limpieza
):
    parametros_limpieza["eliminar_ventas_sin_fecha"] = False
    limpias = limpiar_ventas(tablas_raw["ventas"], parametros_limpieza)
    assert 4 in set(limpias["venta_id"])


def test_ventas_primarias_calcula_ingreso_y_une_dimensiones(
    tablas_raw, parametros_limpieza
):
    limpias = limpiar_ventas(tablas_raw["ventas"], parametros_limpieza)
    t = tablas_raw
    primarias = construir_ventas_primarias(
        limpias, t["productos"], t["categorias"], t["clientes"], t["ciudades"]
    )
    fila = primarias.set_index("venta_id").loc[2]
    # Leche: 50 x 5 unidades con 10% de descuento.
    assert fila["ingreso"] == pytest.approx(225.0)
    assert fila["categoria"] == "Dairy"
    assert fila["ciudad"] == "Dayton"
    assert len(primarias) == len(limpias)
