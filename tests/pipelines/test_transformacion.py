import pandas as pd
import pytest

from valor_clientes.clientes import nombre_columna_categoria, resumir_clientes
from valor_clientes.pipelines.transformacion.nodes import construir_tabla_modelo


def test_nombre_columna_categoria():
    assert nombre_columna_categoria("Shell fish") == "cat_shell_fish"
    assert nombre_columna_categoria("Confections") == "cat_confections"


def test_resumir_clientes_calcula_rasgos(ventas_sinteticas, ventanas):
    r = resumir_clientes(
        ventas_sinteticas, ventanas["inicio_observacion"], ventanas["corte"]
    )
    assert r.index.is_unique
    assert (r["unidades_por_compra"] == r["unidades_por_compra"].round()).all()
    assert r["ticket_medio"].mul(r["n_compras"]).sub(r["ingreso"]).abs().max() < 1e-6
    cats = [c for c in r.columns if c.startswith("cat_")]
    assert r[cats].sum(axis=1).round(6).eq(1).all()
    assert (r["recencia_dias"] >= 0).all()
    assert (r["dias_activos"] <= r["n_compras"]).all()


def test_tabla_modelo_no_usa_el_futuro_en_las_variables(ventas_sinteticas, ventanas):
    tabla = construir_tabla_modelo(ventas_sinteticas, ventanas)
    # Alterar el futuro no puede cambiar ninguna variable de entrada.
    alterada = ventas_sinteticas.copy()
    futuro = alterada["fecha"] >= pd.Timestamp(ventanas["corte"])
    alterada.loc[futuro, "ingreso"] *= 10
    tabla_alterada = construir_tabla_modelo(alterada, ventanas)
    variables = [c for c in tabla.columns if not c.endswith("_objetivo")]
    pd.testing.assert_frame_equal(tabla[variables], tabla_alterada[variables])
    assert tabla_alterada["ingreso_objetivo"].sum() == pytest.approx(
        tabla["ingreso_objetivo"].sum() * 10
    )


def test_tabla_modelo_objetivo_coincide_con_las_ventas(ventas_sinteticas, ventanas):
    tabla = construir_tabla_modelo(ventas_sinteticas, ventanas).set_index("cliente_id")
    v = ventas_sinteticas
    en_objetivo = v[
        (v["fecha"] >= ventanas["corte"]) & (v["fecha"] < ventanas["fin_objetivo"])
    ]
    esperado = en_objetivo.groupby("cliente_id")["ingreso"].sum()
    comunes = esperado.index.intersection(tabla.index)
    pd.testing.assert_series_equal(
        tabla.loc[comunes, "ingreso_objetivo"], esperado.loc[comunes], check_names=False
    )
    assert tabla["ciudad"].notna().all()
