import matplotlib.pyplot as plt

from valor_clientes.pipelines.exploracion.nodes import (
    evaluar_segmentacion,
    graficar_estabilidad,
    graficar_regiones,
    graficar_segmentacion,
    medir_estabilidad,
    perfilar_regiones,
    perfilar_tiempo,
)

PARAMETROS = {
    "semilla": 0,
    "k_segmentos": [3],
    "muestra_silueta": 200,
    "conjuntos_segmentacion": {
        "habitos": ["recencia_dias", "n_compras", "cat_*"],
        "valor": ["unidades_por_compra", "ticket_medio"],
    },
}


def test_estabilidad_separa_senal_de_azar(ventas_sinteticas, ventanas):
    # En los datos simulados la cantidad es fija por cliente y lo demás es azar.
    e = medir_estabilidad(ventas_sinteticas, ventanas)
    corr = e["correlacion_entre_periodos"]
    assert corr["unidades_por_compra"] > 0.99
    assert abs(corr["n_compras"]) < 0.3
    assert abs(corr["mezcla_categorias"]) < 0.3
    assert e["pct_clientes_cantidad_constante"] == 100.0
    assert 0.5 < e["indice_dispersion_frecuencia"] < 1.5


def test_segmentacion_por_valor_es_estable_y_por_habitos_no(
    ventas_sinteticas, ventanas
):
    s = evaluar_segmentacion(ventas_sinteticas, ventanas, PARAMETROS)
    assert s["valor"]["por_k"]["3"]["ari_entre_periodos"] > 0.5
    assert s["habitos"]["por_k"]["3"]["ari_entre_periodos"] < 0.2
    assert "cat_dairy" in s["habitos"]["variables"]


def test_perfil_regional_y_temporal(ventas_sinteticas):
    resumen, tabla = perfilar_regiones(ventas_sinteticas)
    assert resumen["ciudades"] == 3
    assert set(tabla.columns) >= {"ciudad", "clientes", "gasto_por_cliente"}
    t = perfilar_tiempo(ventas_sinteticas)
    assert set(t["unidades_promedio_por_descuento"]) == {"0.0", "0.1", "0.2"}


def test_graficas_se_generan(ventas_sinteticas, ventanas):
    e = medir_estabilidad(ventas_sinteticas, ventanas)
    s = evaluar_segmentacion(ventas_sinteticas, ventanas, PARAMETROS)
    _, tabla = perfilar_regiones(ventas_sinteticas)
    for fig in (
        graficar_estabilidad(e),
        graficar_segmentacion(s, 3),
        graficar_regiones(tabla),
    ):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
