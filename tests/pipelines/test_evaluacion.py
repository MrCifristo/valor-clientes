import matplotlib.pyplot as plt
import pytest

from valor_clientes.pipelines.entrenamiento.nodes import (
    dividir_clientes,
    optimizar_gradient_boosting,
    optimizar_lineal,
    seleccionar_modelo,
)
from valor_clientes.pipelines.evaluacion.nodes import (
    auditar_ciudades,
    calcular_metricas,
    explicar_modelo,
    graficar_captura,
    graficar_prediccion,
    predecir_prueba,
    puntuar_clientes,
)
from valor_clientes.pipelines.transformacion.nodes import construir_tabla_modelo

from .test_entrenamiento import OPTUNA, PARAMETROS

EVALUACION = {
    "semilla": 0,
    "fraccion_campana": 0.2,
    "fracciones_curva": [0.1, 0.2, 0.5],
    "n_bootstrap": 50,
    "repeticiones_importancia": 2,
    "muestra_importancia": 100,
    "segmentos_valor": {"Top": 0.2, "Alto": 0.3, "Medio": 0.3, "Bajo": 0.2},
}


@pytest.fixture(scope="module")
def entrenado(ventas_sinteticas, ventanas):
    ventas = ventas_sinteticas
    tabla = construir_tabla_modelo(ventas, ventanas)
    entrenamiento, prueba = dividir_clientes(tabla, PARAMETROS)
    lin = optimizar_lineal(entrenamiento, PARAMETROS, OPTUNA)
    gb = optimizar_gradient_boosting(entrenamiento, PARAMETROS, OPTUNA)
    final, seleccion, _ = seleccionar_modelo(
        lin[0], lin[1], gb[0], gb[1], lin[2], gb[2], PARAMETROS
    )
    pred = predecir_prueba(
        prueba, entrenamiento, lin[0], gb[0], seleccion, PARAMETROS, ventanas
    )
    return {
        "ventas": ventas,
        "ventanas": ventanas,
        "tabla": tabla,
        "prueba": prueba,
        "final": final,
        "seleccion": seleccion,
        "pred": pred,
    }


def test_predicciones_tienen_todos_los_enfoques(entrenado):
    pred = entrenado["pred"]
    assert len(pred) == len(entrenado["prueba"])
    for col in ["real", "linea_base", "lineal", "gradient_boosting", "final"]:
        assert pred[col].notna().all()


def test_metricas_ordenan_azar_modelo_y_oraculo(entrenado):
    m = calcular_metricas(
        entrenado["pred"], entrenado["tabla"], entrenado["seleccion"], EVALUACION
    )
    final = m["por_modelo"][m["modelo_final"]]["captura"]["0.2"]
    assert 0.2 < final <= m["captura_oraculo"]["0.2"]
    # En los datos simulados la cantidad por cliente es la señal: el modelo
    # debe ganarle a la línea base, que arrastra el azar de la frecuencia.
    assert (
        m["por_modelo"][m["modelo_final"]]["rmse"]
        < m["por_modelo"]["linea_base"]["rmse"]
    )
    ic = m["mejora_captura_vs_linea_base"]
    assert ic["ic95_inferior"] <= ic["media"] <= ic["ic95_superior"]
    assert "gasto_adicional_cubierto_por_la_campana" in m["impacto"]
    plt.close(graficar_captura(m))


def test_unidades_por_compra_es_la_variable_principal(entrenado):
    imp = explicar_modelo(
        entrenado["prueba"],
        entrenado["final"],
        entrenado["seleccion"],
        PARAMETROS,
        EVALUACION,
    )
    assert next(iter(imp["importancia_permutacion"])) == "unidades_por_compra"


def test_auditoria_y_lista_de_campana(entrenado):
    audit = auditar_ciudades(entrenado["pred"], EVALUACION)
    assert audit["clientes"].sum() == len(entrenado["pred"])
    lista = puntuar_clientes(
        entrenado["ventas"],
        entrenado["final"],
        entrenado["seleccion"],
        entrenado["ventanas"],
        EVALUACION,
    )
    assert lista["cliente_id"].is_unique
    assert lista["valor_predicho"].is_monotonic_decreasing
    assert lista["en_campana"].mean() == pytest.approx(0.2, abs=0.01)
    assert set(lista["segmento"]) == {"Top", "Alto", "Medio", "Bajo"}
    assert (lista.loc[lista["en_campana"], "segmento"] == "Top").all()
    plt.close(graficar_prediccion(entrenado["pred"]))
