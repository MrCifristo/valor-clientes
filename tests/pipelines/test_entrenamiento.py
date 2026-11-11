import numpy as np
import pandas as pd
import pytest

from valor_clientes.modelo import captura_top, construir_modelo, seleccionar_variables
from valor_clientes.pipelines.entrenamiento.nodes import (
    dividir_clientes,
    optimizar_gradient_boosting,
    optimizar_lineal,
    seleccionar_modelo,
)
from valor_clientes.pipelines.transformacion.nodes import construir_tabla_modelo

PARAMETROS = {
    "semilla": 0,
    "proporcion_prueba": 0.25,
    "objetivo": "ingreso_objetivo",
    "variables": ["n_compras", "unidades_por_compra", "ingreso", "cat_*"],
    "tolerancia_rmse": 0.01,
}
OPTUNA = {
    "n_trials": 3,
    "n_splits": 3,
    "espacios": {
        "lineal": {
            "alpha": {"tipo": "float", "bajo": 1e-3, "alto": 10.0, "log": True},
            "l1_ratio": {"tipo": "float", "bajo": 0.1, "alto": 1.0},
        },
        "gradient_boosting": {
            "learning_rate": {"tipo": "float", "bajo": 0.05, "alto": 0.2},
            "max_iter": {"tipo": "int", "bajo": 20, "alto": 50},
            "max_depth": {"tipo": "int", "bajo": 2, "alto": 4},
            "min_samples_leaf": {"tipo": "int", "bajo": 5, "alto": 20},
            "l2_regularization": {"tipo": "float", "bajo": 1e-6, "alto": 1.0},
        },
    },
}


@pytest.fixture
def tabla(ventas_sinteticas, ventanas):
    return construir_tabla_modelo(ventas_sinteticas, ventanas)


def test_seleccionar_variables_expande_categorias():
    columnas = pd.Index(["a", "cat_x", "cat_y", "ciudad"])
    assert seleccionar_variables(columnas, ["a", "cat_*", "falta"]) == [
        "a",
        "cat_x",
        "cat_y",
    ]


def test_construir_modelo_rechaza_familia_desconocida():
    with pytest.raises(ValueError, match="Familia desconocida"):
        construir_modelo("red_neuronal", {}, 0)


def test_captura_top():
    real = np.array([10.0, 0.0, 30.0, 60.0])
    assert captura_top(real, real, 0.25) == pytest.approx(0.6)
    assert captura_top(real, -real, 0.25) == pytest.approx(0.0)
    assert captura_top(real, real, 1.0) == pytest.approx(1.0)


def test_dividir_clientes_sin_traslape(tabla):
    entrenamiento, prueba = dividir_clientes(tabla, PARAMETROS)
    assert len(entrenamiento) + len(prueba) == len(tabla)
    assert set(entrenamiento["cliente_id"]).isdisjoint(prueba["cliente_id"])
    assert len(prueba) == pytest.approx(0.25 * len(tabla), abs=1)


def test_optimizar_registra_intentos_y_excluye_identificadores(tabla):
    entrenamiento, _ = dividir_clientes(tabla, PARAMETROS)
    modelo, resumen, intentos = optimizar_lineal(entrenamiento, PARAMETROS, OPTUNA)
    assert len(intentos) == OPTUNA["n_trials"]
    assert set(resumen["mejores_hiperparametros"]) == {"alpha", "l1_ratio"}
    assert "cliente_id" not in resumen["variables"]
    assert "ciudad" not in resumen["variables"]
    assert resumen["rmse_cv"] == pytest.approx(intentos["rmse_cv"].min())
    pred = modelo.predict(entrenamiento[resumen["variables"]])
    assert np.isfinite(pred).all()


def test_optimizacion_es_reproducible(tabla):
    entrenamiento, _ = dividir_clientes(tabla, PARAMETROS)
    _, a, _ = optimizar_gradient_boosting(entrenamiento, PARAMETROS, OPTUNA)
    _, b, _ = optimizar_gradient_boosting(entrenamiento, PARAMETROS, OPTUNA)
    assert a["mejores_hiperparametros"] == b["mejores_hiperparametros"]


def test_seleccion_prefiere_el_lineal_salvo_mejora_clara():
    intentos = pd.DataFrame({"rmse_cv": [1.0]})
    lineal = {"familia": "lineal", "rmse_cv": 100.0}
    gb_poco = {"familia": "gradient_boosting", "rmse_cv": 99.5}
    gb_mucho = {"familia": "gradient_boosting", "rmse_cv": 95.0}
    _, sel, todos = seleccionar_modelo(
        "L", lineal, "G", gb_poco, intentos, intentos, PARAMETROS
    )
    assert sel["modelo_final"] == "lineal"
    assert len(todos) == 2
    final, sel, _ = seleccionar_modelo(
        "L", lineal, "G", gb_mucho, intentos, intentos, PARAMETROS
    )
    assert sel["modelo_final"] == "gradient_boosting"
    assert final == "G"
