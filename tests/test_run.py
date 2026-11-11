from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


def test_pipelines_registrados():
    bootstrap_project(Path(__file__).resolve().parents[1])
    assert {
        "carga_limpieza",
        "transformacion",
        "exploracion",
        "entrenamiento",
        "evaluacion",
        "__default__",
    } <= set(pipelines)
    nodos = {n.name for n in pipelines["__default__"].nodes}
    assert {
        "generar_reporte_calidad",
        "limpiar_ventas",
        "construir_ventas_primarias",
        "construir_tabla_modelo",
        "medir_estabilidad",
        "evaluar_segmentacion",
        "dividir_clientes",
        "optimizar_lineal",
        "optimizar_gradient_boosting",
        "seleccionar_modelo",
        "predecir_prueba",
        "calcular_metricas",
        "puntuar_clientes",
    } <= nodos
