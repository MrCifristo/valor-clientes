from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


def test_pipelines_registrados():
    bootstrap_project(Path(__file__).resolve().parents[1])
    assert {"carga_limpieza", "transformacion", "exploracion", "__default__"} <= set(
        pipelines
    )
    nodos = {n.name for n in pipelines["__default__"].nodes}
    assert {
        "generar_reporte_calidad",
        "limpiar_ventas",
        "construir_ventas_primarias",
        "construir_tabla_modelo",
        "medir_estabilidad",
        "evaluar_segmentacion",
    } <= nodos
