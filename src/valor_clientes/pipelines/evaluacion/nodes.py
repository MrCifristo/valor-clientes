"""Nodos de evaluación: métricas técnicas y de negocio, interpretación y campaña.

Toda comparación se hace sobre los clientes de prueba, que no se usaron ni para
entrenar ni para elegir hiperparámetros. Además del modelo se reportan cuatro
referencias: la línea base («gastará lo mismo que antes»), elegir al azar, una
referencia casi óptima (el promedio de entrenamiento según las unidades por
compra) y el oráculo (el gasto real, inalcanzable).
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from valor_clientes import estilo
from valor_clientes.clientes import resumir_clientes
from valor_clientes.modelo import captura_top

logger = logging.getLogger(__name__)

PREDICTORES = ["linea_base", "lineal", "gradient_boosting", "referencia_casi_optima"]


def _dias(inicio: str, fin: str) -> int:
    """Número de días de la ventana ``[inicio, fin)``."""
    return (pd.Timestamp(fin) - pd.Timestamp(inicio)).days


def predecir_prueba(  # noqa: PLR0913
    prueba: pd.DataFrame,
    entrenamiento: pd.DataFrame,
    modelo_lineal: Pipeline,
    modelo_gb: Pipeline,
    seleccion: dict[str, Any],
    parametros: dict[str, Any],
    ventanas: dict[str, str],
) -> pd.DataFrame:
    """Predice el gasto de mar–abr de los clientes de prueba con cada enfoque.

    Args:
        prueba: Clientes de prueba.
        entrenamiento: Clientes de entrenamiento (para la referencia casi óptima).
        modelo_lineal: ElasticNet optimizado.
        modelo_gb: Gradient boosting optimizado.
        seleccion: Salida de ``seleccionar_modelo`` (variables y modelo final).
        parametros: Parámetros ``modelo``.
        ventanas: Ventanas de observación y objetivo.

    Returns:
        Una fila por cliente de prueba con el gasto real y cada predicción;
        ``final`` es la del modelo elegido.
    """
    variables = seleccion["candidatos"]["lineal"]["variables"]
    objetivo = parametros["objetivo"]
    factor = _dias(ventanas["corte"], ventanas["fin_objetivo"]) / _dias(
        ventanas["inicio_observacion"], ventanas["corte"]
    )
    # Casi óptimo: si solo importa la cantidad fija del cliente, el mejor pronóstico
    # posible es el gasto medio de los clientes de entrenamiento con esa cantidad.
    upc = entrenamiento["unidades_por_compra"].round()
    medias = entrenamiento[objetivo].groupby(upc).mean()
    casi_optima = (
        prueba["unidades_por_compra"]
        .round()
        .map(medias)
        .fillna(entrenamiento[objetivo].mean())
    )
    pred = pd.DataFrame(
        {
            "cliente_id": prueba["cliente_id"],
            "ciudad": prueba["ciudad"],
            "unidades_por_compra": prueba["unidades_por_compra"],
            "real": prueba[objetivo],
            "linea_base": prueba["ingreso"] * factor,
            "lineal": modelo_lineal.predict(prueba[variables]),
            "gradient_boosting": modelo_gb.predict(prueba[variables]),
            "referencia_casi_optima": casi_optima.to_numpy(),
        }
    )
    pred["final"] = pred[seleccion["modelo_final"]]
    return pred


def _metricas(real: pd.Series, pred: pd.Series, fracciones: list[float]) -> dict:
    """MAE, RMSE, R² y captura del top para un predictor."""
    return {
        "mae": float(mean_absolute_error(real, pred)),
        "rmse": float(np.sqrt(mean_squared_error(real, pred))),
        "r2": float(r2_score(real, pred)),
        "captura": {str(f): captura_top(real, pred, f) for f in fracciones},
    }


def _bootstrap_mejora(  # noqa: PLR0913
    real: np.ndarray,
    modelo: np.ndarray,
    base: np.ndarray,
    fraccion: float,
    n: int,
    semilla: int,
) -> dict[str, float]:
    """Intervalo del 95% (bootstrap pareado) de captura modelo − línea base."""
    rng = np.random.default_rng(semilla)
    difs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(real), len(real))
        difs[i] = captura_top(real[idx], modelo[idx], fraccion) - captura_top(
            real[idx], base[idx], fraccion
        )
    return {
        "media": float(difs.mean()),
        "ic95_inferior": float(np.percentile(difs, 2.5)),
        "ic95_superior": float(np.percentile(difs, 97.5)),
        "pct_remuestreos_con_mejora": float((difs > 0).mean() * 100),
    }


def calcular_metricas(
    predicciones: pd.DataFrame,
    tabla_modelo: pd.DataFrame,
    seleccion: dict[str, Any],
    parametros: dict[str, Any],
) -> dict[str, Any]:
    """Compara al modelo contra la línea base, el azar y el óptimo.

    La métrica de negocio es la captura: el % del gasto futuro total que queda
    en el top ``fraccion_campana`` de clientes elegido por cada enfoque. Se
    expresa también como % de la brecha entre el azar y el oráculo que se cierra.

    Args:
        predicciones: Salida de :func:`predecir_prueba`.
        tabla_modelo: Tabla completa (para llevar el impacto a toda la base).
        seleccion: Salida de ``seleccionar_modelo``.
        parametros: Parámetros ``evaluacion``.

    Returns:
        Métricas por predictor, referencias, intervalo bootstrap de la mejora e
        impacto estimado.
    """
    fracciones = parametros["fracciones_curva"]
    f = parametros["fraccion_campana"]
    real = predicciones["real"]
    por_modelo = {p: _metricas(real, predicciones[p], fracciones) for p in PREDICTORES}
    oraculo = {str(x): captura_top(real, real, x) for x in fracciones}
    for m in por_modelo.values():
        m["pct_brecha_cerrada"] = {
            k: float((c - float(k)) / (oraculo[k] - float(k)) * 100)
            for k, c in m["captura"].items()
        }
    final = seleccion["modelo_final"]
    mejora = _bootstrap_mejora(
        real.to_numpy(),
        predicciones["final"].to_numpy(),
        predicciones["linea_base"].to_numpy(),
        f,
        parametros["n_bootstrap"],
        parametros["semilla"],
    )
    gasto_total = float(tabla_modelo["ingreso_objetivo"].sum())
    clave = str(f)
    dif = (
        por_modelo[final]["captura"][clave] - por_modelo["linea_base"]["captura"][clave]
    )
    resultado = {
        "modelo_final": final,
        "clientes_prueba": len(predicciones),
        "fraccion_campana": f,
        "por_modelo": por_modelo,
        "captura_aleatorio": {str(x): x for x in fracciones},
        "captura_oraculo": oraculo,
        "mejora_captura_vs_linea_base": mejora,
        "impacto": {
            "puntos_porcentuales_adicionales": float(dif * 100),
            "mejora_relativa_pct": float(
                dif / por_modelo["linea_base"]["captura"][clave] * 100
            ),
            "gasto_total_base_mar_abr": gasto_total,
            "gasto_adicional_cubierto_por_la_campana": float(dif * gasto_total),
            "nota": (
                "Gasto futuro adicional que queda dentro de la campaña al elegir a "
                "los clientes con el modelo y no con la línea base. No es ingreso "
                "incremental: no hay datos de respuesta a campañas."
            ),
        },
    }
    logger.info(
        "Captura top %.0f%%: línea base %.1f%%, %s %.1f%%, oráculo %.1f%%",
        f * 100,
        por_modelo["linea_base"]["captura"][clave] * 100,
        final,
        por_modelo[final]["captura"][clave] * 100,
        oraculo[clave] * 100,
    )
    return resultado


def explicar_modelo(
    prueba: pd.DataFrame,
    modelo_final: Pipeline,
    seleccion: dict[str, Any],
    parametros_modelo: dict[str, Any],
    parametros: dict[str, Any],
) -> dict[str, Any]:
    """Qué variables usa el modelo: permutation importance y coeficientes.

    La permutation importance mide cuánto empeora el RMSE en prueba al barajar
    cada variable. Si el modelo final es lineal se agregan sus coeficientes
    estandarizados: cambio en el gasto predicho por una desviación estándar.

    Args:
        prueba: Clientes de prueba.
        modelo_final: Modelo elegido.
        seleccion: Salida de ``seleccionar_modelo``.
        parametros_modelo: Parámetros ``modelo``.
        parametros: Parámetros ``evaluacion``.

    Returns:
        Importancias ordenadas de mayor a menor y, si aplica, coeficientes.
    """
    variables = seleccion["candidatos"]["lineal"]["variables"]
    muestra = prueba.sample(
        min(len(prueba), parametros["muestra_importancia"]),
        random_state=parametros["semilla"],
    )
    imp = permutation_importance(
        modelo_final,
        muestra[variables],
        muestra[parametros_modelo["objetivo"]],
        scoring="neg_root_mean_squared_error",
        n_repeats=parametros["repeticiones_importancia"],
        random_state=parametros["semilla"],
    )
    orden = np.argsort(-imp.importances_mean)
    resultado: dict[str, Any] = {
        "metodo": "Aumento del RMSE en prueba al barajar la variable",
        "importancia_permutacion": {
            variables[i]: {
                "aumento_rmse": float(imp.importances_mean[i]),
                "desviacion": float(imp.importances_std[i]),
            }
            for i in orden
        },
    }
    if seleccion["modelo_final"] == "lineal":
        coef = pd.Series(modelo_final.named_steps["modelo"].coef_, index=variables)
        coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
        resultado["coeficientes_estandarizados"] = {
            k: float(v) for k, v in coef.items()
        }
        resultado["variables_descartadas_por_l1"] = [
            k for k, v in coef.items() if v == 0
        ]
    return resultado


def auditar_ciudades(
    predicciones: pd.DataFrame, parametros: dict[str, Any]
) -> pd.DataFrame:
    """Revisa que la campaña no favorezca ni castigue a ninguna ciudad.

    Args:
        predicciones: Salida de :func:`predecir_prueba`.
        parametros: Parámetros ``evaluacion`` (``fraccion_campana``).

    Returns:
        Por ciudad: clientes, % elegido para la campaña, gasto real medio,
        error absoluto medio y sesgo medio (predicho − real).
    """
    p = predicciones.copy()
    corte = p["final"].quantile(1 - parametros["fraccion_campana"])
    p["elegido"] = p["final"] >= corte
    p["error_abs"] = (p["final"] - p["real"]).abs()
    p["sesgo"] = p["final"] - p["real"]
    tabla = p.groupby("ciudad").agg(
        clientes=("cliente_id", "size"),
        pct_elegidos=("elegido", "mean"),
        gasto_real_medio=("real", "mean"),
        mae=("error_abs", "mean"),
        sesgo_medio=("sesgo", "mean"),
    )
    tabla["pct_elegidos"] *= 100
    return tabla.round(2).reset_index().sort_values("pct_elegidos", ascending=False)


def puntuar_clientes(
    ventas: pd.DataFrame,
    modelo_final: Pipeline,
    seleccion: dict[str, Any],
    ventanas: dict[str, str],
    parametros: dict[str, Any],
) -> pd.DataFrame:
    """Lista de la próxima campaña: valor predicho de may–jun para cada cliente.

    Usa el comportamiento más reciente (mar–abr, ventana de duración similar
    a la de entrenamiento: 61 vs. 59 días) para pronosticar los dos meses siguientes.

    Args:
        ventas: Ventas primarias.
        modelo_final: Modelo elegido.
        seleccion: Salida de ``seleccionar_modelo``.
        ventanas: Ventanas del proyecto; ``corte`` a ``fin_objetivo`` es la
            nueva ventana de observación.
        parametros: Parámetros ``evaluacion`` (``segmentos_valor``).

    Returns:
        Un cliente por fila, ordenados de mayor a menor valor predicho, con su
        percentil, segmento y si entra a la campaña.
    """
    variables = seleccion["candidatos"]["lineal"]["variables"]
    recientes = resumir_clientes(ventas, ventanas["corte"], ventanas["fin_objetivo"])
    x = recientes.reindex(columns=variables, fill_value=0.0)
    lista = pd.DataFrame(
        {"valor_predicho": modelo_final.predict(x)}, index=recientes.index
    ).sort_values("valor_predicho", ascending=False)
    lista["percentil"] = (np.arange(len(lista)) + 1) / len(lista) * 100
    limites = np.cumsum(list(parametros["segmentos_valor"].values())) * 100
    lista["segmento"] = pd.cut(
        lista["percentil"],
        bins=[0, *limites[:-1], np.inf],
        labels=list(parametros["segmentos_valor"]),
    ).astype(str)
    lista["en_campana"] = lista["percentil"] <= parametros["fraccion_campana"] * 100
    logger.info("Lista de campaña: %d clientes puntuados", len(lista))
    return lista.round({"valor_predicho": 2, "percentil": 3}).reset_index()


def graficar_captura(metricas: dict[str, Any]) -> plt.Figure:
    """Barras: % del gasto futuro que capta el top elegido por cada enfoque.

    Args:
        metricas: Salida de :func:`calcular_metricas`.

    Returns:
        Figura de matplotlib.
    """
    clave = str(metricas["fraccion_campana"])
    final = metricas["modelo_final"]
    nombre_final = {"lineal": "Modelo lineal", "gradient_boosting": "Gradient boosting"}
    barras = {
        "Al azar": metricas["captura_aleatorio"][clave],
        "Línea base\n(mismo gasto que antes)": metricas["por_modelo"]["linea_base"][
            "captura"
        ][clave],
        f"{nombre_final[final]}\n(Optuna)": metricas["por_modelo"][final]["captura"][
            clave
        ],
        "Oráculo\n(conoce el futuro)": metricas["captura_oraculo"][clave],
    }
    colores = [
        estilo.COLOR_RUIDO,
        estilo.COLOR_RUIDO,
        estilo.COLOR_SENAL,
        estilo.COLOR_EJE,
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=estilo.COLOR_FONDO)
    valores = [v * 100 for v in barras.values()]
    ax.bar(list(barras), valores, color=colores, width=0.55)
    for x, v in enumerate(valores):
        ax.text(x, v + 0.6, f"{v:.1f}%", ha="center", color=estilo.COLOR_TEXTO)
    ax.set_ylim(0, max(valores) * 1.18)
    ax.set_ylabel("% del gasto futuro captado", color=estilo.COLOR_TEXTO_SECUNDARIO)
    estilo.estilo_ejes(ax)
    ax.tick_params(axis="x", colors=estilo.COLOR_TEXTO)
    estilo.titulo(
        ax,
        f"El modelo concentra más gasto en el {float(clave):.0%} de clientes elegido",
        f"Gasto de mar–abr 2018 que capta el top {float(clave):.0%} de cada enfoque "
        f"({metricas['clientes_prueba']:,} clientes de prueba)",
    )
    fig.tight_layout()
    return fig


def graficar_prediccion(predicciones: pd.DataFrame) -> plt.Figure:
    """Gasto real y predicho según las unidades por compra del cliente.

    Args:
        predicciones: Salida de :func:`predecir_prueba`.

    Returns:
        Figura de matplotlib.
    """
    g = predicciones.groupby(predicciones["unidades_por_compra"].round())
    real = g["real"]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=estilo.COLOR_FONDO)
    x = np.asarray(real.mean().index)
    ax.fill_between(
        x,
        real.quantile(0.1) / 1000,
        real.quantile(0.9) / 1000,
        color=estilo.COLOR_RUIDO,
        alpha=0.35,
        linewidth=0,
        label="Gasto real (10%–90% de los clientes)",
    )
    ax.plot(
        x,
        real.mean() / 1000,
        color=estilo.COLOR_TEXTO,
        linewidth=1.5,
        label="Gasto real promedio",
    )
    ax.plot(
        x,
        g["final"].mean() / 1000,
        color=estilo.COLOR_SENAL,
        linewidth=2.5,
        linestyle="--",
        label="Predicción del modelo",
    )
    ax.set_xlabel("Unidades por compra en ene–feb", color=estilo.COLOR_TEXTO_SECUNDARIO)
    ax.set_ylabel("Gasto mar–abr (miles)", color=estilo.COLOR_TEXTO_SECUNDARIO)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    estilo.estilo_ejes(ax)
    estilo.titulo(
        ax,
        "El modelo sigue al gasto promedio; el resto es azar",
        "Clientes de prueba agrupados por la cantidad que compran por línea",
    )
    fig.tight_layout()
    return fig
