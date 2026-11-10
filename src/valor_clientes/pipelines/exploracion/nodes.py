"""Nodos de exploración: qué del comportamiento del cliente se repite y qué es azar.

La pregunta que responden es la que define el proyecto: ¿se puede segmentar a los
clientes por hábitos (frecuencia, recencia, categorías, región) o solo por valor?
La prueba central compara al **mismo cliente** en dos periodos: si un rasgo es
real, quien lo muestra en enero-febrero debe mostrarlo también en marzo-abril.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from valor_clientes import estilo
from valor_clientes.clientes import resumir_clientes

logger = logging.getLogger(__name__)

UMBRAL_SENAL = 0.3
CERO_VISUAL = 0.005
ETIQUETAS = {
    "unidades_por_compra": "Unidades por compra",
    "ticket_medio": "Gasto por compra",
    "ingreso": "Gasto total",
    "precio_medio": "Precio medio comprado",
    "pct_descuento": "% de compras con descuento",
    "mezcla_categorias": "Categorías preferidas",
    "dias_activos": "Días con compra",
    "n_compras": "Número de compras",
    "recencia_dias": "Días desde la última compra",
}


def _fmt(valor: float) -> str:
    """Formatea con dos decimales sin mostrar ``-0.00``."""
    return f"{0.0 if abs(valor) < CERO_VISUAL else valor:.2f}"


def _periodos(
    ventas: pd.DataFrame, ventanas: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resume a los clientes presentes en ambos periodos (A = observación, B = objetivo)."""
    a = resumir_clientes(ventas, ventanas["inicio_observacion"], ventanas["corte"])
    b = resumir_clientes(ventas, ventanas["corte"], ventanas["fin_objetivo"])
    comunes = a.index.intersection(b.index)
    return a.loc[comunes], b.loc[comunes].reindex(columns=a.columns, fill_value=0.0)


def medir_estabilidad(ventas: pd.DataFrame, ventanas: dict[str, str]) -> dict[str, Any]:
    """Correlaciona cada rasgo del cliente entre el periodo A y el periodo B.

    Args:
        ventas: Ventas primarias.
        ventanas: ``inicio_observacion``, ``corte`` y ``fin_objetivo``.

    Returns:
        ``correlacion_entre_periodos`` por rasgo (la mezcla de categorías es el
        promedio de las 11 categorías), el índice de dispersión de la frecuencia
        (varianza / media: 1 = conteo puramente aleatorio tipo Poisson) y el
        porcentaje de clientes que siempre compran la misma cantidad.
    """
    a, b = _periodos(ventas, ventanas)
    rasgos = [r for r in ETIQUETAS if r != "mezcla_categorias"]
    correlaciones = {r: float(a[r].corr(b[r])) for r in rasgos}
    categorias = [c for c in a.columns if c.startswith("cat_")]
    correlaciones["mezcla_categorias"] = float(
        np.mean([a[c].corr(b[c]) for c in categorias])
    )
    cantidades_distintas = ventas.groupby("cliente_id")["cantidad"].nunique()
    n = a["n_compras"]
    resultado = {
        "clientes_comparados": int(len(a)),
        "correlacion_entre_periodos": {
            k: round(v, 4)
            for k, v in sorted(correlaciones.items(), key=lambda kv: -kv[1])
        },
        "indice_dispersion_frecuencia": round(float(n.var() / n.mean()), 3),
        "compras_promedio_periodo_a": round(float(n.mean()), 2),
        "pct_clientes_cantidad_constante": round(
            float((cantidades_distintas == 1).mean()) * 100, 2
        ),
    }
    logger.info(
        "Estabilidad entre periodos: %s", resultado["correlacion_entre_periodos"]
    )
    return resultado


def _columnas(conjunto: list[str], disponibles: pd.Index) -> list[str]:
    """Expande comodines como ``cat_*`` a las columnas existentes."""
    columnas: list[str] = []
    for c in conjunto:
        if c.endswith("*"):
            columnas += [d for d in disponibles if d.startswith(c[:-1])]
        else:
            columnas.append(c)
    return columnas


def evaluar_segmentacion(
    ventas: pd.DataFrame, ventanas: dict[str, str], parametros: dict[str, Any]
) -> dict[str, Any]:
    """Prueba si los segmentos de K-Means se repiten de un periodo al otro.

    Se segmenta a los mismos clientes por separado en A y en B. Si los segmentos
    describen algo real del cliente, ambas segmentaciones coinciden (índice de
    Rand ajustado, ARI, cercano a 1). Un ARI cercano a 0 significa que el
    segmento de un cliente cambia como si se sorteara.

    Args:
        ventas: Ventas primarias.
        ventanas: ``inicio_observacion``, ``corte`` y ``fin_objetivo``.
        parametros: ``exploracion`` de ``parameters_exploracion.yml``.

    Returns:
        Para cada conjunto de variables y cada ``k``: ARI entre periodos y
        coeficiente de silueta en A (separación de los grupos).
    """
    a, b = _periodos(ventas, ventanas)
    semilla = parametros["semilla"]
    muestra = min(parametros["muestra_silueta"], len(a))
    resultado: dict[str, Any] = {}
    for nombre, conjunto in parametros["conjuntos_segmentacion"].items():
        cols = _columnas(conjunto, a.columns)
        xa = StandardScaler().fit_transform(a[cols])
        xb = StandardScaler().fit_transform(b[cols])
        resultado[nombre] = {"variables": cols, "por_k": {}}
        for k in parametros["k_segmentos"]:
            modelo_a = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(xa)
            modelo_b = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(xb)
            resultado[nombre]["por_k"][str(k)] = {
                "ari_entre_periodos": round(
                    float(adjusted_rand_score(modelo_a.labels_, modelo_b.labels_)), 4
                ),
                "silueta_periodo_a": round(
                    float(
                        silhouette_score(
                            xa,
                            modelo_a.labels_,
                            sample_size=muestra,
                            random_state=semilla,
                        )
                    ),
                    4,
                ),
            }
    logger.info("Segmentación: %s", {k: v["por_k"] for k, v in resultado.items()})
    return resultado


def perfilar_regiones(ventas: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compara el gasto por cliente y la mezcla de categorías entre ciudades.

    Args:
        ventas: Ventas primarias.

    Returns:
        Un resumen (variación entre ciudades, prueba F de diferencia de medias y
        la mayor desviación de una categoría respecto a la mezcla nacional) y la
        tabla por ciudad.
    """
    por_cliente = (
        ventas.groupby(["ciudad", "cliente_id"])["ingreso"].sum().reset_index()
    )
    tabla = por_cliente.groupby("ciudad").agg(
        clientes=("cliente_id", "size"), gasto_por_cliente=("ingreso", "mean")
    )
    tabla["ingreso_total"] = ventas.groupby("ciudad")["ingreso"].sum()
    tabla = tabla.sort_values("gasto_por_cliente", ascending=False)

    grupos = [g["ingreso"].to_numpy() for _, g in por_cliente.groupby("ciudad")]
    prueba = stats.f_oneway(*grupos)
    mezcla_ciudad = pd.crosstab(
        ventas["ciudad"], ventas["categoria"], normalize="index"
    )
    mezcla_pais = ventas["categoria"].value_counts(normalize=True)
    desviacion = (mezcla_ciudad - mezcla_pais).abs().to_numpy().max() * 100

    g = tabla["gasto_por_cliente"]
    resumen = {
        "ciudades": int(len(tabla)),
        "clientes_por_ciudad_min_max": [
            int(tabla["clientes"].min()),
            int(tabla["clientes"].max()),
        ],
        "gasto_por_cliente_cv_pct": round(float(g.std() / g.mean()) * 100, 2),
        "gasto_por_cliente_max_sobre_min": round(float(g.max() / g.min()), 3),
        "prueba_f_gasto_por_ciudad": {
            "f": round(float(prueba.statistic), 3),
            "p_valor": round(float(prueba.pvalue), 4),
        },
        "max_desviacion_categoria_pp": round(float(desviacion), 2),
    }
    return resumen, tabla.reset_index()


def perfilar_tiempo(ventas: pd.DataFrame) -> dict[str, Any]:
    """Mide si las ventas cambian por hora, día de la semana o descuento.

    Args:
        ventas: Ventas primarias.

    Returns:
        Participación mínima y máxima por hora y por día, variación del ingreso
        diario y unidades promedio por nivel de descuento.
    """
    por_hora = ventas["fecha"].dt.hour.value_counts(normalize=True) * 100
    diario = ventas.groupby(ventas["fecha"].dt.normalize())["ingreso"].sum()
    # Un día de la semana se compara por su promedio diario: el periodo no tiene
    # el mismo número de lunes que de domingos.
    dia_semana = diario.groupby(diario.index.dayofweek).mean()
    return {
        "pct_ventas_por_hora_min_max": [
            round(float(por_hora.min()), 3),
            round(float(por_hora.max()), 3),
        ],
        "ingreso_diario_cv_pct": round(float(diario.std() / diario.mean()) * 100, 3),
        "ingreso_por_dia_semana_max_sobre_min": round(
            float(dia_semana.max() / dia_semana.min()), 4
        ),
        "unidades_promedio_por_descuento": {
            str(k): round(float(v), 3)
            for k, v in ventas.groupby("descuento")["cantidad"].mean().items()
        },
        "dias": int(len(diario)),
    }


def graficar_estabilidad(estabilidad: dict[str, Any]) -> plt.Figure:
    """Barra horizontal: correlación del mismo rasgo entre los dos periodos.

    Args:
        estabilidad: Salida de :func:`medir_estabilidad`.

    Returns:
        Figura de matplotlib.
    """
    corr = pd.Series(estabilidad["correlacion_entre_periodos"]).sort_values()
    colores = [
        estilo.COLOR_SENAL if v >= UMBRAL_SENAL else estilo.COLOR_RUIDO for v in corr
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=estilo.COLOR_FONDO)
    ax.barh(
        [ETIQUETAS[i] for i in corr.index], corr.to_numpy(), color=colores, height=0.6
    )
    for y, v in enumerate(corr):
        ax.text(
            max(v, 0) + 0.015,
            y,
            _fmt(v),
            va="center",
            color=estilo.COLOR_TEXTO_SECUNDARIO,
            fontsize=9,
        )
    ax.axvline(0, color=estilo.COLOR_EJE, linewidth=1)
    ax.set_xlim(min(-0.05, corr.min() - 0.05), 1.1)
    estilo.estilo_ejes(ax, grilla="x")
    ax.tick_params(axis="y", colors=estilo.COLOR_TEXTO)
    estilo.titulo(
        ax,
        "Del cliente solo se repite cuánto compra, no cuándo ni qué",
        "Correlación del mismo cliente entre ene–feb y mar–abr 2018 "
        f"({estabilidad['clientes_comparados']:,} clientes)",
    )
    fig.tight_layout()
    return fig


def graficar_segmentacion(segmentacion: dict[str, Any], k: int) -> plt.Figure:
    """Barras: qué tanto coincide la segmentación de un periodo con la del otro.

    Args:
        segmentacion: Salida de :func:`evaluar_segmentacion`.
        k: Número de segmentos que se muestra.

    Returns:
        Figura de matplotlib.
    """
    nombres = {
        "rfm_clasico": "RFM clásico\n(recencia, frecuencia, gasto)",
        "habitos": "Hábitos\n(cuándo y qué compra)",
        "valor": "Valor\n(unidades y gasto por compra)",
    }
    ari = {n: v["por_k"][str(k)]["ari_entre_periodos"] for n, v in segmentacion.items()}
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=estilo.COLOR_FONDO)
    etiquetas = [nombres.get(n, n) for n in ari]
    colores = [
        estilo.COLOR_SENAL if v >= UMBRAL_SENAL else estilo.COLOR_RUIDO
        for v in ari.values()
    ]
    ax.bar(etiquetas, list(ari.values()), color=colores, width=0.55)
    for x, v in enumerate(ari.values()):
        ax.text(
            x,
            max(v, 0) + 0.02,
            _fmt(v),
            ha="center",
            color=estilo.COLOR_TEXTO,
            fontsize=10,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(
        "Coincidencia entre periodos (ARI)", color=estilo.COLOR_TEXTO_SECUNDARIO
    )
    estilo.estilo_ejes(ax)
    ax.tick_params(axis="x", colors=estilo.COLOR_TEXTO)
    estilo.titulo(
        ax,
        "Solo los segmentos por valor se mantienen de un periodo a otro",
        f"K-Means con {k} segmentos, entrenado por separado en cada periodo. "
        "1 = mismos segmentos, 0 = como al azar",
    )
    fig.tight_layout()
    return fig


def graficar_regiones(tabla_regional: pd.DataFrame) -> plt.Figure:
    """Gasto promedio por cliente en cada ciudad, ordenado, con la media nacional.

    Args:
        tabla_regional: Tabla por ciudad de :func:`perfilar_regiones`.

    Returns:
        Figura de matplotlib.
    """
    t = tabla_regional.sort_values("gasto_por_cliente", ascending=False)
    media = t["gasto_por_cliente"].mean()
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=estilo.COLOR_FONDO)
    ax.bar(
        range(len(t)),
        t["gasto_por_cliente"] / 1000,
        color=estilo.COLOR_SENAL,
        width=0.8,
    )
    ax.axhline(media / 1000, color=estilo.COLOR_ACENTO, linewidth=2)
    ax.text(
        len(t) - 0.5,
        media / 1000 * 1.02,
        f"Promedio: {media / 1000:,.1f} mil",
        ha="right",
        va="bottom",
        color=estilo.COLOR_TEXTO,
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_xlabel(
        f"{len(t)} ciudades, de mayor a menor", color=estilo.COLOR_TEXTO_SECUNDARIO
    )
    ax.set_ylabel("Gasto por cliente (miles)", color=estilo.COLOR_TEXTO_SECUNDARIO)
    ax.set_ylim(0, t["gasto_por_cliente"].max() / 1000 * 1.15)
    estilo.estilo_ejes(ax)
    estilo.titulo(
        ax,
        "Un cliente gasta casi lo mismo en cualquier ciudad",
        "Gasto promedio por cliente, ene–may 2018.",
    )
    fig.tight_layout()
    return fig
