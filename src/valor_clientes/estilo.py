"""Colores y estilo comunes de las gráficas del proyecto.

Paleta categórica validada para daltonismo (azul y naranja) sobre fondo claro.
Azul = lo que se repite en el cliente (señal); gris = lo que se comporta como azar.
"""

COLOR_SENAL = "#2a78d6"
COLOR_ACENTO = "#eb6834"
COLOR_RUIDO = "#b4b2a9"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_SECUNDARIO = "#52514e"
COLOR_FONDO = "#fcfcfb"
COLOR_EJE = "#d8d7d2"
COLOR_GRILLA = "#ecebe7"


def estilo_ejes(ax, grilla: str = "y") -> None:
    """Fondo claro, sin bordes superiores ni laterales y grilla tenue.

    Args:
        ax: Ejes de matplotlib a los que se aplica el estilo.
        grilla: Eje que lleva grilla (``"x"``, ``"y"`` o ``"both"``).
    """
    ax.set_facecolor(COLOR_FONDO)
    ax.tick_params(colors=COLOR_TEXTO_SECUNDARIO, length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_EJE)
    ax.grid(axis=grilla, color=COLOR_GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def titulo(ax, texto: str, subtitulo: str | None = None) -> None:
    """Título alineado a la izquierda en negrita, con subtítulo opcional.

    Args:
        ax: Ejes de matplotlib.
        texto: Título principal (el mensaje de la gráfica).
        subtitulo: Línea secundaria con la definición o la fuente.
    """
    ax.set_title(
        texto,
        loc="left",
        color=COLOR_TEXTO,
        fontweight="bold",
        pad=22 if subtitulo else 8,
    )
    if subtitulo:
        ax.text(
            0,
            1.02,
            subtitulo,
            transform=ax.transAxes,
            color=COLOR_TEXTO_SECUNDARIO,
            fontsize=9,
            va="bottom",
        )
