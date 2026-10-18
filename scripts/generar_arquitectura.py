"""Genera el diagrama de arquitectura de la aplicación en assets/arquitectura.png y assets/arquitectura.svg.

Ejecutar desde la raíz:
    python -m scripts.generar_arquitectura

Dibuja con Matplotlib sobre el lienzo Agg, con una `Figure` creada sin pyplot, así que no deja
figuras abiertas ni necesita pantalla. Las cajas y las flechas son datos (`CAJAS`, `FLECHAS`):
para cambiar el diagrama se editan esas tablas, no el código de dibujo. El PNG sale a 200 ppp
con unos 1.800 px de ancho; el SVG conserva el texto convertido en trazos para verse igual en
cualquier equipo.

Autor: Kevin Alexis Barrera Llerena 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text

Lado = Literal["arriba", "abajo", "izquierda", "derecha"]

CARPETA_ASSETS: Final = Path(__file__).resolve().parents[1] / "assets"
LEYENDA: Final = "Python calcula; el modelo interpreta; la persona verifica"
DPI: Final = 200
TAMANO_PULGADAS: Final = (9.0, 6.5)
ANCHO_UNIDADES: Final = 180.0
ALTO_UNIDADES: Final = 130.0
COLOR_TEXTO: Final = "#1F2A33"
COLOR_FLECHA: Final = "#4B5863"
_PX_POR_UNIDAD: Final = TAMANO_PULGADAS[0] * DPI / ANCHO_UNIDADES

# Columnas de la cuadrícula: cuatro cajas de 34 unidades con 11,33 de separación.
_ANCHO_CAJA: Final = 34.0
_COLUMNAS: Final = (5.0, 50.33, 95.67, 141.0)

GRUPOS: Final[dict[str, tuple[str, str, str]]] = {
    "interfaz": ("#E3ECF5", "#3E5C76", "Persona e interfaz"),
    "python": ("#E6F0E8", "#3F6B4A", "Cálculo en Python"),
    "agente": ("#F6EEDF", "#8A6A35", "Agente acotado y modelo local"),
    "control": ("#EEEAF4", "#5B5478", "Verificación y trazabilidad"),
}


@dataclass(frozen=True)
class Caja:
    """Componente del diagrama: rectángulo redondeado con título y hasta dos líneas de detalle."""

    clave: str
    titulo: str
    detalle: tuple[str, ...]
    x: float
    y: float
    ancho: float
    alto: float
    grupo: str

    @property
    def centro(self) -> tuple[float, float]:
        """Centro geométrico de la caja en unidades del diagrama."""
        return self.x + self.ancho / 2, self.y + self.alto / 2


@dataclass(frozen=True)
class Flecha:
    """Dependencia dirigida entre dos cajas; el desplazamiento corre el anclaje sobre el borde."""

    origen: str
    lado_origen: Lado
    destino: str
    lado_destino: Lado
    etiqueta: str = ""
    desplazamiento_origen: float = 0.0
    desplazamiento_destino: float = 0.0
    etiqueta_encima: bool = False


def _columna(indice: int, fila_y: float, alto: float, clave: str, titulo: str,
             detalle: tuple[str, ...], grupo: str) -> Caja:
    """Caja de una columna de la cuadrícula."""
    return Caja(clave, titulo, detalle, _COLUMNAS[indice], fila_y, _ANCHO_CAJA, alto, grupo)


_FILA_2: Final = 56.0
_FILA_3: Final = 27.0
_ALTO_FILA: Final = 16.0
_ANCHO_APP: Final = _COLUMNAS[2] + _ANCHO_CAJA - _COLUMNAS[0]


def _desde_centro_app(indice: int) -> float:
    """Desplazamiento horizontal desde el centro de app.py hasta el centro de la columna `indice`."""
    return _COLUMNAS[indice] + _ANCHO_CAJA / 2 - (_COLUMNAS[0] + _ANCHO_APP / 2)


CAJAS: Final[tuple[Caja, ...]] = (
    Caja("usuario", "Usuario", ("persona que analiza",), 47.33, 107.0, 40.0, 9.0, "interfaz"),
    Caja("app", "app.py · Streamlit · 7 pestañas",
         ("Datos · Resumen · Explorar · Relaciones y segmentos · Agente local · Observabilidad · Arquitectura",
          "Solo compone la vista: no calcula ni habla HTTP; el CSV se lee una vez"),
         _COLUMNAS[0], 85.0, _ANCHO_APP, 15.0, "interfaz"),
    _columna(3, 85.0, 15.0, "observabilidad", "observabilidad",
             ("JSONL con trace_id y spans", "SLIs y SLO del servicio"), "control"),
    _columna(0, _FILA_2, _ALTO_FILA, "visualizations", "visualizations",
             ("Plotly + Seaborn", "gráfico según el tipo de variable"), "python"),
    _columna(1, _FILA_2, _ALTO_FILA, "fachada", "AnalisisIncidentes",
             ("fachada de eda.py", "Strategy y Factory"), "python"),
    _columna(2, _FILA_2, _ALTO_FILA, "agente", "AgenteIncidentes",
             ("guarda en Python", "máx. 2 llamadas al modelo"), "agente"),
    _columna(3, _FILA_2, _ALTO_FILA, "verificacion", "verificacion",
             ("verificación de cifras", "contra la evidencia enviada"), "control"),
    _columna(0, _FILA_3, _ALTO_FILA, "csv", "CSV de incidentes",
             ("Incident Response Log", "119.998 eventos"), "python"),
    _columna(1, _FILA_3, _ALTO_FILA, "data_loader", "data_loader",
             ("control SHA-256", "20.769 incidentes únicos"), "python"),
    _columna(2, _FILA_3, _ALTO_FILA, "registro", "RegistroHerramientas",
             ("lista blanca", "5 herramientas de lectura"), "agente"),
    _columna(3, _FILA_3, _ALTO_FILA, "cliente", "ClienteOllama",
             ("Adapter (ClienteChat)", "solo hosts locales"), "agente"),
    _columna(3, 4.0, 12.0, "ollama", "Ollama local",
             ("llama3.2:3b", "127.0.0.1:11434"), "agente"),
)

FLECHAS: Final[tuple[Flecha, ...]] = (
    Flecha("usuario", "abajo", "app", "arriba", "filtra y pregunta"),
    Flecha("app", "abajo", "visualizations", "arriba", "tablas a figuras",
           desplazamiento_origen=_desde_centro_app(0)),
    Flecha("app", "abajo", "fachada", "arriba", "selección"),
    Flecha("app", "abajo", "agente", "arriba", "pregunta", desplazamiento_origen=_desde_centro_app(2)),
    Flecha("observabilidad", "izquierda", "app", "derecha", "lee", etiqueta_encima=True),
    Flecha("agente", "arriba", "observabilidad", "abajo", "spans y eventos",
           desplazamiento_origen=10.0, desplazamiento_destino=-8.0),
    Flecha("agente", "derecha", "verificacion", "izquierda", "cifras", etiqueta_encima=True),
    Flecha("agente", "abajo", "registro", "arriba", "elige y valida"),
    Flecha("registro", "arriba", "fachada", "abajo", "ejecuta lecturas",
           desplazamiento_origen=-12.0, desplazamiento_destino=12.0),
    Flecha("registro", "derecha", "cliente", "izquierda", "esquemas", etiqueta_encima=True),
    Flecha("csv", "derecha", "data_loader", "izquierda", "bytes", etiqueta_encima=True),
    Flecha("data_loader", "arriba", "fachada", "abajo", "incidentes y control"),
    Flecha("cliente", "abajo", "ollama", "arriba", "HTTP local"),
)


def _anclaje(caja: Caja, lado: Lado, desplazamiento: float) -> tuple[float, float]:
    """Punto sobre el borde `lado` de la caja, corrido `desplazamiento` unidades a lo largo del borde."""
    cx, cy = caja.centro
    if lado == "arriba":
        return cx + desplazamiento, caja.y + caja.alto
    if lado == "abajo":
        return cx + desplazamiento, caja.y
    if lado == "izquierda":
        return caja.x, cy + desplazamiento
    return caja.x + caja.ancho, cy + desplazamiento


def _dibujar_caja(ax: Axes, caja: Caja) -> None:
    """Rectángulo redondeado con el título en negrita y el detalle debajo, centrados."""
    relleno, borde, _ = GRUPOS[caja.grupo]
    ax.add_patch(FancyBboxPatch((caja.x, caja.y), caja.ancho, caja.alto,
                                boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=relleno, edgecolor=borde, linewidth=1.1))
    # Paso 1: alturas de línea en unidades del diagrama (título más alto que el detalle).
    alturas = [3.4] + [2.6] * len(caja.detalle)
    cx, cy = caja.centro
    tope = cy + sum(alturas) / 2
    lineas = [(caja.titulo, 8.0, "bold")] + [(texto, 6.3, "normal") for texto in caja.detalle]
    # Paso 2: cada línea se centra dentro de su franja.
    for (texto, tamano, peso), alto in zip(lineas, alturas):
        ax.text(cx, tope - alto / 2, texto, ha="center", va="center", fontsize=tamano,
                fontweight=peso, color=COLOR_TEXTO)
        tope -= alto


def _dibujar_flecha(ax: Axes, flecha: Flecha, cajas: dict[str, Caja]) -> None:
    """Flecha recta entre anclajes con su etiqueta en el punto medio (o encima, si es corta)."""
    inicio = _anclaje(cajas[flecha.origen], flecha.lado_origen, flecha.desplazamiento_origen)
    fin = _anclaje(cajas[flecha.destino], flecha.lado_destino, flecha.desplazamiento_destino)
    ax.annotate("", xy=fin, xytext=inicio,
                arrowprops={"arrowstyle": "-|>", "color": COLOR_FLECHA, "lw": 1.1,
                            "shrinkA": 0, "shrinkB": 1.5, "mutation_scale": 9})
    if not flecha.etiqueta:
        return
    medio_x, medio_y = (inicio[0] + fin[0]) / 2, (inicio[1] + fin[1]) / 2
    if flecha.etiqueta_encima:
        ax.text(medio_x, medio_y + 2.2, flecha.etiqueta, ha="center", va="center",
                fontsize=5.8, fontstyle="italic", color=COLOR_FLECHA)
        return
    ax.text(medio_x, medio_y, flecha.etiqueta, ha="center", va="center", fontsize=6.0,
            fontstyle="italic", color=COLOR_FLECHA,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none"})


def _dibujar_leyenda(ax: Axes) -> None:
    """Recuadro inferior con el lema de responsabilidades y la clave de colores de los grupos."""
    x, y, ancho, alto = _COLUMNAS[0], 4.0, _ANCHO_APP, 12.0
    ax.add_patch(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor="#FAFAFA", edgecolor="#9AA5AE", linewidth=0.9))
    ax.text(x + ancho / 2, y + 8.6, LEYENDA, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=COLOR_TEXTO)
    # Paso 1: muestras de color con su rótulo, en una fila bajo el lema; el ancho real del rótulo
    # se mide con el renderizador para que el siguiente no se superponga.
    renderer = ax.figure.canvas.get_renderer()
    cursor = x + 6.0
    for relleno, borde, rotulo in GRUPOS.values():
        ax.add_patch(Rectangle((cursor, y + 2.0), 3.0, 3.0, facecolor=relleno, edgecolor=borde, linewidth=0.9))
        etiqueta = ax.text(cursor + 4.2, y + 3.5, rotulo, ha="left", va="center", fontsize=6.3, color=COLOR_TEXTO)
        cursor += 4.2 + etiqueta.get_window_extent(renderer).width / _PX_POR_UNIDAD + 5.0


def crear_figura() -> Figure:
    """Figura de Matplotlib (lienzo Agg, sin pyplot) con el diagrama completo.

    Unidades del dibujo: 180 de ancho por 130 de alto, con la misma proporción que la figura,
    de modo que una unidad equivale a 10 px al exportar a 200 ppp.
    """
    figura = Figure(figsize=TAMANO_PULGADAS, dpi=DPI, facecolor="white")
    FigureCanvasAgg(figura)
    ax = figura.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, ANCHO_UNIDADES)
    ax.set_ylim(0, ALTO_UNIDADES)
    ax.axis("off")
    # Paso 1: título y subtítulo.
    ax.text(ANCHO_UNIDADES / 2, 125.5, "Arquitectura de la aplicación de análisis de incidentes TI",
            ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_TEXTO)
    ax.text(ANCHO_UNIDADES / 2, 120.8, "Todo corre en el equipo local: interfaz, cálculo, modelo y registros",
            ha="center", va="center", fontsize=7.2, color="#4B5863")
    # Paso 2: flechas primero, para que las cajas queden por encima de las puntas.
    cajas = {caja.clave: caja for caja in CAJAS}
    for flecha in FLECHAS:
        _dibujar_flecha(ax, flecha, cajas)
    for caja in CAJAS:
        _dibujar_caja(ax, caja)
    # Paso 3: leyenda de responsabilidades.
    _dibujar_leyenda(ax)
    return figura


def textos_figura(figura: Figure) -> list[str]:
    """Todos los textos visibles de la figura, en el orden en que se dibujaron."""
    return [t.get_text() for t in figura.findobj(Text) if t.get_text()]


def exportar(carpeta: Path = CARPETA_ASSETS) -> tuple[Path, Path]:
    """Escribe arquitectura.png (200 ppp) y arquitectura.svg en `carpeta` y devuelve sus rutas.

    Los metadatos variables (fecha, versión) se omiten y los identificadores del SVG usan una sal
    fija, para que dos ejecuciones sin cambios produzcan los mismos archivos.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    figura = crear_figura()
    png, svg = carpeta / "arquitectura.png", carpeta / "arquitectura.svg"
    with matplotlib.rc_context({"svg.hashsalt": "arquitectura-incidentes"}):
        figura.savefig(png, dpi=DPI, facecolor="white", metadata={"Software": None})
        figura.savefig(svg, format="svg", facecolor="white", metadata={"Date": None, "Creator": None})
    return png, svg


def main() -> None:
    """Genera los dos archivos en assets/ e informa su tamaño."""
    for ruta in exportar():
        print(f"{ruta.relative_to(CARPETA_ASSETS.parent)}: {ruta.stat().st_size:,} bytes".replace(",", "."))


if __name__ == "__main__":
    main()
