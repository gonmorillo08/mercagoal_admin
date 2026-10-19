"""
Módulo de visualización de datos.

Reúne funciones "de propósito general" que reciben un DataFrame de pandas
(y los nombres de las columnas a representar) y devuelven una figura de
Plotly lista para pintarse con `st.plotly_chart(fig, use_container_width=True)`.

El objetivo es desacoplar la lógica de generación de gráficas de la interfaz
de Streamlit (`app.py`), de forma que:
  - `app.py` solo se encarga de pedir al usuario qué quiere ver (widgets) y
    de llamar a la función correspondiente.
  - Este módulo se puede reutilizar en notebooks, tests o futuras páginas
    sin depender de Streamlit.

Tipos de gráfica disponibles:
  - Boxplot (diagrama de caja)         -> boxplot()
  - Densidad (estimación KDE)          -> densidad()
  - Barras de frecuencia (categórica)  -> barras_conteo()
  - Barras de agregación numérica      -> barras_agregado()
  - Serie temporal agregada            -> serie_temporal()

La red dirigida de traspasos entre equipos/ligas vive aparte, en
`visualizacion/red_de_traspasos.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Utilidades: detección de tipos de columna
# ---------------------------------------------------------------------------


def columnas_numericas(df: pd.DataFrame, excluir_ids: bool = True) -> list[str]:
    """Columnas numéricas del DataFrame.

    Si `excluir_ids` es True, descarta columnas cuyo nombre sugiera que son
    identificadores (p. ej. `id_player`) en lugar de variables de análisis.
    """
    cols = df.select_dtypes(include=["number"]).columns.tolist()
    if excluir_ids:
        cols = [c for c in cols if "id" not in c.lower()]
    return cols


def columnas_categoricas(df: pd.DataFrame, max_categorias: int = 300) -> list[str]:
    """Columnas categóricas/texto con un número razonable de valores únicos
    (evita ofrecer columnas tipo "nombre" con miles de categorías distintas
    para agrupar/colorear)."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return [c for c in cat_cols if df[c].nunique(dropna=True) <= max_categorias]


def columnas_fecha(df: pd.DataFrame) -> list[str]:
    """Columnas de tipo fecha (datetime) del DataFrame."""
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


def top_n_categorias(df: pd.DataFrame, columna: str, n: int = 15) -> list:
    """Las `n` categorías más frecuentes de `columna` (para no saturar
    gráficos con decenas de equipos, ligas, países, etc.)."""
    return df[columna].value_counts().nlargest(n).index.tolist()


# ---------------------------------------------------------------------------
# Boxplot
# ---------------------------------------------------------------------------


def boxplot(
    df: pd.DataFrame,
    columna_num: str,
    columna_cat: str | None = None,
    escala_log_y: bool = False,
    mostrar_puntos: bool = False,
    titulo: str | None = None,
) -> go.Figure:
    """Boxplot de una variable numérica, opcionalmente desglosado por una
    variable categórica (una caja por categoría)."""
    fig = px.box(
        df,
        x=columna_cat,
        y=columna_num,
        color=columna_cat,
        points="outliers" if not mostrar_puntos else "all",
        title=titulo
        or (f"Boxplot de {columna_num}" + (f" por {columna_cat}" if columna_cat else "")),
    )
    if escala_log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(showlegend=False, xaxis_title=columna_cat or "")
    return fig


# ---------------------------------------------------------------------------
# Densidad (estimación KDE)
# ---------------------------------------------------------------------------


def densidad(
    df: pd.DataFrame,
    columna: str,
    color_por: str | None = None,
    escala_log_x: bool = False,
    titulo: str | None = None,
) -> go.Figure:
    """Curva de densidad (Kernel Density Estimation) de una variable numérica.

    Si `color_por` es una columna categórica, se dibuja una curva de
    densidad por cada categoría (superpuestas), lo que permite comparar la
    forma de la distribución entre grupos (p. ej. valor de mercado por
    posición o por liga).
    """
    fig = go.Figure()

    cols = [columna] + ([color_por] if color_por else [])
    datos = df[cols].dropna()
    if escala_log_x:
        datos = datos[datos[columna] > 0]

    def _curva_kde(valores: pd.Series):
        valores = valores.astype(float)
        base = np.log10(valores) if escala_log_x else valores
        if len(base) < 2 or base.nunique() < 2:
            return None, None
        kde = gaussian_kde(base, bw_method="scott")
        # El ancho de banda automático (regla de Scott) deja la curva con
        # varios picos locales muy pegados entre sí (p. ej. en
        # 'valor_mercado_eur', por el redondeo habitual del mercado a
        # cifras como 100k/500k/1M): se ensancha el kernel (factor x3, en
        # vez de tocar el nº de puntos de `xs`, que solo cambia la
        # resolución con la que se DIBUJA la misma curva, no su forma) para
        # que la densidad se vea como una única curva suave.
        kde.set_bandwidth(kde.factor * 3)
        xs = np.linspace(base.min(), base.max(), 300)
        ys = kde(xs)
        xs_originales = 10**xs if escala_log_x else xs
        return xs_originales, ys

    if color_por:
        for grupo in sorted(datos[color_por].unique(), key=str):
            xs, ys = _curva_kde(datos.loc[datos[color_por] == grupo, columna])
            if xs is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines", name=str(grupo), fill="tozeroy", opacity=0.5
                )
            )
    else:
        xs, ys = _curva_kde(datos[columna])
        if xs is not None:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy", name=columna))

    if escala_log_x:
        fig.update_xaxes(type="log")
    fig.update_layout(
        title=titulo or f"Densidad de {columna}",
        xaxis_title=columna,
        yaxis_title="Densidad",
    )
    return fig

# ---------------------------------------------------------------------------
# Barras: frecuencia de una variable categórica
# ---------------------------------------------------------------------------


def barras_conteo(
    df: pd.DataFrame,
    columna: str,
    top_n: int = 15,
    titulo: str | None = None,
) -> go.Figure:
    """Gráfico de barras con el número de registros por categoría (top N
    categorías más frecuentes, para no saturar el eje)."""
    conteo = df[columna].value_counts().nlargest(top_n).reset_index()
    conteo.columns = [columna, "conteo"]
    conteo = conteo.sort_values("conteo", ascending=True)

    fig = px.bar(
        conteo,
        x="conteo",
        y=columna,
        orientation="h",
        text="conteo",
        title=titulo or f"Frecuencia de {columna} (top {top_n})",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title=columna, xaxis_title="Nº de registros")
    return fig


# ---------------------------------------------------------------------------
# Barras: agregación de una variable numérica por categoría
# ---------------------------------------------------------------------------

_AGREGACIONES = {
    "Media": "mean",
    "Mediana": "median",
    "Suma": "sum",
    "Máximo": "max",
    "Mínimo": "min",
    "Nº de registros": "count",
}


def barras_agregado(
    df: pd.DataFrame,
    columna_cat: str,
    columna_num: str,
    agregacion: str = "Media",
    top_n: int = 15,
    titulo: str | None = None,
) -> go.Figure:
    """Gráfico de barras con una agregación (media, mediana, suma...) de una
    variable numérica, calculada por cada categoría de `columna_cat`.

    `agregacion` debe ser una de las claves de `_AGREGACIONES`
    (p. ej. "Media", "Mediana", "Suma"...).
    """
    func = _AGREGACIONES.get(agregacion, "mean")
    resumen = (
        df.groupby(columna_cat)[columna_num]
        .agg(func)
        .nlargest(top_n)
        .reset_index()
        .sort_values(columna_num, ascending=True)
    )

    fig = px.bar(
        resumen,
        x=columna_num,
        y=columna_cat,
        orientation="h",
        text=columna_num,
        title=titulo or f"{agregacion} de {columna_num} por {columna_cat} (top {top_n})",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(yaxis_title=columna_cat, xaxis_title=f"{agregacion} de {columna_num}")
    return fig


# ---------------------------------------------------------------------------
# Serie temporal agregada
# ---------------------------------------------------------------------------

_FRECUENCIAS = {
    "Día": "D",
    "Semana": "W",
    "Mes": "MS",
    "Trimestre": "QS",
    "Año": "YS",
}


def serie_temporal(
    df: pd.DataFrame,
    columna_fecha: str,
    columna_num: str,
    columna_cat: str | None = None,
    frecuencia: str = "Año",
    agregacion: str = "Media",
    titulo: str | None = None,
) -> go.Figure:
    """Evolución temporal de una variable numérica, agregada por periodo
    (día/semana/mes/trimestre/año) y, opcionalmente, desglosada por una
    variable categórica (una línea por categoría).
    """
    freq = _FRECUENCIAS.get(frecuencia, "YS")
    func = _AGREGACIONES.get(agregacion, "mean")

    cols = [columna_fecha, columna_num] + ([columna_cat] if columna_cat else [])
    datos = df[cols].dropna(subset=[columna_fecha, columna_num]).copy()
    datos[columna_fecha] = pd.to_datetime(datos[columna_fecha])

    agrupadores = [pd.Grouper(key=columna_fecha, freq=freq)]
    if columna_cat:
        agrupadores.append(columna_cat)

    resumen = datos.groupby(agrupadores)[columna_num].agg(func).reset_index()

    fig = px.line(
        resumen,
        x=columna_fecha,
        y=columna_num,
        color=columna_cat,
        markers=True,
        title=titulo
        or f"Evolución temporal ({agregacion.lower()}) de {columna_num} por {frecuencia.lower()}",
    )
    fig.update_layout(xaxis_title="Fecha", yaxis_title=f"{agregacion} de {columna_num}")
    return fig
