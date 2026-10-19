"""
Errores en el propio `valor_mercado_eur` de `tabla_valores_mercado` — a
diferencia de `valores_faltantes.py` (que trata la AUSENCIA de datos) o de
`duplicidad_equipos.py`/`nacionalidad_y_liga.py` (que tratan errores en
`equipo`/`liga`/`nacionalidad`), este módulo se centra exclusivamente en si
la propia cifra de valor de mercado es plausible.

Dos comprobaciones, cada una en su propia función (ambas devuelven un
`pd.DataFrame`, una fila por caso detectado, vacío si no hay ninguno):

  1. `cambios_anomalos`: entre dos valoraciones CONSECUTIVAS del mismo
     jugador, un cambio de valor muy brusco (el nuevo valor multiplica o
     divide al anterior por encima de un umbral) en muy poco tiempo.
  2. `picos_aislados`: un caso más específico que el anterior y con menos
     falsos positivos — el valor SUBE (o baja) para, en la valoración
     inmediatamente siguiente, volver EXACTAMENTE a la cifra anterior. Una
     revalorización real (ascenso, gran temporada, traspaso...) no suele
     deshacerse por completo de un día para otro; que la cifra vuelva a ser
     idéntica a la de antes del "pico" es más compatible con un error de
     introducción de datos puntual que con un cambio real de mercado.

(Se descartaron, tras revisarlas, dos comprobaciones adicionales que sí se
llegaron a implementar: valores <= 0 y varios registros del mismo jugador
en el mismo instante — ninguna de las dos aparece ni una sola vez en
`tabla_valores_mercado`, así que no aportaban nada a esta pestaña.)

Igual que el resto de `limpieza/`, es solo de DETECCIÓN: ninguna función
modifica `ingest/datos/` ni decide por sí sola que un valor esté mal, se
limita a señalar candidatos a revisar.
"""

from __future__ import annotations

import pandas as pd

from . import _comun

# ---------------------------------------------------------------------------
# 1. Cambios de valor anómalos en un periodo corto
# ---------------------------------------------------------------------------
def cambios_anomalos(
    df: pd.DataFrame | None = None, dias_max: int = 30, ratio_umbral: float = 3.0
) -> pd.DataFrame:
    """Entre cada valoración de un jugador y la INMEDIATAMENTE anterior:
    si pasaron `dias_max` días o menos y el valor se multiplicó (o dividió)
    por `ratio_umbral` o más, se marca como cambio anómalo. Ambos parámetros
    son ajustables porque no hay un umbral "correcto" objetivo: cuanto más
    corto `dias_max` y más alto `ratio_umbral`, más estricta (menos casos,
    más claramente atípicos) es la detección."""
    if df is None:
        df = _comun.cargar_tabla_final()

    datos = (
        df.dropna(subset=["id_player", "fecha", "valor_mercado_eur"])
        .sort_values(["id_player", "fecha"], kind="mergesort")
        .copy()
    )
    datos["valor_anterior"] = datos.groupby("id_player")["valor_mercado_eur"].shift(1)
    datos["fecha_anterior"] = datos.groupby("id_player")["fecha"].shift(1)
    datos["equipo_anterior"] = datos.groupby("id_player")["equipo"].shift(1)
    datos = datos.dropna(subset=["valor_anterior", "fecha_anterior"])

    datos["dias"] = (datos["fecha"] - datos["fecha_anterior"]).dt.days
    datos["ratio"] = datos["valor_mercado_eur"] / datos["valor_anterior"]

    candidatos = datos[datos["dias"] <= dias_max].copy()
    candidatos = candidatos[
        (candidatos["ratio"] >= ratio_umbral) | (candidatos["ratio"] <= 1 / ratio_umbral)
    ].copy()
    candidatos["tipo"] = candidatos["ratio"].map(lambda r: "Subida brusca" if r >= 1 else "Bajada brusca")
    candidatos["multiplicador"] = candidatos["ratio"].map(lambda r: r if r >= 1 else 1 / r)

    columnas = [
        "id_player", "nombre", "equipo_anterior", "equipo", "fecha_anterior", "fecha",
        "dias", "valor_anterior", "valor_mercado_eur", "multiplicador", "tipo",
    ]
    return (
        candidatos[columnas]
        .rename(columns={"equipo": "equipo_actual", "valor_mercado_eur": "valor_actual"})
        .sort_values("multiplicador", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 2. Picos aislados: sube (o baja) y vuelve exactamente al valor anterior
# ---------------------------------------------------------------------------
def picos_aislados(df: pd.DataFrame | None = None, dias_max: int = 180) -> pd.DataFrame:
    """De cada TRÍO de valoraciones consecutivas de un jugador (anterior,
    actual, siguiente): si la anterior y la siguiente coinciden EXACTAMENTE
    pero la actual es distinta, y las tres caben dentro de `dias_max` días,
    la del medio es un "pico" aislado que se deshace por completo — la señal
    más específica de las dos: una revalorización real no suele volver a
    la cifra exacta de antes en la siguiente actualización."""
    if df is None:
        df = _comun.cargar_tabla_final()

    datos = (
        df.dropna(subset=["id_player", "fecha", "valor_mercado_eur"])
        .sort_values(["id_player", "fecha"], kind="mergesort")
        .copy()
    )
    datos["valor_anterior"] = datos.groupby("id_player")["valor_mercado_eur"].shift(1)
    datos["valor_siguiente"] = datos.groupby("id_player")["valor_mercado_eur"].shift(-1)
    datos["fecha_anterior"] = datos.groupby("id_player")["fecha"].shift(1)
    datos["fecha_siguiente"] = datos.groupby("id_player")["fecha"].shift(-1)
    datos = datos.dropna(subset=["valor_anterior", "valor_siguiente", "fecha_anterior", "fecha_siguiente"])

    datos["dias_totales"] = (datos["fecha_siguiente"] - datos["fecha_anterior"]).dt.days
    candidatos = datos[
        (datos["valor_anterior"] == datos["valor_siguiente"])
        & (datos["valor_mercado_eur"] != datos["valor_anterior"])
        & (datos["dias_totales"] <= dias_max)
    ].copy()
    candidatos["tipo"] = candidatos.apply(
        lambda fila: "Pico al alza" if fila["valor_mercado_eur"] > fila["valor_anterior"] else "Pico a la baja",
        axis=1,
    )

    columnas = [
        "id_player", "nombre", "equipo", "fecha_anterior", "fecha", "fecha_siguiente",
        "valor_anterior", "valor_mercado_eur", "dias_totales", "tipo",
    ]
    return (
        candidatos[columnas]
        .rename(columns={"valor_mercado_eur": "valor_pico"})
        .sort_values("dias_totales")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 3. Resumen conjunto (para las métricas/KPIs de la app)
# ---------------------------------------------------------------------------
def resumen_valores_erroneos(
    df: pd.DataFrame | None = None, dias_max_cambios: int = 30, ratio_umbral: float = 3.0,
    dias_max_picos: int = 180,
) -> pd.DataFrame:
    """Nº de casos detectados por cada una de las 2 comprobaciones."""
    if df is None:
        df = _comun.cargar_tabla_final()
    filas = [
        {
            "caso": "Cambios de valor anómalos en periodo corto",
            "n_casos": len(cambios_anomalos(df, dias_max_cambios, ratio_umbral)),
        },
        {"caso": "Picos aislados (revertidos)", "n_casos": len(picos_aislados(df, dias_max_picos))},
    ]
    return pd.DataFrame(filas)


if __name__ == "__main__":
    _df = _comun.cargar_tabla_final()
    print("== Cambios anómalos ==")
    print(cambios_anomalos(_df).head(20))
    print("\n== Picos aislados ==")
    print(picos_aislados(_df).head(20))
    print("\n== Resumen ==")
    print(resumen_valores_erroneos(_df))
