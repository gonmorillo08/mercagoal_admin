"""
Detección de valores faltantes en `tabla_valores_mercado`, tanto NA
"auténticos" (`NaN`, que ya detecta `df.isna()`) como NA "disfrazados": un
valor de texto que en realidad significa "sin dato" pero que, al no ser
`NaN`, pasa desapercibido para cualquier análisis basado en `isna()` —el
caso de `pagina_limpieza()` en `app.py`, que documenta el propio proyecto
como paso previo a este módulo—. Por cada tipo de hueco se evalúa además si
hay alguna forma de recuperarlo a partir de otra columna/fuente ya presente
en el dataset, en vez de limitarse a contarlo.

Los tres huecos conocidos de este dataset (ver docstring de
`ingest/procesar_datos.py` y la exploración que motivó este módulo):

  - `posicion == "Missing"`: la columna `position` de `players.csv` usa el
    valor de texto literal `"Missing"` en vez de un NA cuando Transfermarkt
    no tiene la posición del jugador (no es un fallo de nuestro pipeline:
    ya viene así en el dataset de origen). `sub_position` sí es NA en esas
    mismas filas, así que no permite recuperar `posicion`.
  - `equipo == "Unknown"`: mismo patrón que el anterior pero en
    `current_club_name` de `player_valuations.csv`. En esa misma columna
    aparecen también `"Without Club"`, `"Retired"` y `"Career break"`: no
    son un club, sino la situación del jugador en esa fecha, y se tratan
    igualmente como NA.
  - `liga` es NaN en ~46 % de las filas: NO es un valor disfrazado, es un
    NA real y documentado (el club de esa fecha no aparece en `clubs.csv`,
    normalmente por jugar en una liga menor no rastreada por el dataset).
    Se evalúa aquí, junto a `nacionalidad`, cuánto de ese hueco es
    recuperable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _comun
from . import duplicidad_equipos as _dup

# Valores de texto habituales para indicar "sin dato" en datasets en inglés
# y español; se buscan (tras normalizar) en cualquier columna de texto. En
# este dataset aparecen "missing" en `posicion` y "unknown", "without club",
# "retired" y "career break" en `equipo` (estos tres últimos no son un club
# desconocido sino una situación del jugador sin club que asignar); el resto
# se mantiene para que el detector generalice si el dataset de origen
# añadiera otro disfraz en una actualización futura.
VALORES_NA_DISFRAZADO = {
    "missing", "unknown", "n a", "na", "none", "desconocido",
    "sin datos", "sin dato", "no disponible", "-", "retired", 'without club', "career break",
}


def detectar_na_reales(df: pd.DataFrame) -> pd.DataFrame:
    """Recuento y porcentaje de `NaN` auténticos por columna."""
    nulos = df.isna().sum()
    return pd.DataFrame(
        {"n_na": nulos, "pct_na": (nulos / len(df) * 100).round(3)}
    ).loc[lambda t: t["n_na"] > 0].sort_values("n_na", ascending=False)


def detectar_na_disfrazados(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada columna de texto, valores presentes (no NaN) cuya forma
    normalizada coincide con `VALORES_NA_DISFRAZADO`: NA reales que no
    cuenta `df.isna()` porque llevan un texto en vez de estar vacíos."""
    filas = []
    columnas_texto = df.select_dtypes(include=["object", "string", "category"]).columns
    for columna in columnas_texto:
        valores = df[columna].dropna().unique()
        sospechosos = [v for v in valores if _comun.normalizar_texto(v) in VALORES_NA_DISFRAZADO]
        for valor in sospechosos:
            filas.append(
                {
                    "columna": columna,
                    "valor_disfrazado": valor,
                    "n_filas": int((df[columna] == valor).sum()),
                }
            )
    resultado = pd.DataFrame(filas, columns=["columna", "valor_disfrazado", "n_filas"])
    return resultado.sort_values("n_filas", ascending=False).reset_index(drop=True)


def combinar_na_reales_y_disfrazados(
    na_reales: pd.DataFrame, na_disfrazados: pd.DataFrame, n_filas_total: int
) -> pd.DataFrame:
    """Une el resultado de `detectar_na_reales` y `detectar_na_disfrazados`
    en una única tabla ("todo lo que hoy significa 'sin dato', antes de
    convertir nada"): una fila por columna + tipo de hueco, para verlos
    juntos de un vistazo en vez de en dos tablas separadas."""
    reales = na_reales.reset_index().rename(
        columns={"index": "columna", "n_na": "n_filas", "pct_na": "pct_filas"}
    )
    reales["tipo"] = "NaN real"
    reales["valor_detectado"] = "NaN"

    disfrazados = na_disfrazados.rename(columns={"valor_disfrazado": "valor_detectado"}).copy()
    disfrazados["tipo"] = "NaN disfrazado"
    disfrazados["pct_filas"] = (disfrazados["n_filas"] / n_filas_total * 100).round(3)

    columnas = ["columna", "tipo", "valor_detectado", "n_filas", "pct_filas"]
    combinado = pd.concat([reales[columnas], disfrazados[columnas]], ignore_index=True)
    return combinado.sort_values("n_filas", ascending=False).reset_index(drop=True)


def convertir_na_disfrazados(
    df: pd.DataFrame, na_disfrazados: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Copia de `df` donde cada valor detectado por `detectar_na_disfrazados`
    (p. ej. `"Missing"` en `posicion`, `"Unknown"` en `equipo`) se sustituye
    por un `NaN` auténtico, de forma que pase a contar como tal en
    `df.isna()` y en cualquier análisis basado en él (incluido
    `scikit_na.summary()`)."""
    if na_disfrazados is None:
        na_disfrazados = detectar_na_disfrazados(df)

    df_convertido = df.copy()
    for columna, grupo in na_disfrazados.groupby("columna"):
        mascara = df_convertido[columna].isin(grupo["valor_disfrazado"])
        df_convertido.loc[mascara, columna] = np.nan
    return df_convertido


def imputar_equipo_por_historial(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """PROPUESTA (todavía no aplicada en el pipeline ni en la app): recupera
    los `equipo` faltantes —NaN reales y NA disfrazados de
    `VALORES_NA_DISFRAZADO` ("Unknown", "Without Club", "Retired", "Career
    break"...)— a partir del historial de clubes del propio jugador:

      1. Cada registro sin equipo recibe el equipo del registro CON equipo
         del mismo jugador más cercano en fecha (en días, en valor
         absoluto), mire hacia atrás o hacia delante.
      2. En caso de empate (misma distancia al registro previo y al
         posterior), tiene preferencia el equipo PREVIO.
      3. Si el jugador no tiene ningún registro con equipo en todo su
         historial, se eliminan todos sus registros (no hay de dónde
         imputar).

    Devuelve `(df_imputado, detalle, eliminados)`:
      - `df_imputado`: copia de `df` con los equipos imputados y sin los
        jugadores eliminados.
      - `detalle`: una fila por registro imputado (`id_player, nombre,
        fecha, valor_original, equipo_imputado, direccion` —"previo" o
        "posterior"—, `fecha_referencia, distancia_dias`).
      - `eliminados`: una fila por jugador eliminado (`id_player, nombre,
        n_registros, valores_originales`).
    """
    columnas_originales = list(df.columns)
    datos = df.copy()
    datos["_fecha"] = pd.to_datetime(datos["fecha"])
    datos["_valor_original"] = datos["equipo"]
    datos = convertir_na_disfrazados(datos, detectar_na_disfrazados(datos[["equipo"]]))
    datos = datos.sort_values(["id_player", "_fecha"], kind="mergesort")

    # Equipo y fecha del registro con equipo más cercano hacia atrás (ffill)
    # y hacia delante (bfill), siempre dentro del mismo jugador.
    fecha_con_equipo = datos["_fecha"].where(datos["equipo"].notna())
    por_jugador = datos.assign(_fecha_equipo=fecha_con_equipo).groupby("id_player", sort=False)
    equipo_previo = por_jugador["equipo"].ffill()
    fecha_previa = por_jugador["_fecha_equipo"].ffill()
    equipo_posterior = por_jugador["equipo"].bfill()
    fecha_posterior = por_jugador["_fecha_equipo"].bfill()

    distancia_previa = (datos["_fecha"] - fecha_previa).dt.days
    distancia_posterior = (fecha_posterior - datos["_fecha"]).dt.days
    # Regla 2: empate -> previo (<=). Si solo existe uno de los dos lados,
    # se usa ese.
    usar_previo = equipo_previo.notna() & (equipo_posterior.isna() | (distancia_previa <= distancia_posterior))

    sin_equipo = datos["equipo"].isna()
    imputable = sin_equipo & (equipo_previo.notna() | equipo_posterior.notna())

    detalle = pd.DataFrame(
        {
            "id_player": datos["id_player"],
            "nombre": datos["nombre"],
            "fecha": datos["_fecha"],
            "valor_original": datos["_valor_original"],
            "equipo_imputado": equipo_previo.where(usar_previo, equipo_posterior),
            "direccion": np.where(usar_previo, "previo", "posterior"),
            "fecha_referencia": fecha_previa.where(usar_previo, fecha_posterior),
            "distancia_dias": distancia_previa.where(usar_previo, distancia_posterior),
        }
    )[imputable].reset_index(drop=True)
    detalle["distancia_dias"] = detalle["distancia_dias"].astype("int64")

    # Regla 3: jugadores sin ningún equipo en todo su historial.
    jugadores_sin_equipo = datos.groupby("id_player")["equipo"].count().loc[lambda s: s == 0].index
    filas_eliminadas = datos[datos["id_player"].isin(jugadores_sin_equipo)]
    eliminados = (
        filas_eliminadas.groupby("id_player")
        .agg(
            nombre=("nombre", "first"),
            n_registros=("nombre", "size"),
            valores_originales=("_valor_original", lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))),
        )
        .reset_index()
        .sort_values("n_registros", ascending=False)
        .reset_index(drop=True)
    )

    datos.loc[imputable, "equipo"] = detalle.set_index(datos.index[imputable])["equipo_imputado"]
    df_imputado = datos[~datos["id_player"].isin(jugadores_sin_equipo)][columnas_originales]
    return df_imputado.sort_index(), detalle, eliminados


def evaluar_recuperacion_posicion(df: pd.DataFrame, players: pd.DataFrame | None = None) -> dict:
    """`posicion` viene de la columna `position` de `players.csv`; la
    alternativa más fina, `sub_position`, existe pero
    `ingest/procesar_datos.py` nunca la usa como respaldo fila a fila (elige
    UNA columna para todo el dataset, ver `_primera_columna_existente`).
    Aquí se comprueba si, para las filas con `posicion == "Missing"`,
    `sub_position` SÍ tiene un valor aprovechable."""
    if players is None:
        players = _comun.cargar_players()

    ids_sin_posicion = df.loc[df["posicion"] == "Missing", "id_player"].unique()
    sub = players.loc[players["player_id"].isin(ids_sin_posicion), "sub_position"]
    recuperables = int(sub.notna().sum())

    return {
        "n_jugadores_afectados": len(ids_sin_posicion),
        "n_filas_afectadas": int((df["posicion"] == "Missing").sum()),
        "recuperables_via_sub_position": recuperables,
        "conclusion": (
            "No recuperable dentro del propio dataset: sub_position también "
            "es NA para esos mismos jugadores (Transfermarkt no tiene "
            "registrada ninguna posición para ellos). Haría falta una "
            "fuente externa."
            if recuperables == 0 else
            f"Recuperable para {recuperables} de {len(ids_sin_posicion)} "
            "jugadores usando sub_position como respaldo."
        ),
    }


def evaluar_recuperacion_nacionalidad(df: pd.DataFrame, players: pd.DataFrame | None = None) -> dict:
    """`nacionalidad` usa `country_of_citizenship`; `country_of_birth` es la
    alternativa disponible en `players.csv` (mismo patrón que
    `posicion`/`sub_position`: la elige `_primera_columna_existente` y
    nunca se usa como respaldo)."""
    if players is None:
        players = _comun.cargar_players()

    ids_sin_nacionalidad = df.loc[df["nacionalidad"].isna(), "id_player"].unique()
    nacimiento = players.loc[players["player_id"].isin(ids_sin_nacionalidad), "country_of_birth"]
    recuperables = int(nacimiento.notna().sum())

    return {
        "n_jugadores_afectados": len(ids_sin_nacionalidad),
        "recuperables_via_country_of_birth": recuperables,
        "conclusion": (
            f"Recuperación marginal: solo {recuperables} de "
            f"{len(ids_sin_nacionalidad)} jugadores tienen country_of_birth "
            "relleno; en la inmensa mayoría de los casos ambas columnas "
            "faltan a la vez, así que no compensa como estrategia general."
        ),
    }


def evaluar_recuperacion_liga(df: pd.DataFrame, clubs: pd.DataFrame | None = None) -> dict:
    """`liga` es NaN cuando el nombre de `equipo` de esa fecha no cruza
    (por nombre normalizado EXACTO) con `clubs.csv`. Reutiliza
    `duplicidad_equipos.detectar_posibles_duplicados` para saber cuántas de
    esas filas corresponden en realidad a un club que SÍ está en
    `clubs.csv`, solo que con una grafía ligeramente distinta (ver ese
    módulo) — el resto son clubes genuinamente ausentes de `clubs.csv`
    (ligas menores no rastreadas) y no se pueden recuperar por este medio.
    """
    if clubs is None:
        clubs = _comun.cargar_clubs()

    filas_sin_liga = df[df["liga"].isna()]
    duplicados = _dup.detectar_posibles_duplicados(df, clubs)
    recuperables = duplicados[duplicados["tipo"] == "posible variante de club en clubs.csv"]
    equipos_recuperables = set(recuperables["nombre_en_datos"])

    n_filas_recuperables = int(filas_sin_liga["equipo"].isin(equipos_recuperables).sum())

    return {
        "n_filas_sin_liga": len(filas_sin_liga),
        "pct_filas_sin_liga": round(len(filas_sin_liga) / len(df) * 100, 2),
        "n_equipos_distintos_sin_liga": filas_sin_liga["equipo"].nunique(),
        "n_equipos_recuperables_por_grafia": len(equipos_recuperables),
        "n_filas_recuperables_por_grafia": n_filas_recuperables,
        "conclusion": (
            f"De {len(filas_sin_liga):,} filas sin liga, {n_filas_recuperables:,} "
            f"({n_filas_recuperables / max(len(filas_sin_liga), 1) * 100:.1f} %) "
            "corresponden a un club que SÍ está en clubs.csv con otra grafía "
            "(ver duplicidad_equipos.py): son recuperables mejorando el cruce "
            "de nombres. El resto son clubes genuinamente fuera de clubs.csv "
            "(ligas menores no rastreadas por el dataset de origen) y no "
            "tienen recuperación posible dentro de este dataset."
        ),
    }


def informe_valores_faltantes(df: pd.DataFrame | None = None) -> dict:
    """Agrupa las cuatro comprobaciones anteriores en un único informe."""
    if df is None:
        df = _comun.cargar_tabla_final()

    players = _comun.cargar_players()
    clubs = _comun.cargar_clubs()

    return {
        "na_reales": detectar_na_reales(df),
        "na_disfrazados": detectar_na_disfrazados(df),
        "recuperacion_posicion": evaluar_recuperacion_posicion(df, players),
        "recuperacion_nacionalidad": evaluar_recuperacion_nacionalidad(df, players),
        "recuperacion_liga": evaluar_recuperacion_liga(df, clubs),
    }


if __name__ == "__main__":
    informe = informe_valores_faltantes()

    print("=== NA reales (NaN) por columna ===")
    print(informe["na_reales"].to_string())

    print("\n=== NA disfrazados (valor de texto que en realidad es 'sin dato') ===")
    print(informe["na_disfrazados"].to_string(index=False))

    for clave, titulo in [
        ("recuperacion_posicion", "posicion"),
        ("recuperacion_nacionalidad", "nacionalidad"),
        ("recuperacion_liga", "liga"),
    ]:
        print(f"\n=== Recuperabilidad de '{titulo}' ===")
        for k, v in informe[clave].items():
            print(f"  {k}: {v}")
