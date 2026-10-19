"""
Compilación: aplica, en orden, todo lo visto en los módulos de `limpieza/`
sobre `tabla_valores_mercado` y guarda la tabla limpia resultante en
`ingest/datos/tabla_valores_mercado_limpia.{parquet,csv}`.

A diferencia del resto del paquete (solo DETECCIÓN), este módulo sí
TRANSFORMA los datos. Recorrido (una función por paso, cada una devuelve la
tabla transformada y un resumen de lo que ha cambiado):

  1. `paso_1_na_disfrazados`: los falsos valores faltantes de "Datos
     faltantes" (`VALORES_NA_DISFRAZADO`) pasan a NaN real, y se eliminan
     las filas sin `edad` (jugador sin fecha de nacimiento en `players.csv`),
     con edad atípica (<= 14 años) o con altura atípica (< 150 cm).
  2. `paso_2_arsenal`: los jugadores de "Arsenal FC" clasificados como
     "Argentina (Sarandí)" (tabla del caso Arsenal, con las revisiones
     manuales) pasan a "Arsenal Fútbol Club", con liga argentina.
  3. `paso_3_equipo_y_nacionalidad`: recuperación de `equipo` faltante por
     historial del jugador (`valores_faltantes.imputar_equipo_por_historial`),
     fusión de las nacionalidades que son el mismo país con dos grafías
     (conflicto de duplicidad en "Banderas por nacionalidad") y eliminación
     de los registros sin nacionalidad.
  4. `paso_4_liga_a_pais`: `liga` pasa a ser el PAÍS de la competición
     ("Banderas por liga") y se añade la columna `Bandera`.
  5. `paso_5_ligas_y_fusion_equipos`:
       5.1 liga por nacionalidad a los clubes sin liga con >= 50 registros;
       5.2 redes de duplicidad/cantera, las MISMAS que muestra la pestaña de
           duplicidad (tabla original + revisiones manuales): los duplicados
           del líder toman su nombre y el resto de la red pasa a
           "<líder> II";
       5.3 liga por nacionalidad al resto de clubes sin liga.
  6. `paso_6_valores_erroneos`: de los cambios de valor anómalos, se
     elimina el primer registro del jugador si el salto parte de él, y un
     pico aislado (entre dos valores iguales) se sustituye por la media de
     sus vecinos.

Uso: `python -m limpieza.compilacion` desde la raíz del proyecto, o
`compilar()` (lo que hace el botón de la pestaña "Compilación" de la app).
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from . import _comun
from . import duplicidad_equipos as dup
from . import nacionalidad_y_liga as nyl
from . import valores_erroneos as ve
from . import valores_faltantes as vf

ARCHIVO_LIMPIA_PARQUET = _comun.DIR_DATOS / "tabla_valores_mercado_limpia.parquet"
ARCHIVO_LIMPIA_CSV = _comun.DIR_DATOS / "tabla_valores_mercado_limpia.csv"
# Resumen de cada paso (tablas pequeñas de cambios), para que la app lo
# muestre sin volver a compilar. Regenerable, igual que la tabla limpia.
ARCHIVO_INFORME = _comun.DIR_DATOS / "compilacion_informe.pkl"

EQUIPO_ARSENAL_LONDRES = "Arsenal FC"
EQUIPO_ARSENAL_SARANDI = "Arsenal Fútbol Club"
LIGA_ARGENTINA = "torneo-apertura"

# Nacionalidades que son el MISMO país con dos grafías (las marcadas como
# tales en `nacionalidad_y_liga.NOTA_PAIS`): se unifican en el nombre actual
# / en inglés, que es el que usa `competitions.csv`.
FUSION_NACIONALIDADES = {"Turkey": "Türkiye", "Neukaledonien": "New Caledonia"}

# Paso 1: valores atípicos de edad y altura que se eliminan.
EDAD_MINIMA = 14          # se eliminan edades <= 14 años
ALTURA_MINIMA_CM = 150    # se eliminan alturas < 150 cm

UMBRAL_REGISTROS_LIGA_5_1 = 50
SUFIJO_FILIAL = " II"

# Mismos valores por defecto que la pestaña "Valores erróneos".
DIAS_MAX_CAMBIOS = 30
RATIO_UMBRAL_CAMBIOS = 3.0
DIAS_MAX_PICOS = 180

COLUMNAS_FINALES = [
    "id_player", "nombre", "equipo", "liga", "Bandera", "posicion", "nacionalidad",
    "fecha_nacimiento", "edad", "altura_cm", "pie", "fecha", "valor_mercado_eur",
]


# ---------------------------------------------------------------------------
# Paso 1
# ---------------------------------------------------------------------------
def paso_1_na_disfrazados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    na_disfrazados = vf.detectar_na_disfrazados(df)
    resultado = vf.convertir_na_disfrazados(df, na_disfrazados)
    # Columna interna (se descarta al final): el texto original de `equipo`
    # ('Unknown', 'Without Club'...), para que el resumen del paso 3 pueda
    # decir qué se imputó en lugar de qué.
    resultado["_equipo_original"] = df["equipo"]

    # Filas eliminadas por edad o altura:
    #   - sin edad (el jugador no tiene fecha de nacimiento en players.csv);
    #   - edad atípica: <= EDAD_MINIMA años (casi seguro, fecha de nacimiento
    #     mal introducida: hay registros con 2 años);
    #   - altura atípica: < ALTURA_MINIMA_CM (p. ej. 17 cm). Una altura VACÍA
    #     no es atípica: esas filas se conservan.
    ninguno = pd.Series(False, index=resultado.index)
    edad = resultado["edad"] if "edad" in resultado.columns else None
    altura = resultado["altura_cm"] if "altura_cm" in resultado.columns else None
    motivos = {
        "Sin edad (sin fecha de nacimiento)": edad.isna() if edad is not None else ninguno,
        f"Edad atípica (≤ {EDAD_MINIMA} años)": (edad <= EDAD_MINIMA).fillna(False) if edad is not None else ninguno,
        f"Altura atípica (< {ALTURA_MINIMA_CM} cm)": (altura < ALTURA_MINIMA_CM).fillna(False) if altura is not None else ninguno,
    }
    eliminar = ninguno.copy()
    filas_motivo = []
    for motivo, mascara in motivos.items():
        mascara = mascara.astype(bool)
        filas_motivo.append({"motivo": motivo, "registros": int(mascara.sum()),
                             "jugadores": int(resultado.loc[mascara, "id_player"].nunique())})
        eliminar |= mascara
    detalle = resultado[eliminar].assign(
        motivo=np.select([m.astype(bool)[eliminar] for m in motivos.values()], list(motivos), default="")
    )
    registros_eliminados = (
        detalle.groupby(["id_player", "motivo"])
        .agg(nombre=("nombre", "first"), registros=("nombre", "size"),
             edad=("edad", "min"), altura_cm=("altura_cm", "min"))
        .reset_index().sort_values(["motivo", "registros"], ascending=[True, False]).reset_index(drop=True)
    )
    resultado = resultado[~eliminar]
    return resultado, {
        "na_disfrazados": na_disfrazados,
        "n_celdas": int(na_disfrazados["n_filas"].sum()),
        "eliminados_por_motivo": pd.DataFrame(filas_motivo),
        "registros_eliminados": registros_eliminados,
        "n_eliminados": int(eliminar.sum()),
    }


# ---------------------------------------------------------------------------
# Paso 2
# ---------------------------------------------------------------------------
def paso_2_arsenal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tabla = nyl.tabla_jugadores_arsenal_fc(df)
    revisiones = nyl.cargar_revisiones_arsenal_fc()
    tabla["clasificacion"] = [revisiones.get(i, c) for i, c in zip(tabla["id_player"], tabla["clasificacion"])]
    tabla["revisado"] = tabla["id_player"].isin(revisiones)

    argentinos = tabla.loc[tabla["clasificacion"] == "Argentina (Sarandí)", "id_player"]
    mascara = df["id_player"].isin(argentinos) & (df["equipo"] == EQUIPO_ARSENAL_LONDRES)
    resultado = df.copy()
    resultado.loc[mascara, "equipo"] = EQUIPO_ARSENAL_SARANDI
    resultado.loc[mascara, "liga"] = LIGA_ARGENTINA

    movidos = (
        df[mascara].groupby("id_player").size().rename("registros_cambiados").reset_index()
        .merge(tabla[["id_player", "nombre", "nacionalidad", "revisado"]], on="id_player")
    )
    return resultado, {
        "por_clasificacion": tabla["clasificacion"].value_counts().rename_axis("clasificacion").reset_index(name="jugadores"),
        "jugadores_movidos": movidos,
        "n_registros": int(mascara.sum()),
    }


# ---------------------------------------------------------------------------
# Paso 3
# ---------------------------------------------------------------------------
def paso_3_equipo_y_nacionalidad(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # 3a. Equipo faltante por historial del jugador. La liga del registro
    # imputado pasa a ser la liga habitual del club imputado (hasta ahora
    # era NaN: un 'Unknown'/'Without Club' no tiene liga).
    liga_habitual = df.dropna(subset=["equipo", "liga"]).groupby("equipo")["liga"].agg(lambda s: s.mode().iat[0])
    imputado, detalle, eliminados = vf.imputar_equipo_por_historial(df)
    if "_equipo_original" in df.columns:
        original = df.assign(_f=pd.to_datetime(df["fecha"])).set_index(["id_player", "_f"])["_equipo_original"]
        detalle["valor_original"] = [original.get(c) for c in zip(detalle["id_player"], detalle["fecha"])]
        originales_por_jugador = df.groupby("id_player")["_equipo_original"].agg(
            lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))
        )
        eliminados["valores_originales"] = eliminados["id_player"].map(originales_por_jugador)
    claves_detalle = set(zip(detalle["id_player"], detalle["fecha"]))
    fechas = pd.to_datetime(imputado["fecha"])
    es_imputado = pd.Series(
        [clave in claves_detalle for clave in zip(imputado["id_player"], fechas)], index=imputado.index
    )
    imputado.loc[es_imputado, "liga"] = imputado.loc[es_imputado, "equipo"].map(liga_habitual)

    # 3b. Nacionalidades duplicadas (mismo país, dos grafías).
    fusion = imputado["nacionalidad"].isin(FUSION_NACIONALIDADES.keys())
    fusion_resumen = (
        imputado.loc[fusion, "nacionalidad"].value_counts().rename_axis("nacionalidad_original")
        .reset_index(name="registros")
    )
    fusion_resumen["nacionalidad_nueva"] = fusion_resumen["nacionalidad_original"].map(FUSION_NACIONALIDADES)
    imputado["nacionalidad"] = imputado["nacionalidad"].replace(FUSION_NACIONALIDADES)

    # 3c. Registros sin nacionalidad.
    sin_nacionalidad = imputado["nacionalidad"].isna()
    jugadores_sin_nac = (
        imputado[sin_nacionalidad].groupby("id_player")
        .agg(nombre=("nombre", "first"), registros=("nombre", "size")).reset_index()
    )
    resultado = imputado[~sin_nacionalidad]

    return resultado, {
        "imputacion_resumen": detalle.groupby("valor_original").agg(
            registros=("direccion", "size"),
            pct_equipo_previo=("direccion", lambda s: round((s == "previo").mean() * 100, 1)),
            mediana_dias=("distancia_dias", "median"),
        ).reset_index(),
        "n_imputados": len(detalle),
        "n_imputados_con_liga": int(imputado.loc[es_imputado, "liga"].notna().sum()),
        "jugadores_sin_equipo": eliminados,
        "fusion_nacionalidades": fusion_resumen,
        "jugadores_sin_nacionalidad": jugadores_sin_nac,
        "equipo_nan_final": int(resultado["equipo"].isna().sum()),
        "nacionalidad_nan_final": int(resultado["nacionalidad"].isna().sum()),
    }


# ---------------------------------------------------------------------------
# Paso 4
# ---------------------------------------------------------------------------
def _pais_por_club() -> dict[str, str]:
    """Nombre de club normalizado -> país de su competición en `clubs.csv`
    (por `competition_id`, sin la ambigüedad del nombre de liga)."""
    clubs = _comun.cargar_clubs()
    competitions = _comun.cargar_competitions()
    pais_por_comp = competitions.set_index("competition_id")["country_name"]
    clubs = clubs.assign(
        norm=_comun.normalizar_nombre_club(clubs["name"]),
        pais=clubs["domestic_competition_id"].map(pais_por_comp),
    ).drop_duplicates("norm")
    return clubs.set_index("norm")["pais"].dropna().to_dict()


def paso_4_liga_a_pais(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`liga` -> país según "Banderas por liga" (`tabla_banderas_ligas`). En
    las ligas cuyo nombre comparten dos países (p. ej. "bundesliga":
    Alemania/Austria) el país se decide por el CLUB de cada registro,
    cruzándolo con `clubs.csv` (su `competition_id` sí es único)."""
    tabla_ligas = nyl.tabla_banderas_ligas(df, _comun.cargar_competitions())
    paises_por_liga = tabla_ligas[tabla_ligas["pais"] != "—"].groupby("liga")["pais"].agg(list).to_dict()
    pais_club = _pais_por_club()

    pares = df.loc[df["liga"].notna(), ["equipo", "liga"]].drop_duplicates()
    asignacion: dict[tuple[str, str], str] = {}
    sin_resolver = []
    for equipo, liga in zip(pares["equipo"], pares["liga"]):
        candidatos = paises_por_liga.get(liga, [])
        if len(candidatos) == 1:
            asignacion[(equipo, liga)] = candidatos[0]
            continue
        pais = pais_club.get(_comun.normalizar_texto(equipo))
        if pais in candidatos:
            asignacion[(equipo, liga)] = pais
        else:
            sin_resolver.append((equipo, liga))
            if candidatos:
                asignacion[(equipo, liga)] = candidatos[0]

    resultado = df.copy()
    claves = list(zip(resultado["equipo"], resultado["liga"]))
    resultado["liga"] = [asignacion.get(clave) if pd.notna(clave[1]) else None for clave in claves]
    resultado["liga"] = resultado["liga"].astype("object").where(resultado["liga"].notna(), None)
    resultado["Bandera"] = resultado["liga"].map(lambda p: nyl.url_bandera(p) or None if pd.notna(p) else None)

    conversion = (
        pd.DataFrame({"liga_original": df["liga"], "pais": resultado["liga"]})
        .dropna().value_counts().rename("registros").reset_index()
        .sort_values(["liga_original", "registros"], ascending=[True, False])
    )
    conversion["conflicto"] = conversion["liga_original"].map(lambda l: len(paises_por_liga.get(l, [])) > 1)
    return resultado, {"conversion": conversion.reset_index(drop=True), "sin_resolver": sin_resolver}


# ---------------------------------------------------------------------------
# Paso 5
# ---------------------------------------------------------------------------
def _pais_por_nacionalidad(df: pd.DataFrame, equipos: list[str]) -> pd.DataFrame:
    """País aproximado de cada club de `equipos` como en "Recuperar liga por
    nacionalidad del equipo": revisión manual > tabla de excepciones
    conocidas (`CLUBES_EXCEPCION_PAIS_LIGA`) > nacionalidad más repetida de
    su plantilla."""
    revisiones = {
        e: FUSION_NACIONALIDADES.get(n, n) for e, n in nyl.cargar_revisiones_nacionalidad().items() if n
    }
    datos = df[df["equipo"].isin(equipos)]
    nacionalidad = datos.groupby("equipo").apply(nyl._nacionalidad_representativa)
    filas = []
    for equipo in equipos:
        if equipo in revisiones:
            pais, fuente = revisiones[equipo], "revisión manual"
        elif equipo in nyl.CLUBES_EXCEPCION_PAIS_LIGA:
            pais, fuente = nyl.CLUBES_EXCEPCION_PAIS_LIGA[equipo][0], "tabla de excepciones"
        else:
            pais, fuente = nacionalidad.get(equipo), "nacionalidad"
        filas.append({"equipo": equipo, "pais": pais, "fuente": fuente})
    return pd.DataFrame(filas, columns=["equipo", "pais", "fuente"])


def _completar_liga_por_club(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Registros sin liga de un club que SÍ la tiene en otros registros (p.
    ej. un club '<líder> II' formado por varios filiales, unos con liga ya
    asignada y otros no): toman la liga habitual de ese club, para que cada
    club tenga una única liga."""
    liga_habitual = df.dropna(subset=["liga"]).groupby("equipo")["liga"].agg(lambda s: s.mode().iat[0])
    resultado = df.copy()
    liga = resultado["equipo"].map(liga_habitual)
    mascara = resultado["liga"].isna() & liga.notna()
    resultado.loc[mascara, "liga"] = liga[mascara]
    resultado.loc[mascara, "Bandera"] = liga[mascara].map(lambda p: nyl.url_bandera(p) or None)
    return resultado, int(mascara.sum())


def _asignar_liga_por_nacionalidad(df: pd.DataFrame, min_registros: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sin_liga = df[df["liga"].isna()]
    n_registros = sin_liga.groupby("equipo").size()
    equipos = n_registros[n_registros >= min_registros].index.tolist()
    asignacion = _pais_por_nacionalidad(df, equipos).dropna(subset=["pais"])
    asignacion["registros"] = asignacion["equipo"].map(n_registros).astype(int)

    resultado = df.copy()
    pais = resultado["equipo"].map(asignacion.set_index("equipo")["pais"])
    mascara = resultado["liga"].isna() & pais.notna()
    resultado.loc[mascara, "liga"] = pais[mascara]
    resultado.loc[mascara, "Bandera"] = pais[mascara].map(lambda p: nyl.url_bandera(p) or None)
    return resultado, asignacion.sort_values("registros", ascending=False).reset_index(drop=True)


def _aplicar_revisiones_duplicidad(pares: pd.DataFrame, equipos: set[str], valor_por_club: pd.Series) -> tuple[pd.DataFrame, int]:
    """Mismas reglas que la tabla editable de la app: la revisión manual
    manda (probabilidad 1, `origen_regla = "manual"`) y un par revisado como
    filial/duplicidad que el algoritmo ya no propone se añade igualmente."""
    revisiones = dup.cargar_revisiones_manuales()
    pares = pares.copy()
    claves = [dup.clave_par(a, b) for a, b in zip(pares["club_a"], pares["club_b"])]
    revisado = [c in revisiones for c in claves]
    pares["tipo"] = [revisiones[c] if r else t for c, r, t in zip(claves, revisado, pares["tipo"])]
    pares["probabilidad"] = [1.0 if r else p for r, p in zip(revisado, pares["probabilidad"])]
    pares["origen_regla"] = [("manual" if r else o) for r, o in zip(revisado, pares["origen_regla"])]

    presentes = set(claves)
    nuevas = []
    for clave, tipo in revisiones.items():
        if clave in presentes or tipo == dup.NOMBRE_NINGUNA_RELACION or not set(clave) <= equipos:
            continue
        club_a, club_b = sorted(clave, key=lambda c: valor_por_club.get(c, 0), reverse=True)
        nuevas.append({"club_a": club_a, "club_b": club_b, "tipo": tipo, "probabilidad": 1.0,
                       "conflicto": "", "origen_regla": "manual", "origen_candidato": "revisión manual"})
    if nuevas:
        pares = pd.concat([pares, pd.DataFrame(nuevas)], ignore_index=True)
    return pares, sum(revisado) + len(nuevas)


def _evidencia_por_club(pares: pd.DataFrame, redes: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Para cada club de una red, la relación MÁS FUERTE que lo une a otro
    club de su misma red: regla estricta o revisión manual (evidencia
    determinista) antes que fórmula, y a igualdad, mayor probabilidad. Se
    muestra junto a cada fusión para poder localizar las que se apoyan solo
    en una fórmula de probabilidad baja (y corregirlas en la tabla de pares
    de 'Duplicidad de equipos y canteras', lo que cambia la fusión en la
    siguiente compilación)."""
    red_de = dict(zip(redes["club"], redes["red_id"]))
    rel = pares[pares["tipo"].isin([dup.NOMBRE_FILIAL, dup.NOMBRE_DUPLICIDAD])]
    mejor: dict[str, tuple[int, float, str]] = {}
    for a, b, origen, prob in zip(rel["club_a"], rel["club_b"], rel["origen_regla"], rel["probabilidad"]):
        if a not in red_de or red_de.get(a) != red_de.get(b):
            continue
        rango = 1 if origen in ("estricta", "manual") else 0
        texto = {"estricta": "regla estricta", "manual": "revisión manual"}.get(origen, "fórmula")
        for club in (a, b):
            if (rango, prob) > mejor.get(club, (-1, -1.0, ""))[:2]:
                mejor[club] = (rango, float(prob), texto)
    return {club: (texto, prob) for club, (_, prob, texto) in mejor.items()}


def _fusionar_redes(
    df: pd.DataFrame, redes: pd.DataFrame, aristas: pd.DataFrame, pares: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dentro de cada red: los clubes unidos al líder por aristas de
    DUPLICIDAD (directa o encadenadamente) toman el nombre del líder; el
    resto de la red pasa a '<líder> II'. Todos heredan la liga/bandera del
    líder si este la tiene (más fiable que la aproximación por
    nacionalidad, ver análisis de la pestaña de nacionalidades)."""
    renombre: dict[str, tuple[str, str, str]] = {}  # club -> (nuevo nombre, líder, motivo)
    duplicidades = aristas[aristas["tipo"] == dup.NOMBRE_DUPLICIDAD]
    for red_id, miembros in redes.groupby("red_id"):
        lider = miembros["club_lider"].iat[0]
        vecinos: dict[str, set[str]] = {}
        for a, b in zip(*duplicidades.loc[duplicidades["red_id"] == red_id, ["club_a", "club_b"]].T.values):
            vecinos.setdefault(a, set()).add(b)
            vecinos.setdefault(b, set()).add(a)
        mismo_club, cola = {lider}, deque([lider])
        while cola:
            for v in vecinos.get(cola.popleft(), ()):
                if v not in mismo_club:
                    mismo_club.add(v)
                    cola.append(v)
        for club in miembros["club"]:
            if club == lider:
                continue
            if club in mismo_club:
                renombre[club] = (lider, lider, "duplicidad del líder")
            else:
                renombre[club] = (lider + SUFIJO_FILIAL, lider, "filial / resto de la red")

    liga_lider = df.dropna(subset=["liga"]).groupby("equipo")["liga"].agg(lambda s: s.mode().iat[0])
    resultado = df.copy()
    registros = resultado["equipo"].value_counts()
    nuevo = resultado["equipo"].map({c: n for c, (n, _, _) in renombre.items()})
    lider_de = resultado["equipo"].map({c: l for c, (_, l, _) in renombre.items()})
    mascara = nuevo.notna()
    liga_heredada = lider_de[mascara].map(liga_lider)
    con_liga = liga_heredada.notna()
    resultado.loc[mascara, "equipo"] = nuevo[mascara]
    indices_liga = liga_heredada[con_liga].index
    resultado.loc[indices_liga, "liga"] = liga_heredada[con_liga]
    resultado.loc[indices_liga, "Bandera"] = liga_heredada[con_liga].map(lambda p: nyl.url_bandera(p) or None)

    evidencia = _evidencia_por_club(pares, redes)
    cambios = pd.DataFrame(
        [
            {"equipo_original": c, "equipo_nuevo": n, "club_lider": l, "motivo": m,
             "registros": int(registros.get(c, 0)), "liga_del_lider": liga_lider.get(l),
             "evidencia": evidencia.get(c, ("—", float("nan")))[0],
             "probabilidad": evidencia.get(c, ("—", float("nan")))[1]}
            for c, (n, l, m) in renombre.items()
        ],
        columns=["equipo_original", "equipo_nuevo", "club_lider", "motivo", "registros",
                 "liga_del_lider", "evidencia", "probabilidad"],
    ).sort_values(["club_lider", "motivo", "registros"], ascending=[True, True, False]).reset_index(drop=True)
    return resultado, cambios


def redes_de_la_pestana(original: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Pares y redes de duplicidad/cantera EXACTAMENTE como los muestra la
    pestaña "Duplicidad de equipos y canteras": calculados sobre la tabla
    ORIGINAL (`tabla_valores_mercado`, sin ninguna transformación de la
    compilación), con las revisiones manuales aplicadas y el líder elegido
    por el valor de mercado de la tabla original. Devuelve `(pares, redes,
    aristas, n_revisiones_aplicadas)`."""
    clubs = _comun.cargar_clubs()
    traspasos = _comun.cargar_traspasos()
    valor_por_club = dup.valor_mercado_por_club(original)
    pares = dup.detectar_relaciones_equipos(original, clubs, traspasos)
    pares, n_revisiones = _aplicar_revisiones_duplicidad(pares, set(original["equipo"].dropna()), valor_por_club)
    redes, aristas = dup.resolver_redes_relaciones(pares, valor_por_club)
    return pares, redes, aristas, n_revisiones


def paso_5_ligas_y_fusion_equipos(df: pd.DataFrame, original: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # 5.1
    df_51, asignacion_51 = _asignar_liga_por_nacionalidad(df, UMBRAL_REGISTROS_LIGA_5_1)

    # 5.2 — Las fusiones siguen SIEMPRE los grafos de la pestaña de
    # duplicidad (tabla original + revisiones manuales), no los de la tabla ya
    # transformada: si se recalculasen aquí, las ligas imputadas (pasos 4 y
    # 5.1, p. ej. todos los clubes españoles con liga "Spain") cambiarían las
    # probabilidades de la fórmula y podrían desaparecer pares "ninguna
    # relación" que mantienen separadas dos familias (caso real: RCD
    # Espanyol acababa fusionado como "FC Barcelona II"). Así lo que se
    # fusiona es exactamente lo que se ve y se revisa en la pestaña, con su
    # regla de no unir por fórmula familias con evidencia negativa cruzada.
    pares, redes, aristas, n_revisiones = redes_de_la_pestana(original)
    df_52, cambios_52 = _fusionar_redes(df_51, redes, aristas, pares)
    df_52, n_completados_club = _completar_liga_por_club(df_52)

    # 5.3
    df_53, asignacion_53 = _asignar_liga_por_nacionalidad(df_52, 1)

    return df_53, {
        "asignacion_51": asignacion_51,
        "n_pares": len(pares),
        "n_revisiones_aplicadas": n_revisiones,
        "n_redes": int(redes["red_id"].nunique()) if not redes.empty else 0,
        "cambios_52": cambios_52,
        "n_completados_club": n_completados_club,
        "asignacion_53": asignacion_53,
        "liga_nan_final": int(df_53["liga"].isna().sum()),
    }


# ---------------------------------------------------------------------------
# Paso 6
# ---------------------------------------------------------------------------
def paso_6_valores_erroneos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    datos = df.copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos = datos.sort_values(["id_player", "fecha"], kind="mergesort")
    datos["_pos"] = datos.groupby("id_player").cumcount()
    datos["_valor_ant"] = datos.groupby("id_player")["valor_mercado_eur"].shift(1)
    datos["_valor_sig"] = datos.groupby("id_player")["valor_mercado_eur"].shift(-1)
    datos["_fecha_ant"] = datos.groupby("id_player")["fecha"].shift(1)
    datos["_fecha_sig"] = datos.groupby("id_player")["fecha"].shift(-1)

    anomalos = ve.cambios_anomalos(datos, DIAS_MAX_CAMBIOS, RATIO_UMBRAL_CAMBIOS)
    # Cada cambio anómalo es el salto entre el registro (id, fecha_anterior)
    # y el registro (id, fecha).
    clave = datos.set_index(["id_player", "fecha"]).index
    pos = pd.Series(datos["_pos"].values, index=clave)
    anomalos["pos_anterior"] = [pos.get((i, f)) for i, f in zip(anomalos["id_player"], anomalos["fecha_anterior"])]

    # a) El salto parte del PRIMER registro del jugador -> se elimina ese registro.
    primeros = anomalos[anomalos["pos_anterior"] == 0]
    eliminar = set(zip(primeros["id_player"], primeros["fecha_anterior"]))

    # b) El salto cae en un registro entre dos valores iguales (pico aislado,
    #    ida y vuelta en un periodo corto) -> media de sus dos vecinos.
    pico = (
        (datos["_valor_ant"] == datos["_valor_sig"])
        & (datos["valor_mercado_eur"] != datos["_valor_ant"])
        & ((datos["_fecha_sig"] - datos["_fecha_ant"]).dt.days <= DIAS_MAX_PICOS)
    )
    claves_anomalas = set(zip(anomalos["id_player"], anomalos["fecha"])) | set(
        zip(anomalos["id_player"], anomalos["fecha_anterior"])
    )
    en_anomalo = pd.Series([c in claves_anomalas for c in zip(datos["id_player"], datos["fecha"])], index=datos.index)
    corregir = pico & en_anomalo & ~pd.Series(
        [c in eliminar for c in zip(datos["id_player"], datos["fecha"])], index=datos.index
    )
    # Dos picos SEGUIDOS del mismo jugador (valores alternos, p. ej.
    # 80k-300k-80k-300k) no pueden corregirse a la vez con los valores
    # originales (se intercambiarían): solo se corrige el primero; tras
    # corregirlo, el segundo deja de ser un pico.
    anterior_corregido = corregir.shift(1, fill_value=False) & (
        datos["id_player"] == datos["id_player"].shift(1)
    )
    corregir = corregir & ~anterior_corregido
    corregidos = datos.loc[corregir, ["id_player", "nombre", "equipo", "fecha", "valor_mercado_eur", "_valor_ant", "_valor_sig"]].copy()
    corregidos["valor_nuevo"] = ((corregidos["_valor_ant"] + corregidos["_valor_sig"]) / 2).round().astype("int64")
    datos.loc[corregir, "valor_mercado_eur"] = corregidos["valor_nuevo"]

    borrar = pd.Series([c in eliminar for c in zip(datos["id_player"], datos["fecha"])], index=datos.index)
    eliminados = datos.loc[borrar, ["id_player", "nombre", "equipo", "fecha", "valor_mercado_eur"]]
    resultado = datos[~borrar].drop(columns=[c for c in datos.columns if c.startswith("_")])

    return resultado, {
        "n_anomalos": len(anomalos),
        "eliminados_primer_registro": eliminados.reset_index(drop=True),
        "picos_corregidos": corregidos.rename(
            columns={"valor_mercado_eur": "valor_original", "_valor_ant": "valor_anterior", "_valor_sig": "valor_siguiente"}
        ).reset_index(drop=True),
        "n_sin_regla": len(anomalos) - len(primeros) - int(corregir.sum()),
    }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
# Pasos que necesitan, además de la tabla en curso, la tabla ORIGINAL.
PASOS_CON_ORIGINAL = {"paso_5_ligas_y_fusion_equipos"}

PASOS = [
    ("1. Falsos valores faltantes a NaN", paso_1_na_disfrazados),
    ("2. Separación de 'Arsenal FC'", paso_2_arsenal),
    ("3. Equipo y nacionalidad faltantes", paso_3_equipo_y_nacionalidad),
    ("4. Liga -> país y bandera", paso_4_liga_a_pais),
    ("5. Ligas por nacionalidad y fusión de equipos", paso_5_ligas_y_fusion_equipos),
    ("6. Valores de mercado erróneos", paso_6_valores_erroneos),
]


def compilar(df: pd.DataFrame | None = None, guardar: bool = True, progreso=None) -> tuple[pd.DataFrame, dict]:
    """Ejecuta los 6 pasos en orden. Devuelve `(tabla_limpia, informe)`,
    donde `informe` tiene, por paso, su resumen de cambios y el nº de
    filas antes/después. Si `guardar`, escribe la tabla en parquet y CSV y
    el informe en `ARCHIVO_INFORME`. `progreso(i, nombre)` es un callback
    opcional (la app lo usa para su barra de progreso)."""
    if df is None:
        df = _comun.cargar_tabla_final()
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    original = df.copy()
    informe: dict = {"filas_inicial": len(df), "jugadores_inicial": int(df["id_player"].nunique()), "pasos": {}}

    for i, (nombre, funcion) in enumerate(PASOS):
        if progreso:
            progreso(i, nombre)
        filas_antes = len(df)
        if funcion.__name__ in PASOS_CON_ORIGINAL:
            df, resumen = funcion(df, original)
        else:
            df, resumen = funcion(df)
        resumen["filas_antes"], resumen["filas_despues"] = filas_antes, len(df)
        informe["pasos"][nombre] = resumen

    limpia = df.reindex(columns=COLUMNAS_FINALES).sort_values(["id_player", "fecha"]).reset_index(drop=True)
    informe["filas_final"] = len(limpia)
    informe["jugadores_final"] = int(limpia["id_player"].nunique())
    informe["nulos_final"] = limpia.isna().sum().to_dict()

    if guardar:
        limpia.to_parquet(ARCHIVO_LIMPIA_PARQUET, index=False)
        limpia.to_csv(ARCHIVO_LIMPIA_CSV, index=False, encoding="utf-8")
        ARCHIVO_INFORME.write_bytes(pickle.dumps(informe))
    return limpia, informe


def cargar_informe() -> dict | None:
    if not ARCHIVO_INFORME.exists():
        return None
    return pickle.loads(ARCHIVO_INFORME.read_bytes())


def cargar_tabla_limpia() -> pd.DataFrame | None:
    if not ARCHIVO_LIMPIA_PARQUET.exists():
        return None
    df = pd.read_parquet(ARCHIVO_LIMPIA_PARQUET)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


if __name__ == "__main__":
    import time

    inicio = time.time()
    tabla, inf = compilar(progreso=lambda i, n: print(f"[{time.time() - inicio:6.0f}s] {n}", flush=True))
    print(f"\nFilas: {inf['filas_inicial']:,} -> {inf['filas_final']:,} | jugadores: "
          f"{inf['jugadores_inicial']:,} -> {inf['jugadores_final']:,}")
    print("Nulos finales:", inf["nulos_final"])
    print(f"Guardado en {ARCHIVO_LIMPIA_PARQUET} y {ARCHIVO_LIMPIA_CSV} ({time.time() - inicio:.0f}s)")
