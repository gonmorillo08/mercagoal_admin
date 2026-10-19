"""
Construye la tabla final:

    id_player | nombre | equipo | liga | posicion | nacionalidad | fecha_nacimiento | edad
              | altura_cm | pie | fecha | valor_mercado_eur

a partir de los ficheros crudos descargados por descargar_datos.py:

    - players.csv.gz            -> datos estáticos del jugador (nombre, posición, nacionalidad,
                                   fecha de nacimiento, altura y pie dominante)
    - player_valuations.csv.gz  -> histórico de valores de mercado (una fila por fecha)
    - clubs.csv.gz               -> nombre del club y su competición doméstica a partir de su id
    - competitions.csv.gz        -> nombre de la liga/competición a partir de su id

--------------------------------------------------------------------------
BUG DETECTADO Y CORREGIDO: el "club" quedaba fijo al último club conocido
--------------------------------------------------------------------------
En versiones anteriores, el equipo se obtenía uniendo
player_valuations.current_club_id -> clubs.csv (por id), y la liga a
partir de player_valuations.player_club_domestic_competition_id.

El problema es que en el propio dataset de origen (transfermarkt-datasets)
esas dos columnas son, para la inmensa mayoría de jugadores, el id del
ÚLTIMO club/competición conocido del jugador, y NO el club que tenía en la
fecha concreta de cada valoración. Es un problema del dataset original, no
de nuestro código: se puede comprobar en las propias filas de
player_valuations que "current_club_id" no cambia entre valoraciones,
aunque el jugador haya cambiado de equipo varias veces.

Ejemplo real (Mauro Zárate, player_id 26456): las ~35 valoraciones de su
carrera tienen TODAS current_club_id = 430 (ACF Fiorentina, uno de sus
últimos clubes registrados), a pesar de que jugó en Vélez, Birmingham,
Al-Sadd, Lazio, Inter, West Ham, QPR, Fiorentina, Watford, Boca Juniors...

Comprobación sobre el dataset descargado: de 31.507 jugadores con
histórico de valoraciones, 25.023 (~79 %) tienen un único
current_club_id para todas sus filas, aunque su club real haya cambiado.
Por tanto el join por id no es fiable para el objetivo de este proyecto
(equipo EXACTO en la fecha de la valoración).

--------------------------------------------------------------------------
LA SOLUCIÓN: existe un campo de texto que SÍ es correcto por fecha
--------------------------------------------------------------------------
player_valuations.csv incluye además "current_club_name": un campo de
texto que, a diferencia del id, sí registra el nombre del club que tenía
el jugador en cada valoración concreta (es el dato que Transfermarkt
muestra en su propia gráfica histórica de valor de mercado, fila a fila).
Para Mauro Zárate esta columna sí varía correctamente fila a fila. Esta es
la fuente que usamos ahora para "equipo", y está presente en el 100 % de
las filas de player_valuations.

Para "liga" no existe un campo de texto equivalente, así que la
reconstruimos: normalizamos el nombre de club de cada fila y lo cruzamos
con clubs.csv (también normalizado) para obtener su
domestic_competition_id y, con eso, el nombre de la competición. Cuando el
club de esa fecha no está en clubs.csv (aprox. un 42 % de las filas,
normalmente clubes de ligas menores o de fuera de las grandes competiciones
que rastrea el dataset) la liga queda como nula: es preferible dejarla como
"dato no disponible" (se imputa después como "Desconocido" en
limpieza_datos.py) a rellenarla con una liga incorrecta y fija, que es
justo el bug que estamos corrigiendo.

--------------------------------------------------------------------------
¿Se podría afinar más con appearances / games / game_events?
--------------------------------------------------------------------------
Se han explorado como alternativa (ver README del dataset en
https://github.com/dcaribou/transfermarkt-datasets). Conclusión: no
convienen como fuente PRINCIPAL para este caso:

  - "transfers.csv" (altas/bajas) solo cubre 4.345 jugadores de los 31.507
    que tienen valoraciones (~14 %): usarlo como fuente principal nos
    haría perder cobertura para la mayoría de jugadores.
  - "appearances.csv" tiene huecos conocidos y documentados por el propio
    mantenedor del dataset (p. ej. la temporada 2024/25 completa falta
    para una parte importante de jugadores, ver issue #349 del
    repositorio), por lo que "el partido más cercano a la fecha de
    valoración" no siempre existiría o sería fiable.
  - "current_club_name" en player_valuations, en cambio, está presente en
    el 100 % de las filas y es la misma fuente que usa Transfermarkt para
    mostrar el club histórico en su gráfica de valor de mercado.

Por tanto, current_club_name es la fuente correcta y con mejor cobertura
para "equipo". Si en el futuro se quisiera afinar aún más "liga" para el
~42 % de filas sin match en clubs.csv, sí tendría sentido cruzar con
appearances/games (partido más próximo a la fecha) como fuente
complementaria; se deja preparado el punto de extensión de este módulo
por si se añaden esas tablas a config.TABLAS_NECESARIAS en el futuro.

--------------------------------------------------------------------------
¿Y cruzar "equipo"/"liga" por identificador en vez de por nombre
normalizado? Se probó y se descartó: reintroduce el mismo bug.
--------------------------------------------------------------------------
player_valuations.csv sí trae dos columnas de id además del texto:
"current_club_id" y "player_club_domestic_competition_id". En apariencia
sería preferible cruzar por ahí y evitarnos la normalización de nombre.
Comprobado sobre el dataset descargado, NO lo son: ninguna de las dos
varía correctamente fila a fila para el mismo jugador.

  - "current_club_id" solo cambia de valor, a lo largo de toda su carrera,
    para el 45 % de los jugadores (18.820 de 41.528).
  - "player_club_domestic_competition_id" es todavía peor: solo varía
    para el 19 % (7.890 de 41.528). Para Mauro Zárate (el mismo caso de
    más arriba) vale "IT1" (Italia) en sus 35 valoraciones sin excepción,
    a pesar de haber jugado en Argentina, Inglaterra, Catar, Italia,
    Emiratos Árabes y Brasil — es decir, arrastra el mismo problema de
    "queda fijo al último club/competición conocido" que ya se corrigió
    para "equipo", porque ambas columnas de id derivan de esa misma
    referencia estancada del dataset de origen.
  - "current_club_name" (texto), en cambio, sí varía correctamente fila a
    fila (88 % de los jugadores, 36.489 de 41.528) porque es el campo que
    Transfermarkt sí actualiza en cada valoración.

Por tanto, un cruce por id para "equipo"/"liga" sería más "limpio" en
apariencia (sin normalizar texto) pero peor en la práctica: congelaría de
nuevo ambas columnas al último valor conocido del jugador para la mayoría
de las filas. Cruzar por nombre normalizado contra clubs.csv, aunque menos
elegante, es la única forma fiable de recuperar la liga correcta en cada
fecha con los datos de origen disponibles. El resto de columnas (nombre,
posición, nacionalidad) sí se cruzan por id (player_id) sin este problema,
porque players.csv no tiene el defecto de las dos columnas anteriores.
"""

import logging
import os
import re
import unicodedata

import numpy as np
import pandas as pd

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _ruta_raw(nombre_fichero: str) -> str:
    return os.path.join(config.DIR_RAW, nombre_fichero)


def _leer_csv(nombre_fichero: str) -> pd.DataFrame:
    ruta = _ruta_raw(nombre_fichero)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró {ruta}. Ejecuta primero descargar_datos.py"
        )
    return pd.read_csv(ruta, compression="infer", low_memory=False)


def _primera_columna_existente(df: pd.DataFrame, candidatas: list) -> str:
    """Devuelve la primera columna de 'candidatas' que exista en df."""
    for col in candidatas:
        if col in df.columns:
            return col
    raise KeyError(
        f"Ninguna de las columnas {candidatas} está presente en el DataFrame "
        f"(columnas disponibles: {list(df.columns)})"
    )


def _columna_existente_o_none(df: pd.DataFrame, candidatas: list):
    """Como _primera_columna_existente pero devuelve None (en vez de lanzar
    una excepción) si ninguna candidata existe, para poder aplicar un
    fallback en vez de romper la ejecución."""
    for col in candidatas:
        if col in df.columns:
            return col
    return None


def _normalizar_nombre_club(serie: pd.Series) -> pd.Series:
    """
    Normaliza nombres de club para poder cruzarlos de forma robusta entre
    player_valuations (texto libre) y clubs.csv (texto libre también, pero
    con formato ligeramente distinto): minúsculas, sin acentos/tildes y
    sin puntuación/espacios repetidos.
    """

    def _normalizar(valor):
        if pd.isna(valor):
            return valor
        texto = unicodedata.normalize("NFKD", str(valor))
        texto = texto.encode("ascii", "ignore").decode("ascii")
        texto = re.sub(r"[^a-z0-9]+", " ", texto.lower()).strip()
        return texto

    return serie.map(_normalizar)


def calcular_edad(nacimiento: pd.Series, fecha: pd.Series) -> pd.Series:
    """Edad en años cumplidos en la fecha de cada registro: diferencia de años,
    menos uno si en esa fecha todavía no había cumplido años. Nula si falta
    la fecha de nacimiento (entero con NA de pandas, `Int64`)."""
    no_ha_cumplido = (fecha.dt.month * 100 + fecha.dt.day) < (nacimiento.dt.month * 100 + nacimiento.dt.day)
    edad = fecha.dt.year - nacimiento.dt.year - no_ha_cumplido.astype(int)
    return edad.where(nacimiento.notna() & fecha.notna()).astype("Int64")


def construir_tabla_final() -> pd.DataFrame:
    logger.info("Leyendo ficheros crudos...")
    players = _leer_csv(config.TABLAS_NECESARIAS["players"])
    valuations = _leer_csv(config.TABLAS_NECESARIAS["player_valuations"])
    clubs = _leer_csv(config.TABLAS_NECESARIAS["clubs"])
    competitions = _leer_csv(config.TABLAS_NECESARIAS["competitions"])

    # --- normalizamos nombres de columnas clave (por si cambian entre versiones) ---
    col_posicion = _primera_columna_existente(players, ["position", "sub_position"])
    col_nacionalidad = _primera_columna_existente(
        players, ["country_of_citizenship", "country_of_birth"]
    )
    col_nombre_jugador = _primera_columna_existente(players, ["name", "player_name"])
    # Datos físicos y fecha de nacimiento: opcionales (si una versión futura
    # del dataset no los trae, las columnas quedan vacías en vez de romper).
    col_nacimiento = _columna_existente_o_none(players, ["date_of_birth"])
    col_altura = _columna_existente_o_none(players, ["height_in_cm"])
    col_pie = _columna_existente_o_none(players, ["foot"])
    columnas_players = ["player_id", col_nombre_jugador, col_posicion, col_nacionalidad] + [
        c for c in (col_nacimiento, col_altura, col_pie) if c is not None
    ]

    # 'current_club_name' es, dentro de player_valuations, el nombre del
    # club que tenía el jugador EN CADA FECHA de valoración (ver docstring
    # del módulo). Es la columna clave para corregir el bug de "siempre el
    # último club". Si en el futuro el dataset cambiase de nombre de
    # columna, probamos alternativas conocidas; si ninguna existe, caemos
    # (con aviso) al método antiguo por id para no romper la ejecución,
    # aunque vuelva a tener el problema original.
    col_club_texto = _columna_existente_o_none(
        valuations, ["current_club_name", "club_name"]
    )

    logger.info("Uniendo valoraciones + jugadores...")
    tabla = valuations.merge(
        players[columnas_players],
        on="player_id",
        how="left",
    )

    if col_club_texto is not None:
        logger.info(
            "Usando '%s' como club exacto en la fecha de cada valoración "
            "(corrige el bug de 'último club conocido').",
            col_club_texto,
        )
        tabla["equipo"] = tabla[col_club_texto]

        logger.info(
            "Cruzando el nombre de club (normalizado) con clubs.csv para "
            "obtener su competición doméstica en esa época..."
        )
        clubs_norm = clubs[["name", "domestic_competition_id"]].copy()
        clubs_norm["nombre_club_normalizado"] = _normalizar_nombre_club(
            clubs_norm["name"]
        )
        clubs_norm = clubs_norm.drop_duplicates(subset="nombre_club_normalizado")

        tabla["nombre_club_normalizado"] = _normalizar_nombre_club(tabla["equipo"])
        tabla = tabla.merge(
            clubs_norm[["nombre_club_normalizado", "domestic_competition_id"]],
            on="nombre_club_normalizado",
            how="left",
        )
        col_valuation_liga = "domestic_competition_id"
    else:
        # Fallback: no existe ninguna columna de texto con el club por
        # fecha en esta versión del dataset. Avisamos y usamos el método
        # antiguo (por id) para no interrumpir la actualización, aunque
        # vuelva a arrastrar el problema de "último club conocido".
        logger.warning(
            "No se encontró ninguna columna de texto con el club por fecha "
            "en player_valuations (se probaron: current_club_name, "
            "club_name). Se usa el club_id como alternativa, lo cual puede "
            "reproducir el bug de 'siempre el último club conocido' para "
            "algunos jugadores."
        )
        col_valuation_club = _primera_columna_existente(
            valuations, ["current_club_id", "club_id"]
        )
        col_valuation_liga = _primera_columna_existente(
            valuations,
            ["player_club_domestic_competition_id", "domestic_competition_id"],
        )
        tabla = tabla.merge(
            clubs[["club_id", "name"]].rename(columns={"name": "equipo"}),
            left_on=col_valuation_club,
            right_on="club_id",
            how="left",
        )

    logger.info("Uniendo con competiciones (para obtener el nombre de la liga)...")
    tabla = tabla.merge(
        competitions[["competition_id", "name"]].rename(columns={"name": "liga"}),
        left_on=col_valuation_liga,
        right_on="competition_id",
        how="left",
    )

    fecha_registro = pd.to_datetime(tabla["date"], errors="coerce")
    nacimiento = (
        pd.to_datetime(tabla[col_nacimiento], errors="coerce") if col_nacimiento
        else pd.Series(pd.NaT, index=tabla.index)
    )
    tabla_final = pd.DataFrame(
        {
            "id_player": tabla["player_id"],
            "nombre": tabla[col_nombre_jugador],
            "equipo": tabla["equipo"],
            "liga": tabla["liga"],
            "posicion": tabla[col_posicion],
            "nacionalidad": tabla[col_nacionalidad],
            "fecha_nacimiento": nacimiento.dt.date,
            "edad": calcular_edad(nacimiento, fecha_registro),
            "altura_cm": tabla[col_altura] if col_altura else np.nan,
            "pie": tabla[col_pie] if col_pie else None,
            "fecha": fecha_registro.dt.date,
            "valor_mercado_eur": tabla["market_value_in_eur"],
        }
    )

    tabla_final = tabla_final.dropna(subset=["id_player", "fecha"])
    tabla_final = tabla_final.sort_values(["id_player", "fecha"]).reset_index(drop=True)

    return tabla_final


def construir_historial_transferencias(
    df: pd.DataFrame,
    columna_id: str = "id_player",
    columna_equipo: str = "equipo",
    columna_liga: str = "liga",
    columna_fecha: str = "fecha",
    columna_nombre: str = "nombre",
    columna_valor: str = "valor_mercado_eur",
) -> pd.DataFrame:
    """Reconstruye los traspasos individuales (uno por jugador y cambio de
    equipo) a partir del historial de registros de cada jugador (la tabla
    final construida por `construir_tabla_final()`).

    Devuelve un DataFrame con una fila por traspaso detectado:
    `id_player, nombre, equipo_origen, equipo_destino, liga_origen,
    liga_destino, fecha, valor_mercado` (fecha del registro en el que se
    detecta el cambio de equipo; `valor_mercado` es el valor de mercado
    del jugador en ESE registro, ya que este historial no conoce la
    cuantía real de ningún fichaje).

    Es la base de la tabla de traspasos (`construir_traspasos()`).
    """
    columnas_necesarias = [
        columna_id, columna_equipo, columna_liga, columna_fecha, columna_nombre, columna_valor,
    ]
    datos = df[columnas_necesarias].dropna(
        subset=[columna_id, columna_equipo, columna_fecha]
    ).copy()
    datos[columna_fecha] = pd.to_datetime(datos[columna_fecha])
    datos = datos.sort_values([columna_id, columna_fecha], kind="mergesort")

    # "Equipo actual" = equipo del registro inmediatamente anterior del
    # mismo jugador. Si el equipo del registro actual difiere de él, es un
    # traspaso directo (equipo_actual -> nuevo equipo).
    equipo_anterior = datos.groupby(columna_id, sort=False)[columna_equipo].shift(1)
    liga_anterior = datos.groupby(columna_id, sort=False)[columna_liga].shift(1)
    hay_cambio = equipo_anterior.notna() & (datos[columna_equipo] != equipo_anterior)

    traspasos = pd.DataFrame(
        {
            "id_player": datos.loc[hay_cambio, columna_id].values,
            "nombre": datos.loc[hay_cambio, columna_nombre].values,
            "equipo_origen": equipo_anterior.loc[hay_cambio].values,
            "equipo_destino": datos.loc[hay_cambio, columna_equipo].values,
            "liga_origen": liga_anterior.loc[hay_cambio].values,
            "liga_destino": datos.loc[hay_cambio, columna_liga].values,
            "fecha": datos.loc[hay_cambio, columna_fecha].values,
            "valor_mercado": datos.loc[hay_cambio, columna_valor].values,
        }
    ).reset_index(drop=True)
    return traspasos


def construir_traspasos(tabla_final: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de traspasos (`traspasos.csv`) usando
    ÚNICAMENTE el historial de valoraciones: un traspaso por cada cambio
    de equipo entre dos valoraciones consecutivas del mismo jugador
    (`construir_historial_transferencias()`).

    Por qué no se usa `transfers.csv`: una versión anterior
    (`movimientos.csv`) combinaba este historial con `transfers.csv`, pero
    `transfers.csv` cita muchos clubes que nunca aparecen en
    `player_valuations.csv` (y por tanto tampoco en la tabla final): 16.031
    de los 24.036 equipos de aquella tabla eran equipos de los que no
    teníamos ninguna otra información. Aquí `equipo_origen`/`equipo_destino` salen
    siempre de `player_valuations.current_club_name` y `liga_origen`/
    `liga_destino` de cruzar ese nombre con `clubs.csv` +
    `competitions.csv` (lo mismo que ya hace `construir_tabla_final()`),
    así que todo equipo de esta tabla existe también en la tabla final.
    `nombre` es solo la etiqueta del jugador (por `player_id`) que ya trae
    la tabla final.

    Columnas: `id_player, nombre, equipo_origen, equipo_destino,
    liga_origen, liga_destino, fecha, valor_mercado, fuente` (`fuente` =
    `"historial"` en todas las filas; no hay importe real de traspaso, que
    solo lo conocía `transfers.csv`).
    """
    traspasos = construir_historial_transferencias(tabla_final)
    traspasos["fuente"] = "historial"
    traspasos["fecha"] = pd.to_datetime(traspasos["fecha"])
    traspasos["valor_mercado"] = pd.to_numeric(traspasos["valor_mercado"], errors="coerce")
    return traspasos.sort_values(["id_player", "fecha"], kind="mergesort").reset_index(drop=True)


def guardar_traspasos(traspasos: pd.DataFrame) -> None:
    os.makedirs(config.DIR_DATOS, exist_ok=True)
    traspasos.to_csv(config.ARCHIVO_TRASPASOS, index=False, encoding="utf-8")
    logger.info(
        "Tabla de traspasos guardada en %s (%d filas)",
        config.ARCHIVO_TRASPASOS,
        len(traspasos),
    )


def guardar_tabla_final(tabla_final: pd.DataFrame) -> None:
    os.makedirs(config.DIR_DATOS, exist_ok=True)
    tabla_final.to_csv(config.ARCHIVO_TABLA_FINAL, index=False, encoding="utf-8")
    logger.info("Tabla guardada en %s (%d filas)", config.ARCHIVO_TABLA_FINAL, len(tabla_final))

    try:
        tabla_final.to_parquet(config.ARCHIVO_TABLA_FINAL_PARQUET, index=False)
        logger.info("Tabla también guardada en %s", config.ARCHIVO_TABLA_FINAL_PARQUET)
    except ImportError:
        logger.warning(
            "No se pudo guardar en formato parquet (falta 'pyarrow' o 'fastparquet'). "
            "Se mantiene solo el CSV. Instala con: pip install pyarrow"
        )


if __name__ == "__main__":
    tabla = construir_tabla_final()
    guardar_tabla_final(tabla)
    print(tabla.head(10).to_string(index=False))
    print(f"\nTotal de filas: {len(tabla)}")

    traspasos = construir_traspasos(tabla)
    guardar_traspasos(traspasos)
    print(traspasos.head(10).to_string(index=False))
    print(f"\nTotal de traspasos: {len(traspasos)}")
