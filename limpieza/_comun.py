"""
Utilidades compartidas por todos los módulos de `limpieza/`: rutas a los
datos (crudos y ya procesados), carga de tablas y normalización de nombres
de club.

Sigue el mismo patrón que `visualizacion/red_de_traspasos.py` para
localizar los ficheros de `ingest/datos/` (rutas absolutas basadas en la
ubicación de este fichero, para no depender del directorio desde el que se
ejecute Python) en vez de importar `ingest.config`: los scripts de
`ingest/` usan `import config` a secas, lo que solo funciona cuando se
ejecutan directamente (Python añade su propio directorio a `sys.path`), no
cuando se importan como paquete desde otra carpeta.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DIR_INGEST = Path(__file__).resolve().parent.parent / "ingest"
DIR_DATOS = DIR_INGEST / "datos"
DIR_RAW = DIR_DATOS / "raw"

ARCHIVO_TABLA_FINAL_CSV = DIR_DATOS / "tabla_valores_mercado.csv"
ARCHIVO_TABLA_FINAL_PARQUET = DIR_DATOS / "tabla_valores_mercado.parquet"
ARCHIVO_TRASPASOS = DIR_DATOS / "traspasos.csv"
ARCHIVO_CLUBS = DIR_RAW / "clubs.csv.gz"
ARCHIVO_COMPETITIONS = DIR_RAW / "competitions.csv.gz"
ARCHIVO_PLAYERS = DIR_RAW / "players.csv.gz"

_INSTRUCCION_INGEST = (
    "Ejecuta `python ingest/actualizar.py` (o `python ingest/descargar_datos.py` "
    "seguido de `python ingest/procesar_datos.py`) para generarlo."
)


def _requerir(ruta: Path) -> Path:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}. {_INSTRUCCION_INGEST}")
    return ruta


def cargar_tabla_final() -> pd.DataFrame:
    """Carga `tabla_valores_mercado` (parquet si existe, más rápido; si no,
    el CSV)."""
    if ARCHIVO_TABLA_FINAL_PARQUET.exists():
        df = pd.read_parquet(ARCHIVO_TABLA_FINAL_PARQUET)
    else:
        df = pd.read_csv(_requerir(ARCHIVO_TABLA_FINAL_CSV))
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def cargar_traspasos() -> pd.DataFrame:
    df = pd.read_csv(_requerir(ARCHIVO_TRASPASOS))
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def cargar_clubs() -> pd.DataFrame:
    return pd.read_csv(_requerir(ARCHIVO_CLUBS), compression="infer", low_memory=False)


def cargar_competitions() -> pd.DataFrame:
    return pd.read_csv(_requerir(ARCHIVO_COMPETITIONS), compression="infer", low_memory=False)


def cargar_players() -> pd.DataFrame:
    return pd.read_csv(_requerir(ARCHIVO_PLAYERS), compression="infer", low_memory=False)


# ---------------------------------------------------------------------------
# Normalización y comparación de nombres de club
# ---------------------------------------------------------------------------
# Siglas/palabras de forma jurídica o de "tipo de club" que aparecen en
# muchos idiomas y no aportan identidad propia (a diferencia de "Real",
# "Atlético" o "Deportivo", que SÍ distinguen clubes distintos de una misma
# ciudad, p. ej. Real Madrid / Atlético Madrid, y por tanto NO se incluyen
# aquí): se eliminan al comparar nombres para no fallar por "FC Barcelona"
# vs "Barcelona". Coincide con la subcategoría "futbol" de `DICCIONARIO_
# FORMAS_JURIDICAS` (ver más abajo): se mantiene como constante aparte
# porque, a diferencia de esa subcategoría, ÉSTA se usa también para
# recortar el "núcleo" del nombre en `palabras_nucleo()`.
ACRONIMOS_CLUB = {
    "fc", "cf", "cd", "ac", "sc", "sd", "ud", "rc", "rcd", "afc", "cfc",
    "ssc", "ss", "ca", "ce", "fk", "sk", "nk", "bk", "if", "sv", "ff", "bc",
    "club", "calcio", "futbol", "football", "fussballclub", "fussball",
    "clube", "futebol",
}

# Palabras/sufijos que, cuando aparecen SOLO en uno de los dos nombres
# comparados, delatan casi siempre un equipo filial (segundo equipo,
# reserva o cantera) del club que comparte el resto del nombre — p. ej.
# "FC Barcelona" -> "FC Barcelona Atlètic"/"FC Barcelona B"/"...U19". Vive
# aquí porque `duplicidad_equipos.py` la necesita para no confundir un
# filial con un duplicado por errata. Es el "diccionario de filiales" del
# enunciado (a diferencia de `DICCIONARIO_FORMAS_JURIDICAS`, no hace falta
# subdividirlo: todas sus palabras cumplen el mismo papel, marcar un
# segundo equipo/cantera, así que forman una única categoría "filial").
DICCIONARIO_FILIALES = {
    "b", "c", "ii", "iii", "2", "3", "atletic", "castilla",
    "castellana", "juvenil", "cantera", "youth", "reserves", "reserva",
    "reservas", "academy", "promesas",
    "primavera", "colts", "development", "jong",
    "u14", "u15", "u16", "u17", "u18", "u19", "u20", "u21", "u22", "u23",
    "sub14", "sub15", "sub16", "sub17", "sub18", "sub19", "sub20",
    "sub21", "sub22", "sub23",
    "mestalla", "imperial", "vetusta", "fabril", "fortuna",
}
# "deportivo", "atletico", "juniors" y "academia" se retiraron a propósito
# de este diccionario: además de sufijo de segundo equipo, son con
# frecuencia la identidad propia de un club real distinto ("Deportivo
# Guadalajara", "Club Atlético X", "Boca Juniors", "Academia Puerto
# Cabello"...), así que su presencia por sí sola no es evidencia fiable de
# cantera — y, ya fuera del diccionario, tampoco activan la exclusión de
# pares "ambos con palabra de filiales" de `duplicidad_equipos.
# generar_pares_candidatos`.

# Palabras que aparecen en el nombre de decenas de clubes SIN relación
# alguna entre sí (genéricas del vocabulario futbolístico de un idioma:
# "Town"/"City"/"Athletic" en inglés, "Deportes"/"Esporte" en
# español/portugués...) y que por tanto no son una señal útil de
# "ambigüedad": toda palabra de bloqueo ya exige >= 4 letras, pero estas
# además son demasiado frecuentes como para señalar un riesgo de
# confusión concreto entre DOS clubes en particular.
PALABRAS_NUCLEO_GENERICAS = {
    "town", "city", "united", "athletic", "rovers", "sport", "sporting",
    "spor", "esporte", "futebol", "deportes", "deportivo", "belediyespor",
    "clube", "balompie",
}

# Lista blanca DENTRO de `PALABRAS_NUCLEO_GENERICAS`: palabras genéricas que,
# comprobado con los datos reales del dataset (`duplicidad_equipos.
# generar_pares_por_nombre`), SÍ pueden usarse como clave de bloqueo sin
# generar falsos positivos — es decir, entre los clubes que la comparten,
# los pares que la fórmula clasifica "posible filial"/"posible duplicidad"
# son correctos, no coincidencias.
#
# Se probó cada palabra de `PALABRAS_NUCLEO_GENERICAS` por separado (dejando
# el resto excluidas) y se revisaron a mano sus nuevas clasificaciones. Solo
# "sporting" salió limpia: sus 9 nuevos pares son todos "Sporting CP" con
# sus propias canteras (B/U17/U19/U23/Sub-15), y el caso que motivó esta
# lista blanca — "Sporting CP"/"Sporting FC" — queda correctamente
# clasificado "ninguna relación" (52 %). El resto de palabras probadas
# generó falsos positivos reales, algunos graves — p. ej. "united":
# "Manchester United"/"West Ham United U18" como "posible filial" (52,5 %);
# "sport": "Sport Club Corinthians Paulista"/"Sport Club Internacional B"
# como "posible filial"; "city": "Leicester City"/"Worcester City",
# "Manchester City"/"Worcester City"; "belediyespor"/"esporte"/"futebol"/
# "clube": decenas de clubes turcos/brasileños sin relación cruzados entre
# sí; "deportes": "Deportes Limache"/"Deportes Linares" como "posible
# duplicidad" (62,8 %); "balompie" (el ejemplo que motivó originalmente
# excluir palabras genéricas, ver más arriba): "Real Betis Balompié"/
# "Albacete Balompié" como "posible filial" (43,9 %) — así que se quedan
# fuera de esta lista blanca.
PALABRAS_NUCLEO_GENERICAS_PERMITIDAS = frozenset({"sporting"})

# ---------------------------------------------------------------------------
# Diccionario de formas jurídicas (segundo diccionario del enunciado)
# ---------------------------------------------------------------------------
# Palabras (ya normalizadas: minúsculas, sin acentos) que, dentro de una
# misma subcategoría, se consideran formas intercambiables de EXACTAMENTE
# la misma cosa — no traducciones/variantes "parecidas" de conceptos
# distintos. Se usa en `duplicidad_equipos.py` para decidir si la única
# palabra que diferencia a dos nombres de equipo es una simple abreviatura
# de forma jurídica (señal de DUPLICIDAD: mismo club, dos grafías) o, en
# combinación con `DICCIONARIO_FILIALES`, un sufijo de cantera sustituyendo
# a una forma jurídica (señal de FILIAL).
#
# Deliberadamente conservador: cada subcategoría solo incluye variantes de
# las que se puede estar razonablemente seguro de que SIEMPRE significan lo
# mismo, para no repetir el error que motiva este módulo (p. ej. "Sporting
# CP" / "Sporting FC" son clubes distintos que comparten "Sporting" + una
# sigla jurídica genérica; agrupar "cp" y "fc" en la misma subcategoría
# convertiría ese caso en un falso positivo de duplicidad en vez de
# evitarlo — lo que realmente los distingue es que cada uno cruza con un
# `club_id` distinto de `clubs.csv`, ver `duplicidad_equipos.
# detectar_relaciones_equipos`).
#
# Subcategorías. Algunas palabras son, en realidad, FRASES de varias
# palabras ("futbol club", "sociedad deportiva"...): se guardan aquí con
# un guion bajo en vez de espacio ("futbol_club") y `tokenizar_nombre()`
# las funde en un único token ANTES de partir el nombre en palabras (ver
# más abajo), para que se comparen como una unidad, igual que "fc".
#
#   a) "united": "Manchester United" / "Man Utd" / "Man Unite" / "Union"...
#   b) "universidad": "Universitatea" / "University" / "Universidad" / "U"...
#   c) "dinamo": transliteración cirílica vs. latina del mismo término.
#   d) "futbol_palabra": palabras completas que significan literalmente
#      "fútbol" en distintos idiomas ("football"/"fussball"/"futbol"/
#      "futebol"/"fodbold").
#   e) "club": "club"/"clube" (genérico, sin especificar deporte).
#   f) "futbol_club": "club de fútbol" en varios idiomas, como sigla o
#      como frase completa ("fc"/"ff"/"fk"/"fussballclub"/"futbol club").
#   g) "saint": "Saint-Étienne" / "St Etienne" / "Sant Andreu" / "San
#      Lorenzo"...
#   h) "sociedad_deportiva": "Sociedad Deportiva"/"SD" (español) y
#      "Sociedade Esportiva"/"SE" (portugués) — misma forma jurídica en
#      dos idiomas.
#   i) "inter": "Inter Milan" / "Internacional" / "Internazionale"...
#   j) "real": "Real Madrid" / "Royal Antwerp" — "real"/"royal" son la
#      misma forma jurídica (monárquica) en español/inglés.
#   k) "club_atletico": "Club Atlético"/"CA" (español) y "Clube Atlético"
#      (portugués) como sigla o frase completa.
#   l) "siglas_italianas": "SS"/"US" — siglas jurídicas de club italianas
#      (Società Sportiva/Unione Sportiva). "AC" (Associazione Calcio) se
#      quitó de aquí: coincide con la sigla de "atlético" en portugués/
#      italiano en demasiados nombres reales de club sin relación entre
#      sí, así que se ha excluido de este diccionario por completo (ya no
#      es una palabra reconocida en `DICCIONARIO_FORMAS_JURIDICAS`).
#   m) "asociacion_deportiva": "Asociación Deportiva"/"AD" (español) y
#      "SV" (alemán, "Sportverein" = "asociación deportiva").
#   n) "otras_formas_juridicas": el resto de palabras de forma jurídica,
#      partícula de nombre o sigla de club que no encajan en ninguna
#      subcategoría anterior. A diferencia de las demás, estas palabras SÍ
#      son con frecuencia identidad propia de un club real distinto, no
#      solo forma jurídica — por eso quedan excluidas de la regla de
#      DUPLICIDAD/NINGUNA RELACIÓN por "(mismas o distintas) subcategoría"
#      (ver `duplicidad_equipos.py`).
#
# "cf"/"cfc" quedan deliberadamente en (n) y NO en (f) "futbol_club": son
# la sigla inversa ("Club de Fútbol" vs. "Fútbol Club") y, a diferencia de
# fc/fk, se usan con frecuencia como identidad propia intercambiada entre
# clubes DISTINTOS de una misma ciudad (p. ej. no hay evidencia de que
# "cf" y "fc" sean siempre intercambiables sin riesgo, al contrario que
# fc/ff/fk, variantes puramente ortográficas del mismo acrónimo).
DICCIONARIO_FORMAS_JURIDICAS: dict[str, frozenset[str]] = {
    "united": frozenset({"unite", "united", "utd", "union"}),
    "universidad": frozenset({"u", "universidad", "universitaria", "universitario", "universitatea", "university"}),
    "dinamo": frozenset({"dinamo", "dynamo"}),
    "futbol_palabra": frozenset({"fodbold", "football", "fussball", "futbol", "futebol"}),
    "club": frozenset({"club", "clube"}),
    "futbol_club": frozenset({"fc", "ff", "fk", "fussballclub", "futbol_club"}),
    "saint": frozenset({"saint", "sant", "st", "san"}),
    "sociedad_deportiva": frozenset({"sociedad_deportiva", "sd", "sociedade_esportiva", "se"}),
    "inter": frozenset({"inter", "internacional", "internazionale"}),
    "real": frozenset({"real", "royal"}),
    "club_atletico": frozenset({"club_atletico", "ca", "clube_atletico"}),
    "siglas_italianas": frozenset({"ss", "us"}),
    "asociacion_deportiva": frozenset({"asociacion_deportiva", "ad", "sv"}),
    "otras_formas_juridicas": frozenset({
        "afc", "balompie", "bc", "bk", "calcio", "cd", "ce", "cf", "cfc", "city",
        "cs", "hapoel", "if", "independiente", "maccabi", "nacional", "nk", "rc",
        "rcd", "sc", "sk", "sporting", "ssc", "town", "ud",
    }),
}

# Índice combinado de los DOS diccionarios (filiales + formas jurídicas,
# con sus subcategorías) palabra -> categoría, para no tener que mirar cada
# diccionario por separado al clasificar una palabra de diferencia entre
# dos nombres.
_PALABRA_A_CATEGORIA: dict[str, str] = {palabra: "filial" for palabra in DICCIONARIO_FILIALES}
for _categoria, _palabras in DICCIONARIO_FORMAS_JURIDICAS.items():
    for _palabra in _palabras:
        _PALABRA_A_CATEGORIA[_palabra] = _categoria


def categoria_de_palabra(palabra: str) -> str | None:
    """Categoría del TOKEN `palabra` (ya normalizada y, si es una frase de
    varias palabras del diccionario, ya fusionada con guion bajo por
    `tokenizar_nombre`): "filial" (`DICCIONARIO_FILIALES`), una de las
    subcategorías de `DICCIONARIO_FORMAS_JURIDICAS`, o `None` si no
    pertenece a ninguna."""
    return _PALABRA_A_CATEGORIA.get(palabra)


# Frases (varias palabras) del diccionario de formas jurídicas, más largas
# primero (para que "futbol club" se funda ANTES que un eventual "club"
# suelto dentro de ella lo intente fundir como token independiente).
# `tokenizar_nombre()` las usa para fundir cada frase en un único token
# ("futbol club" -> "futbol_club") antes de partir el nombre en palabras.
FRASES_DICCIONARIO: tuple[str, ...] = tuple(
    sorted(
        (
            palabra.replace("_", " ")
            for palabras in DICCIONARIO_FORMAS_JURIDICAS.values()
            for palabra in palabras
            if "_" in palabra
        ),
        key=lambda frase: -len(frase.split()),
    )
)


def tokenizar_nombre(nombre) -> list[str]:
    """Nombre normalizado (sin acentos/puntuación, ver `normalizar_texto`)
    y partido en tokens, fundiendo antes cada frase de `FRASES_DICCIONARIO`
    que aparezca literalmente en el texto en un único token con guion bajo
    (p. ej. "real sociedad deportiva alaves" -> "real sociedad_deportiva
    alaves" -> ["real", "sociedad_deportiva", "alaves"]), para que una
    frase de varias palabras del diccionario de formas jurídicas se
    compare como una unidad, igual que una sigla de una sola palabra."""
    texto = normalizar_texto(nombre)
    if not isinstance(texto, str) or not texto:
        return []
    for frase in FRASES_DICCIONARIO:
        texto = texto.replace(frase, frase.replace(" ", "_"))
    return texto.split()


def normalizar_texto(valor) -> str | float:
    """minúsculas, sin acentos/diacríticos, sin puntuación ni espacios
    repetidos. Devuelve NaN tal cual si `valor` ya es NaN."""
    if pd.isna(valor):
        return valor
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = texto.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", texto.lower()).strip()


def normalizar_nombre_club(serie: pd.Series) -> pd.Series:
    """Versión vectorizada de `normalizar_texto` para una columna de
    nombres de club (misma normalización que usa `ingest/procesar_datos.py`
    para cruzar el histórico de valoraciones con `clubs.csv`)."""
    return serie.map(normalizar_texto)


def palabras_nucleo(nombre: str) -> list[str]:
    """Palabras del nombre normalizado, sin las siglas/formas jurídicas de
    `ACRONIMOS_CLUB` (el "núcleo" identificativo del club)."""
    texto = normalizar_texto(nombre)
    if pd.isna(texto):
        return []
    return [palabra for palabra in texto.split() if palabra not in ACRONIMOS_CLUB]


def es_relacion_filial(nombre_a: str, nombre_b: str) -> bool:
    """Heurística (solo por nombre, sin mirar traspasos) de si `nombre_b`
    podría ser un equipo filial de `nombre_a` (o viceversa): las palabras
    núcleo de uno y otro deben ser IDÉNTICAS salvo, como mucho, por
    palabras de `DICCIONARIO_FILIALES`.

    Importante que sea un SUBCONJUNTO (`issubset`) y no una intersección:
    con intersección, "Barcelona SC Guayaquil" (núcleo {barcelona,
    guayaquil}) se marcaba como filial de "FC Barcelona Atlètic" (núcleo
    {barcelona, atletic}) solo por compartir una palabra con
    `DICCIONARIO_FILIALES` ("atletic"), aunque la otra palabra distinta
    ("guayaquil") no tuviera nada que ver — un falso positivo real,
    detectado al validar esta función contra ese caso conocido. Exigir
    que TODA la diferencia esté en
    `DICCIONARIO_FILIALES` lo evita, a costa de no reconocer relaciones
    donde el filial sustituye una palabra en vez de añadirla (p. ej. "Real
    Betis Balompié" -> "Betis Deportivo Balompié"): esta heurística, al
    basarse solo en el nombre, no detecta esos casos.
    """
    palabras_a = set(palabras_nucleo(nombre_a))
    palabras_b = set(palabras_nucleo(nombre_b))
    palabras_extra = palabras_a ^ palabras_b
    return bool(palabras_extra) and palabras_extra.issubset(DICCIONARIO_FILIALES)


def inferir_transiciones_equipo(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye, a partir de `tabla_valores_mercado` (una fila por
    jugador y fecha de valoración), cada cambio de equipo entre dos
    registros consecutivos del mismo jugador: mismo mecanismo que
    `ingest/procesar_datos.construir_historial_transferencias`, pero
    reducido a solo `origen`/`destino` (sin fechas ni valores), que es todo
    lo que necesitan los módulos de este paquete para medir qué fracción de
    las salidas de un club van a parar a otro club concreto.
    """
    datos = df[["id_player", "equipo", "fecha"]].dropna(subset=["id_player", "equipo", "fecha"])
    datos = datos.sort_values(["id_player", "fecha"], kind="mergesort")
    equipo_anterior = datos.groupby("id_player", sort=False)["equipo"].shift(1)
    hay_cambio = equipo_anterior.notna() & (datos["equipo"] != equipo_anterior)
    return pd.DataFrame(
        {
            "origen": equipo_anterior.loc[hay_cambio].values,
            "destino": datos.loc[hay_cambio, "equipo"].values,
        }
    )
