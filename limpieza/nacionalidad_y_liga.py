"""
Bandera asociada a cada país de `liga` (vía `competitions.csv`) y de
`nacionalidad` en `tabla_valores_mercado`, y detección de los casos en los
que esa asociación es ambigua o directamente imposible.

--------------------------------------------------------------------------
Por qué cada bandera se ofrece en DOS formatos (imagen Y emoji)
--------------------------------------------------------------------------
La primera versión de este módulo generaba solo el emoji-bandera Unicode
(dos "regional indicator symbols", el mecanismo con el que Unicode compone
banderas de país). Es correcto a nivel de datos —se comprobó carácter a
carácter—, pero SU DIBUJO depende de que el sistema operativo de quien
mira la app tenga una fuente de emoji con esos glifos. Comprobado con
capturas reales de la app en Windows: en vez de la bandera, se ve el
código de 2 letras dentro de una cajita ("AT", "DE"...) — Windows,
históricamente, no incluye los dibujos de bandera en su fuente de emoji
(`Segoe UI Emoji`) y usa esa cajita como alternativa; el componente de
tabla de Streamlit, al dibujar el texto en un `canvas`, depende todavía
más estrictamente de esa fuente del sistema que el texto HTML normal.

La solución para tener SIEMPRE una bandera visible: además del emoji
(`emoji_bandera`, columna `bandera_emoji`), cada país tiene una URL a una
imagen real (PNG) servida por flagcdn.com (https://flagcdn.com/), un
servicio público y gratuito de banderas por código de país
(`url_bandera`, columna `bandera_url`). Una imagen se ve igual en
cualquier sistema operativo/navegador, sin depender de ninguna fuente
local — a cambio, la app necesita conexión a internet para cargarla (no
se descargan/empaquetan banderas en el repositorio). Se mantienen AMBAS
columnas a propósito: el emoji es texto plano (copiable, sin depender de
red) y sí se ve bien fuera de Windows; la imagen es la que garantiza
verse en cualquier sistema. `flagcdn.com`, además de los códigos ISO
3166-1 normales, cubre también códigos de territorio NO oficiales que
hacen falta aquí: "gb-eng"/"gb-sct"/"gb-wls"/"gb-nir" (las cuatro
naciones del Reino Unido, Irlanda del Norte incluida — algo que el emoji
Unicode no puede cubrir del todo, ver `emoji_bandera`) y "xk" (Kosovo).

--------------------------------------------------------------------------
Por qué `liga` necesita una tabla propia (no basta con un diccionario)
--------------------------------------------------------------------------
`tabla_valores_mercado.liga` es el NOMBRE de la competición
(`competitions.name`, p. ej. `"bundesliga"`), no su id. Ese nombre NO es
único por país dentro de `competitions.csv`: varias competiciones
nacionales de países distintos comparten el mismo nombre corto (con los
datos reales de este proyecto: Alemania Y Austria tienen las dos una
"bundesliga"; Rusia Y Ucrania una "premier-liga"; Dinamarca Y Rumanía una
"superliga"). Como `ingest/procesar_datos.py` guarda en la tabla final solo
el nombre de la competición, no el `competition_id` que sí conocía en el
momento del cruce (ver docstring de `construir_tabla_final`), a partir de
`tabla_valores_mercado` ya no se puede saber con certeza a qué país
pertenece una fila con una liga ambigua de estas. `tabla_banderas_ligas`
no intenta "adivinar" cuál de los dos países es: genera una fila por cada
país coincidente (con su propia bandera) y las marca `conflicto=True`
para que se traten a mano.

--------------------------------------------------------------------------
Por qué `nacionalidad` necesita, además, una tabla de "problemas"
--------------------------------------------------------------------------
Aquí no hay ambigüedad de nombre (cada país real tiene su propio texto).
Gracias a la imagen de flagcdn.com, Inglaterra/Escocia/Gales/Irlanda del
Norte YA NO son un caso problemático en `bandera_url` (tienen su bandera
propia real, exacta, vía "gb-eng"/"gb-sct"/"gb-wls"/"gb-nir"); en
`bandera_emoji` las tres primeras también tienen su secuencia Unicode
propia, pero Irlanda del Norte no tiene NINGUNA secuencia de bandera
asignada en el estándar, así que en esa columna cae a la bandera del
Reino Unido (ver `emoji_bandera`) — una limitación solo del emoji, no de
la imagen. Quedan, aun así, otros casos que sí conviene revisar a mano:

  - Códigos no oficiales: Kosovo usa "XK" (no es un código ISO 3166-1
    real, por ser un territorio de estatus disputado), aunque flagcdn.com
    sí sirve su bandera sin problema.
  - Designaciones deportivas en vez de políticas: "Chinese Taipei" es el
    nombre con el que compite Taiwán; se aproxima a la bandera de Taiwán
    (TW), una aproximación con más carga política que las demás.
  - Territorios sin bandera propia: "Tahiti" es una isla de la Polinesia
    Francesa, no un territorio con código ISO propio; se aproxima a la
    bandera de la Polinesia Francesa (PF).
  - El mismo país o territorio real repetido con dos grafías distintas en
    el propio dataset: `"Turkey"` / `"Türkiye"` (nombre internacional
    anterior/posterior a 2022) y `"New Caledonia"` / `"Neukaledonien"`
    (inglés/alemán). Comprobado a mano que NO es un problema de
    codificación (los bytes de "Türkiye"/"Réunion" en los ficheros de
    `ingest/datos/` son UTF-8 correcto): son dos textos distintos de
    verdad, no una misma cadena corrupta.

Este módulo NO corrige nada de `ingest/`: solo construye, a partir de los
datos ya generados, la tabla de banderas para cada liga y cada
nacionalidad, y una tabla aparte con los casos de nacionalidad que
conviene revisar a mano.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import _comun

# ---------------------------------------------------------------------------
# Construcción de la URL de la imagen de bandera a partir de un código
# ---------------------------------------------------------------------------
_URL_FLAGCDN = "https://flagcdn.com/40x30/{codigo}.png"


def url_desde_codigo(codigo: str) -> str:
    """URL (PNG, 40x30 px SIEMPRE) de la bandera de `codigo` en
    flagcdn.com: normalmente un código ISO 3166-1 alpha-2 (p. ej. "es"),
    pero también admite los códigos de territorio NO oficiales que
    necesita este módulo ("gb-eng", "gb-sct", "gb-wls", "gb-nir", "xk").
    Se pide un tamaño FIJO (ancho Y alto) en vez de solo el ancho (que
    era lo que hacía esta función en la versión anterior): con solo el
    ancho fijo, flagcdn.com respeta la proporción real de cada bandera, y
    banderas casi cuadradas como la de Suiza salían visiblemente más
    "grandes" (más altas) que el resto (proporción típica 3:2) en la
    tabla de la app — con ancho Y alto fijos, todas ocupan exactamente el
    mismo espacio (a costa de estirar ligeramente las pocas banderas que
    no son 3:2, como la de Suiza). flagcdn.com se encarga de servir
    siempre la misma imagen para el mismo código, así que no hace falta
    descargar/cachear nada aquí."""
    return _URL_FLAGCDN.format(codigo=codigo.lower())


# ---------------------------------------------------------------------------
# Construcción del emoji-bandera Unicode a partir de un código (ver
# docstring del módulo: se ofrece COMO COMPLEMENTO de la URL de imagen,
# no como sustituto)
# ---------------------------------------------------------------------------
def bandera_desde_iso2(codigo_iso2: str) -> str:
    """Emoji-bandera Unicode a partir de un código de 2 letras (normalmente
    ISO 3166-1 alpha-2, p. ej. "ES" -> 🇪🇸): cada letra se traduce a su
    "regional indicator symbol" (U+1F1E6 = 'A'), el mecanismo con el que
    Unicode compone banderas de país sin necesitar un carácter dedicado
    por bandera. El mecanismo no distingue si el código es un ISO 3166-1
    real o no, así que también sirve, técnicamente, para códigos NO
    oficiales de 2 letras como el de Kosovo ("XK")."""
    codigo_iso2 = codigo_iso2.upper()
    return "".join(chr(0x1F1E6 + ord(letra) - ord("A")) for letra in codigo_iso2)


def _bandera_subdivision_reino_unido(codigo_5: str) -> str:
    """Emoji-bandera de una nación NO soberana del Reino Unido (Inglaterra,
    Escocia, Gales), que no tiene código ISO 3166-1 propio (las tres
    comparten "GB"): Unicode define en su lugar una secuencia de
    "etiqueta" (`tag sequence`) sobre la bandera negra 🏴 (U+1F3F4) seguida
    del código ISO 3166-2 en minúsculas codificado letra a letra como
    "tag" (U+E0061 = 'a') y cerrada con el carácter de cancelación
    (U+E007F). `codigo_5` es ese código de 5 caracteres: "gbeng", "gbsct"
    o "gbwls". A diferencia de una bandera de país normal, el soporte de
    renderizado de esta secuencia varía según sistema operativo y fuente
    (fuera de Windows, en general, sí se ve bien)."""
    inicio = "\U0001F3F4"
    fin = "\U000E007F"
    cuerpo = "".join(chr(0xE0000 + ord(letra)) for letra in codigo_5.lower())
    return inicio + cuerpo + fin


def emoji_bandera(pais) -> str:
    """Emoji-bandera Unicode de `pais`, a partir del mismo código de
    `PAIS_A_CODIGO` que usa `url_bandera` (o cadena vacía si `pais` es
    NaN, no está catalogado, o está catalogado con `None`). Único caso en
    el que el emoji no puede replicar exactamente lo que sí logra la
    imagen: Irlanda del Norte ("gb-nir") no tiene ninguna secuencia de
    bandera asignada en Unicode (ver docstring del módulo), así que aquí
    cae a la bandera del Reino Unido (GB) como aproximación, mientras que
    su imagen (`url_bandera`) sí es la suya propia."""
    if pd.isna(pais):
        return ""
    codigo = PAIS_A_CODIGO.get(pais)
    if not codigo:
        return ""
    if codigo in ("gb-eng", "gb-sct", "gb-wls"):
        return _bandera_subdivision_reino_unido(codigo.replace("-", ""))
    if codigo == "gb-nir":
        return bandera_desde_iso2("GB")
    return bandera_desde_iso2(codigo)


# ---------------------------------------------------------------------------
# País/territorio (tal cual aparece en `nacionalidad` y en
# `competitions.country_name`) -> código de flagcdn.com para su bandera.
# `None`: no existe ninguna bandera fiable para este valor.
#
# Las claves con nota en `NOTA_PAIS` (más abajo) son los casos que
# `tabla_problemas_nacionalidades` destaca aparte.
# ---------------------------------------------------------------------------
PAIS_A_CODIGO: dict[str, str | None] = {
    "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "Andorra": "AD",
    "Angola": "AO", "Antigua and Barbuda": "AG", "Argentina": "AR", "Armenia": "AM",
    "Aruba": "AW", "Australia": "AU", "Austria": "AT", "Azerbaijan": "AZ",
    "Bahrain": "BH", "Bangladesh": "BD", "Barbados": "BB", "Belarus": "BY",
    "Belgium": "BE", "Benin": "BJ", "Bermuda": "BM", "Bolivia": "BO",
    "Bonaire": "BQ", "Bosnia-Herzegovina": "BA", "Botswana": "BW", "Brazil": "BR",
    "Brunei Darussalam": "BN", "Bulgaria": "BG", "Burkina Faso": "BF", "Burundi": "BI",
    "Cambodia": "KH", "Cameroon": "CM", "Canada": "CA", "Cape Verde": "CV",
    "Central African Republic": "CF", "Chad": "TD", "Chile": "CL", "China": "CN",
    "Chinese Taipei": "TW", "Colombia": "CO", "Comoros": "KM", "Congo": "CG",
    "Costa Rica": "CR", "Cote d'Ivoire": "CI", "Croatia": "HR", "Cuba": "CU",
    "Curacao": "CW", "Cyprus": "CY", "Czech Republic": "CZ", "DR Congo": "CD",
    "Denmark": "DK", "Dominican Republic": "DO", "Ecuador": "EC", "Egypt": "EG",
    "El Salvador": "SV", "England": "gb-eng", "Equatorial Guinea": "GQ", "Eritrea": "ER",
    "Estonia": "EE", "Ethiopia": "ET", "Faroe Islands": "FO", "Fiji": "FJ",
    "Finland": "FI", "France": "FR", "French Guiana": "GF", "Gabon": "GA",
    "Georgia": "GE", "Germany": "DE", "Ghana": "GH", "Gibraltar": "GI",
    "Greece": "GR", "Grenada": "GD", "Guadeloupe": "GP", "Guatemala": "GT",
    "Guinea": "GN", "Guinea-Bissau": "GW", "Guyana": "GY", "Haiti": "HT",
    "Honduras": "HN", "Hongkong": "HK", "Hungary": "HU", "Iceland": "IS",
    "India": "IN", "Indonesia": "ID", "Iran": "IR", "Iraq": "IQ",
    "Ireland": "IE", "Israel": "IL", "Italy": "IT", "Jamaica": "JM",
    "Japan": "JP", "Jordan": "JO", "Kazakhstan": "KZ", "Kenya": "KE",
    "Korea, North": "KP", "Korea, South": "KR", "Kosovo": "XK", "Kyrgyzstan": "KG",
    "Laos": "LA", "Latvia": "LV", "Lebanon": "LB", "Liberia": "LR",
    "Libya": "LY", "Liechtenstein": "LI", "Lithuania": "LT", "Luxembourg": "LU",
    "Macao": "MO", "Madagascar": "MG", "Malawi": "MW", "Malaysia": "MY",
    "Mali": "ML", "Malta": "MT", "Martinique": "MQ", "Mauritania": "MR",
    "Mauritius": "MU", "Mexico": "MX", "Moldova": "MD", "Monaco": "MC",
    "Montenegro": "ME", "Montserrat": "MS", "Morocco": "MA", "Mozambique": "MZ",
    "Myanmar": "MM", "Netherlands": "NL", "Neukaledonien": "NC", "New Caledonia": "NC",
    "New Zealand": "NZ", "Nicaragua": "NI", "Niger": "NE", "Nigeria": "NG",
    "North Macedonia": "MK", "Northern Ireland": "gb-nir", "Norway": "NO", "Oman": "OM",
    "Pakistan": "PK", "Palestine": "PS", "Panama": "PA", "Papua New Guinea": "PG",
    "Paraguay": "PY", "Peru": "PE", "Philippines": "PH", "Poland": "PL",
    "Portugal": "PT", "Puerto Rico": "PR", "Qatar": "QA", "Romania": "RO",
    "Russia": "RU", "Rwanda": "RW", "Réunion": "RE", "Saint-Martin": "MF",
    "San Marino": "SM", "Sao Tome and Principe": "ST", "Saudi Arabia": "SA", "Scotland": "gb-sct",
    "Senegal": "SN", "Serbia": "RS", "Seychelles": "SC", "Sierra Leone": "SL",
    "Singapore": "SG", "Sint Maarten": "SX", "Slovakia": "SK", "Slovenia": "SI",
    "Somalia": "SO", "South Africa": "ZA", "Southern Sudan": "SS", "Spain": "ES",
    "St. Kitts & Nevis": "KN", "St. Lucia": "LC", "Sudan": "SD", "Suriname": "SR",
    "Sweden": "SE", "Switzerland": "CH", "Syria": "SY", "Tahiti": "PF",
    "Tajikistan": "TJ", "Tanzania": "TZ", "Thailand": "TH", "The Gambia": "GM",
    "Togo": "TG", "Trinidad and Tobago": "TT", "Tunisia": "TN", "Turkey": "TR",
    "Turkmenistan": "TM", "Türkiye": "TR", "Uganda": "UG", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ",
    "Vanuatu": "VU", "Venezuela": "VE", "Vietnam": "VN", "Wales": "gb-wls",
    "Yemen": "YE", "Zambia": "ZM", "Zimbabwe": "ZW",
}

# Aviso explicativo SOLO para las claves de `PAIS_A_CODIGO` con algún matiz
# (bandera inexistente, aproximada, no oficial, o país real duplicado con
# otra grafía): son las que `tabla_problemas_nacionalidades` destaca.
NOTA_PAIS: dict[str, str] = {
    "Kosovo": (
        "Usa el código 'XK', no oficial (no es ISO 3166-1) por ser territorio en disputa."
    ),
    "Chinese Taipei": (
        "Taiwan, terriotorio en disputa. En algunos sistemas no se renderiza por motivos políticos."
    ),
    "Tahiti": (
        "Isla de la Polinesia Francesa, sin bandera-emoji propia. Usa el emoji de esta región"
    ),
    "Turkey": "Mismo país que 'Türkiye', con otra grafía (nombre internacional anterior a 2022).",
    "Türkiye": "Mismo país que 'Turkey', con otra grafía (nombre internacional adoptado en 2022).",
    "New Caledonia": "Mismo territorio que 'Neukaledonien', con otra grafía (inglés/alemán).",
    "Neukaledonien": "Mismo territorio que 'New Caledonia', con otra grafía (alemán/inglés).",
}


def url_bandera(pais) -> str:
    """URL de la imagen de bandera de `pais` según `PAIS_A_CODIGO` (ver
    `url_desde_codigo`), o cadena vacía si `pais` es NaN, no está
    catalogado, o está catalogado con `None` (sin ninguna bandera, ni
    siquiera aproximada, asignable)."""
    if pd.isna(pais):
        return ""
    codigo = PAIS_A_CODIGO.get(pais)
    if not codigo:
        return ""
    return url_desde_codigo(codigo)


# ---------------------------------------------------------------------------
# Tabla 1: banderas por liga
# ---------------------------------------------------------------------------
def tabla_banderas_ligas(
    df: pd.DataFrame | None = None, competitions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Una fila por cada nombre de `liga` presente en `tabla_valores_mercado`
    Y país coincidente en `competitions.csv`. Cuando el nombre de la liga
    es ambiguo (más de un `competition_id` con el mismo nombre, p. ej.
    "bundesliga" en Alemania Y Austria — ver docstring del módulo), se
    genera UNA FILA POR PAÍS (en vez de una única fila con los países
    juntos) para que cada una muestre su propia bandera — antes, al
    intentar meter dos banderas en una sola celda de imagen, la imagen se
    dejaba en blanco (una celda de imagen solo admite una); como fila
    aparte, sí puede tener la suya. Todas las filas de una misma liga
    ambigua repiten el mismo `n_registros` (el total de esa liga: no se
    sabe, y no se puede saber, cuántos de esos registros son de cada
    país — ese es precisamente el problema) y se marcan
    `conflicto=True`."""
    if df is None:
        df = _comun.cargar_tabla_final()
    if competitions is None:
        competitions = _comun.cargar_competitions()

    conteo = df["liga"].value_counts()
    filas = []
    for liga, n_registros in conteo.items():
        coincidencias = competitions.loc[competitions["name"] == liga]
        hay_conflicto = len(coincidencias) > 1

        if coincidencias.empty:
            filas.append(
                {
                    "liga": liga,
                    "id_liga": "(sin coincidencia en competitions.csv)",
                    "pais": "—",
                    "bandera_emoji": "",
                    "bandera_url": "",
                    "n_registros": int(n_registros),
                    "conflicto": False,
                }
            )
            continue

        for _, fila_competicion in coincidencias.iterrows():
            pais = fila_competicion["country_name"]
            filas.append(
                {
                    "liga": liga,
                    "id_liga": fila_competicion["competition_id"],
                    "pais": pais if pd.notna(pais) else "—",
                    "bandera_emoji": emoji_bandera(pais),
                    "bandera_url": url_bandera(pais),
                    "n_registros": int(n_registros),
                    "conflicto": hay_conflicto,
                }
            )

    resultado = pd.DataFrame(filas)
    return resultado.sort_values(
        ["conflicto", "liga", "pais"], ascending=[False, True, True]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabla 2: banderas por nacionalidad, y tabla 3: solo las problemáticas
# ---------------------------------------------------------------------------
def tabla_banderas_nacionalidades(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por cada valor de `nacionalidad` presente en
    `tabla_valores_mercado`, con su bandera en emoji (`bandera_emoji`) y en
    imagen (`bandera_url`, si existe una fiable) y si presenta algún
    problema conocido para asignársela: código no oficial, bandera
    aproximada, mismo país real repetido con otra grafía (`NOTA_PAIS`), o,
    si en el futuro apareciera un valor nuevo en el dataset, una
    nacionalidad directamente no catalogada todavía en `PAIS_A_CODIGO`. La
    columna `conflicto` es la que usa `tabla_problemas_nacionalidades`
    para aislar esos casos."""
    if df is None:
        df = _comun.cargar_tabla_final()

    conteo = df["nacionalidad"].value_counts()
    filas = []
    for nacionalidad, n_registros in conteo.items():
        catalogada = nacionalidad in PAIS_A_CODIGO
        motivo = (
            NOTA_PAIS.get(nacionalidad, "")
            if catalogada
            else "Nacionalidad no catalogada todavía en PAIS_A_CODIGO: revisar manualmente."
        )
        filas.append(
            {
                "nacionalidad": nacionalidad,
                "bandera_emoji": emoji_bandera(nacionalidad),
                "bandera_url": url_bandera(nacionalidad),
                "n_registros": int(n_registros),
                "conflicto": bool(motivo),
                "motivo": motivo,
            }
        )

    resultado = pd.DataFrame(filas)
    return resultado.sort_values(
        ["conflicto", "nacionalidad"], ascending=[False, True]
    ).reset_index(drop=True)


def tabla_problemas_nacionalidades(
    tabla_nacionalidades: pd.DataFrame | None = None, df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Subconjunto de `tabla_banderas_nacionalidades` con solo las filas
    marcadas como `conflicto`, para mostrarlas en una tabla aparte sin
    recalcular la tabla completa dos veces."""
    if tabla_nacionalidades is None:
        tabla_nacionalidades = tabla_banderas_nacionalidades(df)
    return (
        tabla_nacionalidades[tabla_nacionalidades["conflicto"]]
        .drop(columns="conflicto")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Tabla 4: clubes SIN liga conocida, aproximando su país por la
# nacionalidad más repetida de sus jugadores (con revisión manual)
# ---------------------------------------------------------------------------
RUTA_REVISIONES_NACIONALIDAD = Path(__file__).parent / "datos" / "revisiones_manuales_liga_nacionalidad.json"


def cargar_revisiones_nacionalidad(ruta: Path = RUTA_REVISIONES_NACIONALIDAD) -> dict[str, str]:
    """Revisiones manuales guardadas en disco de la nacionalidad
    aproximada de `tabla_clubes_sin_liga`: `{equipo: nacionalidad}`.
    Devuelve un diccionario vacío si el fichero todavía no existe (mismo
    patrón que `duplicidad_equipos.cargar_revisiones_manuales`)."""
    if not ruta.exists():
        return {}
    registros = json.loads(ruta.read_text(encoding="utf-8"))
    return {registro["equipo"]: registro["nacionalidad"] for registro in registros}


def guardar_revisiones_nacionalidad(
    revisiones: dict[str, str], ruta: Path = RUTA_REVISIONES_NACIONALIDAD
) -> None:
    """Persiste `revisiones` (mismo formato que devuelve
    `cargar_revisiones_nacionalidad`) en disco, en `ruta`, como JSON
    legible."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    registros = [
        {"equipo": equipo, "nacionalidad": nacionalidad}
        for equipo, nacionalidad in sorted(revisiones.items())
    ]
    ruta.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")


def _nacionalidad_representativa(grupo: pd.DataFrame) -> str | None:
    """Nacionalidad más repetida entre los jugadores DISTINTOS de `grupo`
    (una fila por `id_player`, para que un jugador con muchas filas de
    valoración no pese más que uno con pocas). En caso de EMPATE entre dos
    o más nacionalidades, se desempata tomando la nacionalidad del jugador
    de MENOR valor de mercado (su valor máximo/pico registrado en este
    club) entre todos los jugadores de las nacionalidades empatadas —
    criterio arbitrario pero determinista, a falta de uno mejor. `None` si
    ningún jugador del grupo tiene `nacionalidad` conocida."""
    jugadores = grupo.drop_duplicates("id_player")
    nacionalidades = jugadores["nacionalidad"].dropna()
    if nacionalidades.empty:
        return None

    conteo = nacionalidades.value_counts()
    empatadas = conteo[conteo == conteo.max()].index.tolist()
    if len(empatadas) == 1:
        return empatadas[0]

    candidatos = jugadores[jugadores["nacionalidad"].isin(empatadas)]
    valor_pico = (
        grupo[grupo["id_player"].isin(candidatos["id_player"])]
        .groupby("id_player")["valor_mercado_eur"]
        .max()
    )
    id_player_menor_valor = valor_pico.idxmin()
    return candidatos.set_index("id_player").loc[id_player_menor_valor, "nacionalidad"]


def tabla_clubes_sin_liga(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por cada `equipo` sin `liga` conocida en
    `tabla_valores_mercado`, con el número de filas registradas y una
    APROXIMACIÓN de su país mediante la nacionalidad más repetida de sus
    jugadores (ver `_nacionalidad_representativa`).

    Importante — esto NO recupera la liga exacta ni distingue primera
    división de otras categorías: un filial o un club descendido puede
    tener una nacionalidad mayoritaria perfectamente correcta sin jugar en
    la competición de primera de ese país (ver el análisis de este mismo
    módulo con Racing Santander/FC Barcelona B/Real Madrid Castilla, los
    tres con nacionalidad mayoritaria "Spain" sin jugar en `laliga`). Es
    solo una aproximación de PAÍS pensada para revisarse a mano — de ahí
    `cargar_revisiones_nacionalidad`/`guardar_revisiones_nacionalidad` y
    la tabla de excepciones conocidas, `tabla_clubes_excepcion`."""
    if df is None:
        df = _comun.cargar_tabla_final()

    sin_liga = df[df["liga"].isna()]
    filas = []
    for equipo, grupo in sin_liga.groupby("equipo"):
        nacionalidad = _nacionalidad_representativa(grupo)
        filas.append(
            {
                "equipo": equipo,
                "n_registros": len(grupo),
                "nacionalidad": nacionalidad or "",
                "bandera_emoji": emoji_bandera(nacionalidad) if nacionalidad else "",
                "bandera_url": url_bandera(nacionalidad) if nacionalidad else "",
            }
        )

    resultado = pd.DataFrame(filas)
    return resultado.sort_values("n_registros", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tabla de excepciones conocidas: clubes donde "nacionalidad más repetida
# de la plantilla" NO es un buen proxy de su país/liga real, detectados al
# validar esta idea contra clubes de país YA conocido (comparando contra
# `tabla_valores_mercado.liga` cuando sí existe). Dos motivos distintos:
#
#   - Colisión de nombre: el mismo texto de `equipo` se usa en los datos
#     de origen para DOS clubes reales distintos (p. ej. "Arsenal FC" es a
#     la vez el Arsenal de Londres y el Arsenal de Sarandí, Argentina:
#     varios jugadores de este último aparecen en el dataset con las dos
#     grafías, "Arsenal FC" Y "Arsenal Fútbol Club").
#   - Liga en otro país o mercado de fichajes atípico: el club es de un
#     país pero compite en la competición de OTRO (clubes canadienses en
#     la Major League Soccer de EE.UU., neozelandeses en la A-League de
#     Australia) o recluta jugadores mayoritariamente de un país vecino o
#     con lazo histórico (muy frecuente entre Portugal y Brasil).
#
# Es una tabla CURADA a mano (no calculada), pensada como aviso: si uno de
# estos nombres aparece en `tabla_clubes_sin_liga` con una nacionalidad
# calculada automáticamente, esa nacionalidad es sospechosa.
# ---------------------------------------------------------------------------
CLUBES_EXCEPCION_PAIS_LIGA: dict[str, tuple[str, str]] = {
    "Arsenal FC": (
        "England",
        "Colisión de nombre: el mismo texto también se usa para el Arsenal "
        "de Sarandí (Argentina) — algunos jugadores de este último club "
        "aparecen con las dos grafías, 'Arsenal FC' y 'Arsenal Fútbol Club'.",
    ),
    "Vancouver Whitecaps FC": (
        "United States",
        "Club canadiense que compite en la Major League Soccer, "
        "clasificada como competición de Estados Unidos.",
    ),
    "Toronto FC": (
        "United States",
        "Club canadiense que compite en la Major League Soccer, "
        "clasificada como competición de Estados Unidos.",
    ),
    "Wellington Phoenix": (
        "Australia",
        "Club neozelandés que compite en la A-League, clasificada como "
        "competición de Australia.",
    ),
    "Auckland FC": (
        "Australia",
        "Club neozelandés que compite en la A-League, clasificada como "
        "competición de Australia.",
    ),
    "CD Nacional": (
        "Portugal",
        "Recluta jugadores mayoritariamente de Brasil (lazo histórico "
        "luso-brasileño): su nacionalidad más repetida no es portuguesa.",
    ),
    "CS Marítimo": (
        "Portugal",
        "Recluta jugadores mayoritariamente de Brasil (lazo histórico "
        "luso-brasileño): su nacionalidad más repetida no es portuguesa.",
    ),
    "CD Santa Clara": (
        "Portugal",
        "Recluta jugadores mayoritariamente de Brasil (lazo histórico "
        "luso-brasileño): su nacionalidad más repetida no es portuguesa.",
    ),
    "Portimonense SC": (
        "Portugal",
        "Recluta jugadores mayoritariamente de Brasil (lazo histórico "
        "luso-brasileño): su nacionalidad más repetida no es portuguesa.",
    ),
    "Avs Futebol": (
        "Portugal",
        "Recluta jugadores mayoritariamente de Brasil (lazo histórico "
        "luso-brasileño): su nacionalidad más repetida no es portuguesa.",
    ),
    "FC Alverca": (
        "Portugal",
        "Recluta jugadores mayoritariamente de Brasil (lazo histórico "
        "luso-brasileño): su nacionalidad más repetida no es portuguesa.",
    ),
    "RAAL La Louvière": (
        "Belgium",
        "Club valón que recluta mayoritariamente de Francia (país vecino, "
        "mismo idioma): su nacionalidad más repetida no es belga.",
    ),
    "Houston Dynamo": (
        "United States",
        "Recluta históricamente muchos jugadores hondureños: su "
        "nacionalidad más repetida no es estadounidense.",
    ),
}


def tabla_clubes_excepcion() -> pd.DataFrame:
    """Tabla curada (no calculada) de clubes conocidos donde la
    nacionalidad más repetida de la plantilla NO coincide con su país/liga
    real — ver `CLUBES_EXCEPCION_PAIS_LIGA`."""
    filas = [
        {
            "equipo": equipo,
            "pais_real": pais_real,
            "motivo": motivo,
            "bandera_emoji": emoji_bandera(pais_real),
            "bandera_url": url_bandera(pais_real),
        }
        for equipo, (pais_real, motivo) in CLUBES_EXCEPCION_PAIS_LIGA.items()
    ]
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Limpieza del caso "Arsenal FC" (ver CLUBES_EXCEPCION_PAIS_LIGA): el mismo
# texto de `equipo` se usa en los datos de origen tanto para el Arsenal de
# Londres como para el Arsenal de Sarandí (Argentina) — comprobado con el
# historial de varios jugadores (p. ej. Iván Marcone, que aparece en el
# dataset con las dos grafías del club argentino, "Arsenal FC" Y "Arsenal
# Fútbol Club"). Como el resto de su carrera SÍ distingue bien Inglaterra
# de Argentina (via `liga`), se puede usar como pista para separar los dos
# grupos de jugadores mezclados bajo "Arsenal FC".
# ---------------------------------------------------------------------------
def tabla_jugadores_arsenal_fc(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por cada jugador DISTINTO con al menos un registro de
    `equipo == "Arsenal FC"`, con su nombre y, contando SOLO en sus OTROS
    clubes (nunca en las propias filas de "Arsenal FC"), el número de
    registros en equipos ingleses (`liga == "premier-league"`) y en
    equipos argentinos (`liga == "torneo-apertura"`).

    Importante por qué se excluyen las propias filas de "Arsenal FC": como
    ese club siempre tiene `liga == "premier-league"` (por construcción,
    ver docstring del módulo), contarlas habría sido circular — CUALQUIER
    jugador con una sola fila de "Arsenal FC" habría sumado
    automáticamente esa fila a su cuenta de "premier-league" sin aportar
    ninguna evidencia real de haber jugado en Inglaterra. Se comprobó con
    Iván Marcone: antes de esta exclusión salía con 11 registros de
    'premier-league' pese a que el resto de su carrera (Trapani, Vicenza,
    Südtirol, Lanús, Boca Juniors, Independiente, Cruz Azul, Elche) no
    tiene ninguno — los 11 eran, precisamente, sus propias filas de
    "Arsenal FC" (el club argentino, a juzgar por su trayectoria).

    Un jugador con registros de 'torneo-apertura' en otros clubes y
    ninguno de 'premier-league' es, con toda probabilidad, del Arsenal
    argentino colado bajo el mismo texto que el inglés — y viceversa.
    Comprobado con los datos reales: ningún jugador tiene señal de los DOS
    lados a la vez (0 casos), así que la columna `clasificacion` puede
    asignarse sin ambigüedad como una de `CLASIFICACIONES_ARSENAL_FC`
    (jugador cuyos otros clubes no están en ninguna de las dos ligas
    rastreadas, p. ej. clubes argentinos que no cruzan con
    `competitions.csv`, como le pasa a Iván Marcone con Boca
    Juniors/Independiente/Lanús, cae en "Sin evidencia"). Se incluye
    también `id_player` (clave para las revisiones manuales, ver
    `cargar_revisiones_arsenal_fc`) y `nacionalidad`. Ordenada para que
    los casos más claramente argentinos aparezcan primero."""
    if df is None:
        df = _comun.cargar_tabla_final()

    jugadores_arsenal = df.loc[df["equipo"] == "Arsenal FC", "id_player"].unique()
    datos = df[df["id_player"].isin(jugadores_arsenal)]
    datos_otros_clubes = datos[datos["equipo"] != "Arsenal FC"]

    datos_por_jugador = datos.drop_duplicates("id_player").set_index("id_player")
    n_premier = datos_otros_clubes.loc[datos_otros_clubes["liga"] == "premier-league"].groupby("id_player").size()
    n_apertura = datos_otros_clubes.loc[datos_otros_clubes["liga"] == "torneo-apertura"].groupby("id_player").size()

    resultado = pd.DataFrame(
        {"nombre": datos_por_jugador["nombre"], "nacionalidad": datos_por_jugador["nacionalidad"]}
    )
    resultado["n_registros_premier_league"] = n_premier.reindex(resultado.index, fill_value=0).astype(int)
    resultado["n_registros_torneo_apertura"] = n_apertura.reindex(resultado.index, fill_value=0).astype(int)

    resultado["clasificacion"] = "Sin evidencia"
    resultado.loc[resultado["n_registros_torneo_apertura"] > 0, "clasificacion"] = "Argentina (Sarandí)"
    resultado.loc[resultado["n_registros_premier_league"] > 0, "clasificacion"] = "Inglaterra (Londres)"

    resultado = resultado.reset_index().rename(columns={"index": "id_player"})
    return resultado.sort_values(
        ["n_registros_torneo_apertura", "n_registros_premier_league"], ascending=[False, False]
    ).reset_index(drop=True)


CLASIFICACIONES_ARSENAL_FC: tuple[str, ...] = ("Argentina (Sarandí)", "Inglaterra (Londres)", "Sin evidencia")

RUTA_REVISIONES_ARSENAL_FC = Path(__file__).parent / "datos" / "revisiones_manuales_arsenal_fc.json"


def cargar_revisiones_arsenal_fc(ruta: Path = RUTA_REVISIONES_ARSENAL_FC) -> dict[int, str]:
    """Revisiones manuales guardadas en disco de la `clasificacion` de
    `tabla_jugadores_arsenal_fc`: `{id_player: clasificacion}` (mismo
    patrón que `cargar_revisiones_nacionalidad`, pero con `id_player`
    como clave en vez de `equipo`, porque aquí lo que se corrige es por
    JUGADOR, no por club). Devuelve un diccionario vacío si el fichero
    todavía no existe."""
    if not ruta.exists():
        return {}
    registros = json.loads(ruta.read_text(encoding="utf-8"))
    return {int(registro["id_player"]): registro["clasificacion"] for registro in registros}


def guardar_revisiones_arsenal_fc(
    revisiones: dict[int, str], ruta: Path = RUTA_REVISIONES_ARSENAL_FC
) -> None:
    """Persiste `revisiones` (mismo formato que devuelve
    `cargar_revisiones_arsenal_fc`) en disco, en `ruta`, como JSON
    legible."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    registros = [
        {"id_player": int(id_player), "clasificacion": clasificacion}
        for id_player, clasificacion in sorted(revisiones.items())
    ]
    ruta.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    tabla_ligas = tabla_banderas_ligas()
    print("=== Banderas por liga ===")
    print(tabla_ligas.to_string(index=False))
    print(f"\nFilas en conflicto: {int(tabla_ligas['conflicto'].sum())} de {len(tabla_ligas)}")

    tabla_nacionalidades = tabla_banderas_nacionalidades()
    print("\n=== Banderas por nacionalidad ===")
    print(tabla_nacionalidades.to_string(index=False))

    problemas = tabla_problemas_nacionalidades(tabla_nacionalidades)
    print(f"\n=== Nacionalidades con posible problema ({len(problemas)} de {len(tabla_nacionalidades)}) ===")
    print(problemas.to_string(index=False))

    tabla_sin_liga = tabla_clubes_sin_liga()
    print(f"\n=== Clubes sin liga, con nacionalidad aproximada ({len(tabla_sin_liga)}) ===")
    print(tabla_sin_liga.head(30).to_string(index=False))

    print("\n=== Clubes donde esta aproximación es conocida por ser errónea ===")
    print(tabla_clubes_excepcion().to_string(index=False))

    tabla_arsenal = tabla_jugadores_arsenal_fc()
    print(f"\n=== Jugadores de 'Arsenal FC' (Londres + Sarandí mezclados, {len(tabla_arsenal)}) ===")
    print(tabla_arsenal.to_string(index=False))
