"""
Duplicidad de nombres de equipo y relaciones de cantera: dos problemas que
comparten la misma causa raíz (el nombre de club es texto libre, no un
id) y se detectan con el mismo mecanismo, así que se tratan juntos aquí.

  1. "Posible duplicidad": dos cadenas de texto distintas de `equipo` que
     en realidad son el MISMO club real (variante de grafía, abreviatura,
     transliteración...), p. ej. "Manchester United" / "Man Utd".
  2. "Posible filial": dos cadenas de texto de `equipo` que corresponden a
     dos clubes JERÁRQUICAMENTE relacionados del mismo club (primer
     equipo y su segundo equipo/cantera), p. ej. "Real Betis Balompié" /
     "Betis Deportivo Balompié".

Enfoque (a partir de aquí, todo por comparación ENTRE nombres del propio
dataset, no contra `clubs.csv`; el cruce con `clubs.csv` se usa solo como
una SEÑAL más, ver `detectar_relaciones_equipos`):

  1. `frecuencia_palabras_equipo`: qué palabras se repiten más en `equipo`
     (para decidir manualmente qué agrupar en `_comun.DICCIONARIO_
     FORMAS_JURIDICAS`).
  2. `generar_pares_candidatos`: se generan pares (A, B), con A SIEMPRE el
     de mayor valor de mercado, por dos vías — nombre (comparten palabra
     de bloqueo y superan un umbral de `SequenceMatcher`) y traspasos
     (la mayoría de las salidas de B van hacia A o su red) — con una
     exclusión: nunca se compara un par donde AMBOS nombres contengan ya
     una palabra de `_comun.DICCIONARIO_FILIALES` (sería comparar dos
     canteras entre sí, redundante).
  3. Cada par se clasifica en "posible filial", "posible duplicidad" o
     "ninguna relación": primero se comprueban reglas ESTRICTAS (100 %) de
     los tres grupos — si más de un grupo las cumple a la vez, se marca
     como conflicto en la columna `conflicto`; si ninguna regla estricta
     aplica, se calcula una probabilidad CONTINUA (0-1) por fórmula
     explícita para cada grupo (ver `detectar_relaciones_equipos`) y se
     usa la de mayor probabilidad.
  4. `resolver_redes_relaciones` agrupa en componentes conexas TANTO los
     pares "posible filial" COMO "posible duplicidad" (una duplicidad de
     A puede tener, a su vez, sus propios filiales) y marca como "líder"
     al de mayor valor de mercado de plantilla.

Aparte de este algoritmo, el módulo conserva `detectar_posibles_
duplicados` (comparación CONTRA `clubs.csv`): no lo usa ya la pestaña de
"Limpieza de datos" (sustituido por `detectar_relaciones_equipos`), pero
sigue siendo la base de `valores_faltantes.evaluar_recuperacion_liga`
(cuántas filas sin `liga` son recuperables porque su club SÍ está en
`clubs.csv` con otra grafía) — un problema distinto (recuperar un campo
del propio cruce con `clubs.csv`) que no encaja en la comparación
dataset-contra-dataset de más arriba.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _comun

# ---------------------------------------------------------------------------
# Revisiones manuales (correcciones del usuario a la tabla de la app,
# ver `app.py`): persistidas en disco (NO en `st.session_state`, que se
# pierde al recargar la página/cerrar el navegador) para que sobrevivan a
# una actualización de la pantalla o un reinicio del servidor de
# Streamlit. Es trabajo humano, no un dato regenerable del pipeline de
# `ingest/` (por eso vive en `limpieza/datos/`, fuera de `ingest/datos/`,
# que sí está en `.gitignore` por ser regenerable).
# ---------------------------------------------------------------------------
# Revisiones de los pares calculados con `traspasos.csv`. (El antiguo
# `revisiones_manuales_duplicidad.json` corresponde a la tabla
# `movimientos.csv`, ya descartada, y no se usa.)
RUTA_REVISIONES_MANUALES = Path(__file__).parent / "datos" / "revisiones_manuales_duplicidad_traspasos.json"


def clave_par(club_1: str, club_2: str) -> tuple[str, str]:
    """Clave de un par de clubes INDEPENDIENTE del orden: las revisiones
    manuales se guardan y se buscan con ella. El orden A/B de un par
    candidato depende del valor de mercado de cada club (A = mayor), que
    puede cambiar al actualizar los datos; con una clave ordenada una
    revisión sigue encontrando su par aunque A y B se hayan invertido."""
    return tuple(sorted((club_1, club_2)))  # type: ignore[return-value]


def cargar_revisiones_manuales(ruta: Path = RUTA_REVISIONES_MANUALES) -> dict[tuple[str, str], str]:
    """Revisiones manuales guardadas en disco: `{clave_par(club_a, club_b):
    tipo}`. Devuelve un diccionario vacío si el fichero todavía no existe
    (no se ha hecho ninguna revisión manual todavía)."""
    if not ruta.exists():
        return {}
    registros = json.loads(ruta.read_text(encoding="utf-8"))
    return {clave_par(registro["club_a"], registro["club_b"]): registro["tipo"] for registro in registros}


def guardar_revisiones_manuales(
    revisiones: dict[tuple[str, str], str], ruta: Path = RUTA_REVISIONES_MANUALES
) -> None:
    """Persiste `revisiones` (mismo formato que devuelve
    `cargar_revisiones_manuales`) en disco, en `ruta`, como JSON legible
    (para poder revisarlo/versionarlo a mano si hace falta)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    registros = [
        {"club_a": club_a, "club_b": club_b, "tipo": tipo}
        for (club_a, club_b), tipo in sorted(
            (clave_par(*clave), tipo) for clave, tipo in revisiones.items()
        )
    ]
    ruta.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Variantes de grafía CONTRA `clubs.csv` (usado por
#    `valores_faltantes.evaluar_recuperacion_liga`; ver docstring del módulo)
# ---------------------------------------------------------------------------
def _indice_por_palabra(
    nombres: list[str], longitud_minima_palabra: int = 4, excluir_genericas: bool = False
) -> dict[str, set[str]]:
    """Índice invertido palabra-núcleo -> nombres que la contienen, para no
    tener que comparar cada nombre contra TODOS los demás (con miles de
    nombres de equipo, el producto cartesiano completo es demasiado
    lento): dos nombres solo se comparan si comparten al menos una palabra
    significativa.

    `excluir_genericas=True` (usado en la comparación dataset-contra-
    dataset de `detectar_relaciones_equipos`, NO en el cruce contra
    `clubs.csv` de la sección 1) descarta además las palabras de
    `_comun.PALABRAS_NUCLEO_GENERICAS` como clave de bloqueo: sin esto,
    dos clubes SIN relación alguna que comparten una palabra genérica del
    vocabulario futbolístico (p. ej. "Balompié", que solo significa
    "fútbol" y usan varios clubes andaluces sin relación entre sí, como
    "Albacete Balompié" y "Real Betis Balompié") se convertían en un par
    candidato y, si el resto de señales daban falsos positivos (un club
    ausente de `clubs.csv` + traspasos domésticos con nacionalidad
    compartida, algo esperable entre dos clubes cualquiera del mismo
    país), acababan puntuando alto como filial sin serlo — caso detectado
    validando el módulo contra ese par."""
    indice: dict[str, set[str]] = {}
    for nombre in nombres:
        for palabra in set(_comun.palabras_nucleo(nombre)):
            if len(palabra) < longitud_minima_palabra:
                continue
            if excluir_genericas and palabra in _comun.PALABRAS_NUCLEO_GENERICAS:
                continue
            indice.setdefault(palabra, set()).add(nombre)
    return indice


def _mejor_coincidencia_bloqueada(nombre: str, indice_clubs: dict[str, set[str]]) -> tuple[str, float]:
    """Nombre de club (original, tal cual en `clubs.csv`) más parecido a
    `nombre`, restringido a los clubes que comparten alguna palabra núcleo
    con él (ver `_indice_por_palabra`) en vez de comparar contra los 796
    clubes de `clubs.csv` uno a uno. Si ninguna palabra de `nombre`
    aparece en el índice, no hay candidato razonable y se descarta."""
    candidatos: set[str] = set()
    for palabra in set(_comun.palabras_nucleo(nombre)):
        candidatos |= indice_clubs.get(palabra, set())
    if not candidatos:
        return "", 0.0

    norm = _comun.normalizar_texto(nombre)
    mejor_nombre, mejor_ratio = "", 0.0
    for candidato in candidatos:
        ratio = difflib.SequenceMatcher(None, norm, _comun.normalizar_texto(candidato)).ratio()
        if ratio > mejor_ratio:
            mejor_nombre, mejor_ratio = candidato, ratio
    return mejor_nombre, mejor_ratio


def detectar_posibles_duplicados(
    df: pd.DataFrame | None = None,
    clubs: pd.DataFrame | None = None,
    frecuencia_minima: int = 10,
    similitud_minima: float = 0.90,
) -> pd.DataFrame:
    """Para cada nombre de `equipo` que NO cruza de forma exacta con
    `clubs.csv`, busca su coincidencia más parecida dentro de `clubs.csv` y
    la reporta como candidata a "misma club, distinta grafía" si supera
    `similitud_minima` y no es, en realidad, una relación de cantera.

    Uso actual: exclusivamente `valores_faltantes.evaluar_recuperacion_
    liga` (estimar cuántas filas sin `liga` son recuperables). La pestaña
    de "Limpieza de datos" usa en su lugar `detectar_relaciones_equipos`
    (comparación entre nombres del propio dataset, no contra `clubs.csv`).
    """
    if df is None:
        df = _comun.cargar_tabla_final()
    if clubs is None:
        clubs = _comun.cargar_clubs()

    normalizados_clubs = set(_comun.normalizar_nombre_club(clubs["name"]))
    indice_clubs = _indice_por_palabra(clubs["name"].tolist())

    conteo = df["equipo"].value_counts()
    candidatos = conteo[conteo >= frecuencia_minima]
    candidatos = candidatos[~_comun.normalizar_nombre_club(pd.Series(candidatos.index)).isin(normalizados_clubs).values]

    filas = []
    for nombre, n_filas in candidatos.items():
        nombre_clubs_csv, similitud = _mejor_coincidencia_bloqueada(nombre, indice_clubs)
        if similitud < similitud_minima:
            continue
        tipo = (
            "filial (no es duplicado, es relación de cantera)"
            if _comun.es_relacion_filial(nombre, nombre_clubs_csv)
            else "posible variante de club en clubs.csv"
        )
        filas.append(
            {
                "nombre_en_datos": nombre,
                "club_en_clubs_csv": nombre_clubs_csv,
                "similitud": round(similitud, 3),
                "n_filas": int(n_filas),
                "tipo": tipo,
            }
        )

    resultado = pd.DataFrame(
        filas, columns=["nombre_en_datos", "club_en_clubs_csv", "similitud", "n_filas", "tipo"]
    )
    return resultado.sort_values(["tipo", "n_filas"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Palabras más repetidas en `equipo` (para diseñar los diccionarios de
#    `_comun.py`)
# ---------------------------------------------------------------------------
def frecuencia_palabras_equipo(df: pd.DataFrame | None = None, top_n: int = 40) -> pd.DataFrame:
    """Palabras (normalizadas, SIN quitar siglas de forma jurídica: al
    contrario que `_comun.palabras_nucleo`, aquí interesa precisamente ver
    lo frecuentes que son "fc"/"united"/"b"/"ii"...) más repetidas entre
    los nombres de `equipo` DISTINTOS del dataset, con el nº de clubes
    (no de filas: un club con miles de valoraciones no debe pesar más que
    uno con pocas) que la contienen. Es el punto de partida para decidir
    qué palabras agrupar en `_comun.DICCIONARIO_FILIALES` / `_comun.
    DICCIONARIO_FORMAS_JURIDICAS`."""
    if df is None:
        df = _comun.cargar_tabla_final()

    contador: dict[str, int] = {}
    for nombre in df["equipo"].dropna().unique():
        texto = _comun.normalizar_texto(nombre)
        if not texto:
            continue
        for palabra in set(texto.split()):
            contador[palabra] = contador.get(palabra, 0) + 1

    resultado = pd.DataFrame(
        {"palabra": list(contador.keys()), "n_clubes": list(contador.values())}
    )
    return resultado.sort_values(["n_clubes", "palabra"], ascending=[False, True]).head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Datos precalculados UNA VEZ para todos los pares candidatos
# ---------------------------------------------------------------------------
def valor_mercado_por_club(df: pd.DataFrame | None = None) -> pd.Series:
    """Valor de mercado "actual" aproximado de cada `equipo`: suma, para
    cada jugador que ese equipo tuvo alguna vez, de su ÚLTIMO valor de
    mercado conocido MIENTRAS estaba en ese equipo (una fila por jugador y
    equipo). Se usa tanto para ORDENAR cada par candidato (`club_a` =
    mayor valor, `club_b` = menor) como en la fórmula de probabilidad de
    FILIAL/DUPLICIDAD y para elegir el "líder" de cada red de relaciones."""
    if df is None:
        df = _comun.cargar_tabla_final()

    datos = df.dropna(subset=["equipo", "valor_mercado_eur"]).sort_values("fecha")
    ultimo_por_jugador_equipo = datos.groupby(["equipo", "id_player"], sort=False).tail(1)
    return ultimo_por_jugador_equipo.groupby("equipo")["valor_mercado_eur"].sum()


def _registro_en_clubs_csv(nombres: list[str], clubs: pd.DataFrame) -> dict[str, tuple[object, float]]:
    """Para cada nombre de `nombres`, su mejor coincidencia dentro de
    `clubs.csv`: `(club_id o None, similitud)`. Usa el mismo bloqueo por
    palabra núcleo que `_mejor_coincidencia_bloqueada` para no comparar
    cada nombre contra los 796 clubes uno a uno.

    Es la base de la regla estricta de NINGUNA RELACIÓN: dos nombres que
    casan cada uno al 100 % con un `club_id` DISTINTO de `clubs.csv` son
    dos clubes reales ya identificados sin ambigüedad — los filiales, al
    no tener ficha propia en `clubs.csv`, nunca alcanzan una similitud
    cercana al 100 % con él.
    """
    clubs_validos = clubs.dropna(subset=["name"])
    indice_clubs = _indice_por_palabra(clubs_validos["name"].tolist(), excluir_genericas=True)

    clubs_norm = clubs_validos[["club_id", "name"]].copy()
    clubs_norm["norm"] = _comun.normalizar_nombre_club(clubs_norm["name"])
    clubs_norm = clubs_norm.drop_duplicates(subset="norm")
    id_por_norm = clubs_norm.set_index("norm")["club_id"]

    resultado: dict[str, tuple[object, float]] = {}
    for nombre in nombres:
        nombre_match, ratio = _mejor_coincidencia_bloqueada(nombre, indice_clubs)
        club_id = id_por_norm.get(_comun.normalizar_texto(nombre_match)) if nombre_match else None
        resultado[nombre] = (club_id, ratio)
    return resultado


def _indice_traspasos(traspasos: pd.DataFrame) -> tuple[dict[tuple[str, str], int], dict[str, dict[str, int]], dict[str, int]]:
    """A partir de `traspasos.csv` (la tabla de la pestaña "Red de
    traspasos"), precalcula: `flujo` (nº de traspasos por par DIRIGIDO
    origen->destino), `flujo_por_origen` (lo mismo, pero indexado por
    origen -> {destino: nº}, para sumar rápido "traspasos de B hacia toda
    una red de clubes") y `salidas_totales` (nº total de traspasos de
    SALIDA de cada equipo)."""
    datos = traspasos.dropna(subset=["equipo_origen", "equipo_destino", "id_player"])
    agrupado = datos.groupby(["equipo_origen", "equipo_destino"]).size()

    flujo: dict[tuple[str, str], int] = {}
    flujo_por_origen: dict[str, dict[str, int]] = {}
    for (origen, destino), cantidad in agrupado.items():
        flujo[(origen, destino)] = int(cantidad)
        flujo_por_origen.setdefault(origen, {})[destino] = int(cantidad)

    salidas_totales = datos.groupby("equipo_origen").size().to_dict()
    return flujo, flujo_por_origen, salidas_totales


def _nacionalidad_por_jugador(df: pd.DataFrame) -> pd.Series:
    """Nacionalidad de cada `id_player` (primer valor no nulo; es estática
    a lo largo de la carrera del jugador)."""
    return (
        df.dropna(subset=["nacionalidad"])
        .drop_duplicates(subset="id_player")
        .set_index("id_player")["nacionalidad"]
    )


def _nacionalidad_dominante_por_club(
    jugadores_por_club: dict[str, set], nacionalidad_por_jugador: pd.Series
) -> dict[str, tuple[str, int]]:
    """Nacionalidad más repetida entre los jugadores de cada club, junto
    con su nº de apariciones: `(nacionalidad, n_jugadores)`."""
    resultado: dict[str, tuple[str, int]] = {}
    for club, jugadores in jugadores_por_club.items():
        nacs = nacionalidad_por_jugador.reindex(list(jugadores)).dropna()
        if nacs.empty:
            continue
        conteo = nacs.value_counts()
        resultado[club] = (conteo.idxmax(), int(conteo.iloc[0]))
    return resultado


def _liga_dominante_por_club(df: pd.DataFrame) -> dict[str, str]:
    """Liga más habitual (moda) de cada nombre de `equipo`."""
    datos = df.dropna(subset=["equipo", "liga"])
    return (
        datos.groupby("equipo")["liga"]
        .agg(lambda serie: serie.mode().iat[0] if not serie.mode().empty else None)
        .dropna()
        .to_dict()
    )


class _DatosPrecalculados:
    """Contenedor de todo lo que `detectar_relaciones_equipos` calcula UNA
    VEZ y reutiliza en el bloqueo, las reglas estrictas y las fórmulas de
    probabilidad."""

    def __init__(self, df: pd.DataFrame, clubs: pd.DataFrame, traspasos: pd.DataFrame):
        self.conteo = df["equipo"].value_counts()
        self.nombres = self.conteo.index.tolist()
        self.tokens = {nombre: _comun.tokenizar_nombre(nombre) for nombre in self.nombres}
        self.tokens_set = {nombre: set(tokens) for nombre, tokens in self.tokens.items()}

        self.valor_por_club = valor_mercado_por_club(df)
        self.registro = _registro_en_clubs_csv(self.nombres, clubs)
        self.flujo, self.flujo_por_origen, self.salidas_totales = _indice_traspasos(traspasos)
        self.liga_dominante = _liga_dominante_por_club(df)

        self.jugadores_por_club = (
            df.dropna(subset=["equipo", "id_player"]).groupby("equipo")["id_player"].apply(set).to_dict()
        )
        self.nacionalidad_por_jugador = _nacionalidad_por_jugador(df)
        self.nacionalidad_dominante = _nacionalidad_dominante_por_club(
            self.jugadores_por_club, self.nacionalidad_por_jugador
        )

    def valor(self, nombre: str) -> float:
        return float(self.valor_por_club.get(nombre, 0.0))

    def ordenar_por_valor(self, nombre_1: str, nombre_2: str) -> tuple[str, str]:
        v1, v2 = self.valor(nombre_1), self.valor(nombre_2)
        if v1 > v2 or (v1 == v2 and nombre_1 < nombre_2):
            return nombre_1, nombre_2
        return nombre_2, nombre_1


# ---------------------------------------------------------------------------
# 4. Pares candidatos: por nombre y por patrón de traspasos
# ---------------------------------------------------------------------------
UMBRAL_SIMILITUD_ALTO = 0.70
UMBRAL_SIMILITUD_BAJO = 0.50
UMBRAL_TAM_MUESTRA_TRASPASOS = 5
UMBRAL_PCT_TRASPASOS_CANDIDATO = 0.5


def _contiene_palabra_filial(tokens: set[str]) -> bool:
    return bool(tokens & _comun.DICCIONARIO_FILIALES)


def _cumple_condiciones_de_nombre(ratio: float, filial_a: bool, filial_b: bool) -> bool:
    """Condiciones 1 y 2 del enunciado: `ratio` (SequenceMatcher sobre el
    nombre completo normalizado) >= 70 %, o >= 50 % si además uno de los
    dos nombres contiene una palabra de `_comun.DICCIONARIO_FILIALES`
    (para no perder pares como "Ajax" / "Jong Ajax", cuyo SequenceMatcher
    puede rondar el 55-65 % pero cuya diferencia es limpiamente un sufijo
    de cantera)."""
    if ratio >= UMBRAL_SIMILITUD_ALTO:
        return True
    if ratio >= UMBRAL_SIMILITUD_BAJO and (filial_a or filial_b):
        return True
    return False


def _indice_bloqueo(datos: _DatosPrecalculados, longitud_minima_palabra: int) -> dict[str, set[str]]:
    """Índice palabra/token de bloqueo -> nombres que lo contienen: NUNCA
    se usa como clave de bloqueo una palabra de `_comun.DICCIONARIO_
    FILIALES` (comparar dos equipos solo porque ambos tienen "B" o "U19"
    en el nombre no aporta nada útil, hay cientos de "equipo B" sin
    relación entre sí) ni de `_comun.PALABRAS_NUCLEO_GENERICAS` (palabras
    del vocabulario futbolístico compartidas por decenas de clubes SIN
    relación, p. ej. "Town"/"City"/"Belediyespor" — comprobado con datos
    reales: sin esta exclusión, "Real Betis Balompié" y "Albacete
    Balompié", dos clubes españoles sin ninguna relación, se acaban
    emparejando y la fórmula los clasifica como "posible filial" al
    43,87 %, porque comparten liga/nacionalidad por pura coincidencia
    geográfica, no por relación real) — SALVO las de `_comun.
    PALABRAS_NUCLEO_GENERICAS_PERMITIDAS`, la lista blanca DENTRO de esa
    lista de palabras genéricas ya comprobada caso por caso (ver su
    docstring en `_comun.py`): esas sí se dejan usar como clave de
    bloqueo, porque no generan ese tipo de falso positivo."""
    indice: dict[str, set[str]] = {}
    for nombre in datos.nombres:
        for token in datos.tokens_set[nombre]:
            if len(token) < longitud_minima_palabra:
                continue
            if token in _comun.DICCIONARIO_FILIALES:
                continue
            if token in _comun.PALABRAS_NUCLEO_GENERICAS and token not in _comun.PALABRAS_NUCLEO_GENERICAS_PERMITIDAS:
                continue
            indice.setdefault(token, set()).add(nombre)
    return indice


# Límite de tamaño de bloque específico para las palabras de la lista
# blanca (`_comun.PALABRAS_NUCLEO_GENERICAS_PERMITIDAS`), independiente de
# `tamano_maximo_bloque`: son palabras ya vetadas caso por caso como
# seguras, así que se les da algo más de margen que a una palabra
# cualquiera (p. ej. "sporting" tiene 34 clubes, por encima del límite
# general de 30) sin por ello aflojar el límite general, que sigue siendo
# la salvaguarda contra bloques enormes para el resto de palabras.
TAMANO_MAXIMO_BLOQUE_GENERICAS_PERMITIDAS = 40


def generar_pares_por_nombre(
    datos: _DatosPrecalculados, longitud_minima_palabra: int = 4, tamano_maximo_bloque: int = 30
) -> dict[tuple[str, str], float]:
    """Pares candidatos por similitud de NOMBRE (condiciones 1 y 2 del
    enunciado), ya ordenados por valor de mercado (`club_a` = mayor
    valor). Devuelve un diccionario `(club_a, club_b) -> ratio` (el ratio
    de `SequenceMatcher`, útil más adelante para la fórmula de
    DUPLICIDAD).

    Aplica también la exclusión pedida: NUNCA se genera un par donde
    AMBOS nombres contengan ya una palabra de `_comun.DICCIONARIO_
    FILIALES` (comparar dos canteras entre sí es redundante — la relación
    relevante es cada una con su primer equipo, no entre ellas)."""
    indice = _indice_bloqueo(datos, longitud_minima_palabra)
    pares: dict[tuple[str, str], float] = {}
    vistos: set[tuple[str, str]] = set()

    for token, grupo in indice.items():
        cap = (
            TAMANO_MAXIMO_BLOQUE_GENERICAS_PERMITIDAS
            if token in _comun.PALABRAS_NUCLEO_GENERICAS_PERMITIDAS
            else tamano_maximo_bloque
        )
        if not (1 < len(grupo) <= cap):
            continue
        ordenado = sorted(grupo)
        for i, nombre_1 in enumerate(ordenado):
            for nombre_2 in ordenado[i + 1 :]:
                clave_vista = (nombre_1, nombre_2)
                if clave_vista in vistos:
                    continue
                vistos.add(clave_vista)

                filial_1 = _contiene_palabra_filial(datos.tokens_set[nombre_1])
                filial_2 = _contiene_palabra_filial(datos.tokens_set[nombre_2])
                if filial_1 and filial_2:
                    continue  # exclusión: nunca comparar dos "filiales" entre sí

                ratio = difflib.SequenceMatcher(
                    None,
                    _comun.normalizar_texto(nombre_1) or "",
                    _comun.normalizar_texto(nombre_2) or "",
                ).ratio()
                if not _cumple_condiciones_de_nombre(ratio, filial_1, filial_2):
                    continue

                club_a, club_b = datos.ordenar_por_valor(nombre_1, nombre_2)
                pares[(club_a, club_b)] = ratio
    return pares


def _construir_red_preliminar(pares_filial_o_duplicidad: list[tuple[str, str]]) -> dict[str, frozenset[str]]:
    """Componentes conexas (unión-búsqueda) de los pares ya identificados
    como "posible filial" o "posible duplicidad" SOLO por patrón de
    nombre (antes de generar los pares por traspasos, que necesitan saber
    qué clubes forman ya una "familia" para poder agregar los traspasos
    de TODOS sus miembros — ver `generar_pares_por_traspasos`). Devuelve
    un diccionario club -> conjunto de todos los clubes de su red
    (incluido él mismo)."""
    padre: dict[str, str] = {}

    def _raiz(nombre: str) -> str:
        padre.setdefault(nombre, nombre)
        while padre[nombre] != nombre:
            padre[nombre] = padre[padre[nombre]]
            nombre = padre[nombre]
        return nombre

    for club_a, club_b in pares_filial_o_duplicidad:
        raiz_a, raiz_b = _raiz(club_a), _raiz(club_b)
        if raiz_a != raiz_b:
            padre[raiz_b] = raiz_a

    grupos: dict[str, set[str]] = {}
    for nombre in padre:
        grupos.setdefault(_raiz(nombre), set()).add(nombre)

    red_por_club: dict[str, frozenset[str]] = {}
    for miembros in grupos.values():
        congelado = frozenset(miembros)
        for miembro in miembros:
            red_por_club[miembro] = congelado
    return red_por_club


def generar_pares_por_traspasos(
    datos: _DatosPrecalculados,
    red_preliminar: dict[str, frozenset[str]],
    pares_existentes: set[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    """Condición 3 del enunciado: para cada club B con al menos `UMBRAL_
    TAM_MUESTRA_TRASPASOS` traspasos de salida registrados, si al menos el
    `UMBRAL_PCT_TRASPASOS_CANDIDATO` (50 %) de esos traspasos van hacia UN
    MISMO club A o hacia la red de A ya confirmada por nombre (`red_
    preliminar`, ver `_construir_red_preliminar`), se genera el par
    candidato (A, B) — esto captura relaciones SIN ningún parecido de
    nombre, apoyándose solo en el patrón de traspasos (`ingest/datos/
    traspasos.csv`, la tabla de la pestaña "Red de traspasos").

    El representante "A" de una red multi-club es el de mayor valor de
    mercado (su "líder" natural). Aplica la misma exclusión que
    `generar_pares_por_nombre` (nunca ambos con palabra de filiales) y no
    repite pares ya generados por nombre. Devuelve `(club_a, club_b) ->
    fracción de traspasos hacia la red de A` (para la fórmula de FILIAL)."""
    nuevos: dict[tuple[str, str], float] = {}
    for club_b, total in datos.salidas_totales.items():
        if total < UMBRAL_TAM_MUESTRA_TRASPASOS or club_b not in datos.tokens_set:
            continue  # comprobación defensiva: todo equipo de `traspasos.csv` sale de `equipo`, pero la tabla podría haberse generado con otra versión de los datos
        destinos = datos.flujo_por_origen.get(club_b, {})
        if not destinos:
            continue

        suma_por_red: dict[frozenset[str], int] = {}
        for destino, cantidad in destinos.items():
            if destino not in datos.tokens_set:
                continue
            red = red_preliminar.get(destino, frozenset({destino}))
            suma_por_red[red] = suma_por_red.get(red, 0) + cantidad

        for red, cantidad in suma_por_red.items():
            if club_b in red:
                continue
            fraccion = cantidad / total
            if fraccion < UMBRAL_PCT_TRASPASOS_CANDIDATO:
                continue

            representante = max(red, key=datos.valor)
            club_a, club_b_ordenado = datos.ordenar_por_valor(representante, club_b)

            if (club_a, club_b_ordenado) in pares_existentes or (club_a, club_b_ordenado) in nuevos:
                continue
            filial_a = _contiene_palabra_filial(datos.tokens_set.get(club_a, set()))
            filial_b = _contiene_palabra_filial(datos.tokens_set.get(club_b_ordenado, set()))
            if filial_a and filial_b:
                continue

            nuevos[(club_a, club_b_ordenado)] = fraccion
    return nuevos


# ---------------------------------------------------------------------------
# 5. Reglas ESTRICTAS (probabilidad 100 %) de cada grupo
# ---------------------------------------------------------------------------
_CATEGORIAS_FORMAS_JURIDICAS_COMPARABLES: frozenset[str] = (
    frozenset(_comun.DICCIONARIO_FORMAS_JURIDICAS) - {"otras_formas_juridicas"}
)


def _diff_tokens(nombre_a: str, nombre_b: str, datos: _DatosPrecalculados) -> set[str]:
    return datos.tokens_set[nombre_a] ^ datos.tokens_set[nombre_b]


def _es_filial_una_palabra(diff: set[str]) -> bool:
    """Regla estricta 1 de FILIAL: el nombre de A y B se diferencia
    ÚNICAMENTE en una palabra/token, y esa palabra está en `_comun.
    DICCIONARIO_FILIALES` (p. ej. "Ajax" -> "Jong Ajax")."""
    return len(diff) == 1 and diff.issubset(_comun.DICCIONARIO_FILIALES)


def _es_filial_por_todos_los_traspasos(
    club_a: str, club_b: str, red_preliminar: dict[str, frozenset[str]], datos: _DatosPrecalculados
) -> bool:
    """Regla estricta 2 de FILIAL: con al menos `UMBRAL_TAM_MUESTRA_
    TRASPASOS` traspasos de salida registrados para B, el 100 % de ellos
    van hacia A o hacia algún miembro de la red de A ya confirmada."""
    total = datos.salidas_totales.get(club_b, 0)
    if total < UMBRAL_TAM_MUESTRA_TRASPASOS:
        return False
    red_a = red_preliminar.get(club_a, frozenset({club_a}))
    destinos = datos.flujo_por_origen.get(club_b, {})
    hacia_red_a = sum(cantidad for destino, cantidad in destinos.items() if destino in red_a)
    return hacia_red_a == total


def _solo_letras(nombre: str) -> str:
    texto = _comun.normalizar_texto(nombre)
    if not isinstance(texto, str):
        return ""
    return "".join(caracter for caracter in texto if caracter.isalpha())


def _caracteres_diferentes_si_distancia_uno(texto_a: str, texto_b: str) -> str | None:
    """Si `texto_a`/`texto_b` (cualquier cadena — letras, o letras y
    dígitos/espacios, según lo que pase quien llama) difieren en, como
    mucho, UNA operación de edición de un carácter (sustitución,
    inserción o eliminación) — distancia de Levenshtein <= 1 —, devuelve
    los caracteres implicados (los dos de una sustitución concatenados, o
    el insertado/eliminado; cadena vacía si son idénticas — distinguible
    de `None`, que significa distancia > 1). Calculada a mano explotando
    que solo hace falta distinguir "<= 1" de "> 1", no la distancia
    exacta."""
    if texto_a == texto_b:
        return ""
    if abs(len(texto_a) - len(texto_b)) > 1:
        return None
    if len(texto_a) == len(texto_b):
        diferencias = [(x, y) for x, y in zip(texto_a, texto_b) if x != y]
        if len(diferencias) != 1:
            return None
        return diferencias[0][0] + diferencias[0][1]

    larga, corta = (texto_a, texto_b) if len(texto_a) > len(texto_b) else (texto_b, texto_a)
    i = j = 0
    saltos = 0
    caracter_extra = ""
    while i < len(larga) and j < len(corta):
        if larga[i] == corta[j]:
            i += 1
            j += 1
            continue
        saltos += 1
        if saltos > 1:
            return None
        caracter_extra = larga[i]
        i += 1
    if i < len(larga):
        caracter_extra = larga[i]
    return caracter_extra


def _diferencia_es_solo_2_o_3(resto_a: str, resto_b: str) -> bool:
    """True si, tras quitar '2', '3' y espacios de AMBOS "restos" no
    alfabéticos (dígitos/espacios; ver `_es_duplicidad_normalizacion_o_
    caracteres`), quedan idénticos — es decir, la ÚNICA diferencia entre
    los dos nombres son esos dígitos, con o sin espacio/guion (los guiones
    ya se convierten en espacio en `_comun.normalizar_texto`, así que el
    "resto" solo contiene dígitos y espacios). P. ej. "Arsenal Tula"
    (resto " ") / "Arsenal-2 Tula" (resto " 2 ") -> al quitar '2' y
    espacios, ambos quedan vacíos. En cambio "Taizhou Yuanda" / "Taizhou
    Yuanda (2017-2021)" NO activa esta excepción: sobran demasiados
    dígitos aparte de un posible '2'."""
    limpio_a = resto_a.replace("2", "").replace("3", "").replace(" ", "")
    limpio_b = resto_b.replace("2", "").replace("3", "").replace(" ", "")
    return limpio_a == limpio_b


def _es_duplicidad_normalizacion_o_caracteres(nombre_a: str, nombre_b: str) -> bool:
    """Reglas estrictas 1, 2 y 3 de DUPLICIDAD:

      1. Si los nombres son IDÉNTICOS tras la normalización COMPLETA
         (`_comun.normalizar_texto`: minúsculas, sin acentos/diacríticos,
         dígitos y espacios incluidos) — p. ej. "Malmö"/"Malmo".
      2. Si no, pero SÍ tienen el mismo nº de tokens (`_comun.
         tokenizar_nombre`) y difieren en un único CARÁCTER sobre ESE
         MISMO texto normalizado COMPLETO (letras Y dígitos/espacios, no
         solo letras — ver nota) — distancia de edición <= 1
         (`_caracteres_diferentes_si_distancia_uno`), p. ej. "Dynamo
         Kyiv" -> "Dinamo Kyiv" — Y ese carácter NO es '2' ni '3': un
         '2'/'3' suele marcar un segundo o tercer equipo (están en
         `_comun.DICCIONARIO_FILIALES`), no una errata de grafía.
      3. Si no, pero SÍ tienen las mismas letras (`_solo_letras`, que AQUÍ
         SÍ descarta dígitos/espacios/signos A PROPÓSITO: es la regla
         pensada justo para tolerar esa clase de diferencia) y el resto
         de la diferencia son solo elementos que no son letras (números,
         espacios, guiones...), EXCEPTO si esa diferencia es únicamente
         '2', '3' o combinaciones de estos con espacios/guiones
         (`_diferencia_es_solo_2_o_3`) — p. ej. "Arsenal Tula" ->
         "Arsenal-2 Tula": la diferencia es solo un "-2" que en realidad
         marca un segundo equipo, no una duplicidad.

    La regla 2 compara el texto COMPLETO (no solo letras) a propósito, a
    diferencia de la 3: su "normalización" solo debe plegar acentos (à ->
    a), no descartar dígitos — si lo hiciera, un sufijo puramente
    numérico podría reducirse a un carácter suelto y colar como "un
    carácter de diferencia" sin serlo. Caso real detectado al validar
    este módulo: "Bucaspor 1928" (año de fundación, ninguna letra) frente
    a "Bucaspor U21" (solo la "u" es letra) parecían diferir en una sola
    letra al quitar los dígitos de AMBOS ("bucaspor" vs "bucasporu"),
    aunque sean sufijos de naturaleza totalmente distinta; comparando el
    texto completo ("bucaspor 1928" vs "bucaspor u21") la diferencia real
    (4 caracteres) queda visible y la regla ya no se dispara. La regla 3,
    en cambio, SÍ necesita descartar dígitos/espacios: es la que
    reconoce que "Taizhou Yuanda" / "Taizhou Yuanda (2017-2021)" son el
    mismo nombre con una anotación de año añadida — su tolerancia a
    dígitos es intencionada, no un error, así que sigue usando `_solo_
    letras` sin cambios.

    El requisito de "mismo nº de tokens" en la regla 2 es una salvaguarda
    explícita (aunque con el texto completo ya queda cubierto casi
    siempre por la propia distancia de edición): así "Rio Ave FC" (3
    tokens) -> "Rio Ave FC U23" (4 tokens) no se compara nunca por esta
    regla, por si acaso coincidiera en longitud con algún nombre no
    relacionado."""
    texto_a = _comun.normalizar_texto(nombre_a) or ""
    texto_b = _comun.normalizar_texto(nombre_b) or ""
    if texto_a == texto_b:
        return True

    if len(_comun.tokenizar_nombre(nombre_a)) == len(_comun.tokenizar_nombre(nombre_b)):
        caracteres = _caracteres_diferentes_si_distancia_uno(texto_a, texto_b)
        if caracteres is not None and not any(c in "23" for c in caracteres):
            return True

    letras_a, letras_b = _solo_letras(nombre_a), _solo_letras(nombre_b)
    if letras_a != letras_b:
        return False

    resto_a = "".join(c for c in texto_a if not c.isalpha())
    resto_b = "".join(c for c in texto_b if not c.isalpha())
    return not _diferencia_es_solo_2_o_3(resto_a, resto_b)


def _categoria_comun_no_otras(diff: set[str]) -> str | None:
    """Si `diff` tiene exactamente 2 tokens y ambos pertenecen a la MISMA
    subcategoría de `_comun.DICCIONARIO_FORMAS_JURIDICAS` (sin contar
    "otras_formas_juridicas"), devuelve esa categoría; si no, `None`."""
    if len(diff) != 2:
        return None
    categorias = {_comun.categoria_de_palabra(token) for token in diff}
    if len(categorias) != 1:
        return None
    categoria = next(iter(categorias))
    if categoria is None or categoria not in _CATEGORIAS_FORMAS_JURIDICAS_COMPARABLES:
        return None
    return categoria


def _es_duplicidad_misma_subcategoria(diff: set[str]) -> bool:
    """Regla estricta 4 de DUPLICIDAD: la diferencia son 2 tokens
    (bloques de palabra ya fusionados por `_comun.tokenizar_nombre`) de la
    MISMA subcategoría de formas jurídicas, sin contar "otras_formas_
    juridicas" (p. ej. "united"/"utd", o "fc"/"fk" dentro de "futbol_
    club")."""
    return _categoria_comun_no_otras(diff) is not None


def _es_forma_juridica_distinta_categoria(diff: set[str]) -> bool:
    """Regla estricta 1 de NINGUNA RELACIÓN: la diferencia son 2 tokens de
    formas jurídicas (sin contar "otras_formas_juridicas") de DISTINTA
    subcategoría (p. ej. "united" frente a "dinamo": "Manchester United" /
    "Dynamo Kyiv" no son ni cantera ni grafías del mismo club)."""
    if len(diff) != 2:
        return False
    token_a, token_b = tuple(diff)
    categoria_a, categoria_b = _comun.categoria_de_palabra(token_a), _comun.categoria_de_palabra(token_b)
    if categoria_a is None or categoria_b is None:
        return False
    if categoria_a not in _CATEGORIAS_FORMAS_JURIDICAS_COMPARABLES:
        return False
    if categoria_b not in _CATEGORIAS_FORMAS_JURIDICAS_COMPARABLES:
        return False
    return categoria_a != categoria_b


def _es_liga_distinta(club_a: str, club_b: str, datos: _DatosPrecalculados) -> bool:
    """Regla estricta 2 de NINGUNA RELACIÓN: la liga habitual (moda) de A
    y B es distinta (solo si se conoce la de ambos)."""
    liga_a, liga_b = datos.liga_dominante.get(club_a), datos.liga_dominante.get(club_b)
    if liga_a is None or liga_b is None:
        return False
    return liga_a != liga_b


def _es_ninguna_relacion_por_registro(club_a: str, club_b: str, datos: _DatosPrecalculados) -> bool:
    """Regla estricta 3 de NINGUNA RELACIÓN: A y B casan cada uno al
    ~100 % (SequenceMatcher) con un `club_id` DISTINTO de `clubs.csv` —
    dos clubes reales ya identificados sin ambigüedad — A NO SER que la
    diferencia entre A y B sea solo de caracteres que no son letras
    (números, espacios, guiones... `_solo_letras` coincide en ambos): en
    ese caso no son necesariamente dos clubes distintos, sino que
    `clubs.csv` fragmenta el histórico de UN MISMO club real en dos
    fichas (`club_id`) distintas por época — caso real detectado al
    validar este módulo: "Karpaty Lviv" / "Karpaty Lviv (-2021)" son el
    mismo club (el "(-2021)" marca una refundación con nuevo `club_id`
    en `clubs.csv`, no un club distinto)."""
    club_id_a, ratio_a = datos.registro.get(club_a, (None, 0.0))
    club_id_b, ratio_b = datos.registro.get(club_b, (None, 0.0))
    if not (ratio_a >= 0.999 and ratio_b >= 0.999 and club_id_a is not None and club_id_a != club_id_b):
        return False
    return _solo_letras(club_a) != _solo_letras(club_b)


# ---------------------------------------------------------------------------
# 6. Fórmulas de probabilidad (0-1) cuando ninguna regla estricta aplica
# ---------------------------------------------------------------------------
# --- FILIAL: prob = w1*traspasos + w2*valor + w3*nacionalidad + w4*liga ---
PESO_FILIAL_TRASPASOS = 0.40
PESO_FILIAL_VALOR = 0.25
PESO_FILIAL_NACIONALIDAD = 0.20
PESO_FILIAL_LIGA = 0.15

# --- DUPLICIDAD: prob = w1*texto + w2*diccionario + w3*jugadores_comunes
#     + w4*valor + w5*liga + w6*nacionalidad + w7*ausencia_traspasos ---
PESO_DUP_TEXTO = 0.20
PESO_DUP_DICCIONARIO = 0.10
PESO_DUP_JUGADORES_COMUNES = 0.20
PESO_DUP_VALOR = 0.15
PESO_DUP_LIGA = 0.10
PESO_DUP_NACIONALIDAD = 0.10
PESO_DUP_AUSENCIA_TRASPASOS = 0.15

# --- NINGUNA RELACIÓN: prob = w1*nacionalidad_distinta + w2*forma_juridica
#     + w3*ausencia_traspasos + w4*ausencia_jugadores_comunes + w5*liga ---
PESO_NR_NACIONALIDAD = 0.30
PESO_NR_FORMA_JURIDICA = 0.20
PESO_NR_AUSENCIA_TRASPASOS = 0.25
PESO_NR_AUSENCIA_JUGADORES = 0.15
PESO_NR_LIGA = 0.10

NACIONALIDAD_MINIMO_APARICIONES = 10


def _fraccion_traspasos_hacia_red(club_b: str, red: frozenset[str], datos: _DatosPrecalculados) -> float:
    total = datos.salidas_totales.get(club_b, 0)
    if total == 0:
        return 0.0
    destinos = datos.flujo_por_origen.get(club_b, {})
    hacia_red = sum(cantidad for destino, cantidad in destinos.items() if destino in red)
    return hacia_red / total


def _pct_traspasos_directos(club_a: str, club_b: str, datos: _DatosPrecalculados) -> float:
    """% de traspasos DIRECTOS entre A y B (en cualquier sentido),
    relativo al total de salidas de cada uno — usado como evidencia
    NEGATIVA en duplicidad/ninguna relación (dos alias del mismo club, o
    dos clubes sin relación, no deberían "traspasarse" jugadores entre
    sí)."""
    salidas_a, salidas_b = datos.salidas_totales.get(club_a, 0), datos.salidas_totales.get(club_b, 0)
    pct_a_b = datos.flujo.get((club_a, club_b), 0) / salidas_a if salidas_a else 0.0
    pct_b_a = datos.flujo.get((club_b, club_a), 0) / salidas_b if salidas_b else 0.0
    return max(pct_a_b, pct_b_a)


def _fraccion_jugadores_comunes(club_a: str, club_b: str, datos: _DatosPrecalculados) -> float:
    jugadores_a = datos.jugadores_por_club.get(club_a, set())
    jugadores_b = datos.jugadores_por_club.get(club_b, set())
    denominador = min(len(jugadores_a), len(jugadores_b))
    if denominador == 0:
        return 0.0
    return len(jugadores_a & jugadores_b) / denominador


def _fraccion_nacionalidad_coincide_con_dominante(club_objetivo: str, club_referencia: str, datos: _DatosPrecalculados) -> float | None:
    """Fracción de jugadores de `club_objetivo` cuya nacionalidad coincide
    con la nacionalidad DOMINANTE de `club_referencia`. `None` si no hay
    datos suficientes para calcularlo (nacionalidad dominante de
    referencia desconocida, o plantilla objetivo sin nacionalidades)."""
    dominante = datos.nacionalidad_dominante.get(club_referencia)
    jugadores = datos.jugadores_por_club.get(club_objetivo, set())
    if dominante is None or not jugadores:
        return None
    nacionalidad_dominante, _ = dominante
    nacs = datos.nacionalidad_por_jugador.reindex(list(jugadores)).dropna()
    if nacs.empty:
        return None
    return float((nacs == nacionalidad_dominante).mean())


def _misma_liga(club_a: str, club_b: str, datos: _DatosPrecalculados) -> bool | None:
    liga_a, liga_b = datos.liga_dominante.get(club_a), datos.liga_dominante.get(club_b)
    if liga_a is None or liga_b is None:
        return None
    return liga_a == liga_b


def probabilidad_filial(
    club_a: str, club_b: str, red_preliminar: dict[str, frozenset[str]], datos: _DatosPrecalculados
) -> float:
    """Fórmula de probabilidad de FILIAL (0-1) cuando ninguna regla
    estricta aplica:

        P = 0.40 * P_traspasos + 0.25 * P_valor
          + 0.20 * P_nacionalidad + 0.15 * P_liga

      - `P_traspasos`: fracción de las salidas de B que van hacia A o su
        red (`_fraccion_traspasos_hacia_red`), amortiguada por el tamaño
        de la muestra (`min(1, salidas_B / 5)`) para no fiarse de un
        100 % basado en 1 o 2 traspasos.
      - `P_valor = max(0, 1 - valor_B / valor_A)`: cuanto mayor la
        diferencia de valor de mercado (A > B, por construcción), más
        probable que B sea la cantera.
      - `P_nacionalidad`: fracción de jugadores de B cuya nacionalidad es
        la dominante en A (una cantera forma sobre todo jugadores
        locales); 0.5 (neutro) si no hay datos suficientes.
      - `P_liga`: 1 si A y B comparten liga habitual, 0.5 si no se conoce
        la de alguno, 0 si es distinta.
    """
    red_a = red_preliminar.get(club_a, frozenset({club_a}))
    total_b = datos.salidas_totales.get(club_b, 0)
    fraccion = _fraccion_traspasos_hacia_red(club_b, red_a, datos)
    peso_muestra = min(1.0, total_b / UMBRAL_TAM_MUESTRA_TRASPASOS)
    p_traspasos = fraccion * peso_muestra

    valor_a, valor_b = datos.valor(club_a), datos.valor(club_b)
    p_valor = max(0.0, 1.0 - valor_b / valor_a) if valor_a > 0 else 0.0

    p_nacionalidad = _fraccion_nacionalidad_coincide_con_dominante(club_b, club_a, datos)
    if p_nacionalidad is None:
        p_nacionalidad = 0.5

    misma_liga = _misma_liga(club_a, club_b, datos)
    p_liga = 0.5 if misma_liga is None else (1.0 if misma_liga else 0.0)

    return (
        PESO_FILIAL_TRASPASOS * p_traspasos
        + PESO_FILIAL_VALOR * p_valor
        + PESO_FILIAL_NACIONALIDAD * p_nacionalidad
        + PESO_FILIAL_LIGA * p_liga
    )


def probabilidad_duplicidad(club_a: str, club_b: str, ratio: float, diff: set[str], datos: _DatosPrecalculados) -> float:
    """Fórmula de probabilidad de DUPLICIDAD (0-1) cuando ninguna regla
    estricta aplica:

        P = 0.20 * P_texto + 0.10 * P_diccionario
          + 0.20 * P_jugadores_comunes + 0.15 * P_valor
          + 0.10 * P_liga + 0.10 * P_nacionalidad
          + 0.15 * P_ausencia_traspasos

      - `P_texto`: similitud de `SequenceMatcher` reescalada linealmente
        desde `UMBRAL_SIMILITUD_BAJO` (0 %) hasta 1.0 (100 %).
      - `P_diccionario`: 1 si la diferencia son 2 tokens de la MISMA
        subcategoría de formas jurídicas (sin contar "otras"), 0 si no.
      - `P_jugadores_comunes`: jugadores que aparecen en el dataset bajo
        AMBOS nombres, relativo al tamaño de la plantilla más pequeña.
      - `P_valor = min(valor_A, valor_B) / max(valor_A, valor_B)`: cuanto
        más parecido el valor de mercado, más probable que sea el mismo
        club.
      - `P_liga`: 1 si comparten liga (lo normal si son el mismo club),
        0.5 si no se conoce, 0 si difiere.
      - `P_nacionalidad`: fracción de jugadores de B con la nacionalidad
        dominante de A (misma plantilla real -> misma composición).
      - `P_ausencia_traspasos = 1 - % traspasos DIRECTOS entre A y B`: dos
        alias del mismo club no deberían tener traspasos reales entre sí.
    """
    p_texto = max(0.0, min(1.0, (ratio - UMBRAL_SIMILITUD_BAJO) / (1.0 - UMBRAL_SIMILITUD_BAJO)))
    p_diccionario = 1.0 if _es_duplicidad_misma_subcategoria(diff) else 0.0
    p_jugadores = _fraccion_jugadores_comunes(club_a, club_b, datos)

    valor_a, valor_b = datos.valor(club_a), datos.valor(club_b)
    p_valor = (min(valor_a, valor_b) / max(valor_a, valor_b)) if max(valor_a, valor_b) > 0 else 0.0

    misma_liga = _misma_liga(club_a, club_b, datos)
    p_liga = 0.5 if misma_liga is None else (1.0 if misma_liga else 0.0)

    p_nacionalidad = _fraccion_nacionalidad_coincide_con_dominante(club_b, club_a, datos)
    if p_nacionalidad is None:
        p_nacionalidad = 0.5

    p_ausencia_traspasos = 1.0 - min(1.0, _pct_traspasos_directos(club_a, club_b, datos))

    return (
        PESO_DUP_TEXTO * p_texto
        + PESO_DUP_DICCIONARIO * p_diccionario
        + PESO_DUP_JUGADORES_COMUNES * p_jugadores
        + PESO_DUP_VALOR * p_valor
        + PESO_DUP_LIGA * p_liga
        + PESO_DUP_NACIONALIDAD * p_nacionalidad
        + PESO_DUP_AUSENCIA_TRASPASOS * p_ausencia_traspasos
    )


def probabilidad_ninguna_relacion(club_a: str, club_b: str, diff: set[str], datos: _DatosPrecalculados) -> float:
    """Fórmula de probabilidad de NINGUNA RELACIÓN (0-1) cuando ninguna
    regla estricta aplica:

        P = 0.30 * P_nacionalidad_distinta + 0.20 * P_forma_juridica
          + 0.25 * P_ausencia_traspasos + 0.15 * P_ausencia_jugadores
          + 0.10 * P_liga

      - `P_nacionalidad_distinta`: 1 si la nacionalidad dominante de A y
        B es distinta (exigiendo al menos `NACIONALIDAD_MINIMO_
        APARICIONES` jugadores de esa nacionalidad en AMBOS clubes, para
        no fiarse de una plantilla pequeña), 0.3 si no hay datos
        suficientes, 0 si coincide.
      - `P_forma_juridica`: 1 si la diferencia son 2 tokens de formas
        jurídicas de DISTINTA subcategoría (ver `_es_forma_juridica_
        distinta_categoria`), 0 si no.
      - `P_ausencia_traspasos = 1 - % traspasos DIRECTOS entre A y B`: sin
        conexión de traspasos, más evidencia de que no hay relación.
      - `P_ausencia_jugadores = 1 - fracción de jugadores en común`: sin
        plantilla compartida, más evidencia de que no hay relación.
      - `P_liga`: como en este punto ya se sabe (regla estricta 2) que NO
        son de liga distinta, esta señal solo aporta cuando la liga de
        alguno de los dos es desconocida (0.3, leve indicio a favor de
        "sin relación" por falta de verificación) — si ambas se conocen y
        coinciden, 0.
    """
    dominante_a = datos.nacionalidad_dominante.get(club_a)
    dominante_b = datos.nacionalidad_dominante.get(club_b)
    if (
        dominante_a is not None
        and dominante_b is not None
        and dominante_a[1] >= NACIONALIDAD_MINIMO_APARICIONES
        and dominante_b[1] >= NACIONALIDAD_MINIMO_APARICIONES
    ):
        p_nacionalidad = 1.0 if dominante_a[0] != dominante_b[0] else 0.0
    else:
        p_nacionalidad = 0.3

    p_forma_juridica = 1.0 if _es_forma_juridica_distinta_categoria(diff) else 0.0
    p_ausencia_traspasos = 1.0 - min(1.0, _pct_traspasos_directos(club_a, club_b, datos))
    p_ausencia_jugadores = 1.0 - _fraccion_jugadores_comunes(club_a, club_b, datos)

    misma_liga = _misma_liga(club_a, club_b, datos)
    p_liga = 0.3 if misma_liga is None else 0.0

    return (
        PESO_NR_NACIONALIDAD * p_nacionalidad
        + PESO_NR_FORMA_JURIDICA * p_forma_juridica
        + PESO_NR_AUSENCIA_TRASPASOS * p_ausencia_traspasos
        + PESO_NR_AUSENCIA_JUGADORES * p_ausencia_jugadores
        + PESO_NR_LIGA * p_liga
    )


# ---------------------------------------------------------------------------
# 7. Pipeline principal
# ---------------------------------------------------------------------------
COLUMNAS_RELACIONES = [
    "club_a", "club_b", "tipo", "probabilidad", "conflicto",
    "origen_regla", "origen_candidato", "n_filas_a", "n_filas_b",
]

NOMBRE_FILIAL = "posible filial"
NOMBRE_DUPLICIDAD = "posible duplicidad"
NOMBRE_NINGUNA_RELACION = "ninguna relación"
_PRIORIDAD_EMPATE = {NOMBRE_FILIAL: 0, NOMBRE_DUPLICIDAD: 1, NOMBRE_NINGUNA_RELACION: 2}


def _clasificar_par(
    club_a: str,
    club_b: str,
    ratio: float,
    red_preliminar: dict[str, frozenset[str]],
    datos: _DatosPrecalculados,
) -> tuple[str, float, list[str]]:
    """Clasifica un único par ya candidato: evalúa TODAS las reglas
    estrictas de los tres grupos (sin cortocircuitar en la primera que
    aplique, para poder detectar conflictos), y si NINGUNA aplica, calcula
    las tres fórmulas de probabilidad. Devuelve `(tipo, probabilidad,
    grupos_con_regla_estricta)` — esta última lista, con más de un
    elemento, señala un conflicto (ver `detectar_relaciones_equipos`)."""
    diff = _diff_tokens(club_a, club_b, datos)

    reglas_estrictas: list[str] = []
    if _es_filial_una_palabra(diff) or _es_filial_por_todos_los_traspasos(club_a, club_b, red_preliminar, datos):
        reglas_estrictas.append(NOMBRE_FILIAL)
    if _es_duplicidad_normalizacion_o_caracteres(club_a, club_b) or _es_duplicidad_misma_subcategoria(diff):
        reglas_estrictas.append(NOMBRE_DUPLICIDAD)
    if (
        _es_forma_juridica_distinta_categoria(diff)
        or _es_liga_distinta(club_a, club_b, datos)
        or _es_ninguna_relacion_por_registro(club_a, club_b, datos)
    ):
        reglas_estrictas.append(NOMBRE_NINGUNA_RELACION)

    if reglas_estrictas:
        tipo = min(reglas_estrictas, key=lambda grupo: _PRIORIDAD_EMPATE[grupo])
        return tipo, 1.0, reglas_estrictas

    probabilidades = {
        NOMBRE_FILIAL: probabilidad_filial(club_a, club_b, red_preliminar, datos),
        NOMBRE_DUPLICIDAD: probabilidad_duplicidad(club_a, club_b, ratio, diff, datos),
        NOMBRE_NINGUNA_RELACION: probabilidad_ninguna_relacion(club_a, club_b, diff, datos),
    }
    tipo = max(probabilidades, key=probabilidades.get)
    return tipo, probabilidades[tipo], []


def detectar_relaciones_equipos(
    df: pd.DataFrame | None = None,
    clubs: pd.DataFrame | None = None,
    traspasos: pd.DataFrame | None = None,
    longitud_minima_palabra: int = 4,
    tamano_maximo_bloque: int = 30,
) -> pd.DataFrame:
    """Compara los nombres de `equipo` del dataset ENTRE SÍ (no contra
    `clubs.csv`) y clasifica cada par candidato como "posible filial",
    "posible duplicidad" o "ninguna relación", con su probabilidad.

    Pipeline (ver también el docstring del módulo):

      1. `generar_pares_por_nombre`: pares (A, B, A = mayor valor) que
         comparten palabra de bloqueo y cumplen la condición 1 o 2 del
         enunciado (SequenceMatcher >= 70 %, o >= 50 % si alguno contiene
         palabra de filiales), excluyendo los pares donde AMBOS nombres
         ya contienen palabra de filiales.
      2. Con esos pares se evalúan las reglas estrictas de nombre (no las
         que dependen de traspasos) para construir una red PRELIMINAR de
         familias filial/duplicidad (`_construir_red_preliminar`).
      3. `generar_pares_por_traspasos`: condición 3 del enunciado — nuevos
         pares SOLO por patrón de traspasos (>= 50 % de las salidas de B
         van a la red de A, con >= 5 traspasos), usando esa red
         preliminar.
      4. Se clasifica el conjunto COMPLETO de pares (por nombre + por
         traspasos): para cada uno, se comprueban TODAS las reglas
         estrictas de los tres grupos (sin cortocircuito, para detectar
         si más de un grupo las cumple a la vez — columna `conflicto`) y,
         si ninguna aplica, se calculan las tres fórmulas de probabilidad
         (`probabilidad_filial` / `_duplicidad` / `_ninguna_relacion`),
         quedándose con la de mayor probabilidad.

    Devuelve una única fila por par, con el tipo de MAYOR probabilidad.
    La columna `conflicto` está vacía si solo un grupo tenía regla
    estricta aplicable (o ninguno), y si no, lista los grupos en conflicto
    (se ha resuelto a favor del de mayor prioridad: filial > duplicidad >
    ninguna relación).
    """
    if df is None:
        df = _comun.cargar_tabla_final()
    if clubs is None:
        clubs = _comun.cargar_clubs()
    if traspasos is None:
        traspasos = _comun.cargar_traspasos()

    datos = _DatosPrecalculados(df, clubs, traspasos)

    pares_por_nombre = generar_pares_por_nombre(datos, longitud_minima_palabra, tamano_maximo_bloque)
    if not pares_por_nombre:
        return pd.DataFrame(columns=COLUMNAS_RELACIONES)

    pares_filial_o_dup_preliminar = [
        (club_a, club_b)
        for (club_a, club_b) in pares_por_nombre
        if _es_filial_una_palabra(_diff_tokens(club_a, club_b, datos))
        or _es_duplicidad_normalizacion_o_caracteres(club_a, club_b)
        or _es_duplicidad_misma_subcategoria(_diff_tokens(club_a, club_b, datos))
    ]
    red_preliminar = _construir_red_preliminar(pares_filial_o_dup_preliminar)

    pares_por_traspasos = generar_pares_por_traspasos(datos, red_preliminar, set(pares_por_nombre))

    todos_los_pares: dict[tuple[str, str], float] = dict(pares_por_nombre)
    origen_candidato: dict[tuple[str, str], str] = {par: "nombre" for par in pares_por_nombre}
    for par, _fraccion in pares_por_traspasos.items():
        if par not in todos_los_pares:
            ratio = difflib.SequenceMatcher(
                None,
                _comun.normalizar_texto(par[0]) or "",
                _comun.normalizar_texto(par[1]) or "",
            ).ratio()
            todos_los_pares[par] = ratio
            origen_candidato[par] = "traspasos"

    filas = []
    for (club_a, club_b), ratio in todos_los_pares.items():
        tipo, probabilidad, reglas_estrictas = _clasificar_par(club_a, club_b, ratio, red_preliminar, datos)
        conflicto = (
            " / ".join(g for g in reglas_estrictas if g != tipo) if len(reglas_estrictas) > 1 else ""
        )
        filas.append(
            {
                "club_a": club_a,
                "club_b": club_b,
                "tipo": tipo,
                "probabilidad": round(probabilidad, 4),
                "conflicto": conflicto,
                "origen_regla": "estricta" if reglas_estrictas else "formula",
                "origen_candidato": origen_candidato[(club_a, club_b)],
                "n_filas_a": int(datos.conteo.get(club_a, 0)),
                "n_filas_b": int(datos.conteo.get(club_b, 0)),
            }
        )

    resultado = pd.DataFrame(filas, columns=COLUMNAS_RELACIONES)
    if resultado.empty:
        return resultado
    return resultado.sort_values("probabilidad", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Redes de relaciones: filiales Y duplicidades en UN MISMO grafo
# ---------------------------------------------------------------------------
def resolver_redes_relaciones(pares: pd.DataFrame, valor_por_club: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa en componentes conexas TODAS las relaciones "posible filial"
    y "posible duplicidad" que existen (estén confirmadas por regla
    estricta o no — no se aplica ningún umbral de probabilidad), pero
    fusionando las "familias" en dos fases para no unir dos que en
    realidad no tienen relación:

      1. Primero se unen SIEMPRE los pares con regla estricta
        (`origen_regla == "estricta"`, probabilidad 1.0) O revisados a
        mano por el usuario (`origen_regla == "manual"`): en ambos casos
        es evidencia determinista, sin ambigüedad — una corrección manual
        es, por definición, más fiable que cualquier salvaguarda de la
        fase 2, así que se trata exactamente igual que una regla
        estricta del algoritmo.
      2. Los pares por FÓRMULA se procesan agrupados por probabilidad, de
         mayor a menor, con DOS salvaguardas (ninguna es un umbral de
         probabilidad, las dos dependen solo de la ESTRUCTURA de la red
         construida hasta ese momento):
           a. Una fusión entre dos familias (todavía distintas en ese
              momento) solo se acepta si es la ÚNICA propuesta de fusión
              de AMBAS familias A ESE NIVEL de probabilidad — si una
              familia tiene, al mismo nivel, propuestas hacia DOS
              familias distintas, se considera ambigua y NINGUNA de ellas
              se fusiona en ese nivel.
           b. Dos familias que YA tienen 2 o más miembros cada una en ese
              momento solo se fusionan por fórmula si NINGÚN club de una
              está clasificado como "ninguna relación" con NINGÚN club de
              la otra (en CUALQUIER par candidato de la tabla completa,
              no solo entre los dos clubes de la arista que se está
              evaluando): sin esa evidencia negativa, una fórmula puede
              ATRAER un club todavía suelto hacia una familia ya formada
              igual que fusionar dos familias ya consolidadas entre sí;
              con ella, la fusión se bloquea igual que antes.

    La salvaguarda (a) resuelve el conflicto "un club con relaciones hacia
    dos familias distintas" (p. ej. "Atlético de Madrid B" con 100 % de
    ser filial de "Atlético de Madrid" y 48 % de "Real Madrid": al
    procesar primero el 100 %, "Atlético de Madrid B" ya está en la
    familia del Atlético cuando se llega al 48 %, así que esa propuesta
    ya no conecta dos familias distintas del mismo club) y el caso real
    "FC U Craiova 1948", con CUATRO relaciones exactamente empatadas al
    52 %, dos hacia "Universitatea Craiova" y dos hacia "FC Dinamo 1948"
    (sin relación real): ambigüedad detectada, no se fusiona con ninguna.

    La salvaguarda (b) hizo falta porque (a) por sí sola NO basta: una
    arista débil pero NADA ambigua a su propio nivel exacto de
    probabilidad (no hay ninguna otra arista con ese mismo valor exacto)
    seguía fusionando dos familias YA grandes y sin relación real entre
    sí — caso real detectado al validar este módulo: "Real Madrid" y
    "Atlético de Madrid" (o "Manchester City" y "Manchester United")
    acababan en la misma red a través de una única arista de baja
    probabilidad entre dos de sus filiales, simplemente porque esa arista
    no estaba empatada con ninguna otra. En su primera versión, (b)
    bloqueaba TODA fusión entre dos familias de 2+ miembros sin
    excepción — pero eso también bloqueaba casos reales, como "Athletic
    Bilbao" (con su "Athletic Bilbao U19") y "Bilbao Athletic" (con su
    afiliado "CD Basconia"): la relación real entre las dos mitades era,
    precisamente, la arista más débil de las tres, así que para cuando le
    tocaba procesarse ambos lados ya se habían consolidado por su cuenta
    y (b) la descartaba aunque no hubiese ninguna duda real. La condición
    "ningún club de una familia es 'ninguna relación' de ningún club de
    la otra" distingue ambos casos SIN volver a un umbral de probabilidad:
    entre Real Madrid y Atlético de Madrid (o Man City/Man United) SÍ hay
    pares explícitamente clasificados "ninguna relación" (p. ej. "Real
    Madrid" - "Atlético de Madrid C", o el propio "Manchester City" -
    "Manchester United" por regla estricta al 100 %), así que (b) los
    sigue bloqueando; entre Athletic Bilbao y Bilbao Athletic no existe
    ningún par así, así que (b) ya no bloquea esa fusión.

    Si A tiene filiales, B tiene sus propios filiales, y A-B son a su vez
    duplicidad entre sí, los tres grupos de clubes quedan en una única
    red, para ver de un vistazo toda la "familia" de un club (no solo su
    cantera, o solo sus variantes de grafía, por separado).

    Devuelve `(redes, aristas)`:
      - `redes`: columnas `red_id, club, rol ('líder'/'otro'),
        valor_mercado_eur, club_lider`. El líder es el de mayor valor de
        mercado de la red completa.
      - `aristas`: columnas `red_id, club_a, club_b, tipo, probabilidad` —
        TODAS las relaciones (fusionaran o no dos familias) entre clubes
        que acaban en la misma red, para dibujar el grafo distinguiendo
        filial de duplicidad.
    """
    columnas_redes = ["red_id", "club", "rol", "valor_mercado_eur", "club_lider"]
    columnas_aristas = ["red_id", "club_a", "club_b", "tipo", "probabilidad"]

    relaciones = pares[pares["tipo"].isin([NOMBRE_FILIAL, NOMBRE_DUPLICIDAD])]
    if relaciones.empty:
        return pd.DataFrame(columns=columnas_redes), pd.DataFrame(columns=columnas_aristas)

    # Pares "ninguna relación" de TODA la tabla (no solo filial/duplicidad):
    # evidencia negativa explícita para la salvaguarda (b), ver más abajo.
    pares_sin_relacion = {
        frozenset((club_a, club_b))
        for club_a, club_b in zip(
            pares.loc[pares["tipo"] == NOMBRE_NINGUNA_RELACION, "club_a"],
            pares.loc[pares["tipo"] == NOMBRE_NINGUNA_RELACION, "club_b"],
        )
    }

    padre: dict[str, str] = {}
    tamano: dict[str, int] = {}
    miembros_activos: dict[str, set[str]] = {}

    def _raiz(nombre: str) -> str:
        if nombre not in padre:
            padre[nombre] = nombre
            tamano[nombre] = 1
            miembros_activos[nombre] = {nombre}
        while padre[nombre] != nombre:
            padre[nombre] = padre[padre[nombre]]
            nombre = padre[nombre]
        return nombre

    def _unir(nombre_a: str, nombre_b: str) -> None:
        raiz_a, raiz_b = _raiz(nombre_a), _raiz(nombre_b)
        if raiz_a != raiz_b:
            padre[raiz_b] = raiz_a
            tamano[raiz_a] = tamano[raiz_a] + tamano[raiz_b]
            miembros_activos[raiz_a] |= miembros_activos.pop(raiz_b)

    def _hay_ninguna_relacion_cruzada(raiz_a: str, raiz_b: str) -> bool:
        """True si algún club de la familia de `raiz_a` está clasificado
        como "ninguna relación" con algún club de la familia de `raiz_b`,
        en cualquier par candidato de la tabla (no solo entre los dos
        clubes concretos de la arista evaluada)."""
        return any(
            frozenset((club_a, club_b)) in pares_sin_relacion
            for club_a in miembros_activos[raiz_a]
            for club_b in miembros_activos[raiz_b]
        )

    # Fase 1: reglas estrictas y revisiones manuales, sin ambigüedad posible.
    estrictas = relaciones[relaciones["origen_regla"].isin(["estricta", "manual"])]
    for club_a, club_b in zip(estrictas["club_a"], estrictas["club_b"]):
        _unir(club_a, club_b)

    # Fase 2: fórmula, por lotes de igual probabilidad, de mayor a menor.
    # Además de la ambigüedad por empate, dos familias que YA tienen 2 o
    # más miembros cada una en ese momento (`tamano[raiz] >= 2` en AMBOS
    # lados) solo se fusionan por fórmula si NINGÚN club de una está
    # clasificado como "ninguna relación" con NINGÚN club de la otra —
    # ver `_hay_ninguna_relacion_cruzada` y el docstring de la función.
    formula = relaciones[~relaciones["origen_regla"].isin(["estricta", "manual"])]
    for probabilidad in sorted(formula["probabilidad"].unique(), reverse=True):
        lote = formula[formula["probabilidad"] == probabilidad]
        propuestas: dict[str, set[str]] = {}
        pares_lote: list[tuple[str, str]] = []
        for club_a, club_b in zip(lote["club_a"], lote["club_b"]):
            raiz_a, raiz_b = _raiz(club_a), _raiz(club_b)
            if raiz_a == raiz_b:
                continue  # ya son la misma familia, esta arista no aporta una fusión nueva
            if tamano[raiz_a] >= 2 and tamano[raiz_b] >= 2 and _hay_ninguna_relacion_cruzada(raiz_a, raiz_b):
                continue  # dos familias consolidadas Y con evidencia negativa cruzada: no se fusionan
            propuestas.setdefault(raiz_a, set()).add(raiz_b)
            propuestas.setdefault(raiz_b, set()).add(raiz_a)
            pares_lote.append((raiz_a, raiz_b))
        for raiz_a, raiz_b in pares_lote:
            if len(propuestas[raiz_a]) == 1 and len(propuestas[raiz_b]) == 1:
                _unir(raiz_a, raiz_b)

    miembros_por_raiz: dict[str, set[str]] = {}
    for nombre in padre:
        miembros_por_raiz.setdefault(_raiz(nombre), set()).add(nombre)

    filas_redes = []
    red_id_por_club: dict[str, int] = {}
    redes_ordenadas = sorted(miembros_por_raiz.values(), key=len, reverse=True)
    for red_id, miembros in enumerate(redes_ordenadas, start=1):
        if len(miembros) < 2:
            continue
        valores = {club: float(valor_por_club.get(club, 0.0)) for club in miembros}
        lider = max(valores, key=valores.get)
        for club, valor in valores.items():
            red_id_por_club[club] = red_id
            filas_redes.append(
                {
                    "red_id": red_id,
                    "club": club,
                    "rol": "líder" if club == lider else "otro",
                    "valor_mercado_eur": valor,
                    "club_lider": lider,
                }
            )

    filas_aristas = [
        {
            "red_id": red_id_por_club[fila["club_a"]],
            "club_a": fila["club_a"],
            "club_b": fila["club_b"],
            "tipo": fila["tipo"],
            "probabilidad": fila["probabilidad"],
        }
        for _, fila in relaciones.iterrows()
        if fila["club_a"] in red_id_por_club
        and fila["club_b"] in red_id_por_club
        and red_id_por_club[fila["club_a"]] == red_id_por_club[fila["club_b"]]
    ]

    redes = pd.DataFrame(filas_redes, columns=columnas_redes).sort_values(
        ["red_id", "rol"], ascending=[True, False]
    ).reset_index(drop=True)
    aristas = pd.DataFrame(filas_aristas, columns=columnas_aristas).reset_index(drop=True)
    return redes, aristas


def figura_red_relaciones(miembros: pd.DataFrame, aristas: pd.DataFrame, lider: str) -> go.Figure:
    """Dibuja una red completa (líder + todos los clubes relacionados,
    filiales y duplicidades juntos) con un layout de grafo genérico
    (`networkx.spring_layout`, igual mecanismo que `visualizacion.
    red_de_traspasos`): el grafo puede tener varias ramas y no ser una
    simple estrella líder-filiales, al mezclar relaciones de filial y de
    duplicidad. Las aristas de filial se dibujan como línea continua
    gris, las de duplicidad como línea discontinua naranja."""
    import networkx as nx

    grafo = nx.Graph()
    grafo.add_nodes_from(miembros["club"])
    for _, fila in aristas.iterrows():
        grafo.add_edge(fila["club_a"], fila["club_b"], tipo=fila["tipo"])

    k = 1.6 / max(len(miembros), 1) ** 0.5
    pos = nx.spring_layout(grafo, seed=42, k=k, iterations=200)

    valores = miembros.set_index("club")["valor_mercado_eur"]
    valor_maximo = float(valores.max()) if len(valores) else 0.0

    def _tamano(club: str) -> float:
        if valor_maximo <= 0:
            return 30.0
        return 18.0 + (valores.get(club, 0.0) / valor_maximo) * 55.0

    fig = go.Figure()
    for _, fila in aristas.iterrows():
        x0, y0 = pos[fila["club_a"]]
        x1, y1 = pos[fila["club_b"]]
        es_filial = fila["tipo"] == NOMBRE_FILIAL
        fig.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(
                    color="rgba(120,120,120,0.6)" if es_filial else "rgba(230,126,34,0.75)",
                    width=1.5,
                    dash=None if es_filial else "dash",
                ),
                hoverinfo="text",
                text=f"{fila['club_a']} — {fila['club_b']} ({fila['tipo']}, p={fila['probabilidad']:.2f})",
                showlegend=False,
            )
        )

    nodos = list(pos.keys())
    xs = [pos[club][0] for club in nodos]
    ys = [pos[club][1] for club in nodos]
    tamanos = [_tamano(club) for club in nodos]
    colores = ["#d62728" if club == lider else "#1f77b4" for club in nodos]
    hover = [
        f"<b>{club}</b><br>Valor de mercado: {valores.get(club, 0.0):,.0f} €<br>"
        f"{'Club líder' if club == lider else 'Relacionado'}"
        for club in nodos
    ]

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text", text=nodos, textposition="bottom center",
            marker=dict(size=tamanos, color=colores, line=dict(width=1, color="white")),
            hoverinfo="text", hovertext=hover, showlegend=False,
        )
    )
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="rgba(120,120,120,0.6)", width=2), name="Filial"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="rgba(230,126,34,0.75)", width=2, dash="dash"), name="Duplicidad"))
    fig.update_layout(
        title=f"Red de relaciones de «{lider}»",
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


if __name__ == "__main__":
    df = _comun.cargar_tabla_final()
    clubs = _comun.cargar_clubs()
    traspasos = _comun.cargar_traspasos()

    print("=== Palabras más repetidas en 'equipo' ===")
    print(frecuencia_palabras_equipo(df).head(20).to_string(index=False))

    print("\n=== Relaciones entre nombres de equipo (filial / duplicidad) ===")
    relaciones = detectar_relaciones_equipos(df, clubs, traspasos)
    for tipo in [NOMBRE_FILIAL, NOMBRE_DUPLICIDAD, NOMBRE_NINGUNA_RELACION]:
        n = int((relaciones["tipo"] == tipo).sum())
        print(f"{tipo}: {n}")
    n_conflicto = int((relaciones["conflicto"] != "").sum())
    print(f"pares con reglas estrictas de varios grupos a la vez: {n_conflicto}")
    if n_conflicto:
        print(relaciones[relaciones["conflicto"] != ""].head(20).to_string(index=False))
    print(relaciones.head(30).to_string(index=False))

    print("\n=== Redes de relaciones (filiales + duplicidades) ===")
    valor_por_club = valor_mercado_por_club(df)
    redes, aristas = resolver_redes_relaciones(relaciones, valor_por_club)
    print(f"{redes['club_lider'].nunique() if not redes.empty else 0} redes detectadas")
    print(redes.head(30).to_string(index=False))
