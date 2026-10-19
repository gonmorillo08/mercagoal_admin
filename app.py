import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scikit_na as na
from pathlib import Path

from visualizacion import graficas as gr
from visualizacion import red_de_traspasos as rt
from limpieza import _comun as lp
from limpieza import duplicidad_equipos, valores_faltantes, nacionalidad_y_liga, valores_erroneos, compilacion

DIR_ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="Mercagoal · Administración", page_icon=str(DIR_ASSETS / "icono.png"), layout="wide")
st.logo(str(DIR_ASSETS / "logo.png"), size="large", icon_image=str(DIR_ASSETS / "icono.png"))

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
# Ruta relativa: se asume que este script vive en la raíz del proyecto y los
# datos están en la carpeta "datos de ingest". Ajusta DATA_PATH si tu estructura difiere.
DATA_PATH = Path(__file__).parent / "ingest" / "datos" /"tabla_valores_mercado.parquet"


@st.cache_data
def cargar_datos(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def _aplicar_revisiones_manuales(
    pares: pd.DataFrame,
    revisiones: dict[tuple[str, str], str],
    valor_por_club: pd.Series,
    filas_por_club: pd.Series,
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Copia de `pares` con las revisiones manuales del usuario (guardadas en
    disco, clave = `duplicidad_equipos.clave_par`, independiente del orden
    A/B) aplicadas sobre el tipo detectado por el algoritmo: `tipo_original`
    conserva siempre el veredicto del algoritmo (para poder deshacer la
    revisión y para comparar en el editor), mientras que `tipo`/
    `probabilidad`/`origen_regla` reflejan la revisión manual cuando existe
    (probabilidad 100 %, `origen_regla = "manual"` — `resolver_redes_relaciones`
    trata una revisión manual igual que una regla estricta del algoritmo, ya
    que es más fiable que cualquiera de sus heurísticas).

    Pensado para que las revisiones sobrevivan a una actualización de los
    datos:
      - Una revisión de filial/duplicidad cuyo par el algoritmo YA NO
        propone como candidato se añade igualmente como fila (con
        `tipo_original = "ninguna relación"`, que es lo que implica no ser
        candidato), para que siga contando en las métricas y en las redes.
      - Las revisiones con algún club que ya no existe en los datos
        (renombrado o eliminado en el origen) no se pueden aplicar: se
        devuelven aparte para avisar, pero NO se borran del fichero, por si
        el club vuelve a aparecer en una actualización posterior."""
    pares = pares.copy()
    pares["tipo_original"] = pares["tipo"]

    claves_en_tabla = {duplicidad_equipos.clave_par(a, b) for a, b in zip(pares["club_a"], pares["club_b"])}
    no_aplicables: list[tuple[str, str]] = []
    filas_nuevas = []
    for clave, tipo in revisiones.items():
        if clave in claves_en_tabla:
            continue
        if not all(club in filas_por_club.index for club in clave):
            no_aplicables.append(clave)
            continue
        if tipo == duplicidad_equipos.NOMBRE_NINGUNA_RELACION:
            continue  # no ser candidato ya equivale a "ninguna relación"
        club_a, club_b = sorted(clave, key=lambda club: valor_por_club.get(club, 0), reverse=True)
        filas_nuevas.append(
            {
                "club_a": club_a, "club_b": club_b, "tipo": duplicidad_equipos.NOMBRE_NINGUNA_RELACION,
                "probabilidad": 0.0, "conflicto": "", "origen_regla": "manual",
                "origen_candidato": "revisión manual", "n_filas_a": int(filas_por_club[club_a]),
                "n_filas_b": int(filas_por_club[club_b]),
                "tipo_original": duplicidad_equipos.NOMBRE_NINGUNA_RELACION,
            }
        )
    if filas_nuevas:
        pares = pd.concat([pares, pd.DataFrame(filas_nuevas)], ignore_index=True)

    claves = [duplicidad_equipos.clave_par(a, b) for a, b in zip(pares["club_a"], pares["club_b"])]
    revisado = [clave in revisiones for clave in claves]
    pares["tipo"] = [
        revisiones[clave] if es_rev else original
        for clave, es_rev, original in zip(claves, revisado, pares["tipo_original"])
    ]
    pares["probabilidad"] = [
        1.0 if es_rev else prob for es_rev, prob in zip(revisado, pares["probabilidad"])
    ]
    pares["origen_regla"] = [
        "manual" if es_rev else origen for es_rev, origen in zip(revisado, pares["origen_regla"])
    ]
    pares["revisado"] = revisado
    return pares, no_aplicables


@st.dialog("Restablecer revisiones manuales")
def _confirmar_reset_revisiones_manuales() -> None:
    st.warning(
        "Esto elimina PERMANENTEMENTE todas las revisiones manuales guardadas "
        "en disco. ¿Seguro que quieres continuar?"
    )
    c_si, c_no = st.columns(2)
    if c_si.button("Sí, restablecer", type="primary", use_container_width=True, key="confirmar_reset_duplicidad_si"):
        st.session_state["revisiones_manuales_duplicidad"] = {}
        duplicidad_equipos.guardar_revisiones_manuales({})
        st.rerun()
    if c_no.button("Cancelar", use_container_width=True, key="confirmar_reset_duplicidad_no"):
        st.rerun()


@st.dialog("Restablecer revisiones de nacionalidad")
def _confirmar_reset_revisiones_nacionalidad() -> None:
    st.warning(
        "Esto elimina PERMANENTEMENTE todas las revisiones manuales de "
        "nacionalidad guardadas en disco. ¿Seguro que quieres continuar?"
    )
    c_si, c_no = st.columns(2)
    if c_si.button("Sí, restablecer", type="primary", use_container_width=True, key="confirmar_reset_nacionalidad_si"):
        st.session_state["revisiones_manuales_nacionalidad"] = {}
        nacionalidad_y_liga.guardar_revisiones_nacionalidad({})
        st.rerun()
    if c_no.button("Cancelar", use_container_width=True, key="confirmar_reset_nacionalidad_no"):
        st.rerun()


@st.dialog("Restablecer revisiones de clasificación (Arsenal FC)")
def _confirmar_reset_revisiones_arsenal_fc() -> None:
    st.warning(
        "Esto elimina PERMANENTEMENTE todas las revisiones manuales de "
        "clasificación de jugadores de 'Arsenal FC' guardadas en disco. "
        "¿Seguro que quieres continuar?"
    )
    c_si, c_no = st.columns(2)
    if c_si.button("Sí, restablecer", type="primary", use_container_width=True, key="confirmar_reset_arsenal_si"):
        st.session_state["revisiones_manuales_arsenal_fc"] = {}
        nacionalidad_y_liga.guardar_revisiones_arsenal_fc({})
        st.rerun()
    if c_no.button("Cancelar", use_container_width=True, key="confirmar_reset_arsenal_no"):
        st.rerun()


def _avisar_revisiones_no_aplicables(elementos: list[str], que: str, motivo: str) -> None:
    """Aviso común a las tres tablas editables: revisiones manuales guardadas
    en disco que no se pueden aplicar porque `motivo` ya no existe en los
    datos actuales (p. ej. tras una actualización del dataset). No se
    borran: se conservan en el fichero por si vuelven a aparecer."""
    if not elementos:
        return
    with st.expander(f"⚠️ {len(elementos)} {que} revisado(s) que no se aplican con los datos actuales"):
        st.caption(
            f"Guardados en disco, pero {motivo} ya no aparece en los datos actuales "
            "(renombrado o eliminado en una actualización). Se conservan en el "
            "fichero y volverán a aplicarse solos si vuelve a aparecer."
        )
        st.dataframe(pd.DataFrame({"Revisión guardada": sorted(elementos)}), use_container_width=True, hide_index=True)


def _coincide_busqueda(serie: pd.Series, busqueda: str) -> pd.Series:
    """Máscara booleana: `serie` contiene `busqueda`, ambas normalizadas
    (minúsculas y SIN acentos/diacríticos, vía `limpieza._comun.
    normalizar_texto` — la misma normalización que usa el resto del
    proyecto para comparar nombres) para que las barras de búsqueda de la
    app encuentren "1.FC Köln" buscando "koln", sin tener que escribir la
    diéresis."""
    busqueda_normalizada = lp.normalizar_texto(busqueda) or ""
    if not busqueda_normalizada:
        return pd.Series(True, index=serie.index)
    return lp.normalizar_nombre_club(serie).str.contains(busqueda_normalizada, regex=False, na=False)


# ---------------------------------------------------------------------------
# Análisis de valores faltantes con scikit-na
# (https://scikit-na.readthedocs.io/en/latest/index.html)
# ---------------------------------------------------------------------------
# El guion bajo en "_df" le indica a st.cache_data que NO calcule el hash del
# DataFrame (sería muy costoso con >500k filas): al haberse cargado una única
# vez con @st.cache_data en cargar_datos(), su contenido no cambia entre
# reruns, así que basta con cachear por la propia función.
# Clave de caché que cambia al editar `valores_faltantes.VALORES_NA_DISFRAZADO`:
# como `_df` no se hashea, sin ella las funciones cacheadas que dependen de
# qué se considera "NA disfrazado" seguirían devolviendo el resultado
# calculado con el diccionario anterior hasta reiniciar el servidor.
def _clave_na_disfrazados() -> tuple[str, ...]:
    return tuple(sorted(valores_faltantes.VALORES_NA_DISFRAZADO))


@st.cache_data
def calcular_resumen_na_convertido(
    _df: pd.DataFrame, clave_na: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resumen de NA por columna y a nivel de dataset (na.summary), sobre el
    DataFrame ya con los NA disfrazados convertidos a NaN real (ver
    `calcular_df_convertido`). Nombre propio (en vez de reutilizar una
    función genérica) a propósito: como `_df` no participa en el hash de
    `@st.cache_data`, una función que se llamara alguna vez con el
    DataFrame original y otra con el convertido devolvería para uno de los
    dos el resultado cacheado del otro. `clave_na`: ver
    `_clave_na_disfrazados`."""
    resumen_columnas = na.summary(_df).T
    resumen_columnas.index.name = "categoria"
    resumen_global = na.summary(_df, per_column=False)
    return resumen_columnas, resumen_global


@st.cache_data
def calcular_describe_na(
    _df: pd.DataFrame, col_na: str, col_valor: str, clave_na: tuple[str, ...]
) -> pd.DataFrame:
    """Estadísticos de 'col_valor' comparando registros con/sin NA en 'col_na' (na.describe)."""
    return na.describe(_df, col_na=col_na, columns=[col_valor])


@st.cache_data
def cargar_tabla_traspasos() -> pd.DataFrame:
    """`traspasos.csv`, ya precalculada por `ingest/procesar_datos.py`. La
    usan tanto la red de traspasos como `duplicidad_equipos.py` (señal
    secundaria, pilar 3, para distinguir filial de duplicidad)."""
    return rt.cargar_traspasos()


# La tabla de traspasos ya viene precalculada; aquí solo se lee y se
# agrega (equipos/ligas). Se cachea igualmente: no depende de los filtros
# interactivos (nivel de nodo, ligas/equipos concretos...) que elija el
# usuario en la red de transferencias, esos se aplican después sobre el
# resultado ya calculado.
@st.cache_data
def calcular_datos_red_transferencias() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    return rt.preparar_datos_red(cargar_tabla_traspasos())


# ---------------------------------------------------------------------------
# Módulos de limpieza (paquete `limpieza/`)
# ---------------------------------------------------------------------------
@st.cache_data
def calcular_informe_valores_faltantes(_df: pd.DataFrame, clave_na: tuple[str, ...]) -> dict:
    return valores_faltantes.informe_valores_faltantes(_df)


@st.cache_data
def calcular_imputacion_equipo(
    _df: pd.DataFrame, clave_na: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vista previa de `valores_faltantes.imputar_equipo_por_historial`
    (solo el detalle y los eliminados: la tabla imputada NO se usa en la
    app, el proceso está documentado pero no aplicado)."""
    _, detalle, eliminados = valores_faltantes.imputar_equipo_por_historial(_df)
    return detalle, eliminados


@st.cache_data
def calcular_df_convertido(_df: pd.DataFrame, na_disfrazados: pd.DataFrame) -> pd.DataFrame:
    """`_df` con los NA disfrazados (ver `informe_valores_faltantes`)
    convertidos a NaN real, para que `scikit_na` los cuente como tales."""
    return valores_faltantes.convertir_na_disfrazados(_df, na_disfrazados)


@st.cache_data
def cargar_clubs_csv() -> pd.DataFrame:
    """`clubs.csv` crudo: lo necesita `duplicidad_equipos.py` para saber
    qué nombres de `equipo` tienen ficha oficial (y con qué `club_id`)."""
    return lp.cargar_clubs()


@st.cache_data
def cargar_competitions_csv() -> pd.DataFrame:
    """`competitions.csv` crudo: lo necesita `nacionalidad_y_liga.py` para
    saber el país y el `competition_id` de cada nombre de liga."""
    return lp.cargar_competitions()


@st.cache_data
def calcular_tabla_banderas_ligas(_df: pd.DataFrame, _competitions: pd.DataFrame) -> pd.DataFrame:
    return nacionalidad_y_liga.tabla_banderas_ligas(_df, _competitions)


@st.cache_data
def calcular_tabla_banderas_nacionalidades(_df: pd.DataFrame) -> pd.DataFrame:
    return nacionalidad_y_liga.tabla_banderas_nacionalidades(_df)


@st.cache_data
def calcular_tabla_clubes_sin_liga(_df: pd.DataFrame) -> pd.DataFrame:
    return nacionalidad_y_liga.tabla_clubes_sin_liga(_df)


@st.cache_data
def calcular_tabla_jugadores_arsenal_fc(_df: pd.DataFrame) -> pd.DataFrame:
    return nacionalidad_y_liga.tabla_jugadores_arsenal_fc(_df)


@st.cache_data
def calcular_relaciones_equipos(_df: pd.DataFrame, _clubs: pd.DataFrame) -> pd.DataFrame:
    return duplicidad_equipos.detectar_relaciones_equipos(_df, _clubs, cargar_tabla_traspasos())


@st.cache_data
def calcular_valor_mercado_por_club(_df: pd.DataFrame) -> pd.Series:
    return duplicidad_equipos.valor_mercado_por_club(_df)


@st.cache_data
def calcular_cambios_anomalos(_df: pd.DataFrame, dias_max: int, ratio_umbral: float) -> pd.DataFrame:
    return valores_erroneos.cambios_anomalos(_df, dias_max, ratio_umbral)


@st.cache_data
def calcular_picos_aislados(_df: pd.DataFrame, dias_max: int) -> pd.DataFrame:
    return valores_erroneos.picos_aislados(_df, dias_max)


if not DATA_PATH.exists():
    st.error(
        f"No se encontró el archivo de datos en '{DATA_PATH}'. "
        "Comprueba que el parquet está dentro de la carpeta 'ingest' "
        "junto a este script, o edita la variable DATA_PATH."
    )
    st.stop()

df = cargar_datos(DATA_PATH)

# ---------------------------------------------------------------------------
# Navegación (barra lateral)
# ---------------------------------------------------------------------------
st.sidebar.caption("PANEL DE ADMINISTRACIÓN")
pagina = st.sidebar.radio(
    "Navegación",
    ["🔎 Explorador de datos", "📊 Visualización de datos", "🧹 Limpieza de datos"],
)
st.sidebar.divider()


# ---------------------------------------------------------------------------
# Página: Explorador de datos (filtrado + visualización)
# ---------------------------------------------------------------------------
def pagina_explorador(df: pd.DataFrame) -> None:
    st.title("🔎 Explorador de Valores de Mercado")
    st.caption(f"{len(df):,} registros cargados desde `{DATA_PATH.name}`")

    # -- Filtros (barra lateral) --------------------------------------------
    st.sidebar.header("Filtros")

    def opciones(col: str):
        return sorted(df[col].dropna().unique().tolist())

    paises = st.sidebar.multiselect("País (nacionalidad)", opciones("nacionalidad"))
    ligas = st.sidebar.multiselect("Liga", opciones("liga"))
    equipos = st.sidebar.multiselect("Equipo", opciones("equipo"))
    posiciones = st.sidebar.multiselect("Posición", opciones("posicion"))

    id = st.sidebar.multiselect("ID", opciones("id_player"))
    nombre_busqueda = st.sidebar.text_input("Buscar jugador por nombre")

    fecha_min, fecha_max = df["fecha"].min(), df["fecha"].max()
    rango_fechas = st.sidebar.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )

    valor_min, valor_max = int(df["valor_mercado_eur"].min()), int(df["valor_mercado_eur"].max())
    rango_valor = st.sidebar.slider(
        "Valor de mercado (€)",
        min_value=valor_min,
        max_value=valor_max,
        value=(valor_min, valor_max),
        step=100_000,
        format="%d",
    )

    # -- Aplicar filtros ------------------------------------------------------
    df_filtrado = df.copy()

    if paises:
        df_filtrado = df_filtrado[df_filtrado["nacionalidad"].isin(paises)]
    if ligas:
        df_filtrado = df_filtrado[df_filtrado["liga"].isin(ligas)]
    if equipos:
        df_filtrado = df_filtrado[df_filtrado["equipo"].isin(equipos)]
    if posiciones:
        df_filtrado = df_filtrado[df_filtrado["posicion"].isin(posiciones)]
    if id:
        df_filtrado = df_filtrado[
            df_filtrado["id_player"].isin(id)
        ]
    if nombre_busqueda:
        df_filtrado = df_filtrado[_coincide_busqueda(df_filtrado["nombre"], nombre_busqueda)]
    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        inicio, fin = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
        df_filtrado = df_filtrado[(df_filtrado["fecha"] >= inicio) & (df_filtrado["fecha"] <= fin)]
    df_filtrado = df_filtrado[
        (df_filtrado["valor_mercado_eur"] >= rango_valor[0])
        & (df_filtrado["valor_mercado_eur"] <= rango_valor[1])
    ]

    # -- KPIs -------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros", f"{len(df_filtrado):,}")
    col2.metric("Jugadores únicos", f"{df_filtrado['id_player'].nunique():,}")
    col3.metric(
        "Valor medio (€)",
        f"{df_filtrado['valor_mercado_eur'].mean():,.0f}" if len(df_filtrado) else "-",
    )
    col4.metric(
        "Valor máximo (€)",
        f"{df_filtrado['valor_mercado_eur'].max():,.0f}" if len(df_filtrado) else "-",
    )

    st.divider()

    # -- Tabla de datos ------------------------------------------------------
    st.subheader("Tabla de datos")
    st.dataframe(
        df_filtrado.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV filtrado",
        data=csv,
        file_name="valores_mercado_filtrado.csv",
        mime="text/csv",
    )

    # -- Evolución del valor de mercado (por jugador) ------------------------
    st.divider()
    st.subheader("Evolución del valor de mercado")

    jugadores_disponibles = sorted(df_filtrado["nombre"].dropna().unique().tolist())
    if jugadores_disponibles:
        jugadores_grafico = st.multiselect(
            "Selecciona uno o varios jugadores para ver su evolución",
            jugadores_disponibles,
            default=jugadores_disponibles[: min(3, len(jugadores_disponibles))],
        )
        if jugadores_grafico:
            df_grafico = df_filtrado[df_filtrado["nombre"].isin(jugadores_grafico)].sort_values("fecha")
            fig = px.line(
                df_grafico,
                x="fecha",
                y="valor_mercado_eur",
                color="nombre",
                markers=True,
                labels={"fecha": "Fecha", "valor_mercado_eur": "Valor de mercado (€)", "nombre": "Jugador"},
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay jugadores que coincidan con los filtros seleccionados.")


# ---------------------------------------------------------------------------
# Página: Visualización de datos
# ---------------------------------------------------------------------------
def _seccion_red_traspasos() -> None:
    """Configuración + resultado de la red de traspasos (ligas o equipos),
    a partir de `traspasos.csv`."""
    if not rt.RUTA_TRASPASOS.exists():
        st.error(
            f"No se encontró '{rt.RUTA_TRASPASOS}'. Ejecuta "
            "`python ingest/procesar_datos.py` para generar la tabla de traspasos."
        )
        return

    aristas_equipos, liga_por_equipo, aristas_ligas, traspasos = calcular_datos_red_transferencias()
    lista_equipos = rt.equipos_disponibles(aristas_equipos)
    lista_ligas = rt.ligas_disponibles(aristas_equipos)

    st.caption(
        f"Tabla de traspasos: {len(traspasos):,} traspasos inferidos del historial de valoraciones "
        f"(valor de mercado en la fecha del cambio de equipo) · {len(lista_equipos):,} equipos distintos. "
        "Generada por `ingest/procesar_datos.py` (`ingest/datos/traspasos.csv`)."
    )

    nivel_nodo = st.selectbox("Cada nodo del gráfico es un/una...", ["Liga", "Equipo"], key="nivel_nodo")

    if nivel_nodo == "Liga":
        # -- Nodo = LIGA: se eligen directamente las ligas a mostrar --
        ligas_sel = st.multiselect(
            "Ligas a mostrar",
            lista_ligas,
            default=lista_ligas[: min(3, len(lista_ligas))],
            key="ligas_red_ligas",
        )

        top_n_conexiones = None
        if len(ligas_sel) == 1:
            top_n_conexiones = st.slider(
                f"Nº de ligas con más enlaces a «{ligas_sel[0]}» a mostrar", 3, 40, 10,
                key="top_n_ligas",
            )

        peso_minimo = st.slider(
            "Nº mínimo de jugadores traspasados por enlace", 1, 10, 1, key="peso_minimo_ligas"
        )

        fig, aristas_mostradas = rt.red_por_ligas(
            aristas_ligas,
            ligas_sel,
            top_n_conexiones=top_n_conexiones,
            peso_minimo=peso_minimo,
        )
        nivel_registros = "liga"

    else:
        # -- Nodo = EQUIPO: se puede elegir qué equipos aparecen indicando
        # directamente los equipos, o indicando las ligas cuyos equipos
        # se quieren ver --
        filtrar_por = st.selectbox("Elegir equipos por", ["Ligas", "Equipos"], key="filtrar_por")

        equipos_sel: list[str] = []
        ligas_sel = []
        top_n_conexiones = None
        top_n_equipos = 40
        if filtrar_por == "Ligas":
            ligas_sel = st.multiselect(
                "Ligas cuyos equipos se mostrarán",
                lista_ligas,
                default=lista_ligas[: min(3, len(lista_ligas))],
                key="ligas_red_equipos",
            )
            top_n_equipos = st.slider(
                "Nº máximo de equipos a mostrar (los más conectados)", 5, 150, 40,
                key="top_n_equipos",
            )
        else:
            equipos_sel = st.multiselect(
                "Equipos a mostrar",
                lista_equipos,
                default=lista_equipos[: min(3, len(lista_equipos))],
                key="equipos_red",
            )
            if len(equipos_sel) == 1:
                top_n_conexiones = st.slider(
                    f"Nº de equipos con más enlaces a «{equipos_sel[0]}» a mostrar", 3, 40, 10,
                    key="top_n_conexiones_equipo",
                )
            elif len(equipos_sel) > 1:
                # Con dos o más equipos elegidos directamente, la red es
                # siempre exactamente esos equipos: no hace falta ningún
                # límite de "nº máximo a mostrar".
                top_n_equipos = None

        peso_minimo = st.slider(
            "Nº mínimo de jugadores traspasados por enlace", 1, 10, 1, key="peso_minimo_equipos"
        )

        fig, aristas_mostradas = rt.red_por_equipos(
            aristas_equipos,
            liga_por_equipo,
            equipos=equipos_sel or None,
            ligas=ligas_sel or None,
            top_n_conexiones=top_n_conexiones,
            top_n_equipos=top_n_equipos,
            peso_minimo=peso_minimo,
        )
        nivel_registros = "equipo"

    st.divider()
    st.subheader("3. Resultado")
    st.plotly_chart(fig, use_container_width=True)

    col_origen_n = "liga_origen" if nivel_registros == "liga" else "equipo_origen"
    col_destino_n = "liga_destino" if nivel_registros == "liga" else "equipo_destino"
    etiqueta_nodo = "Liga" if nivel_registros == "liga" else "Equipo"

    # -- Desglose del nº de fichajes en cada dirección ------------------
    st.markdown(
        f"#### Desglose de fichajes por dirección ({etiqueta_nodo.lower()} a {etiqueta_nodo.lower()})"
    )
    tabla_desglose = rt.tabla_desglose_direcciones(aristas_mostradas, col_origen_n, col_destino_n)
    if not tabla_desglose.empty:
        tabla_desglose = tabla_desglose.rename(
            columns={"Origen": f"{etiqueta_nodo} origen", "Destino": f"{etiqueta_nodo} destino"}
        )
        st.dataframe(tabla_desglose, use_container_width=True, hide_index=True)
    else:
        st.info("No hay combinaciones con traspasos que mostrar.")

    # -- Registros de jugadores detrás de los enlaces mostrados --------
    registros = rt.registros_de_aristas(traspasos, aristas_mostradas, nivel_registros)
    resumen = rt.resumen_registros(registros)

    st.markdown("#### Resumen de los traspasos representados")
    if resumen:
        m1, m2, m3 = st.columns(3)
        m1.metric("Traspasos representados", f"{resumen['n_traspasos']:,}")
        m2.metric("Jugadores distintos", f"{resumen['n_jugadores']:,}")
        m3.metric(
            "Fichaje más valioso",
            resumen["jugador_mas_valioso"],
            f"{resumen['valor_mas_valioso']:,.0f} €",
        )
        st.caption(
            f"Valor de mercado medio: {resumen['valor_medio']:,.0f} € · "
            f"Valor de mercado total acumulado: {resumen['valor_total']:,.0f} € · "
            f"Periodo: {resumen['fecha_min'].date()} — {resumen['fecha_max'].date()}"
        )
    else:
        st.info("No hay traspasos representados todavía: ajusta la selección anterior.")

    st.markdown("#### Jugadores implicados en estos traspasos")
    busqueda = st.text_input("Buscar jugador por nombre", key="busqueda_jugador_red")
    registros_mostrar = registros
    if busqueda:
        registros_mostrar = registros_mostrar[_coincide_busqueda(registros_mostrar["nombre"], busqueda)]
    nombres_columnas = {
        "nombre": "Jugador",
        "equipo_origen": "Equipo origen",
        "equipo_destino": "Equipo destino",
        "liga_origen": "Liga origen",
        "liga_destino": "Liga destino",
        "fecha": "Fecha",
        "valor_mercado": "Valor de mercado (€)",
        "fuente": "Fuente",
    }
    registros_mostrar = (
        registros_mostrar.sort_values("valor_mercado", ascending=False)[list(nombres_columnas)]
        .rename(columns=nombres_columnas)
    )
    st.dataframe(registros_mostrar, use_container_width=True, hide_index=True)


def pagina_visualizacion(df: pd.DataFrame) -> None:
    st.title("📊 Visualización de datos")
    st.caption(
        "Explora la forma de las variables del dataset: elige el tipo de "
        "gráfica, la(s) variable(s) a representar y, si procede, una "
        "variable categórica para agrupar/comparar. Las funciones que "
        "generan estas gráficas están en `visualizacion/graficas.py`."
    )

    cols_num = gr.columnas_numericas(df)
    cols_cat = gr.columnas_categoricas(df)
    cols_fecha = gr.columnas_fecha(df)

    if not cols_num:
        st.warning("El dataset no contiene columnas numéricas para visualizar.")
        return

    tipos_grafica = [
        "Boxplot",
        "Densidad (KDE)",
        "Barras — frecuencia de categoría",
        "Barras — agregación por categoría",
        "Serie temporal agregada",
        "Red de traspasos (ligas o equipos)",
    ]

    # -- Selección de modalidad -----------------------------------------
    st.subheader("1. Elige qué quieres ver")
    c_tipo, c_desc = st.columns([1, 2])
    with c_tipo:
        tipo = st.selectbox("Tipo de gráfica", tipos_grafica)

    descripciones = {
        "Boxplot": "Resumen de mediana, cuartiles y valores atípicos de una variable numérica.",
        "Densidad (KDE)": "Curva suavizada (estimación de densidad) de la distribución de una variable.",
        "Barras — frecuencia de categoría": "Nº de registros por cada valor de una variable categórica.",
        "Barras — agregación por categoría": "Media/mediana/suma... de una variable numérica, por categoría.",
        "Serie temporal agregada": "Evolución en el tiempo de una variable numérica, agregada por periodo.",
        "Red de traspasos (ligas o equipos)": (
            "Grafo dirigido que resume los movimientos/flujos de jugadores entre clubes o ligas"
        ),
    }
    with c_desc:
        st.info(descripciones[tipo])

    st.divider()

    # -- Parámetros específicos de cada modalidad ------------------------
    st.subheader("2. Configura la gráfica")
    fig = None
    resultado_ya_mostrado = False


    if tipo == "Boxplot":
        c1, c2 = st.columns(2)
        columna_num = c1.selectbox("Variable numérica", cols_num)
        columna_cat = c2.selectbox("Comparar por categoría (opcional)", ["Ninguna"] + cols_cat)
        c3, c4 = st.columns(2)
        escala_log = c3.checkbox("Escala logarítmica en Y", value="valor" in columna_num.lower())
        mostrar_puntos = c4.checkbox("Mostrar todos los puntos", value=False)

        df_plot = df
        columna_cat_final = None if columna_cat == "Ninguna" else columna_cat
        if columna_cat_final:
            top_n = st.slider("Nº de categorías a mostrar (más frecuentes)", 2, 30, 10)
            categorias = gr.top_n_categorias(df, columna_cat_final, top_n)
            df_plot = df[df[columna_cat_final].isin(categorias)]

        fig = gr.boxplot(
            df_plot,
            columna_num,
            columna_cat=columna_cat_final,
            escala_log_y=escala_log,
            mostrar_puntos=mostrar_puntos,
        )

    elif tipo == "Densidad (KDE)":
        c1, c2 = st.columns(2)
        columna = c1.selectbox("Variable numérica", cols_num)
        color_por = c2.selectbox("Comparar por categoría (opcional)", ["Ninguna"] + cols_cat)
        escala_log = st.checkbox("Escala logarítmica en X", value="valor" in columna.lower())

        df_plot = df
        color_final = None if color_por == "Ninguna" else color_por
        if color_final:
            top_n = st.slider("Nº de categorías a comparar (más frecuentes)", 2, 10, 5)
            categorias = gr.top_n_categorias(df, color_final, top_n)
            df_plot = df[df[color_final].isin(categorias)]

        fig = gr.densidad(df_plot, columna, color_por=color_final, escala_log_x=escala_log)


    elif tipo == "Barras — frecuencia de categoría":
        c1, c2 = st.columns(2)
        columna = c1.selectbox("Variable categórica", cols_cat)
        top_n = c2.slider("Nº de categorías (top)", 3, 40, 15)
        fig = gr.barras_conteo(df, columna, top_n=top_n)

    elif tipo == "Barras — agregación por categoría":
        c1, c2, c3 = st.columns(3)
        columna_cat = c1.selectbox("Variable categórica", cols_cat)
        columna_num = c2.selectbox("Variable numérica", cols_num)
        agregacion = c3.selectbox("Agregación", list(gr._AGREGACIONES.keys()))
        top_n = st.slider("Nº de categorías (top)", 3, 40, 15)
        fig = gr.barras_agregado(
            df, columna_cat, columna_num, agregacion=agregacion, top_n=top_n
        )

    elif tipo == "Serie temporal agregada":
        if not cols_fecha:
            st.warning("El dataset no contiene ninguna columna de tipo fecha.")
        else:
            c1, c2 = st.columns(2)
            columna_fecha = c1.selectbox("Variable de fecha", cols_fecha)
            columna_num = c2.selectbox("Variable numérica", cols_num)
            c3, c4, c5 = st.columns(3)
            frecuencia = c3.selectbox("Frecuencia", list(gr._FRECUENCIAS.keys()), index=4)
            agregacion = c4.selectbox("Agregación", list(gr._AGREGACIONES.keys()))
            columna_cat = c5.selectbox("Desglosar por categoría (opcional)", ["Ninguna"] + cols_cat)

            df_plot = df
            columna_cat_final = None if columna_cat == "Ninguna" else columna_cat
            if columna_cat_final:
                top_n = st.slider("Nº de categorías a comparar (más frecuentes)", 2, 10, 5)
                categorias = gr.top_n_categorias(df, columna_cat_final, top_n)
                df_plot = df[df[columna_cat_final].isin(categorias)]

            fig = gr.serie_temporal(
                df_plot,
                columna_fecha,
                columna_num,
                columna_cat=columna_cat_final,
                frecuencia=frecuencia,
                agregacion=agregacion,
            )

    elif tipo == "Red de traspasos (ligas o equipos)":
        _seccion_red_traspasos()
        resultado_ya_mostrado = True

    # -- Resultado ---------------------------------------------------------
    if not resultado_ya_mostrado:
        st.divider()
        st.subheader("3. Resultado")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ajusta las opciones anteriores para generar la gráfica.")


# ---------------------------------------------------------------------------
# Página: Limpieza de datos
# ---------------------------------------------------------------------------
def _tab_valores_faltantes(df: pd.DataFrame) -> None:
    st.caption(
        "Recuento inicial de NA por columna, detección de los NA "
        "'disfrazados' de `limpieza/valores_faltantes.py` (texto que en "
        "realidad significa 'sin dato', p. ej. `'Missing'` en posición o "
        "`'Unknown'`/`'Without Club'` en equipo) y su conversión a NaN real, y finalmente el "
        "resumen y análisis de aleatoriedad de valores faltantes (scikit-na) "
        "ya con esa conversión aplicada."
    )

    # -- 1. NaN por categoría (gráfico de barras, dataset original) ----------
    st.subheader("Valores NaN detectados por categoría")
    st.caption(
        "Incluye las variables del jugador que se toman de `players.csv`: `fecha_nacimiento`, `altura_cm` y "
        "`pie`, y la `edad`, calculada como los años cumplidos en la fecha de cada registro. La edad solo "
        "falta cuando falta la fecha de nacimiento."
    )

    nulos_por_columna = df.isna().sum().reset_index()
    nulos_por_columna.columns = ["categoria", "n_nulos"]
    nulos_por_columna = nulos_por_columna.sort_values("n_nulos", ascending=False)

    fig_nulos = px.bar(
        nulos_por_columna,
        x="categoria",
        y="n_nulos",
        text="n_nulos",
        labels={"categoria": "Categoría", "n_nulos": "Valores NA detectados"},
        title="Valores NA detectados por categoría",
    )
    fig_nulos.update_traces(textposition="outside")
    fig_nulos.update_layout(xaxis_title="Categoría", yaxis_title="Valores NA detectados")
    st.plotly_chart(fig_nulos, use_container_width=True)

    st.divider()

    # -- 2. NA reales + NA disfrazados (tabla única), y conversión de estos --
    clave_na = _clave_na_disfrazados()
    informe = calcular_informe_valores_faltantes(df, clave_na)

    st.subheader("Valores NA detectados (reales y disfrazados)")
    st.caption(
        "`limpieza/valores_faltantes.py`: junto a los NA auténticos "
        "(`NaN`), también busca valores de texto que en realidad significan "
        "'sin dato' pero, al no ser `NaN`, no cuentan como tal en el "
        "gráfico anterior (p. ej. `'Missing'` en posición, `'Unknown'` en "
        "equipo). En `equipo`, `'Without Club'`, `'Retired'` y `'Career "
        "break'` no son un club desconocido sino una SITUACIÓN del jugador "
        "(sin equipo, retirado o en pausa): en esa fecha no hay club que "
        "asignar, así que también se tratan como NA — si no, cada uno "
        "aparecería en el resto de la app como si fuera un club más (con su "
        "liga, su nacionalidad dominante, sus traspasos...)."
    )

    na_disfrazados = informe["na_disfrazados"]
    with st.expander("Valores buscados (`VALORES_NA_DISFRAZADO`)"):
        st.caption(
            "Se comparan tras normalizar (minúsculas, sin acentos ni "
            "puntuación) contra TODAS las columnas de texto. Para añadir o "
            "quitar uno, edita `VALORES_NA_DISFRAZADO` en "
            "`limpieza/valores_faltantes.py`."
        )
        encontrados = {lp.normalizar_texto(v) for v in na_disfrazados["valor_disfrazado"]}
        st.dataframe(
            pd.DataFrame(
                {
                    "Valor buscado": list(clave_na),
                    "Encontrado en el dataset": [v in encontrados for v in clave_na],
                }
            ).sort_values("Encontrado en el dataset", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    tabla_na = valores_faltantes.combinar_na_reales_y_disfrazados(
        informe["na_reales"], na_disfrazados, len(df)
    )
    st.dataframe(tabla_na, use_container_width=True, hide_index=True)

    if na_disfrazados.empty:
        st.success("No se detectó ningún valor NA disfrazado.")
        df_convertido = df
    else:
        df_convertido = calcular_df_convertido(df, na_disfrazados)
        st.success(
            f"Convertidos a NaN real: {int(na_disfrazados['n_filas'].sum()):,} "
            f"celdas en {na_disfrazados['columna'].nunique()} columna(s) "
            f"({', '.join(na_disfrazados['columna'].unique())}). Las "
            "gráficas siguientes ya incluyen esta conversión."
        )

    st.divider()

    # -- 3. Resumen final de valores faltantes (scikit-na: na.summary) ------
    st.subheader("Resumen de valores faltantes (scikit-na)")
    st.caption(
        "Generado con `scikit_na.summary()` sobre el dataset ya con los NA "
        "disfrazados convertidos a NaN real: además del recuento y el "
        "porcentaje de NA por columna, indica qué porcentaje de todos los "
        "NA del dataset corresponde a cada columna y cuántas filas "
        "quedarían si se eliminaran (`dropna`) los NA de esa columna."
    )

    resumen_columnas, resumen_global = calcular_resumen_na_convertido(df_convertido, clave_na)
    st.dataframe(resumen_columnas, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Columnas con NA", f"{int(resumen_global.loc['na_cols', 'dataset']):,}")
    c2.metric("Filas con algún NA", f"{int(resumen_global.loc['na_rows', 'dataset']):,}")
    c3.metric("Celdas con NA", f"{int(resumen_global.loc['na_cells', 'dataset']):,}")
    c4.metric("% de celdas con NA", f"{resumen_global.loc['na_cells_pct', 'dataset']:.2f}%")

    columnas_con_na = df_convertido.columns[df_convertido.isna().any()].tolist()
    if not columnas_con_na:
        st.success("El dataset no contiene valores NA en ninguna columna.")
        return

    st.divider()

    # -- 4. ¿La ausencia de datos es aleatoria? (scikit-na: na.describe) -----
    st.subheader("¿Los valores faltantes son aleatorios?")
    st.caption(
        "Generado con `scikit_na.describe()` sobre el dataset ya convertido: "
        "compara la distribución de `valor_mercado_eur` entre los registros "
        "con y sin NA en la columna seleccionada. Cuanto más se separan la "
        "caja y la mediana de ambos grupos, más clara es la señal de que la "
        "ausencia de datos NO es aleatoria (MCAR), sino que está relacionada "
        "con el propio valor del jugador (MAR/MNAR) — información relevante "
        "antes de decidir cómo tratar esos NA."
    )

    col_na_seleccionada = st.selectbox(
        "Columna con NaN a analizar", columnas_con_na, key="col_na_describe"
    )

    stats_na = calcular_describe_na(df_convertido, col_na_seleccionada, "valor_mercado_eur", clave_na)
    stats_na_flat = (
        stats_na["valor_mercado_eur"]
        .reset_index()
        .rename(columns={"index": "estadístico"})
    )
    stats_na_flat.columns.name = None
    st.dataframe(stats_na_flat, use_container_width=True, hide_index=True)

    grupo_presente = f"{col_na_seleccionada} = presente"
    grupo_na = f"{col_na_seleccionada} = NA"
    df_grupos = df_convertido[["valor_mercado_eur"]].copy()
    df_grupos["grupo"] = np.where(
        df_convertido[col_na_seleccionada].isna(), grupo_na, grupo_presente
    )

    mediana_presente = df_grupos.loc[df_grupos["grupo"] == grupo_presente, "valor_mercado_eur"].median()
    mediana_na = df_grupos.loc[df_grupos["grupo"] == grupo_na, "valor_mercado_eur"].median()

    # Colores muy contrastados (azul = dato presente, rojo = NA) y línea de
    # la media dentro de cada caja (`boxmean`) para que la diferencia entre
    # grupos —y no solo la mediana— salte a la vista de un vistazo.
    fig_box = px.box(
        df_grupos,
        x="grupo",
        y="valor_mercado_eur",
        color="grupo",
        points=False,
        category_orders={"grupo": [grupo_presente, grupo_na]},
        color_discrete_map={grupo_presente: "#1f77b4", grupo_na: "#d62728"},
        labels={"grupo": "", "valor_mercado_eur": "Valor de mercado (€)"},
        title=f"Valor de mercado según si '{col_na_seleccionada}' es NA",
    )
    fig_box.update_traces(boxmean=True)
    fig_box.update_yaxes(type="log", title="Valor de mercado (€, escala log)")
    fig_box.update_layout(showlegend=False)

    if pd.notna(mediana_presente) and pd.notna(mediana_na) and mediana_presente > 0:
        diff_pct = (mediana_presente - mediana_na) / mediana_presente * 100
        fig_box.update_layout(margin=dict(t=100))
        fig_box.add_annotation(
            text=(
                f"Mediana con dato: {mediana_presente:,.0f} €  ·  "
                f"Mediana con NA: {mediana_na:,.0f} €  ·  "
                f"Diferencia: {diff_pct:.0f}%"
            ),
            xref="paper", yref="paper", x=0.5, y=1.15, showarrow=False,
            font=dict(size=13),
        )

    st.plotly_chart(fig_box, use_container_width=True)


def _tab_duplicidad_equipos(df: pd.DataFrame) -> None:
    st.caption(
        "`limpieza/duplicidad_equipos.py`: los nombres de `equipo` se "
        "comparan ENTRE SÍ (no contra `clubs.csv`) para distinguir dos "
        "problemas de raíz común (el nombre de club es texto libre, no un "
        "id): que dos cadenas de texto sean en realidad el MISMO club "
        "(posible duplicidad, p. ej. 'Dinamo Moscow'/'Dynamo Moscow') o que "
        "sean primer equipo y su cantera (posible filial, p. ej. 'Real "
        "Betis Balompié'/'Betis Deportivo Balompié')."
    )

    # -- 0. Recuperación de `equipo` faltante (documentada, NO aplicada) ------
    st.subheader("Recuperación de «equipo» faltante (propuesta, no aplicada)")
    st.info(
        "Proceso DOCUMENTADO para ponerse en marcha más adelante: todavía no se "
        "aplica. El resto de esta pestaña y de la app siguen usando la tabla "
        "original; aquí solo se muestra lo que haría "
        "(`limpieza/valores_faltantes.imputar_equipo_por_historial`)."
    )
    st.markdown(
        "Para cada registro sin `equipo` —NaN real o NA disfrazado de la pestaña "
        "*Datos faltantes* ('Unknown', 'Without Club', 'Retired', 'Career "
        "break'...)— se mira el historial de clubes del propio jugador "
        "(sus demás registros, ordenados por fecha) y:\n"
        "1. Se imputa el equipo del registro **más cercano en fecha** "
        "(en días) que sí tenga equipo, sea anterior o posterior.\n"
        "2. En caso de **empate** de distancia, tiene preferencia el equipo "
        "**previo** sobre el posterior.\n"
        "3. Si el jugador **no tiene ningún equipo** registrado en todo su "
        "historial, se **eliminan** todos sus registros."
    )

    detalle_imp, eliminados_imp = calcular_imputacion_equipo(df, _clave_na_disfrazados())
    n_sin_equipo = len(detalle_imp) + int(eliminados_imp["n_registros"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros sin equipo", f"{n_sin_equipo:,}")
    m2.metric("Se imputarían", f"{len(detalle_imp):,}")
    m3.metric(
        "Jugadores eliminados",
        f"{len(eliminados_imp):,}",
        f"{int(eliminados_imp['n_registros'].sum()):,} registros",
        delta_color="off",
    )
    m4.metric(
        "Distancia mediana",
        f"{detalle_imp['distancia_dias'].median():,.0f} días" if not detalle_imp.empty else "—",
    )

    if not detalle_imp.empty:
        resumen_imp = (
            detalle_imp.groupby("valor_original")
            .agg(
                registros=("direccion", "size"),
                pct_equipo_previo=("direccion", lambda s: round((s == "previo").mean() * 100, 1)),
                mediana_dias=("distancia_dias", "median"),
                max_dias=("distancia_dias", "max"),
            )
            .sort_values("registros", ascending=False)
            .reset_index()
            .rename(
                columns={
                    "valor_original": "Valor original", "registros": "Registros imputados",
                    "pct_equipo_previo": "% con equipo previo", "mediana_dias": "Distancia mediana (días)",
                    "max_dias": "Distancia máxima (días)",
                }
            )
        )
        st.dataframe(resumen_imp, use_container_width=True, hide_index=True)
        st.caption(
            "A tener en cuenta antes de ponerlo en marcha: 'Unknown' suele estar al "
            "INICIO de la carrera (casi siempre se imputa el equipo posterior), "
            "mientras que 'Retired' está al final (siempre el previo: se "
            "imputaría su último club aunque ya no juegue). Las distancias "
            "pueden ser de años: la imputación es tanto más fiable cuanto más "
            "cerca está el registro de referencia."
        )

        with st.expander("Detalle de los registros que se imputarían"):
            busqueda_imp = st.text_input("Buscar jugador por nombre", key="busqueda_imputacion_equipo")
            detalle_mostrar = detalle_imp
            if busqueda_imp:
                detalle_mostrar = detalle_mostrar[_coincide_busqueda(detalle_mostrar["nombre"], busqueda_imp)]
            st.dataframe(
                detalle_mostrar.drop(columns="id_player").rename(
                    columns={
                        "nombre": "Jugador", "fecha": "Fecha", "valor_original": "Valor original",
                        "equipo_imputado": "Equipo imputado", "direccion": "Registro de referencia",
                        "fecha_referencia": "Fecha de referencia", "distancia_dias": "Distancia (días)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    if not eliminados_imp.empty:
        with st.expander(f"Jugadores que se eliminarían ({len(eliminados_imp):,})"):
            st.dataframe(
                eliminados_imp.drop(columns="id_player").rename(
                    columns={
                        "nombre": "Jugador", "n_registros": "Registros",
                        "valores_originales": "Valores de equipo en su historial",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # -- 1. Palabras más repetidas en 'equipo' -------------------------------
    st.subheader("Palabras más repetidas en «equipo»")
    st.caption(
        "Nº de clubes DISTINTOS (no de filas) cuyo nombre contiene cada "
        "palabra tras normalizar (minúsculas, sin acentos/puntuación). "
        "Punto de partida para detectar patrones de siglas/abreviaturas "
        "(¿cuántos clubes usan 'fc'? ¿'united'? ¿'u19'?) antes de decidir "
        "qué agrupar como sinónimos."
    )
    tabla_palabras = duplicidad_equipos.frecuencia_palabras_equipo(df, top_n=50)
    st.dataframe(tabla_palabras, use_container_width=True, hide_index=True)

    with st.expander("Diccionarios usados (`limpieza._comun`)"):
        st.caption(
            "Diccionario de **filiales** (`DICCIONARIO_FILIALES`): sufijos "
            "de segundo equipo/cantera ('b', 'ii', 'u19'...). Diccionario "
            "de **formas jurídicas** (`DICCIONARIO_FORMAS_JURIDICAS`), "
            "dividido en subcategorías: palabras (o bloques de varias "
            "palabras, p. ej. 'sociedad deportiva') que, dentro de una "
            "misma subcategoría, se tratan como formas intercambiables de "
            "lo mismo (p. ej. 'united'/'utd')."
        )
        st.markdown(f"**filiales**: {', '.join(sorted(lp.DICCIONARIO_FILIALES))}")
        st.markdown("**formas jurídicas**, por subcategoría:")
        for categoria, palabras in lp.DICCIONARIO_FORMAS_JURIDICAS.items():
            st.markdown(f"- **{categoria}**: {', '.join(sorted(p.replace('_', ' ') for p in palabras))}")

    with st.expander("Fórmulas de probabilidad usadas"):
        st.markdown(
            "Cada par se clasifica primero por **reglas estrictas** (probabilidad = 1.0); "
            "si ninguna aplica, se calcula una probabilidad continua por fórmula:"
        )
        st.markdown("**Posible filial** (si no aplica regla estricta):")
        st.latex(
            r"P_{filial} = 0.40\,P_{traspasos} + 0.25\,P_{valor} + 0.20\,P_{nacionalidad} + 0.15\,P_{liga}"
        )
        st.caption(
            "P_traspasos = (% de salidas de B hacia A o su red) × min(1, salidas_B / 5) · "
            "P_valor = max(0, 1 − valor_B/valor_A) · "
            "P_nacionalidad = fracción de jugadores de B con la nacionalidad dominante de A · "
            "P_liga = 1 si misma liga, 0.5 si desconocida, 0 si distinta."
        )
        st.markdown("**Posible duplicidad** (si no aplica regla estricta):")
        st.latex(
            r"P_{dup} = 0.20\,P_{texto} + 0.10\,P_{dicc} + 0.20\,P_{jugadores} + 0.15\,P_{valor}"
            r" + 0.10\,P_{liga} + 0.10\,P_{nacionalidad} + 0.15\,P_{sin\ traspasos}"
        )
        st.caption(
            "P_texto = (SequenceMatcher − 0.5) / 0.5 · "
            "P_dicc = 1 si la diferencia son 2 tokens de la misma subcategoría de formas jurídicas · "
            "P_jugadores = jugadores en común / plantilla más pequeña · "
            "P_valor = min(valor_A,valor_B)/max(valor_A,valor_B) · "
            "P_liga = 1 si misma liga, 0.5 si desconocida, 0 si distinta · "
            "P_nacionalidad = fracción de jugadores de B con la nacionalidad dominante de A · "
            "P_sin_traspasos = 1 − % traspasos directos entre A y B."
        )
        st.markdown("**Ninguna relación** (si no aplica regla estricta):")
        st.latex(
            r"P_{sin} = 0.30\,P_{nac\ distinta} + 0.20\,P_{forma\ juridica} + 0.25\,P_{sin\ traspasos}"
            r" + 0.15\,P_{sin\ jugadores} + 0.10\,P_{liga}"
        )
        st.caption(
            "P_nac_distinta = 1 si la nacionalidad dominante de A y B difiere (con ≥10 jugadores de esa "
            "nacionalidad en ambos), 0.3 si no hay datos suficientes, 0 si coincide · "
            "P_forma_juridica = 1 si la diferencia son 2 tokens de formas jurídicas de distinta subcategoría · "
            "P_sin_traspasos = 1 − % traspasos directos entre A y B · "
            "P_sin_jugadores = 1 − jugadores en común / plantilla más pequeña · "
            "P_liga = 0.3 si se desconoce la liga de alguno (ya no puede ser 'distinta liga': eso ya lo "
            "resolvería la regla estricta), 0 si ambas se conocen."
        )
        st.caption(
            "Reglas estrictas completas (código y comentarios) en `limpieza/duplicidad_equipos.py`, "
            "funciones `_es_filial_*`, `_es_duplicidad_*`, `_es_forma_juridica_distinta_categoria`, "
            "`_es_liga_distinta`, `_es_ninguna_relacion_por_registro`."
        )

    st.divider()
    _seccion_relaciones_equipos(df)


def _seccion_relaciones_equipos(df: pd.DataFrame) -> None:
    """Secciones 2 (pares similares) y 3 (red de relaciones de un club) de
    la pestaña "Duplicidad de equipos y canteras": las únicas que dependen de
    `traspasos.csv`."""
    if not rt.RUTA_TRASPASOS.exists():
        st.error(
            f"No se encontró '{rt.RUTA_TRASPASOS}'. Ejecuta "
            "`python ingest/procesar_datos.py` para generar la tabla de traspasos."
        )
        return

    # -- 2. Pares candidatos: filial / duplicidad / ninguna relación --------
    st.subheader("Pares de nombres de equipo similares")
    st.caption(
        "Candidatos (A = club de mayor valor de mercado, B = el de menor): "
        "comparten palabra significativa y superan el 70 % de similitud de "
        "texto, o el 50 % si alguno contiene palabra del diccionario de "
        "filiales, o al menos el 50 % de los traspasos de B (con ≥5 "
        "registros) van hacia A o su red — nunca se compara un par donde "
        "AMBOS nombres ya contengan palabra de filiales. Cada par se "
        "clasifica primero por reglas estrictas (100 %); si ninguna "
        "aplica, por la fórmula de mayor probabilidad (ver 'Fórmulas de "
        "probabilidad usadas' arriba)."
    )

    clubs = cargar_clubs_csv()
    pares_base = calcular_relaciones_equipos(df, clubs)

    if "revisiones_manuales_duplicidad" not in st.session_state:
        # Se cargan de disco (no solo de `st.session_state`) para que una
        # revisión manual sobreviva a un F5/recarga de la página, que abre
        # una sesión de Streamlit nueva y por tanto un `session_state` vacío.
        st.session_state["revisiones_manuales_duplicidad"] = duplicidad_equipos.cargar_revisiones_manuales()
    revisiones_manuales = st.session_state["revisiones_manuales_duplicidad"]

    pares, revisiones_no_aplicables = _aplicar_revisiones_manuales(
        pares_base.reindex(columns=duplicidad_equipos.COLUMNAS_RELACIONES),
        revisiones_manuales,
        calcular_valor_mercado_por_club(df),
        df["equipo"].value_counts(),
    )

    if pares.empty:
        st.info("No se han encontrado pares de nombres de equipo similares con los umbrales actuales.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pares candidatos", f"{len(pares):,}")
        c2.metric("Posibles filiales", int((pares["tipo"] == "posible filial").sum()))
        c3.metric("Posibles duplicidades", int((pares["tipo"] == "posible duplicidad").sum()))
        c4.metric("Sin relación", int((pares["tipo"] == "ninguna relación").sum()))

        n_conflicto = int((pares["conflicto"] != "").sum())
        if n_conflicto:
            st.caption(
                f"⚠️ {n_conflicto} pares cumplían una regla estricta de MÁS DE UN grupo a la vez "
                "(se resolvió a favor de filial > duplicidad > ninguna relación)."
            )

        c_busqueda, c_filtro = st.columns([2, 2])
        busqueda = c_busqueda.text_input("Buscar por nombre de club", key="busqueda_club_duplicidad")
        tipos_disponibles = ["posible filial", "posible duplicidad", "ninguna relación"]
        tipos_sel = c_filtro.multiselect(
            "Filtrar por tipo", tipos_disponibles, default=["posible filial", "posible duplicidad"],
            key="filtro_tipo_duplicidad",
        )

        pares_mostrar = pares[pares["tipo"].isin(tipos_sel)] if tipos_sel else pares.iloc[0:0]
        if busqueda:
            coincide = _coincide_busqueda(pares_mostrar["club_a"], busqueda) | _coincide_busqueda(
                pares_mostrar["club_b"], busqueda
            )
            pares_mostrar = pares_mostrar[coincide]
        pares_mostrar = pares_mostrar.sort_values("probabilidad", ascending=False)

        st.caption(
            "✏️ La columna **Tipo** es editable: haz doble clic para corregir a mano el "
            "tipo de un par que el algoritmo haya clasificado mal. La revisión se guarda "
            f"en disco (`limpieza/datos/{duplicidad_equipos.RUTA_REVISIONES_MANUALES.name}`), así que "
            "sobrevive a recargar la página o reiniciar la app; se trata como una regla "
            "estricta (probabilidad 100 %, columna 'Revisado') y se refleja al instante "
            "tanto en las métricas de arriba "
            "como en el grafo de la sección siguiente. Para deshacerla, vuelve a elegir en "
            "'Tipo' el mismo valor que muestra 'Tipo original'. Solo se pueden revisar "
            "pares ya presentes en esta tabla — ajusta el filtro 'Tipo' para encontrar el "
            "par si lo acabas de reclasificar a un grupo oculto."
        )

        columnas_tabla = {
            "club_a": "Club A", "club_b": "Club B", "tipo": "Tipo",
            "revisado": "Revisado", "probabilidad": "Probabilidad",
            "origen_regla": "Origen",
            "origen_candidato": "Candidato por", "n_filas_a": "Filas club A",
            "n_filas_b": "Filas club B", "tipo_original": "Tipo original",
        }
        pares_editable = pares_mostrar.rename(columns=columnas_tabla)[list(columnas_tabla.values())]

        pares_editado = st.data_editor(
            pares_editable,
            use_container_width=True,
            hide_index=True,
            disabled=[c for c in columnas_tabla.values() if c != "Tipo"],
            column_config={
                "Tipo": st.column_config.SelectboxColumn("Tipo", options=tipos_disponibles, required=True),
                "Revisado": st.column_config.CheckboxColumn("Revisado"),
            },
            key="editor_pares_duplicidad",
        )

        # Solo las celdas que el usuario ha cambiado en ESTA edición: una
        # revisión que tras actualizar los datos coincide con el nuevo
        # veredicto del algoritmo se conserva (si no, se perdería en silencio
        # y no volvería a aplicarse si el algoritmo cambiase otra vez).
        cambios = False
        editadas = pares_editado["Tipo"] != pares_editable["Tipo"]
        for _, fila in pares_editado[editadas].iterrows():
            clave = duplicidad_equipos.clave_par(fila["Club A"], fila["Club B"])
            if fila["Tipo"] == fila["Tipo original"]:
                revisiones_manuales.pop(clave, None)
            else:
                revisiones_manuales[clave] = fila["Tipo"]
            cambios = True

        if cambios:
            duplicidad_equipos.guardar_revisiones_manuales(revisiones_manuales)
            # Las métricas y el grafo de más abajo ya se han dibujado en esta
            # misma pasada con el `pares` ANTERIOR a esta edición (el cambio
            # solo se conoce al leer `pares_editado`, más abajo en el
            # script). Forzamos un segundo rerun inmediato para que todo —
            # métricas incluidas — se recalcule ya con la revisión aplicada,
            # en vez de quedarse "un clic por detrás".
            st.rerun()

        if revisiones_manuales:
            c_resumen, c_reset = st.columns([5, 1])
            c_resumen.info(
                f"🔧 {len(revisiones_manuales)} par(es) con tipo revisado manualmente "
                "(guardado en disco: sobrevive a recargar la página y a actualizar los datos)."
            )
            _avisar_revisiones_no_aplicables(
                [f"{a} — {b}" for a, b in revisiones_no_aplicables], "par(es)", "alguno de sus clubes"
            )
            if c_reset.button(
                "Restablecer", help="Restablecer todas las revisiones manuales", key="reset_revisiones_duplicidad"
            ):
                _confirmar_reset_revisiones_manuales()

    st.divider()

    # -- 3. Redes de relaciones (filiales + duplicidades juntas) ------------
    st.subheader("Red de relaciones de un club")
    st.caption(
        "Componentes conexas de pares 'posible filial' Y 'posible duplicidad' JUNTOS — "
        "TODAS las relaciones detectadas, estén confirmadas por regla estricta o no, sin "
        "umbral mínimo de probabilidad — agrupados TRANSITIVAMENTE (si A-B y B-C son ambos "
        "filial, A, B y C quedan en una misma red aunque A-C nunca se haya comparado "
        "directamente). Cuando un club tiene relaciones hacia dos familias distintas (p. ej. "
        "'Atlético de Madrid B' con 100 % de ser filial de 'Atlético de Madrid' y 48 % de "
        "serlo de 'Real Madrid'), solo cuenta su relación de MAYOR probabilidad para el "
        "grafo — ese club aparece únicamente en la red de su mejor relación, nunca en la más "
        "débil. Línea continua = filial, discontinua = duplicidad. El club líder es el de "
        "mayor valor de mercado acumulado de plantilla de toda la red; tamaño del nodo = "
        "valor de mercado. Las revisiones manuales de la tabla de arriba (sección 2) se "
        "aplican aquí al instante: un par pasado a mano a 'ninguna relación' desaparece del "
        "grafo, y uno reclasificado a 'posible filial'/'posible duplicidad' se une a la red "
        "como si fuera una regla estricta más."
    )

    if pares.empty or not pares["tipo"].isin(["posible filial", "posible duplicidad"]).any():
        st.info("No se han detectado relaciones de filial ni de duplicidad con los umbrales actuales.")
    else:
        valor_por_club = calcular_valor_mercado_por_club(df)
        redes, aristas = duplicidad_equipos.resolver_redes_relaciones(pares, valor_por_club)

        if redes.empty:
            st.info("No se han detectado redes de 2 o más clubes.")
        else:
            lideres = sorted(redes.loc[redes["rol"] == "líder", "club"].unique())
            club_elegido = st.selectbox("Club líder", lideres, key="club_lider")

            miembros = redes[redes["club_lider"] == club_elegido]
            aristas_red = aristas[aristas["red_id"].isin(miembros["red_id"])]
            c_tabla, c_red = st.columns([1, 2])
            with c_tabla:
                st.dataframe(
                    miembros[["club", "rol", "valor_mercado_eur"]]
                    .sort_values("valor_mercado_eur", ascending=False)
                    .rename(
                        columns={
                            "club": "Club", "rol": "Rol",
                            "valor_mercado_eur": "Valor de mercado (€)",
                        }
                    ),
                    use_container_width=True, hide_index=True,
                )
            with c_red:
                fig_red = duplicidad_equipos.figura_red_relaciones(miembros, aristas_red, club_elegido)
                st.plotly_chart(fig_red, use_container_width=True)


def _tab_nacionalidad_y_liga(df: pd.DataFrame) -> None:
    st.caption(
        "`limpieza/nacionalidad_y_liga.py`: bandera asociada a cada país "
        "de `liga` (vía `competitions.csv`) y de `nacionalidad` en "
        "`tabla_valores_mercado`, y detección de los casos en los que esa "
        "asociación es ambigua (mismo nombre de liga en países distintos) "
        "o requiere alguna aproximación (nacionalidad sin bandera fiable "
        "propia). Cada bandera se muestra en dos formatos: como emoji "
        "Unicode y como imagen real (flagcdn.com). El emoji depende de la "
        "fuente de emoji del sistema operativo de quien mira la app — en "
        "Windows, históricamente, se ve el código de 2 letras en vez de "
        "la bandera — así que se añade también la imagen, que se ve igual "
        "en cualquier sistema (a costa de necesitar conexión a internet)."
    )

    # -- 1. Banderas por liga -------------------------------------------------
    st.subheader("Banderas por liga")
    st.caption(
        "`liga` es el NOMBRE de la competición (`competitions.name`), no "
        "su id, y ese nombre no es único por país en `competitions.csv`: "
        "cuando el mismo nombre de liga existe en más de un país (p. ej. "
        "'bundesliga' en Alemania Y Austria) — porque "
        "`ingest/procesar_datos.py` no conserva el `competition_id` "
        "original en la tabla final, así que ya no se puede saber a cuál "
        "de los dos pertenece cada fila generada — se genera UNA FILA POR "
        "PAÍS, cada una con su propia bandera, y se marcan en conflicto en "
        "vez de elegir uno al azar."
    )

    competitions = cargar_competitions_csv()
    tabla_ligas = calcular_tabla_banderas_ligas(df, competitions)

    n_ligas_conflicto = tabla_ligas.loc[tabla_ligas["conflicto"], "liga"].nunique()
    c1, c2 = st.columns(2)
    c1.metric("Ligas distintas", tabla_ligas["liga"].nunique())
    c2.metric("Ligas en conflicto", n_ligas_conflicto)

    st.dataframe(
        tabla_ligas.rename(
            columns={
                "liga": "Liga", "id_liga": "Id. de liga", "pais": "País",
                "bandera_emoji": "Emoji", "bandera_url": "Bandera",
                "n_registros": "Nº registros", "conflicto": "Conflicto",
            }
        ),
        use_container_width=True, hide_index=True,
        column_config={"Bandera": st.column_config.ImageColumn("Bandera", width="small")},
    )

    st.divider()

    # -- 2. Banderas por nacionalidad -----------------------------------------
    st.subheader("Banderas por nacionalidad")
    st.caption(
        "Bandera de cada valor de `nacionalidad`. A diferencia de `liga`, "
        "aquí no hay ambigüedad de nombre, pero sí casos marcados en "
        "**Conflicto** (columna `Motivo`): código no oficial, bandera "
        "aproximada (no exactamente la suya propia), o que en realidad "
        "son el MISMO país/territorio real que otra fila de esta misma "
        "tabla, con otra grafía (ver `NOTA_PAIS` en "
        "`limpieza/nacionalidad_y_liga.py`)."
    )

    tabla_nacionalidades = calcular_tabla_banderas_nacionalidades(df)
    n_nacionalidades_conflicto = int(tabla_nacionalidades["conflicto"].sum())
    c1, c2 = st.columns(2)
    c1.metric("Nacionalidades distintas", len(tabla_nacionalidades))
    c2.metric("Nacionalidades en conflicto", f"{n_nacionalidades_conflicto} de {len(tabla_nacionalidades)}")

    st.dataframe(
        tabla_nacionalidades.rename(
            columns={
                "nacionalidad": "Nacionalidad", "bandera_emoji": "Emoji",
                "bandera_url": "Bandera", "n_registros": "Nº registros",
                "conflicto": "Conflicto", "motivo": "Motivo",
            }
        ),
        use_container_width=True, hide_index=True,
        column_config={"Bandera": st.column_config.ImageColumn("Bandera", width="small")},
    )

    st.divider()

    # -- 3. Clubes sin liga: aproximar país por nacionalidad de plantilla ----
    st.subheader("Recuperar liga por nacionalidad del equipo")
    st.caption(
        "Para cada `equipo` SIN `liga` conocida, se aproxima su país con la "
        "nacionalidad más repetida de sus jugadores (por jugador distinto, "
        "no por fila). En caso de empate, se usa la nacionalidad del "
        "jugador de MENOR valor de mercado (su pico) entre los empatados. "
        "**Esto NO recupera la liga exacta ni distingue primera división "
        "de otras categorías**: un filial o un club descendido pueden "
        "tener nacionalidad mayoritaria perfectamente correcta sin jugar "
        "en la competición de primera de ese país. La columna "
        "**Nacionalidad** es editable a mano para corregir casos "
        "conocidos de error (ver la tabla de excepciones más abajo)."
    )

    tabla_sin_liga = calcular_tabla_clubes_sin_liga(df)

    if "revisiones_manuales_nacionalidad" not in st.session_state:
        # Igual que las revisiones de duplicidad: se cargan de disco (no
        # solo de `st.session_state`) para sobrevivir a un F5/reinicio.
        st.session_state["revisiones_manuales_nacionalidad"] = nacionalidad_y_liga.cargar_revisiones_nacionalidad()
    revisiones_nacionalidad = st.session_state["revisiones_manuales_nacionalidad"]

    tabla_sin_liga = tabla_sin_liga.copy()
    tabla_sin_liga["nacionalidad_original"] = tabla_sin_liga["nacionalidad"]
    tabla_sin_liga["revisado"] = tabla_sin_liga["equipo"].isin(revisiones_nacionalidad)
    tabla_sin_liga["nacionalidad"] = [
        revisiones_nacionalidad.get(equipo, original)
        for equipo, original in zip(tabla_sin_liga["equipo"], tabla_sin_liga["nacionalidad_original"])
    ]
    tabla_sin_liga["bandera_emoji"] = tabla_sin_liga["nacionalidad"].map(nacionalidad_y_liga.emoji_bandera)
    tabla_sin_liga["bandera_url"] = tabla_sin_liga["nacionalidad"].map(nacionalidad_y_liga.url_bandera)

    busqueda_club_sin_liga = st.text_input("Buscar equipo", key="busqueda_club_sin_liga")
    tabla_sin_liga_filtrada = (
        tabla_sin_liga[_coincide_busqueda(tabla_sin_liga["equipo"], busqueda_club_sin_liga)]
        if busqueda_club_sin_liga
        else tabla_sin_liga
    )

    c1, c2 = st.columns(2)
    c1.metric("Equipos sin liga", f"{len(tabla_sin_liga):,}")
    c2.metric("Mostrados con el filtro", f"{len(tabla_sin_liga_filtrada):,}")

    opciones_nacionalidad = [""] + sorted(nacionalidad_y_liga.PAIS_A_CODIGO.keys())
    columnas_tabla_sl = {
        "equipo": "Equipo", "n_registros": "Nº registros", "bandera_url": "Bandera",
        "nacionalidad": "Nacionalidad", "revisado": "Revisado",
        "nacionalidad_original": "Nacionalidad original",
    }
    tabla_sl_editable = tabla_sin_liga_filtrada.rename(columns=columnas_tabla_sl)[list(columnas_tabla_sl.values())]

    tabla_sl_editada = st.data_editor(
        tabla_sl_editable,
        use_container_width=True,
        hide_index=True,
        disabled=[c for c in columnas_tabla_sl.values() if c != "Nacionalidad"],
        column_config={
            "Nacionalidad": st.column_config.SelectboxColumn("Nacionalidad", options=opciones_nacionalidad),
            "Bandera": st.column_config.ImageColumn("Bandera", width="small"),
            "Revisado": st.column_config.CheckboxColumn("Revisado"),
        },
        key="editor_clubes_sin_liga",
    )

    # Solo las celdas cambiadas por el usuario (ver comentario equivalente en
    # la tabla de pares de duplicidad).
    cambios_nacionalidad = False
    editadas = tabla_sl_editada["Nacionalidad"] != tabla_sl_editable["Nacionalidad"]
    for _, fila in tabla_sl_editada[editadas].iterrows():
        equipo = fila["Equipo"]
        if fila["Nacionalidad"] == fila["Nacionalidad original"]:
            revisiones_nacionalidad.pop(equipo, None)
        else:
            revisiones_nacionalidad[equipo] = fila["Nacionalidad"]
        cambios_nacionalidad = True

    if cambios_nacionalidad:
        nacionalidad_y_liga.guardar_revisiones_nacionalidad(revisiones_nacionalidad)
        st.rerun()

    if revisiones_nacionalidad:
        c_resumen, c_reset = st.columns([5, 1])
        c_resumen.info(
            f"🔧 {len(revisiones_nacionalidad)} equipo(s) con nacionalidad revisada manualmente "
            "(guardado en disco: sobrevive a recargar la página y a actualizar los datos)."
        )
        # Un equipo revisado que ya no está en esta tabla pero SÍ en los datos
        # ha pasado a tener liga (la revisión ya no hace falta, pero se
        # conserva); solo se avisa de los que han desaparecido del todo.
        equipos_actuales = set(df["equipo"].dropna())
        _avisar_revisiones_no_aplicables(
            [e for e in revisiones_nacionalidad if e not in equipos_actuales], "equipo(s)", "el equipo"
        )
        if c_reset.button(
            "Restablecer",
            help="Restablecer todas las revisiones manuales de nacionalidad",
            key="reset_revisiones_nacionalidad",
        ):
            _confirmar_reset_revisiones_nacionalidad()

    st.divider()

    # -- 4. Clubes donde este método es conocido por fallar -------------------
    st.subheader("Clubes donde este método no es fiable")
    st.caption(
        "Tabla curada a mano (no calculada): clubes conocidos donde la "
        "nacionalidad más repetida de la plantilla NO coincide con su "
        "país/liga real, por dos motivos distintos — colisión de nombre "
        "con otro club real (p. ej. 'Arsenal FC' también es, en los datos "
        "de origen, el Arsenal de Sarandí argentino) o porque el club "
        "juega en la liga de OTRO país o recluta mayoritariamente de un "
        "país distinto al suyo (clubes canadienses en la MLS de EE.UU., "
        "neozelandeses en la A-League australiana, o el habitual flujo "
        "Portugal-Brasil). Si alguno de estos nombres aparece en la tabla "
        "de arriba, su nacionalidad calculada es sospechosa."
    )
    tabla_excepciones = nacionalidad_y_liga.tabla_clubes_excepcion()
    st.dataframe(
        tabla_excepciones.rename(
            columns={
                "equipo": "Equipo", "pais_real": "País real", "motivo": "Motivo",
                "bandera_emoji": "Emoji", "bandera_url": "Bandera",
            }
        ),
        use_container_width=True, hide_index=True,
        column_config={"Bandera": st.column_config.ImageColumn("Bandera", width="small")},
    )

    st.divider()

    # -- 5. Limpieza del caso "Arsenal FC": ¿Londres o Sarandí? ---------------
    st.subheader("Limpieza del caso «Arsenal FC»: ¿Londres o Sarandí?")
    st.caption(
        "El texto `equipo == 'Arsenal FC'` mezcla en los datos de origen "
        "dos clubes reales distintos: el Arsenal de Londres y el Arsenal "
        "de Sarandí (Argentina). Para cada jugador con algún registro de "
        "'Arsenal FC', se cuenta cuántos registros tiene en equipos "
        "ingleses (`liga == 'premier-league'`) y en equipos argentinos "
        "(`liga == 'torneo-apertura'`) — pero SOLO en sus OTROS clubes: "
        "las propias filas de 'Arsenal FC' siempre llevan "
        "`liga == 'premier-league'`, así que contarlas habría sido "
        "circular (todo jugador habría sumado 'premier-league' solo por "
        "tener una fila de 'Arsenal FC', sin decir nada real). La columna "
        "**Clasificación** es editable a mano para corregir casos que el "
        "criterio automático (sin evidencia en ninguna de las dos ligas, "
        "o alguna otra situación atípica) haya dejado mal resuelto."
    )

    tabla_arsenal = calcular_tabla_jugadores_arsenal_fc(df)

    if "revisiones_manuales_arsenal_fc" not in st.session_state:
        st.session_state["revisiones_manuales_arsenal_fc"] = nacionalidad_y_liga.cargar_revisiones_arsenal_fc()
    revisiones_arsenal = st.session_state["revisiones_manuales_arsenal_fc"]

    tabla_arsenal = tabla_arsenal.copy()
    tabla_arsenal["bandera_url"] = tabla_arsenal["nacionalidad"].map(nacionalidad_y_liga.url_bandera)
    tabla_arsenal["clasificacion_original"] = tabla_arsenal["clasificacion"]
    tabla_arsenal["revisado"] = tabla_arsenal["id_player"].isin(revisiones_arsenal)
    tabla_arsenal["clasificacion"] = [
        revisiones_arsenal.get(id_player, original)
        for id_player, original in zip(tabla_arsenal["id_player"], tabla_arsenal["clasificacion_original"])
    ]

    n_argentina = int((tabla_arsenal["clasificacion"] == "Argentina (Sarandí)").sum())
    n_inglaterra = int((tabla_arsenal["clasificacion"] == "Inglaterra (Londres)").sum())
    n_sin_evidencia = int((tabla_arsenal["clasificacion"] == "Sin evidencia").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jugadores en 'Arsenal FC'", len(tabla_arsenal))
    c2.metric("Arsenal de Sarandí", n_argentina)
    c3.metric("Arsenal de Londres", n_inglaterra)
    c4.metric("Sin evidencia", n_sin_evidencia)

    busqueda_arsenal = st.text_input("Buscar jugador o nacionalidad", key="busqueda_jugador_arsenal")
    tabla_arsenal_filtrada = (
        tabla_arsenal[
            _coincide_busqueda(tabla_arsenal["nombre"], busqueda_arsenal)
            | _coincide_busqueda(tabla_arsenal["nacionalidad"], busqueda_arsenal)
        ]
        if busqueda_arsenal
        else tabla_arsenal
    )

    columnas_arsenal = {
        "nombre": "Nombre", "nacionalidad": "Nacionalidad", "bandera_url": "Bandera",
        "n_registros_premier_league": "Nº registros Premier League",
        "n_registros_torneo_apertura": "Nº registros Torneo Apertura",
        "clasificacion": "Clasificación", "revisado": "Revisado",
        "clasificacion_original": "Clasificación original", "id_player": "id_player",
    }
    tabla_arsenal_editable = tabla_arsenal_filtrada.rename(columns=columnas_arsenal)[list(columnas_arsenal.values())]

    tabla_arsenal_editada = st.data_editor(
        tabla_arsenal_editable,
        use_container_width=True,
        hide_index=True,
        disabled=[c for c in columnas_arsenal.values() if c != "Clasificación"],
        column_config={
            "Clasificación": st.column_config.SelectboxColumn(
                "Clasificación", options=list(nacionalidad_y_liga.CLASIFICACIONES_ARSENAL_FC), required=True
            ),
            "Bandera": st.column_config.ImageColumn("Bandera", width="small"),
            "Revisado": st.column_config.CheckboxColumn("Revisado"),
            "id_player": None,
        },
        key="editor_jugadores_arsenal_fc",
    )

    # Solo las celdas cambiadas por el usuario (ver comentario equivalente en
    # la tabla de pares de duplicidad).
    cambios_arsenal = False
    editadas = tabla_arsenal_editada["Clasificación"] != tabla_arsenal_editable["Clasificación"]
    for _, fila in tabla_arsenal_editada[editadas].iterrows():
        id_player = int(fila["id_player"])
        if fila["Clasificación"] == fila["Clasificación original"]:
            revisiones_arsenal.pop(id_player, None)
        else:
            revisiones_arsenal[id_player] = fila["Clasificación"]
        cambios_arsenal = True

    if cambios_arsenal:
        nacionalidad_y_liga.guardar_revisiones_arsenal_fc(revisiones_arsenal)
        st.rerun()

    if revisiones_arsenal:
        c_resumen, c_reset = st.columns([5, 1])
        c_resumen.info(
            f"🔧 {len(revisiones_arsenal)} jugador(es) con clasificación revisada manualmente "
            "(guardado en disco: sobrevive a recargar la página y a actualizar los datos)."
        )
        ids_actuales = set(tabla_arsenal["id_player"])
        _avisar_revisiones_no_aplicables(
            [f"id_player {i}" for i in revisiones_arsenal if i not in ids_actuales],
            "jugador(es)", "el jugador (en 'Arsenal FC')",
        )
        if c_reset.button(
            "Restablecer",
            help="Restablecer todas las revisiones manuales de clasificación",
            key="reset_revisiones_arsenal_fc",
        ):
            _confirmar_reset_revisiones_arsenal_fc()


def _grafico_evolucion_jugador(
    df: pd.DataFrame, id_player, nombre: str, fechas_destacar: list, titulo_extra: str = ""
) -> "px.Figure":
    """Línea temporal de `valor_mercado_eur` de un único jugador, con los
    puntos de `fechas_destacar` (el caso anómalo bajo revisión) resaltados
    en rojo sobre el resto de su historial, para juzgar de un vistazo si el
    cambio detectado encaja o desentona con el resto de su trayectoria."""
    datos = df[df["id_player"] == id_player].sort_values("fecha")
    fig = px.line(
        datos,
        x="fecha",
        y="valor_mercado_eur",
        markers=True,
        labels={"fecha": "Fecha", "valor_mercado_eur": "Valor de mercado (€)"},
        title=f"Evolución de {nombre}{titulo_extra}",
    )
    destacados = datos[datos["fecha"].isin(fechas_destacar)]
    fig.add_scatter(
        x=destacados["fecha"],
        y=destacados["valor_mercado_eur"],
        mode="markers",
        marker=dict(color="red", size=14, symbol="x"),
        name="Caso señalado",
    )
    return fig


def _tab_valores_erroneos(df: pd.DataFrame) -> None:
    st.caption(
        "`limpieza/valores_erroneos.py`: a diferencia del resto de "
        "pestañas (que tratan errores en `equipo`/`liga`/`nacionalidad` o "
        "la ausencia de datos), aquí se estudia si la propia cifra de "
        "`valor_mercado_eur` es plausible — cambios de valor muy bruscos "
        "en poco tiempo, y picos aislados que se revierten por completo. "
        "Ninguna comprobación corrige nada automáticamente: son "
        "candidatos a revisar, no errores confirmados. (También se "
        "estudiaron los valores negativos/cero y los registros duplicados "
        "del mismo jugador en el mismo instante, pero no aparece ni un "
        "solo caso en `tabla_valores_mercado`, así que se retiraron de "
        "esta pestaña por no aportar nada.)"
    )

    resumen = valores_erroneos.resumen_valores_erroneos(df)
    columnas_resumen = st.columns(len(resumen))
    for columna, (_, fila) in zip(columnas_resumen, resumen.iterrows()):
        columna.metric(fila["caso"], f"{int(fila['n_casos']):,}")

    st.divider()

    # -- 1. Cambios de valor anómalos en un periodo corto ----------------------
    st.subheader("Cambios de valor anómalos en un periodo corto")
    st.caption(
        "Entre dos valoraciones consecutivas del mismo jugador: el valor "
        "se multiplica (o divide) por encima del umbral elegido en, como "
        "mucho, el nº de días elegido. No hay un umbral objetivamente "
        "'correcto' — cuanto más corto el periodo y más alto el "
        "multiplicador, menos casos quedan pero más claramente atípicos "
        "son. **Aviso**: con los parámetros por defecto, casi la mitad de "
        "los casos parten de un valor anterior de exactamente 50.000 €, el "
        "mínimo con el que Transfermarkt valora a un jugador debutante "
        "antes de su primera valoración real — un salto brusco desde esa "
        "cifra concreta no es necesariamente un error, sino ese primer "
        "'estreno' de mercado."
    )

    c1, c2 = st.columns(2)
    dias_max_cambios = c1.slider(
        "Días máximos entre valoraciones", 1, 180, 30, key="dias_max_cambios"
    )
    ratio_umbral_cambios = c2.slider(
        "Multiplicador mínimo (subida o bajada)", 1.5, 10.0, 3.0, step=0.5, key="ratio_umbral_cambios"
    )

    tabla_cambios = calcular_cambios_anomalos(df, dias_max_cambios, ratio_umbral_cambios)
    st.metric("Cambios anómalos detectados", f"{len(tabla_cambios):,}")

    if tabla_cambios.empty:
        st.success("No se ha detectado ningún cambio de valor anómalo con estos parámetros.")
    else:
        fig_hist_cambios = px.histogram(
            tabla_cambios,
            x="multiplicador",
            color="tipo",
            nbins=30,
            labels={"multiplicador": "Multiplicador (subida o bajada)", "tipo": "Tipo"},
            title="Distribución del multiplicador de los cambios anómalos",
        )
        st.plotly_chart(fig_hist_cambios, use_container_width=True)

        columnas_cambios = {
            "nombre": "Nombre", "equipo_anterior": "Equipo anterior", "equipo_actual": "Equipo actual",
            "fecha_anterior": "Fecha anterior", "fecha": "Fecha", "dias": "Días",
            "valor_anterior": "Valor anterior (€)", "valor_actual": "Valor actual (€)",
            "multiplicador": "Multiplicador", "tipo": "Tipo", "id_player": "id_player",
        }
        st.dataframe(
            tabla_cambios.rename(columns=columnas_cambios)[list(columnas_cambios.values())],
            use_container_width=True, hide_index=True,
        )

        opciones_cambios = {
            f"{fila['nombre']} — {fila['fecha_anterior'].date()} → {fila['fecha'].date()} (x{fila['multiplicador']:.1f})": indice
            for indice, fila in tabla_cambios.head(200).iterrows()
        }
        seleccion_cambio = st.selectbox(
            "Ver la evolución completa de un caso (los 200 más extremos)",
            list(opciones_cambios.keys()),
            key="seleccion_cambio_anomalo",
        )
        fila_cambio = tabla_cambios.loc[opciones_cambios[seleccion_cambio]]
        st.plotly_chart(
            _grafico_evolucion_jugador(
                df, fila_cambio["id_player"], fila_cambio["nombre"],
                [fila_cambio["fecha_anterior"], fila_cambio["fecha"]],
            ),
            use_container_width=True,
        )

    st.divider()

    # -- 2. Picos aislados ------------------------------------------------------
    st.subheader("Picos aislados (suben o bajan y se revierten por completo)")
    st.caption(
        "Caso más específico que el anterior: el valor sube o baja para, "
        "en la valoración inmediatamente siguiente, volver EXACTAMENTE a "
        "la cifra de antes. Una revalorización real rara vez se deshace "
        "por completo de un día para otro, así que este patrón es un "
        "candidato más claro a error de introducción de datos que un "
        "cambio brusco cualquiera."
    )

    dias_max_picos = st.slider(
        "Días máximos entre la valoración anterior y la siguiente",
        30, 365, 180, step=10, key="dias_max_picos",
    )
    tabla_picos = calcular_picos_aislados(df, dias_max_picos)
    st.metric("Picos aislados detectados", f"{len(tabla_picos):,}")

    if tabla_picos.empty:
        st.success("No se ha detectado ningún pico aislado con este parámetro.")
    else:
        columnas_picos = {
            "nombre": "Nombre", "equipo": "Equipo", "fecha_anterior": "Fecha anterior",
            "fecha": "Fecha del pico", "fecha_siguiente": "Fecha siguiente",
            "valor_anterior": "Valor anterior (€)", "valor_pico": "Valor del pico (€)",
            "dias_totales": "Días (anterior → siguiente)", "tipo": "Tipo", "id_player": "id_player",
        }
        st.dataframe(
            tabla_picos.rename(columns=columnas_picos)[list(columnas_picos.values())],
            use_container_width=True, hide_index=True,
        )

        opciones_picos = {
            f"{fila['nombre']} — pico en {fila['fecha'].date()}": indice
            for indice, fila in tabla_picos.head(200).iterrows()
        }
        seleccion_pico = st.selectbox(
            "Ver la evolución completa de un caso (los 200 más cortos)",
            list(opciones_picos.keys()),
            key="seleccion_pico_aislado",
        )
        fila_pico = tabla_picos.loc[opciones_picos[seleccion_pico]]
        st.plotly_chart(
            _grafico_evolucion_jugador(
                df, fila_pico["id_player"], fila_pico["nombre"], [fila_pico["fecha"]],
            ),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Pestaña: Compilación (aplica todo lo anterior -> tabla limpia)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_tabla_limpia(marca_tiempo: float) -> pd.DataFrame | None:
    """`marca_tiempo` (fecha de modificación del parquet) invalida la caché
    cada vez que se vuelve a compilar."""
    return compilacion.cargar_tabla_limpia()


def _tabla_compacta(tabla: pd.DataFrame, columnas: dict[str, str] | None = None) -> None:
    if columnas:
        tabla = tabla[list(columnas)].rename(columns=columnas)
    st.dataframe(tabla, use_container_width=True, hide_index=True)


def _resumen_compilacion(informe: dict) -> None:
    pasos = list(informe["pasos"].values())
    p1, p2, p3, p4, p5, p6 = pasos

    def _filas(paso: dict) -> str:
        diferencia = paso["filas_despues"] - paso["filas_antes"]
        return f"Registros: {paso['filas_antes']:,} → {paso['filas_despues']:,}" + (
            f" ({diferencia:+,})" if diferencia else " (sin cambios)"
        )

    # -- 1 ----------------------------------------------------------------
    st.subheader("1. Falsos valores faltantes → NaN, y filas sin edad o con edad/altura atípicas")
    st.caption(
        "Como en *Datos faltantes*: los valores de `VALORES_NA_DISFRAZADO` pasan a NaN real. Además se eliminan "
        f"las filas sin `edad` (jugador sin fecha de nacimiento en `players.csv`), con edad ≤ "
        f"{compilacion.EDAD_MINIMA} años o con altura < {compilacion.ALTURA_MINIMA_CM} cm (las filas sin altura "
        "se conservan). " + _filas(p1)
    )
    por_motivo = p1.get("eliminados_por_motivo", pd.DataFrame(columns=["motivo", "registros", "jugadores"]))
    c1, c2 = st.columns(2)
    c1.metric("Celdas convertidas a NaN", f"{p1['n_celdas']:,}")
    c2.metric("Registros eliminados", f"{p1.get('n_eliminados', 0):,}")
    c_na, c_elim = st.columns(2, gap="large")
    with c_na:
        _tabla_compacta(p1["na_disfrazados"], {"columna": "Columna", "valor_disfrazado": "Valor", "n_filas": "Registros"})
    with c_elim:
        _tabla_compacta(por_motivo, {"motivo": "Motivo de eliminación", "registros": "Registros", "jugadores": "Jugadores"})
    eliminados = p1.get("registros_eliminados")
    if eliminados is not None and len(eliminados):
        with st.expander(f"Jugadores con registros eliminados ({eliminados['id_player'].nunique()})"):
            _tabla_compacta(eliminados, {"nombre": "Jugador", "motivo": "Motivo", "registros": "Registros",
                                         "edad": "Edad mínima", "altura_cm": "Altura (cm)"})

    # -- 2 ----------------------------------------------------------------
    st.subheader("2. Separación de «Arsenal FC»")
    st.caption(
        "Con la tabla del caso Arsenal de *Nacionalidades y ligas* (incluidas tus revisiones manuales): "
        "los registros de 'Arsenal FC' de los jugadores clasificados como «Argentina (Sarandí)» pasan a "
        f"'{compilacion.EQUIPO_ARSENAL_SARANDI}' con liga argentina. " + _filas(p2)
    )
    clasif = p2["por_clasificacion"].set_index("clasificacion")["jugadores"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jugadores en 'Arsenal FC'", int(clasif.sum()))
    c2.metric("Argentina (Sarandí)", int(clasif.get("Argentina (Sarandí)", 0)))
    c3.metric("Inglaterra (Londres)", int(clasif.get("Inglaterra (Londres)", 0)))
    c4.metric("Registros cambiados", f"{p2['n_registros']:,}")
    with st.expander(f"Jugadores movidos a '{compilacion.EQUIPO_ARSENAL_SARANDI}' ({len(p2['jugadores_movidos'])})"):
        _tabla_compacta(
            p2["jugadores_movidos"],
            {"nombre": "Jugador", "nacionalidad": "Nacionalidad", "registros_cambiados": "Registros cambiados",
             "revisado": "Revisado a mano"},
        )

    # -- 3 ----------------------------------------------------------------
    st.subheader("3. Equipo y nacionalidad faltantes")
    st.caption(
        "Recuperación de `equipo` por el historial del jugador (equipo más cercano en fecha; en empate, "
        "el previo; jugadores sin ningún equipo, eliminados) — la liga del registro imputado pasa a ser "
        "la del club imputado —, fusión de nacionalidades que son el mismo país con dos grafías y "
        "eliminación de los registros sin nacionalidad. " + _filas(p3)
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equipos imputados", f"{p3['n_imputados']:,}", f"{p3['n_imputados_con_liga']:,} con liga", delta_color="off")
    sin_equipo = p3["jugadores_sin_equipo"]
    c2.metric("Jugadores sin ningún equipo", f"{len(sin_equipo):,}",
              f"-{int(sin_equipo['n_registros'].sum()):,} registros", delta_color="off")
    c3.metric("Nacionalidades fusionadas", f"{int(p3['fusion_nacionalidades']['registros'].sum()):,} registros")
    sin_nac = p3["jugadores_sin_nacionalidad"]
    c4.metric("Jugadores sin nacionalidad", f"{len(sin_nac):,}",
              f"-{int(sin_nac['registros'].sum()):,} registros", delta_color="off")
    _tabla_compacta(
        p3["imputacion_resumen"],
        {"valor_original": "Valor original de equipo", "registros": "Registros imputados",
         "pct_equipo_previo": "% con equipo previo", "mediana_dias": "Distancia mediana (días)"},
    )
    _tabla_compacta(
        p3["fusion_nacionalidades"],
        {"nacionalidad_original": "Nacionalidad original", "nacionalidad_nueva": "Fusionada en", "registros": "Registros"},
    )
    with st.expander("Jugadores eliminados (sin ningún equipo o sin nacionalidad)"):
        st.markdown("**Sin ningún equipo en su historial**")
        _tabla_compacta(sin_equipo, {"nombre": "Jugador", "n_registros": "Registros", "valores_originales": "Valores de equipo"})
        st.markdown("**Sin nacionalidad**")
        _tabla_compacta(sin_nac, {"nombre": "Jugador", "registros": "Registros"})
    st.caption(f"Tras este paso: {p3['equipo_nan_final']} registros sin equipo y {p3['nacionalidad_nan_final']} sin nacionalidad.")

    # -- 4 ----------------------------------------------------------------
    st.subheader("4. Liga → país, y columna «Bandera»")
    st.caption(
        "Cada liga pasa a ser el país de *Banderas por liga*, y se añade su bandera. En las ligas cuyo nombre "
        "comparten dos países (marcadas en conflicto) el país se decide por el club de cada registro, "
        "cruzándolo con `clubs.csv`. " + _filas(p4)
    )
    _tabla_compacta(
        p4["conversion"],
        {"liga_original": "Liga original", "pais": "País", "registros": "Registros", "conflicto": "Nombre compartido"},
    )
    if p4["sin_resolver"]:
        st.warning(f"{len(p4['sin_resolver'])} combinaciones club/liga no se pudieron resolver por club (se usó el primer país).")

    # -- 5 ----------------------------------------------------------------
    st.subheader("5. Ligas por nacionalidad y fusión de equipos")
    st.caption(_filas(p5))

    st.markdown(f"**5.1 Liga por nacionalidad a clubes sin liga con ≥ {compilacion.UMBRAL_REGISTROS_LIGA_5_1} registros**")
    a51 = p5["asignacion_51"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Clubes", f"{len(a51):,}")
    c2.metric("Registros", f"{int(a51['registros'].sum()):,}")
    c3.metric("Por revisión manual / excepción", f"{int((a51['fuente'] != 'nacionalidad').sum()):,}")
    with st.expander("Clubes y país asignado (5.1)"):
        _tabla_compacta(a51, {"equipo": "Club", "pais": "País asignado", "fuente": "Fuente", "registros": "Registros"})

    st.markdown("**5.2 Duplicidad de equipos y canteras: fusión por redes**")
    st.caption(
        "Las fusiones siguen exactamente los grafos de la pestaña *Duplicidad de equipos y canteras* "
        "(calculados sobre la tabla original, con tus revisiones manuales y su regla de no unir por fórmula "
        "dos familias con algún par 'ninguna relación' entre ellas). Las ligas imputadas en los pasos 4 y "
        "5.1 no influyen en qué clubes se fusionan."
    )
    cambios = p5["cambios_52"]
    es_dup = cambios["motivo"] == "duplicidad del líder"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pares evaluados", f"{p5['n_pares']:,}", f"{p5['n_revisiones_aplicadas']} revisados a mano", delta_color="off")
    c2.metric("Redes", f"{p5['n_redes']:,}")
    c3.metric("Duplicados → nombre del líder", f"{int(es_dup.sum()):,}",
              f"{int(cambios.loc[es_dup, 'registros'].sum()):,} registros", delta_color="off")
    c4.metric(f"Resto → «líder{compilacion.SUFIJO_FILIAL}»", f"{int((~es_dup).sum()):,}",
              f"{int(cambios.loc[~es_dup, 'registros'].sum()):,} registros", delta_color="off")
    debiles = cambios[(cambios["evidencia"] == "fórmula") & (cambios["probabilidad"] < 0.5)]
    st.caption(
        "Evidencia de cada fusión (la relación más fuerte que une al club con su red): "
        + " · ".join(f"{k}: {v:,}" for k, v in cambios["evidencia"].value_counts().items())
        + f". ⚠️ {len(debiles):,} fusiones se apoyan solo en una fórmula con probabilidad < 50 %: "
        "si alguna es incorrecta, márcala como 'ninguna relación' en la tabla de pares de *Duplicidad de "
        "equipos y canteras* y vuelve a compilar."
    )
    with st.expander("Clubes fusionados (5.2)"):
        busqueda_52 = st.text_input("Buscar club", key="comp_busqueda_52")
        cambios_mostrar = cambios
        if busqueda_52:
            cambios_mostrar = cambios[
                _coincide_busqueda(cambios["equipo_original"], busqueda_52)
                | _coincide_busqueda(cambios["club_lider"], busqueda_52)
            ]
        _tabla_compacta(
            cambios_mostrar,
            {"equipo_original": "Club original", "equipo_nuevo": "Nuevo nombre", "motivo": "Motivo",
             "registros": "Registros", "liga_del_lider": "Liga del líder", "evidencia": "Evidencia",
             "probabilidad": "Probabilidad"},
        )

    st.markdown("**5.3 Liga por nacionalidad al resto de clubes sin liga**")
    a53 = p5["asignacion_53"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros completados con la liga de su propio club", f"{p5['n_completados_club']:,}")
    c2.metric("Clubes por nacionalidad", f"{len(a53):,}")
    c3.metric("Registros", f"{int(a53['registros'].sum()):,}")
    with st.expander("Clubes y país asignado (5.3)"):
        _tabla_compacta(a53, {"equipo": "Club", "pais": "País asignado", "fuente": "Fuente", "registros": "Registros"})
    st.caption(f"Tras este paso: {p5['liga_nan_final']} registros sin liga.")

    # -- 6 ----------------------------------------------------------------
    st.subheader("6. Valores de mercado erróneos")
    st.caption(
        f"Cambios anómalos de *Valores erróneos* (≤ {compilacion.DIAS_MAX_CAMBIOS} días, ×{compilacion.RATIO_UMBRAL_CAMBIOS:g}): "
        "si el salto parte del primer registro del jugador, ese registro se elimina; si cae en un pico entre "
        f"dos valores iguales (≤ {compilacion.DIAS_MAX_PICOS} días), se sustituye por la media de ambos. " + _filas(p6)
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cambios anómalos", f"{p6['n_anomalos']:,}")
    c2.metric("Primeros registros eliminados", f"{len(p6['eliminados_primer_registro']):,}")
    c3.metric("Picos corregidos", f"{len(p6['picos_corregidos']):,}")
    c4.metric("Sin regla aplicable (se mantienen)", f"{p6['n_sin_regla']:,}")
    with st.expander("Detalle de los valores eliminados y corregidos"):
        st.markdown("**Primer registro eliminado**")
        _tabla_compacta(
            p6["eliminados_primer_registro"],
            {"nombre": "Jugador", "equipo": "Equipo", "fecha": "Fecha", "valor_mercado_eur": "Valor (€)"},
        )
        st.markdown("**Picos corregidos**")
        _tabla_compacta(
            p6["picos_corregidos"],
            {"nombre": "Jugador", "equipo": "Equipo", "fecha": "Fecha", "valor_anterior": "Valor anterior",
             "valor_original": "Valor del pico", "valor_siguiente": "Valor siguiente", "valor_nuevo": "Valor corregido"},
        )


def _explorador_tabla_limpia(limpia: pd.DataFrame) -> None:
    """Mismo explorador que la página 'Explorador de datos', sobre la tabla
    limpia y con los filtros dentro de la pestaña (la barra lateral es de la
    página entera)."""
    def opciones(col: str):
        return sorted(limpia[col].dropna().unique().tolist())

    c1, c2, c3, c4 = st.columns(4)
    paises = c1.multiselect("País (nacionalidad)", opciones("nacionalidad"), key="comp_f_nacionalidad")
    ligas = c2.multiselect("Liga (país)", opciones("liga"), key="comp_f_liga")
    equipos = c3.multiselect("Equipo", opciones("equipo"), key="comp_f_equipo")
    posiciones = c4.multiselect("Posición", opciones("posicion"), key="comp_f_posicion")

    c1, c2, c3 = st.columns([2, 2, 3])
    nombre_busqueda = c1.text_input("Buscar jugador por nombre", key="comp_f_nombre")
    fecha_min, fecha_max = limpia["fecha"].min(), limpia["fecha"].max()
    rango_fechas = c2.date_input(
        "Rango de fechas", value=(fecha_min, fecha_max), min_value=fecha_min, max_value=fecha_max, key="comp_f_fechas"
    )
    valor_min, valor_max = int(limpia["valor_mercado_eur"].min()), int(limpia["valor_mercado_eur"].max())
    rango_valor = c3.slider(
        "Valor de mercado (€)", min_value=valor_min, max_value=valor_max, value=(valor_min, valor_max),
        step=100_000, format="%d", key="comp_f_valor",
    )

    filtrado = limpia
    if paises:
        filtrado = filtrado[filtrado["nacionalidad"].isin(paises)]
    if ligas:
        filtrado = filtrado[filtrado["liga"].isin(ligas)]
    if equipos:
        filtrado = filtrado[filtrado["equipo"].isin(equipos)]
    if posiciones:
        filtrado = filtrado[filtrado["posicion"].isin(posiciones)]
    if nombre_busqueda:
        filtrado = filtrado[_coincide_busqueda(filtrado["nombre"], nombre_busqueda)]
    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        inicio, fin = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
        filtrado = filtrado[(filtrado["fecha"] >= inicio) & (filtrado["fecha"] <= fin)]
    filtrado = filtrado[
        (filtrado["valor_mercado_eur"] >= rango_valor[0]) & (filtrado["valor_mercado_eur"] <= rango_valor[1])
    ]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros", f"{len(filtrado):,}")
    k2.metric("Jugadores únicos", f"{filtrado['id_player'].nunique():,}")
    k3.metric("Valor medio (€)", f"{filtrado['valor_mercado_eur'].mean():,.0f}" if len(filtrado) else "-")
    k4.metric("Valor máximo (€)", f"{filtrado['valor_mercado_eur'].max():,.0f}" if len(filtrado) else "-")

    st.dataframe(
        filtrado.sort_values("fecha", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={"Bandera": st.column_config.ImageColumn("Bandera", width="small")},
    )
    st.download_button(
        "⬇️ Descargar CSV filtrado",
        data=filtrado.to_csv(index=False).encode("utf-8"),
        file_name="valores_mercado_limpia_filtrado.csv",
        mime="text/csv",
        key="comp_descarga",
    )

    st.markdown("#### Evolución del valor de mercado")
    jugadores = sorted(filtrado["nombre"].dropna().unique().tolist())
    if not jugadores:
        st.info("No hay jugadores que coincidan con los filtros seleccionados.")
        return
    seleccion = st.multiselect(
        "Selecciona uno o varios jugadores para ver su evolución", jugadores,
        default=jugadores[: min(3, len(jugadores))], key="comp_jugadores_grafico",
    )
    if seleccion:
        datos_grafico = filtrado[filtrado["nombre"].isin(seleccion)].sort_values("fecha")
        fig = px.line(
            datos_grafico, x="fecha", y="valor_mercado_eur", color="nombre", markers=True,
            hover_data=["equipo", "liga"],
            labels={"fecha": "Fecha", "valor_mercado_eur": "Valor de mercado (€)", "nombre": "Jugador"},
        )
        st.plotly_chart(fig, use_container_width=True)


def _tab_compilacion() -> None:
    st.caption(
        "`limpieza/compilacion.py`: aplica, en orden, lo visto en las pestañas anteriores (incluidas tus "
        "revisiones manuales) sobre `tabla_valores_mercado` y guarda la tabla limpia en "
        f"`ingest/datos/{compilacion.ARCHIVO_LIMPIA_PARQUET.name}` y `.csv`. Las demás pestañas no se "
        "modifican: siguen mostrando la tabla original. Tras cambiar una revisión manual, vuelve a compilar."
    )

    existe = compilacion.ARCHIVO_LIMPIA_PARQUET.exists() and compilacion.ARCHIVO_INFORME.exists()
    c_boton, c_estado = st.columns([1, 3])
    if c_boton.button("▶️ Compilar" if not existe else "🔄 Volver a compilar", type="primary", key="comp_ejecutar"):
        barra = st.progress(0.0, text="Compilando...")
        total = len(compilacion.PASOS)
        compilacion.compilar(progreso=lambda i, nombre: barra.progress(i / total, text=f"Paso {nombre}"))
        barra.progress(1.0, text="Compilación terminada")
        cargar_tabla_limpia.clear()
        st.rerun()
    if not existe:
        c_estado.info("Todavía no se ha compilado la tabla limpia (tarda alrededor de un minuto).")
        return

    marca = compilacion.ARCHIVO_LIMPIA_PARQUET.stat().st_mtime
    c_estado.caption(f"Última compilación: {pd.Timestamp(marca, unit='s', tz='UTC').tz_convert('Europe/Madrid'):%d/%m/%Y %H:%M}")
    informe = compilacion.cargar_informe()

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{informe['filas_final']:,}", f"{informe['filas_final'] - informe['filas_inicial']:+,}")
    c2.metric("Jugadores", f"{informe['jugadores_final']:,}", f"{informe['jugadores_final'] - informe['jugadores_inicial']:+,}")
    nulos = {k: v for k, v in informe["nulos_final"].items() if v}
    c3.metric("Columnas con NaN en la tabla final", len(nulos))
    if nulos:
        st.caption(
            "NaN restantes: " + ", ".join(f"`{k}` ({v:,})" for k, v in nulos.items())
            + " — ningún paso del recorrido los trata (p. ej. `posicion` = 'Missing', convertido a NaN en el paso 1)."
        )

    st.divider()
    _resumen_compilacion(informe)

    st.divider()
    st.subheader("Tabla final limpia")
    limpia = cargar_tabla_limpia(marca)
    _explorador_tabla_limpia(limpia)


def pagina_limpieza(df: pd.DataFrame) -> None:
    st.title("🧹 Limpieza de datos")
    st.caption(
        "Cada pestaña corresponde a un módulo del paquete `limpieza/` todas son de detección y, cuando es posible, de estimación de recuperabilidad."
    )

    tab_faltantes, tab_nacionalidad_liga, tab_duplicidad, tab_valores_erroneos, tab_compilacion = st.tabs(
        [
            "❓ Datos faltantes",
            "🌍 Nacionalidades y ligas",
            "🔁 Duplicidad de equipos y canteras",
            "📉 Valores erróneos",
            "🧩 Compilación",
        ]
    )

    with tab_faltantes:
        _tab_valores_faltantes(df)
    with tab_nacionalidad_liga:
        _tab_nacionalidad_y_liga(df)
    with tab_duplicidad:
        _tab_duplicidad_equipos(df)
    with tab_valores_erroneos:
        _tab_valores_erroneos(df)
    with tab_compilacion:
        _tab_compilacion()


# ---------------------------------------------------------------------------
# Enrutado
# ---------------------------------------------------------------------------
if pagina == "🔎 Explorador de datos":
    pagina_explorador(df)
elif pagina == "📊 Visualización de datos":
    pagina_visualizacion(df)
elif pagina == "🧹 Limpieza de datos":
    pagina_limpieza(df)
