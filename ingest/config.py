"""
Configuración central del proyecto.

Fuente de datos: repositorio público de dcaribou/transfermarkt-datasets
https://github.com/dcaribou/transfermarkt-datasets

El propio README del repositorio recomienda consumir los CSV ya preparados
directamente desde su bucket público (se actualiza automáticamente cada
semana junto con el repositorio), en lugar de clonar el repo entero:

    https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/<tabla>.csv.gz

Ese es el mecanismo que usamos aquí. Si en el futuro esa URL cambiara,
basta con actualizar BASE_URL_R2 (y/o BASE_URL_GITHUB como alternativa).
"""

import os

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_DATOS = os.path.join(DIR_BASE, "datos")
DIR_RAW = os.path.join(DIR_DATOS, "raw")
ARCHIVO_METADATOS = os.path.join(DIR_RAW, "_metadatos_descarga.json")
ARCHIVO_TABLA_FINAL = os.path.join(DIR_DATOS, "tabla_valores_mercado.csv")
ARCHIVO_TABLA_FINAL_PARQUET = os.path.join(DIR_DATOS, "tabla_valores_mercado.parquet")
# Tabla de traspasos, construida ÚNICAMENTE a partir de
# player_valuations.csv (+ clubs.csv y competitions.csv para la liga); ver
# `construir_traspasos()` en procesar_datos.py. Vive en DIR_DATOS (NO en
# DIR_RAW) porque es un resultado derivado, no un fichero crudo descargado.
ARCHIVO_TRASPASOS = os.path.join(DIR_DATOS, "traspasos.csv")

# --------------------------------------------------------------------------
# Origen de los datos
# --------------------------------------------------------------------------
# Bucket público donde el proyecto publica los CSV ya preparados (dbt).
BASE_URL_R2 = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

# Alternativa (por si el bucket público cambiase de dirección en el futuro):
# los mismos ficheros también se publican como release asset / raw del repo.
BASE_URL_GITHUB_RAW = (
    "https://raw.githubusercontent.com/dcaribou/transfermarkt-datasets/master/data"
)

# Tablas que necesitamos para construir: Id_player, nombre, equipo (en la
# fecha), liga, posicion, nacionalidad, fecha, valor de mercado.
TABLAS_NECESARIAS = {
    "players": "players.csv.gz",
    "player_valuations": "player_valuations.csv.gz",
    "clubs": "clubs.csv.gz",
    "competitions": "competitions.csv.gz",
}

# Columnas finales de la tabla resultante
COLUMNAS_FINALES = [
    "id_player",
    "nombre",
    "equipo",
    "liga",
    "posicion",
    "nacionalidad",
    "fecha_nacimiento",
    "edad",
    "altura_cm",
    "pie",
    "fecha",
    "valor_mercado_eur",
]
