"""
Descarga los ficheros CSV (gzip) necesarios desde el dataset público de
transfermarkt-datasets.

Este script es "inteligente": guarda el ETag / Last-Modified de cada
fichero descargado en datos/raw/_metadatos_descarga.json. En cada
ejecución posterior hace una petición condicional (If-None-Match /
If-Modified-Since) y solo vuelve a descargar el fichero si el repositorio
lo ha actualizado. Esto permite ejecutarlo periódicamente (por ejemplo, a
diario) sin gastar ancho de banda ni tiempo de más cuando no hay novedades.

Uso:
    python ingest/descargar_datos.py             # descarga solo lo que cambió
    python ingest/descargar_datos.py --forzar    # vuelve a descargar todo

Devuelve (a través de descargar_todo) True si al menos un fichero cambió.
"""

import argparse
import json
import logging
import os
import sys

import requests

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _cargar_metadatos() -> dict:
    if os.path.exists(config.ARCHIVO_METADATOS):
        with open(config.ARCHIVO_METADATOS, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _guardar_metadatos(metadatos: dict) -> None:
    os.makedirs(os.path.dirname(config.ARCHIVO_METADATOS), exist_ok=True)
    with open(config.ARCHIVO_METADATOS, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=2, ensure_ascii=False)


def _descargar_fichero(
    nombre_tabla: str,
    nombre_fichero: str,
    metadatos: dict,
    forzar: bool = False,
    base_url: str = config.BASE_URL_R2,
) -> bool:
    """
    Descarga un fichero individual si ha cambiado respecto a la última
    descarga. Devuelve True si se descargó una versión nueva.
    """
    url = f"{base_url}/{nombre_fichero}"
    destino = os.path.join(config.DIR_RAW, nombre_fichero)

    headers = {}
    info_previa = metadatos.get(nombre_tabla, {})
    if not forzar and os.path.exists(destino):
        if info_previa.get("etag"):
            headers["If-None-Match"] = info_previa["etag"]
        if info_previa.get("last_modified"):
            headers["If-Modified-Since"] = info_previa["last_modified"]

    logger.info("Comprobando %s ...", url)
    try:
        respuesta = requests.get(url, headers=headers, stream=True, timeout=60)
    except requests.RequestException as e:
        logger.error("No se pudo contactar con %s: %s", url, e)
        # Si ya existe una copia local previa, seguimos con ella.
        if os.path.exists(destino):
            logger.warning("Se usará la copia local existente de %s", nombre_fichero)
            return False
        raise

    if respuesta.status_code == 304:
        logger.info("%s no ha cambiado desde la última descarga.", nombre_fichero)
        return False

    if respuesta.status_code != 200:
        logger.error(
            "Error %s descargando %s desde %s",
            respuesta.status_code,
            nombre_fichero,
            url,
        )
        if os.path.exists(destino):
            logger.warning("Se usará la copia local existente de %s", nombre_fichero)
            return False
        respuesta.raise_for_status()

    os.makedirs(config.DIR_RAW, exist_ok=True)
    tmp_destino = destino + ".tmp"
    with open(tmp_destino, "wb") as f:
        for chunk in respuesta.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(tmp_destino, destino)

    metadatos[nombre_tabla] = {
        "etag": respuesta.headers.get("ETag"),
        "last_modified": respuesta.headers.get("Last-Modified"),
        "url": url,
        "fichero": nombre_fichero,
    }
    logger.info("Descargado %s (nueva versión).", nombre_fichero)
    return True


def descargar_todo(forzar: bool = False) -> bool:
    """
    Descarga (o actualiza) todas las tablas necesarias.
    Devuelve True si al menos una tabla cambió (o se descargó por primera vez).
    """
    os.makedirs(config.DIR_RAW, exist_ok=True)
    metadatos = _cargar_metadatos()
    hubo_cambios = False

    for nombre_tabla, nombre_fichero in config.TABLAS_NECESARIAS.items():
        try:
            cambio = _descargar_fichero(
                nombre_tabla, nombre_fichero, metadatos, forzar=forzar
            )
        except requests.RequestException:
            logger.error(
                "Fallo al descargar la tabla '%s' y no hay copia local previa. "
                "Abortando esta ejecución.",
                nombre_tabla,
            )
            sys.exit(1)
        hubo_cambios = hubo_cambios or cambio

    _guardar_metadatos(metadatos)
    return hubo_cambios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga los datos de transfermarkt-datasets")
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Vuelve a descargar todos los ficheros aunque no hayan cambiado",
    )
    args = parser.parse_args()

    cambios = descargar_todo(forzar=args.forzar)
    if cambios:
        print("Se descargaron datos nuevos.")
    else:
        print("No había datos nuevos: se mantiene la copia local existente.")
