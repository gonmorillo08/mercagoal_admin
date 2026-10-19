"""
Punto de entrada único del proyecto.

Ejecuta, en orden:
    1. descargar_datos.py  -> trae los CSV más recientes del repositorio
                               (solo descarga lo que haya cambiado)
    2. procesar_datos.py   -> reconstruye datos/tabla_valores_mercado.csv
                               ÚNICAMENTE si hubo datos nuevos (o si se
                               fuerza con --forzar)

Modos de uso:

    # Una sola ejecución (ideal para llamarlo desde cron / Task Scheduler)
    python ingest/actualizar.py

    # Forzar reproceso aunque no haya cambios en el origen
    python ingest/actualizar.py --forzar

    # Modo "demonio": se queda corriendo y se autoactualiza cada N horas
    # (alternativa a usar cron si prefieres no configurar el sistema)
    python ingest/actualizar.py --loop --cada-horas 24
"""

import argparse
import logging
import os
import time
from datetime import datetime

import config
from descargar_datos import descargar_todo
from procesar_datos import (
    construir_tabla_final,
    construir_traspasos,
    guardar_tabla_final,
    guardar_traspasos,
)

os.makedirs(config.DIR_DATOS, exist_ok=True)

logger = logging.getLogger(__name__)


def ejecutar_ciclo(forzar: bool = False) -> None:
    logger.info("=== Iniciando ciclo de actualización ===")
    hubo_cambios = descargar_todo(forzar=forzar)

    if hubo_cambios or forzar:
        logger.info("Reconstruyendo la tabla final...")
        tabla = construir_tabla_final()
        guardar_tabla_final(tabla)
        logger.info("Tabla final actualizada con %d filas.", len(tabla))

        logger.info("Reconstruyendo la tabla de traspasos...")
        traspasos = construir_traspasos(tabla)
        guardar_traspasos(traspasos)
        logger.info("Tabla de traspasos actualizada con %d filas.", len(traspasos))
    else:
        logger.info(
            "No hubo cambios en el repositorio; se mantiene la tabla final existente."
        )
    logger.info("=== Ciclo de actualización finalizado ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga y procesa los datos de transfermarkt-datasets"
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Fuerza la re-descarga y el reproceso aunque no haya cambios",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Deja el script corriendo indefinidamente, repitiendo cada --cada-horas",
    )
    parser.add_argument(
        "--cada-horas",
        type=float,
        default=24.0,
        help="Frecuencia de actualización en horas cuando se usa --loop (default: 24)",
    )
    args = parser.parse_args()

    if args.loop:
        logger.info(
            "Modo bucle activado: se ejecutará cada %.1f horas. Ctrl+C para detener.",
            args.cada_horas,
        )
        while True:
            try:
                ejecutar_ciclo(forzar=args.forzar)
            except Exception:
                logger.exception("Error durante el ciclo de actualización")
            logger.info("Próxima actualización: %s", datetime.now())
            time.sleep(args.cada_horas * 3600)
    else:
        ejecutar_ciclo(forzar=args.forzar)
