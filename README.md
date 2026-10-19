# 1. INGEST (Descarga y procesamiento de datos de repositorio de transfermarkt)

Descarga y procesa los datos públicos de
[dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
para construir una tabla histórica de valores de mercado de jugadores.

## Tabla final (`datos/tabla_valores_mercado.csv`)

| columna              | descripción                                              |
|----------------------|-----------------------------------------------------------|
| `id_player`          | id del jugador en Transfermarkt                            |
| `nombre`             | nombre del jugador                                          |
| `equipo`             | club en el que jugaba **en la fecha de esa valoración**    |
| `liga`               | liga doméstica del club en esa fecha                        |
| `posicion`           | posición del jugador                                        |
| `nacionalidad`       | país de nacionalidad/ciudadanía                              |
| `fecha`              | fecha de la valoración                                       |
| `valor_mercado_eur`  | valor de mercado en euros en esa fecha                       |

Una fila = una valoración histórica de un jugador (un jugador tiene tantas
filas como valoraciones registradas a lo largo de su carrera).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso manual

```bash
python ingest/actualizar.py
```

Esto:
1. Descarga (solo si cambiaron) `players`, `player_valuations`, `clubs` y
   `competitions` desde el bucket público del proyecto.
2. Reconstruye `datos/tabla_valores_mercado.csv` (y su copia en
   `.parquet`) únicamente si hubo datos nuevos.

Para forzar la reconstrucción aunque no haya cambios:

```bash
python ingest/actualizar.py --forzar
```

## Actualización periódica

Tienes dos formas de mantener la tabla al día automáticamente:

### Opción A — cron (Linux/Mac) [recomendada]

```bash
crontab -e
```

Añade, por ejemplo, para actualizar cada día a las 6:00 AM:

```
0 6 * * * cd /ruta/a/transfermarkt_valores && /ruta/a/.venv/bin/python ingest/actualizar.py >> datos/cron.log 2>&1
```

### Opción B — Programador de tareas (Windows)

Crea una tarea que ejecute, por ejemplo a diario:

```
python.exe C:\ruta\a\transfermarkt_valores\actualizar.py
```

### Opción C — modo "demonio" integrado (sin cron/Task Scheduler)

Si prefieres no tocar la configuración del sistema operativo, deja el
propio script corriendo en segundo plano:

```bash
python ingest/actualizar.py --loop --cada-horas 24
```

Quedará ejecutándose indefinidamente y se autoactualizará cada 24 horas
(ajustable). Puedes dejarlo en un `screen`/`tmux`, un servicio `systemd`,
o similar.

## Notas

- El repositorio de origen se actualiza semanalmente, así que no hace
  falta comprobar con más frecuencia que una vez al día.
- `ingest/descargar_datos.py` usa peticiones condicionales (ETag / Last-Modified),
  así que ejecutarlo a menudo no supone descargar de más si no hay
  cambios en el repositorio.
- Si `pyarrow` no está instalado, la tabla se guarda igualmente en CSV
  (el `.parquet` simplemente se omite).
- Fuente pública de los datos, usada tal como recomienda el propio
  README del proyecto:
  `https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/<tabla>.csv.gz`
