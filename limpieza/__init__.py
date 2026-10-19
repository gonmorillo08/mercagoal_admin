"""
Limpieza y detección de errores en `tabla_valores_mercado` (la tabla que
genera `ingest/procesar_datos.py`).

Cada módulo se ocupa de un tipo de problema y se puede ejecutar de forma
independiente (`python -m limpieza.<modulo>`) o importar por separado.
Ninguno modifica los ficheros de `ingest/datos/`: todos son de DETECCIÓN
(y, cuando es posible, de estimación de recuperabilidad), no de corrección
automática — la corrección queda para quien revise cada caso.

  - `valores_faltantes`: valores NA reales y "disfrazados" (texto tipo
    "Missing"/"Unknown" en vez de NaN) y si son recuperables desde otra
    columna del propio dataset.
  - `duplicidad_equipos`: nombres de equipo del propio dataset comparados
    ENTRE SÍ para distinguir el mismo club con distinta grafía (posible
    duplicidad) de primer equipo/cantera (posible filial), con su
    probabilidad; y, dentro de cada red de filiales detectada, cuál es el
    club líder (mayor valor de mercado de plantilla).
  - `nacionalidad_y_liga`: bandera-emoji de cada `liga` (vía
    `competitions.csv`) y de cada `nacionalidad`, y detección de los casos
    en los que esa asociación es ambigua (mismo nombre de liga en países
    distintos) o directamente imposible (nacionalidad sin bandera-emoji
    fiable).
  - `valores_erroneos`: si la propia cifra de `valor_mercado_eur` es
    plausible — valores no positivos, varios registros del mismo jugador
    en el mismo instante, cambios de valor muy bruscos en poco tiempo, y
    picos aislados que se revierten por completo.
"""
