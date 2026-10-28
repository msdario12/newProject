"""Inferencia de líneas de cordón y cuneta a partir de LiDAR aéreo urbano.

Módulos:
    config       lectura y validación de parámetros (YAML/JSON)
    lectura      lectura de ejes (DXF/SHP) y de la nube (LAS/LAZ) por buffer
    eje          geometría del eje: estaciones, proyección (estación, offset)
    apilado      ventanas móviles y perfiles transversales apilados
    ajuste       ajuste robusto de la plantilla calzada–cordón–vereda
    raster       raster de intensidad/RGB y detección del borde de pavimento
    postproceso  filtrado, relleno de huecos y suavizado a lo largo del eje
    exportacion  DXF (POLYLINE 3D), CSV y LandXML
    graficos     gráficos de control (matplotlib)
"""

__version__ = "1.0.0"
