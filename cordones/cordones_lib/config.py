"""Lectura de la configuración (YAML o JSON) con valores por defecto."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

# Valores por defecto. Cualquier clave del archivo de configuración
# sobreescribe la correspondiente de este diccionario.
DEFAULTS: dict = {
    "entrada": {
        "nube": None,            # archivo .las/.laz o patrón glob con tiles ("tiles/*.laz")
        "ejes": None,            # .dxf (polilíneas 2D) o .shp
        "capa_ejes": None,       # DXF: capa a usar (None = todas las polilíneas)
        "campo_nombre": None,    # SHP: campo con el nombre de la calle
        "epsg": None,            # solo informativo (se escribe en el LandXML)
    },
    "salida": {
        "carpeta": "salida",
        "prefijo": "cordones",
        "landxml": True,
    },
    "lectura": {
        "tamano_bloque": 2_000_000,  # puntos por bloque al leer LAS/LAZ
        "cache": True,               # guarda los puntos recortados por eje en .npz
    },
    "clasificacion": {
        "modo": "auto",          # clase2 | smrf | csf | auto
        "clase_suelo": 2,
        "fraccion_min_suelo": 0.05,  # modo auto: mínimo de puntos clase 2 para confiar en la clasificación
        "buffer_extra": 15.0,    # m de contexto extra para SMRF/CSF
        "smrf": {"cell": 1.0, "slope": 0.15, "window": 18.0, "threshold": 0.5, "scalar": 1.25},
        "csf": {"resolution": 0.5, "rigidness": 3, "threshold": 0.5, "smooth": True},
    },
    "geometria": {
        "ancho_calzada": 9.0,        # m, de cordón a cordón
        "ancho_vereda": 3.0,         # m
        "margen": 1.0,               # m extra más allá de la vereda
        "tolerancia_offset": 1.5,    # d se busca en [ancho/2 - tol, ancho/2 + tol]
        "u_min": 0.0,                # m: ignorar puntos más cerca del eje que esto
    },
    "ejes_parametros": {},           # sobreescrituras por eje: {nombre: {ancho_calzada: 8.0, ...}}
    "ventana": {
        "longitud": 4.0,
        "paso": 1.0,
        "min_puntos_calzada": 40,    # puntos mínimos entre eje y cordón (por lado)
        "min_puntos_vereda": 25,     # puntos mínimos detrás del cordón (por lado)
    },
    "ajuste": {
        "loss": "soft_l1",           # soft_l1 | huber | cauchy | linear
        "f_scale": 0.06,             # m, escala del ruido (≈ precisión vertical)
        "h_min": 0.05,
        "h_max": 0.30,
        "ancho_transicion": 0.05,    # m, suavizado del escalón (sigmoide) para que d sea derivable
        "paso_busqueda_d": 0.05,     # m, grilla inicial de d
        "pend_calzada": [-0.08, 0.08],
        "pend_vereda": [-0.08, 0.08],
        "pend_longitudinal": [-0.15, 0.15],
    },
    "confianza": {
        "n_ref": 300,                # puntos por lado para confianza plena por cantidad
        "rms_bueno": 0.05,           # m
        "rms_malo": 0.15,            # m
        "escala_bic": 10.0,          # ΔBIC que da ~63 % de confianza por significancia
        "conf_min": 0.30,            # ventanas por debajo se descartan
        "conf_alta": 0.60,           # tramos por debajo van a BAJA_CONFIANZA
        "se_h_max": 0.04,            # m, error estándar máximo de h para considerarla medida
    },
    "raster": {
        "usar": "auto",              # auto | si | no
        "campo": "auto",             # auto | intensidad | rgb | ambos
        "resolucion": 0.25,
        "ancho_kernel": 2,           # celdas a cada lado para el gradiente
        "correccion_offset": 0.0,    # m sumados al borde detectado (p.ej. ancho de cuneta de H°)
        "signo_contraste": "cualquiera",  # cualquiera | vereda_mas_clara | vereda_mas_oscura
        "ratio_min": 3.0,            # relación gradiente/ruido con confianza 0
        "ratio_max": 8.0,            # relación gradiente/ruido con confianza 1
        "peso": 0.7,                 # peso relativo del raster frente a la geometría
        "tolerancia_combinacion": 0.30,  # m: si difieren más, no se promedian
    },
    "postproceso": {
        "ventana_mediana": 9.0,      # m
        "umbral_offset": 0.25,       # m
        "umbral_z": 0.08,            # m
        "umbral_h": 0.06,            # m
        "hueco_max": 10.0,           # m: huecos menores se interpolan
        "tramo_min": 4.0,            # m: tramos aislados más cortos se descartan
        "suavizado": "savgol",       # savgol | spline | ninguno
        "savgol_ventana": 7.0,       # m
        "savgol_orden": 2,
        "spline_sigma": {"offset": 0.05, "z": 0.02, "h": 0.02},
        "h_tipica": 0.16,
    },
    "graficos": {
        "generar": True,
        "cada_n_ventanas": 50,       # modo normal: 1 gráfico de ventana cada N
        "dpi": 110,
    },
    "debug": {
        "eje": 0,                    # índice o nombre del eje a procesar con --debug
    },
}


def _fusionar(base: dict, extra: dict) -> dict:
    """Fusión recursiva: los valores de `extra` sobreescriben a `base`."""
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "ejes_parametros":
            out[k] = _fusionar(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def cargar_config(ruta: str | Path) -> dict:
    ruta = Path(ruta)
    with open(ruta, encoding="utf-8") as f:
        if ruta.suffix.lower() == ".json":
            usuario = json.load(f)
        else:
            usuario = yaml.safe_load(f) or {}
    cfg = _fusionar(DEFAULTS, usuario)

    # Las rutas relativas se resuelven respecto de la carpeta del archivo de configuración
    base = ruta.resolve().parent
    for seccion, clave in (("entrada", "nube"), ("entrada", "ejes"), ("salida", "carpeta")):
        valor = cfg[seccion][clave]
        if valor and not Path(valor).is_absolute():
            cfg[seccion][clave] = str(base / valor)

    _validar(cfg)
    return cfg


def _validar(cfg: dict) -> None:
    # YAML interpreta `no`/`si`... sin comillas como booleanos: normalizar
    usar = cfg["raster"]["usar"]
    if isinstance(usar, bool):
        cfg["raster"]["usar"] = "si" if usar else "no"
    if cfg["raster"]["usar"] not in ("auto", "si", "no"):
        raise ValueError("raster.usar debe ser auto | si | no")
    if not cfg["entrada"]["nube"]:
        raise ValueError("Falta entrada.nube en la configuración")
    if not cfg["entrada"]["ejes"]:
        raise ValueError("Falta entrada.ejes en la configuración")
    if cfg["clasificacion"]["modo"] not in ("clase2", "smrf", "csf", "auto"):
        raise ValueError("clasificacion.modo debe ser clase2 | smrf | csf | auto")
    a = cfg["ajuste"]
    if not 0 < a["h_min"] < a["h_max"]:
        raise ValueError("Se requiere 0 < ajuste.h_min < ajuste.h_max")
    v = cfg["ventana"]
    if v["paso"] <= 0 or v["longitud"] <= 0:
        raise ValueError("ventana.longitud y ventana.paso deben ser positivos")


def params_geometria(cfg: dict, nombre_eje: str) -> dict:
    """Parámetros geométricos de un eje, aplicando las sobreescrituras por nombre."""
    g = dict(cfg["geometria"])
    g.update(cfg.get("ejes_parametros", {}).get(nombre_eje, {}) or {})
    return g
