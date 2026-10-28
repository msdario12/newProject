"""Raster de intensidad / RGB y detección del borde asfalto–hormigón.

El raster se construye en el sistema "enderezado" del eje (estación, offset)
con celdas de `resolucion` m (0,25 m por defecto). Así una sección transversal
es simplemente un conjunto de filas, y no hace falta rasterizar en X/Y toda la
extensión de la ciudad.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .apilado import NubeEje


@dataclass
class RasterEje:
    valores: np.ndarray     # (ns, no) media del rasgo por celda (NaN = vacía)
    res: float
    o_min: float

    @property
    def o_centros(self) -> np.ndarray:
        return self.o_min + self.res * (np.arange(self.valores.shape[1]) + 0.5)

    def perfil(self, s_c: float, longitud: float) -> np.ndarray:
        """Perfil transversal: mediana de las filas dentro de la ventana."""
        i0 = max(0, int(np.floor((s_c - longitud / 2) / self.res)))
        i1 = min(self.valores.shape[0], int(np.ceil((s_c + longitud / 2) / self.res)))
        filas = self.valores[i0:i1]
        if filas.size == 0:
            return np.full(self.valores.shape[1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # columnas todas NaN
            return np.nanmedian(filas, axis=0)


def construir_raster(nube: NubeEje, longitud_eje: float, offset_max: float, res: float) -> RasterEje | None:
    if nube.rasgo is None or len(nube.s) == 0:
        return None
    ns = int(np.ceil(longitud_eje / res)) + 1
    no = int(np.ceil(2 * offset_max / res))
    i = np.clip((nube.s / res).astype(int), 0, ns - 1)
    j = np.clip(((nube.o + offset_max) / res).astype(int), 0, no - 1)
    lin = i * no + j
    suma = np.bincount(lin, weights=nube.rasgo, minlength=ns * no)
    cnt = np.bincount(lin, minlength=ns * no)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(cnt > 0, suma / cnt, np.nan).reshape(ns, no)
    return RasterEje(val, res, -offset_max)


def detectar_borde(raster: RasterEje, s_c: float, longitud: float, lado: int,
                   dmin: float, dmax: float, prm: dict):
    """Busca en la sección el máximo gradiente del rasgo entre dmin y dmax.

    Devuelve (offset_borde, confianza, ratio). offset_borde ya incluye
    `correccion_offset` y es una distancia (positiva) al eje.
    """
    perfil = raster.perfil(s_c, longitud)
    u = lado * raster.o_centros
    orden = np.argsort(u)
    u, perfil = u[orden], perfil[orden]
    sel = u > 0
    u, perfil = u[sel], perfil[sel]
    k = int(prm["ancho_kernel"])
    if len(u) < 2 * k + 2:
        return np.nan, 0.0, 0.0

    # gradiente en cada frontera de celda: media(k celdas de afuera) - media(k de adentro)
    bordes, grad = [], []
    for b in range(k, len(u) - k + 1):
        adentro = perfil[b - k:b]
        afuera = perfil[b:b + k]
        if np.isnan(adentro).any() or np.isnan(afuera).any():
            continue
        bordes.append(0.5 * (u[b - 1] + u[b]))
        grad.append(afuera.mean() - adentro.mean())
    if len(grad) < 5:
        return np.nan, 0.0, 0.0
    bordes, grad = np.array(bordes), np.array(grad)

    signo = prm.get("signo_contraste", "cualquiera")
    g_eval = {"vereda_mas_clara": grad, "vereda_mas_oscura": -grad}.get(signo, np.abs(grad))
    en_rango = (bordes >= dmin) & (bordes <= dmax)
    if not en_rango.any():
        return np.nan, 0.0, 0.0
    idx = np.flatnonzero(en_rango)[np.argmax(g_eval[en_rango])]
    ruido = 1.4826 * np.median(np.abs(grad - np.median(grad))) + 1e-6
    ratio = float(g_eval[idx] / ruido)

    # refinamiento sub-celda por parábola sobre el pico
    off = bordes[idx]
    if 0 < idx < len(grad) - 1:
        y0, y1, y2 = g_eval[idx - 1], g_eval[idx], g_eval[idx + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            off += 0.5 * (y0 - y2) / den * raster.res

    conf = float(np.clip((ratio - prm["ratio_min"]) / (prm["ratio_max"] - prm["ratio_min"]), 0, 1))
    return float(off + prm["correccion_offset"]), conf, ratio


def combinar(d_geo: float, c_geo: float, d_ras: float, c_ras: float, prm: dict):
    """Combina la estimación geométrica y la radiométrica del offset.

    * si coinciden (|Δ| ≤ tolerancia): promedio ponderado por confianza y la
      confianza combinada es 1 - (1-c_geo)(1-peso·c_ras);
    * si no coinciden: se queda con la de mayor peso y la confianza se reduce
      a la mitad (el tramo queda marcado como dudoso).
    Devuelve (d, conf, fuente).
    """
    w_r = prm["peso"] * c_ras
    if not np.isfinite(d_ras) or w_r <= 0:
        return d_geo, c_geo, "geometria"
    if not np.isfinite(d_geo) or c_geo <= 0:
        return d_ras, w_r, "raster"
    if abs(d_geo - d_ras) <= prm["tolerancia_combinacion"]:
        d = (c_geo * d_geo + w_r * d_ras) / (c_geo + w_r)
        return d, 1 - (1 - c_geo) * (1 - w_r), "combinada"
    if c_geo >= w_r:
        return d_geo, 0.5 * c_geo, "geometria (raster discrepa)"
    return d_ras, 0.5 * w_r, "raster (geometria discrepa)"
