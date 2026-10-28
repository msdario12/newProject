"""Proyección de la nube al sistema (estación, offset, z) del eje y extracción
de perfiles transversales apilados por ventana móvil."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eje import Eje


@dataclass
class NubeEje:
    """Puntos de suelo de un eje en coordenadas locales, ordenados por estación."""
    s: np.ndarray
    o: np.ndarray
    z: np.ndarray
    rasgo: np.ndarray | None     # intensidad / luminancia normalizada (0–1) para el raster

    def ventana(self, s_c: float, longitud: float):
        """Puntos con |s - s_c| <= longitud/2. Devuelve (ds, o, z, rasgo)."""
        i0 = np.searchsorted(self.s, s_c - longitud / 2, side="left")
        i1 = np.searchsorted(self.s, s_c + longitud / 2, side="right")
        r = self.rasgo[i0:i1] if self.rasgo is not None else None
        return self.s[i0:i1] - s_c, self.o[i0:i1], self.z[i0:i1], r


def _normalizar(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, [2, 98])
    if hi <= lo:
        return np.zeros_like(v, dtype=np.float32)
    return np.clip((v - lo) / (hi - lo), 0, 1).astype(np.float32)


def rasgo_radiometrico(datos: dict, campo: str) -> np.ndarray | None:
    """Rasgo escalar para distinguir asfalto/hormigón: intensidad, luminancia RGB
    o el promedio de ambos (normalizados por percentiles)."""
    inten, rgb = datos.get("intensidad"), datos.get("rgb")
    lum = None
    if rgb is not None:
        lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    if campo == "intensidad":
        return _normalizar(inten) if inten is not None else None
    if campo == "rgb":
        return _normalizar(lum) if lum is not None else None
    if campo == "ambos" and inten is not None and lum is not None:
        return 0.5 * (_normalizar(inten) + _normalizar(lum))
    # auto: intensidad si existe, si no RGB
    if inten is not None:
        return _normalizar(inten)
    if lum is not None:
        return _normalizar(lum)
    return None


def proyectar_nube(eje: Eje, datos: dict, offset_max: float, rasgo: np.ndarray | None) -> NubeEje:
    s, o, dentro = eje.proyectar(datos["x"], datos["y"])
    m = dentro & (np.abs(o) <= offset_max)
    orden = np.argsort(s[m], kind="stable")
    r = rasgo[m][orden] if rasgo is not None else None
    return NubeEje(s[m][orden], o[m][orden], datos["z"][m][orden], r)


def estaciones_ventanas(longitud_eje: float, longitud_ventana: float, paso: float) -> np.ndarray:
    """Estaciones centrales de las ventanas (la primera y la última quedan
    completamente dentro del eje)."""
    if longitud_eje <= longitud_ventana:
        return np.array([longitud_eje / 2])
    ini = longitud_ventana / 2
    fin = longitud_eje - longitud_ventana / 2
    n = int(np.floor((fin - ini) / paso + 1e-9)) + 1
    return ini + paso * np.arange(n)
