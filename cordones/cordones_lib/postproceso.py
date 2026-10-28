"""Post-proceso a lo largo del eje para un lado: outliers, huecos y suavizado.

Estados por estación:
    estado_d: 0 = sin dato (línea cortada), 1 = medido, 2 = interpolado
    estado_h: 0 = sin dato, 1 = medido, 2 = interpolado, 3 = asumido (h típica)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

SIN_DATO, MEDIDO, INTERPOLADO, ASUMIDO = 0, 1, 2, 3


def _mediana_movil(s, y, valido, semiancho):
    """Mediana de los valores válidos en |s_j - s_i| <= semiancho (s ordenado)."""
    med = np.full(len(s), np.nan)
    sv, yv = s[valido], y[valido]
    if len(sv) == 0:
        return med
    i0 = np.searchsorted(sv, s - semiancho, side="left")
    i1 = np.searchsorted(sv, s + semiancho, side="right")
    for i in range(len(s)):
        if i1[i] > i0[i]:
            med[i] = np.median(yv[i0[i]:i1[i]])
    return med


def _quitar_outliers(s, y, valido, semiancho, umbral, iteraciones=2):
    valido = valido.copy()
    for _ in range(iteraciones):
        med = _mediana_movil(s, y, valido, semiancho)
        nuevo = valido & (np.abs(y - med) <= umbral)
        if np.array_equal(nuevo, valido):
            break
        valido = nuevo
    return valido


def _tramos(s, valido, hueco_max):
    """Agrupa estaciones válidas en tramos: dos válidas consecutivas pertenecen
    al mismo tramo si están separadas por menos de hueco_max."""
    idx = np.flatnonzero(valido)
    if idx.size == 0:
        return []
    cortes = np.flatnonzero(np.diff(s[idx]) >= hueco_max)
    grupos = np.split(idx, cortes + 1)
    return [(g[0], g[-1]) for g in grupos]


def _suavizar(s, y, metodo, cfgp, sigma):
    n = len(y)
    if metodo == "savgol":
        paso = np.median(np.diff(s)) if n > 1 else 1.0
        ventana = int(round(cfgp["savgol_ventana"] / paso))
        ventana += 1 - ventana % 2           # impar
        orden = cfgp["savgol_orden"]
        ventana = min(ventana, n if n % 2 else n - 1)
        if ventana <= orden:
            return y
        return savgol_filter(y, ventana, orden, mode="interp")
    if metodo == "spline":
        if n <= 3:
            return y
        return UnivariateSpline(s, y, k=3, s=n * sigma ** 2)(s)
    return y


def procesar_lado(s, d, zc, h, conf, h_ok, cfg) -> dict:
    """s ordenado; d, zc, h, conf por ventana (NaN si el ajuste falló)."""
    p = cfg["postproceso"]
    cc = cfg["confianza"]
    semi = p["ventana_mediana"] / 2
    n = len(s)

    valido = np.isfinite(d) & np.isfinite(zc) & (conf >= cc["conf_min"])
    # outliers de offset y de cota de cuneta
    valido = _quitar_outliers(s, d, valido, semi, p["umbral_offset"])
    valido = _quitar_outliers(s, zc, valido, semi, p["umbral_z"])

    # tramos aislados demasiado cortos -> se descartan
    for i0, i1 in _tramos(s, valido, p["hueco_max"]):
        if s[i1] - s[i0] < p["tramo_min"]:
            valido[i0:i1 + 1] = False
    tramos = _tramos(s, valido, p["hueco_max"])

    d_out = np.full(n, np.nan)
    zc_out = np.full(n, np.nan)
    h_out = np.full(n, np.nan)
    est_d = np.zeros(n, dtype=np.int8)
    est_h = np.zeros(n, dtype=np.int8)

    h_valido = valido & h_ok & np.isfinite(h)
    h_valido = _quitar_outliers(s, h, h_valido, semi, p["umbral_h"])
    sig = p["spline_sigma"]

    for i0, i1 in tramos:
        r = slice(i0, i1 + 1)
        ss = s[r]
        v = valido[r]
        dd = np.interp(ss, ss[v], d[r][v])
        zz = np.interp(ss, ss[v], zc[r][v])
        est_d[r] = np.where(v, MEDIDO, INTERPOLADO)

        # altura: interpolar huecos cortos de h; si no, h típica ("asumido")
        hv = h_valido[r]
        hh = np.full(len(ss), p["h_tipica"])
        eh = np.full(len(ss), ASUMIDO, dtype=np.int8)
        if hv.any():
            sh = ss[hv]
            interp = np.interp(ss, sh, h[r][hv])
            # distancia al dato de h más cercano a cada lado
            j = np.searchsorted(sh, ss)
            izq = np.where(j > 0, sh[np.maximum(j - 1, 0)], -np.inf)
            der = np.where(j < len(sh), sh[np.minimum(j, len(sh) - 1)], np.inf)
            exacto = np.isin(ss, sh)
            cubierto = exacto | ((der - izq) < p["hueco_max"])
            hh = np.where(cubierto, interp, p["h_tipica"])
            eh = np.where(exacto & hv, MEDIDO, np.where(cubierto, INTERPOLADO, ASUMIDO)).astype(np.int8)

        metodo = p["suavizado"]
        d_out[r] = _suavizar(ss, dd, metodo, p, sig["offset"])
        zc_out[r] = _suavizar(ss, zz, metodo, p, sig["z"])
        # la h asumida se deja constante; solo se suaviza donde hay medición
        h_s = _suavizar(ss, hh, metodo, p, sig["h"])
        h_out[r] = np.where(eh == ASUMIDO, p["h_tipica"], np.clip(h_s, cfg["ajuste"]["h_min"],
                                                                cfg["ajuste"]["h_max"]))
        est_h[r] = eh

    conf_out = np.where(est_d == MEDIDO, conf, 0.0)
    dudoso = (est_d > 0) & ((est_d == INTERPOLADO) | (est_h >= INTERPOLADO) | (conf_out < cc["conf_alta"]))
    return {
        "s": s, "d": d_out, "z_cuneta": zc_out, "h": h_out,
        "z_cabeza": zc_out + h_out, "estado_d": est_d, "estado_h": est_h,
        "conf": conf_out, "dudoso": dudoso, "valido_bruto": valido,
    }
