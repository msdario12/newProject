"""Ajuste robusto de la plantilla calzada – cordón – vereda a un perfil apilado.

Para un lado del eje se trabaja con u = distancia al eje (u > 0 hacia el
cordón) y ds = estación relativa al centro de la ventana. Modelo:

    calzada (u < d):  z = z0 + g·ds + sc·u
    vereda  (u > d):  z = z0 + g·ds + sc·d + h + sv·(u - d)

    z0  cota de la calzada en el eje (centro de ventana)
    sc  pendiente transversal de calzada (negativa = baja hacia el cordón)
    d   offset del cordón (cara del cordón / línea de cuneta)
    h   altura del cordón
    sv  pendiente transversal de vereda
    g   pendiente longitudinal dentro de la ventana (evita que la rasante
        "ensucie" el perfil apilado: 1 % en 4 m son 4 cm, del orden de h)

El escalón se representa con una sigmoide de ancho `w` (≈5 cm) para que la
función sea derivable respecto de d. Primero se hace una búsqueda en grilla
de d (con los demás parámetros lineales resueltos por IRLS-Huber) y luego se
refina todo con scipy.optimize.least_squares (pérdida robusta).

El modelo nulo es el mismo SIN escalón (h = 0, quiebre de pendiente continuo
en d). La significancia del escalón se mide con ΔBIC = BIC_nulo − BIC_escalón,
calculado sobre los mismos inliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

NOMBRES_ESC = ("z0", "sc", "h", "sv", "d", "g")
NOMBRES_NULO = ("z0", "sc", "sv", "d", "g")


@dataclass
class ResultadoLado:
    ok: bool
    motivo: str = ""
    n: int = 0
    n_calzada: int = 0
    n_vereda: int = 0
    d: float = np.nan
    h: float = np.nan
    z0: float = np.nan
    sc: float = np.nan
    sv: float = np.nan
    g: float = np.nan
    z_cuneta: float = np.nan
    z_cabeza: float = np.nan
    rms: float = np.nan
    delta_bic: float = np.nan
    se_d: float = np.nan
    se_h: float = np.nan
    en_limite: bool = False
    conf: float = 0.0
    conf_partes: dict = field(default_factory=dict)
    params_nulo: np.ndarray | None = None


# ----------------------------------------------------------------------------- modelos
def modelo_escalon(p, u, ds, w):
    z0, sc, h, sv, d, g = p
    S = expit((u - d) / w)
    return z0 + g * ds + sc * u * (1 - S) + (sc * d + h + sv * (u - d)) * S


def modelo_nulo(p, u, ds, w):
    z0, sc, sv, d, g = p
    S = expit((u - d) / w)
    return z0 + g * ds + sc * u * (1 - S) + (sc * d + sv * (u - d)) * S


def _diseno(u, ds, d, con_escalon: bool):
    """Matriz de diseño con escalón rígido en d (para la búsqueda en grilla)."""
    v = u >= d
    cols = [np.ones_like(u), np.where(v, d, u)]
    if con_escalon:
        cols.append(v.astype(float))
    cols += [np.where(v, u - d, 0.0), ds]
    return np.column_stack(cols)


def _irls_huber(A, z, c, iters=6):
    """Mínimos cuadrados con pesos de Huber (escala fija c). Devuelve (coef, costo).
    Se resuelven las ecuaciones normales (pocas columnas): mucho más rápido que lstsq."""
    w = np.ones(len(z))
    reg = 1e-9 * np.eye(A.shape[1])
    coef = None
    for _ in range(iters):
        Aw = A * w[:, None]
        coef = np.linalg.solve(Aw.T @ A + reg, Aw.T @ z)
        r = np.abs(z - A @ coef)
        w = np.where(r <= c, 1.0, c / np.maximum(r, 1e-12))
    costo = np.where(r <= c, 0.5 * r ** 2, c * (r - 0.5 * c)).sum()
    return coef, costo


def _busqueda_grilla(u, ds, z, dmin, dmax, paso, c, nmin_c, nmin_v, con_escalon):
    """Búsqueda de d en dos pasadas: grilla gruesa (4·paso) y luego fina
    (paso) alrededor del mejor valor grueso."""
    def evaluar(candidatos, mejor):
        for d in candidatos:
            nc = np.count_nonzero(u < d)
            if nc < nmin_c or len(u) - nc < nmin_v:
                continue
            try:
                coef, costo = _irls_huber(_diseno(u, ds, d, con_escalon), z, c)
            except np.linalg.LinAlgError:
                continue
            if costo < mejor[0]:
                mejor = (costo, d, coef)
        return mejor

    grueso = 4 * paso
    mejor = evaluar(np.arange(dmin, dmax + 1e-9, grueso), (np.inf, None, None))
    if mejor[1] is None:
        return None, None
    d_g = mejor[1]
    fino = np.arange(max(dmin, d_g - grueso), min(dmax, d_g + grueso) + 1e-9, paso)
    mejor = evaluar(fino, mejor)
    return mejor[1], mejor[2]


def _refinar(fun_modelo, x0, lo, hi, u, ds, z, w, loss, f_scale):
    eps = 1e-6   # least_squares exige x0 estrictamente dentro de los límites
    x0 = np.minimum(np.maximum(x0, lo + eps), hi - eps)
    return least_squares(lambda p: fun_modelo(p, u, ds, w) - z, x0, bounds=(lo, hi),
                         loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=200)


# ----------------------------------------------------------------------------- principal
def ajustar_lado(u, ds, z, dmin, dmax, cfg) -> ResultadoLado:
    """Ajusta la plantilla a un lado del eje.

    u, ds, z  : puntos del lado (u > 0), ya recortados al rango de offsets.
    dmin, dmax: rango admisible para el offset del cordón.
    """
    a, cc, v = cfg["ajuste"], cfg["confianza"], cfg["ventana"]
    w = a["ancho_transicion"]
    c = a["f_scale"]
    n = len(u)
    res = ResultadoLado(ok=False, n=n)
    if n < v["min_puntos_calzada"] + v["min_puntos_vereda"]:
        res.motivo = "pocos puntos"
        return res

    # ---- 1) búsqueda en grilla de d (modelo con escalón y nulo)
    d0, coef = _busqueda_grilla(u, ds, z, dmin, dmax, a["paso_busqueda_d"], c,
                                v["min_puntos_calzada"], v["min_puntos_vereda"], True)
    if d0 is None:
        res.motivo = "sin puntos a ambos lados del rango de d"
        return res
    d0n, coefn = _busqueda_grilla(u, ds, z, dmin, dmax, a["paso_busqueda_d"], c,
                                  v["min_puntos_calzada"], v["min_puntos_vereda"], False)

    # ---- 2) refinamiento no lineal robusto
    pc, pv, pg = a["pend_calzada"], a["pend_vereda"], a["pend_longitudinal"]
    lo = np.array([-np.inf, pc[0], a["h_min"], pv[0], dmin, pg[0]])
    hi = np.array([np.inf, pc[1], a["h_max"], pv[1], dmax, pg[1]])
    x0 = np.array([coef[0], coef[1], coef[2], coef[3], d0, coef[4]])
    sol = _refinar(modelo_escalon, x0, lo, hi, u, ds, z, w, a["loss"], c)
    p = sol.x

    lon = np.array([-np.inf, pc[0], pv[0], dmin, pg[0]])
    hin = np.array([np.inf, pc[1], pv[1], dmax, pg[1]])
    x0n = np.array([coefn[0], coefn[1], coefn[2], d0n, coefn[3]])
    soln = _refinar(modelo_nulo, x0n, lon, hin, u, ds, z, w, a["loss"], c)

    # ---- 3) estadísticos sobre inliers del modelo con escalón
    r = z - modelo_escalon(p, u, ds, w)
    rn = z - modelo_nulo(soln.x, u, ds, w)
    escala = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-3)
    inl = np.abs(r) <= 3 * escala
    m = int(inl.sum())
    rss = float(np.sum(r[inl] ** 2))
    rssn = float(np.sum(rn[inl] ** 2))
    k, kn = len(NOMBRES_ESC), len(NOMBRES_NULO)
    bic = m * np.log(rss / m) + k * np.log(m)
    bicn = m * np.log(rssn / m) + kn * np.log(m)
    rms = np.sqrt(rss / m)

    # errores estándar a partir del jacobiano en la solución (inliers)
    se = np.full(k, np.nan)
    J = sol.jac[inl]
    try:
        cov = np.linalg.pinv(J.T @ J) * rss / max(m - k, 1)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        pass

    z0, sc, h, sv, d, g = p
    tol_lim = 0.02
    en_limite = bool(d - dmin < tol_lim or dmax - d < tol_lim
                     or h - a["h_min"] < 0.005 or a["h_max"] - h < 0.005)

    res.ok = True
    res.n = n
    res.n_calzada = int(np.count_nonzero(u < d))
    res.n_vereda = n - res.n_calzada
    res.d, res.h, res.z0, res.sc, res.sv, res.g = map(float, (d, h, z0, sc, sv, g))
    res.z_cuneta = float(z0 + sc * d)          # calzada evaluada en d, centro de ventana
    res.z_cabeza = res.z_cuneta + res.h
    res.rms = float(rms)
    res.delta_bic = float(bicn - bic)
    res.se_d, res.se_h = float(se[4]), float(se[2])
    res.en_limite = en_limite
    res.params_nulo = soln.x
    res.conf, res.conf_partes = indice_confianza(res, cc)
    return res


def indice_confianza(res: ResultadoLado, cc: dict) -> tuple[float, dict]:
    """Índice 0–1 = producto de factores (cada uno en 0–1):

    n     cantidad de puntos: min(1, n / n_ref)
    rms   residual: 1 en rms_bueno, 0 en rms_malo (lineal)
    bic   significancia del escalón: 1 - exp(-ΔBIC / escala_bic), 0 si ΔBIC ≤ 0
    lim   0.5 si d o h quedaron pegados a un límite (solución forzada)
    lados equilibrio de puntos calzada/vereda: min(1, 4·min(nc, nv)/n)
    """
    f_n = min(1.0, res.n / cc["n_ref"])
    f_rms = float(np.clip((cc["rms_malo"] - res.rms) / (cc["rms_malo"] - cc["rms_bueno"]), 0, 1))
    f_bic = float(1 - np.exp(-res.delta_bic / cc["escala_bic"])) if res.delta_bic > 0 else 0.0
    f_lim = 0.5 if res.en_limite else 1.0
    f_lados = min(1.0, 4.0 * min(res.n_calzada, res.n_vereda) / max(res.n, 1))
    partes = {"n": f_n, "rms": f_rms, "bic": f_bic, "lim": f_lim, "lados": f_lados}
    return float(f_n * f_rms * f_bic * f_lim * f_lados), partes


def h_confiable(res: ResultadoLado, cfg: dict) -> bool:
    cc = cfg["confianza"]
    return bool(res.ok and res.conf >= cc["conf_min"] and np.isfinite(res.se_h)
                and res.se_h <= cc["se_h_max"] and not res.en_limite)
