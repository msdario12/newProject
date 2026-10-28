"""Geometría del eje de calle: estaciones, puntos, normales y proyección.

Convención de signos: el offset es POSITIVO A LA IZQUIERDA del eje mirando en
el sentido de digitalización de la polilínea (estación creciente).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Eje:
    nombre: str
    xy: np.ndarray                      # (M, 2) vértices
    _est: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        xy = np.asarray(self.xy, dtype=float)[:, :2]
        # quitar vértices consecutivos repetidos
        keep = np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 1e-6]
        self.xy = xy[keep]
        if len(self.xy) < 2:
            raise ValueError(f"Eje {self.nombre}: menos de 2 vértices válidos")
        seg = np.diff(self.xy, axis=0)
        self._est = np.r_[0.0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
        self._subseg = None

    # ------------------------------------------------------------------ básicos
    @property
    def longitud(self) -> float:
        return float(self._est[-1])

    def punto(self, s) -> np.ndarray:
        """Coordenadas XY en la(s) estación(es) s."""
        s = np.clip(np.atleast_1d(np.asarray(s, dtype=float)), 0.0, self.longitud)
        return np.column_stack([np.interp(s, self._est, self.xy[:, 0]),
                                np.interp(s, self._est, self.xy[:, 1])])

    def tangente(self, s, delta: float = 1.0) -> np.ndarray:
        """Tangente unitaria suavizada (diferencia centrada de ±delta) para
        evitar quiebres bruscos de la normal en los vértices."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        a = self.punto(s - delta)
        b = self.punto(s + delta)
        t = b - a
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    def normal_izq(self, s, delta: float = 1.0) -> np.ndarray:
        t = self.tangente(s, delta)
        return np.column_stack([-t[:, 1], t[:, 0]])

    def punto_offset(self, s, offset) -> np.ndarray:
        """XY del punto a `offset` (positivo izquierda) de la estación s."""
        return self.punto(s) + self.normal_izq(s) * np.atleast_1d(offset)[:, None]

    def densificar(self, paso: float) -> np.ndarray:
        n = max(2, int(np.ceil(self.longitud / paso)) + 1)
        return self.punto(np.linspace(0.0, self.longitud, n))

    def bbox(self, buffer: float = 0.0):
        mn = self.xy.min(axis=0) - buffer
        mx = self.xy.max(axis=0) + buffer
        return mn[0], mn[1], mx[0], mx[1]

    # --------------------------------------------------------------- proyección
    def _preparar_subsegmentos(self, largo_max: float = 1.0):
        """Divide el eje en subsegmentos cortos e indexa sus puntos medios con
        un cKDTree, para proyectar millones de puntos sin recorrer todo el eje."""
        a_list, b_list, s0_list = [], [], []
        for i in range(len(self.xy) - 1):
            p, q = self.xy[i], self.xy[i + 1]
            L = self._est[i + 1] - self._est[i]
            n = max(1, int(np.ceil(L / largo_max)))
            f = np.linspace(0.0, 1.0, n + 1)
            pts = p + (q - p) * f[:, None]
            a_list.append(pts[:-1])
            b_list.append(pts[1:])
            s0_list.append(self._est[i] + L * f[:-1])
        a = np.vstack(a_list)
        b = np.vstack(b_list)
        self._subseg = {
            "a": a,
            "b": b,
            "s0": np.concatenate(s0_list),
            "arbol": cKDTree(0.5 * (a + b)),
        }

    def proyectar(self, x: np.ndarray, y: np.ndarray, k: int = 8, bloque: int = 1_000_000):
        """Proyecta puntos al eje.

        Devuelve (s, o, dentro):
            s       estación de la proyección,
            o       offset firmado (positivo a la izquierda),
            dentro  False si la proyección cae fuera de los extremos del eje.
        """
        if self._subseg is None:
            self._preparar_subsegmentos()
        ss = self._subseg
        nseg = len(ss["a"])
        k = min(k, nseg)
        n = len(x)
        s_out = np.empty(n)
        o_out = np.empty(n)
        dentro = np.empty(n, dtype=bool)

        for i0 in range(0, n, bloque):
            i1 = min(n, i0 + bloque)
            p = np.column_stack([x[i0:i1], y[i0:i1]])
            _, idx = ss["arbol"].query(p, k=k)
            idx = idx.reshape(len(p), -1)
            a = ss["a"][idx]                  # (m, k, 2)
            b = ss["b"][idx]
            ab = b - a
            L2 = np.einsum("mkj,mkj->mk", ab, ab)
            t_raw = np.einsum("mkj,mkj->mk", p[:, None, :] - a, ab) / L2
            t = np.clip(t_raw, 0.0, 1.0)
            q = a + ab * t[..., None]
            d2 = np.einsum("mkj,mkj->mk", p[:, None, :] - q, p[:, None, :] - q)
            j = np.argmin(d2, axis=1)
            r = np.arange(len(p))
            sel = idx[r, j]
            tj, traw, abj, aj = t[r, j], t_raw[r, j], ab[r, j], a[r, j]
            L = np.sqrt(L2[r, j])
            s_out[i0:i1] = ss["s0"][sel] + tj * L
            dist = np.sqrt(d2[r, j])
            cruz = abj[:, 0] * (p[:, 1] - aj[:, 1]) - abj[:, 1] * (p[:, 0] - aj[:, 0])
            o_out[i0:i1] = np.where(cruz >= 0, dist, -dist)
            fuera = ((sel == 0) & (traw < 0)) | ((sel == nseg - 1) & (traw > 1))
            dentro[i0:i1] = ~fuera
        return s_out, o_out, dentro
