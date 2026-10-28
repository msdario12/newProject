"""Prueba de punta a punta sobre el caso sintético.

Ejecutar desde la carpeta cordones/:   python -m pytest tests   (o: python tests/test_sintetico.py)
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]


def _leer_csv(ruta):
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_caso_sintetico():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(RAIZ / "ejemplo" / "generar_sintetico.py"), tmp], check=True)
        subprocess.run([sys.executable, str(RAIZ / "cordones.py"), "--config", f"{tmp}/config.yaml",
                        "--sin-graficos"], check=True, cwd=RAIZ)
        verdad = _leer_csv(Path(tmp) / "verdad.csv")
        lineas = _leer_csv(Path(tmp) / "salida" / "cordones_lineas.csv")
        for archivo in ("cordones.dxf", "cordones.xml", "cordones_ventanas.csv"):
            assert (Path(tmp) / "salida" / archivo).exists()

        for lado in ("izq", "der"):
            v = [r for r in verdad if r["lado"] == lado]
            sv = np.array([float(r["estacion"]) for r in v])
            dv = np.array([float(r["offset"]) for r in v])
            hv = np.array([float(r["h"]) for r in v])
            L = [r for r in lineas if r["lado"] == lado]
            s = np.array([float(r["estacion"]) for r in L])
            d = np.array([float(r["offset"]) for r in L])
            h = np.array([float(r["h"]) for r in L])
            normal = ~((s > 58) & (s < 68)) & ~((s > 98) & (s < 114))

            # cobertura: la bocacalle (100–112) debe quedar cortada, el resto con línea
            assert not np.any((s > 102) & (s < 110)), "la bocacalle no debería tener cordón"
            assert len(s) > 0.85 * (len(sv) - 16)
            err_d = np.abs(d - np.interp(s, sv, dv))[normal]
            err_h = (h - np.interp(s, sv, hv))[normal]
            assert np.percentile(err_d, 95) < 0.10, f"{lado}: offset p95 {np.percentile(err_d, 95):.3f}"
            assert np.sqrt(np.mean(err_h ** 2)) < 0.03, f"{lado}: h rmse"


if __name__ == "__main__":
    test_caso_sintetico()
    print("OK")
