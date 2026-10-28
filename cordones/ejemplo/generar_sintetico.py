#!/usr/bin/env python3
"""Genera un caso sintético (2 cuadras) para probar el flujo completo.

Crea en la carpeta indicada:
    nube.laz      LiDAR simulado ~20 pts/m², ruido vertical σ=7 cm, clase 2 + no-suelo
    ejes.dxf      eje de calle (LWPOLYLINE con tramo recto y curva)
    verdad.csv    offset y altura reales por estación
    config.yaml   configuración lista para usar

Escenario: calzada de 9 m (4,5 m a la izquierda, 4,6 m a la derecha: el eje no
está perfectamente centrado), cordón de 0,15 m (izq) y 0,17 m (der), bombeo
2,5 %, vereda de 3 m con 2 % hacia la calle, rasante con 1,2 % de pendiente,
un rebaje de garaje y una bocacalle sin cordón.

Uso: python ejemplo/generar_sintetico.py ejemplo/datos
"""

import sys
from pathlib import Path

import ezdxf
import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cordones_lib.eje import Eje  # noqa: E402

RNG = np.random.default_rng(42)
DENSIDAD = 20.0
SIGMA_Z = 0.07
X0, Y0 = 5_520_000.0, 6_150_000.0     # coordenadas tipo POSGAR


def eje_xy():
    recto = np.column_stack([np.linspace(0, 120, 7), np.zeros(7)])
    ang = np.linspace(0, np.deg2rad(25), 12)[1:]
    R = 200.0
    curva = np.column_stack([120 + R * np.sin(ang), R - R * np.cos(ang)])
    return np.vstack([recto, curva]) + [X0, Y0]


def verdad(s, lado):
    """Offset y altura de cordón reales en función de la estación y el lado."""
    d = np.where(lado > 0, 4.5, 4.6) + 0.10 * np.sin(s / 40)       # leve variación
    h = np.where(lado > 0, 0.15, 0.17) * np.ones_like(s)
    rebaje = (lado > 0) & (s > 60) & (s < 66)
    h = np.where(rebaje, 0.03, h)
    bocacalle = (s > 100) & (s < 112)
    return d, h, rebaje, bocacalle


def superficie(s, o):
    lado = np.where(o >= 0, 1, -1)
    u = np.abs(o)
    d, h, _, boca = verdad(s, lado)
    rasante = 25.0 + 0.012 * s + 2e-5 * (s - 100) ** 2
    calzada = rasante - 0.025 * np.minimum(u, d)
    vereda = rasante - 0.025 * d + h + 0.02 * (u - d)
    z = np.where(u < d, calzada, vereda)
    # bocacalle: la calzada continúa con bombeo, sin cordón
    z = np.where(boca, rasante - 0.025 * u, z)
    es_hormigon = (u >= d) & ~boca
    return z, es_hormigon


def main(carpeta: str):
    out = Path(carpeta)
    out.mkdir(parents=True, exist_ok=True)
    xy = eje_xy()
    eje = Eje("sintetico", xy)
    L, W = eje.longitud, 11.0

    n = int(DENSIDAD * L * 2 * W)
    s = RNG.uniform(0, L, n)
    o = RNG.uniform(-W, W, n)
    # huecos por autos estacionados en el carril derecho
    autos = np.zeros(n, bool)
    for s0 in (20, 45, 140, 170):
        autos |= (s > s0) & (s < s0 + 4.5) & (o < -2.4) & (o > -4.3)
    s, o = s[~autos], o[~autos]

    z, hormigon = superficie(s, o)
    z = z + RNG.normal(0, SIGMA_Z, len(z))
    p = eje.punto_offset(s, o)
    inten = np.where(hormigon, 3200, 1700) + RNG.normal(0, 450, len(z))
    clase = np.full(len(z), 2, np.uint8)

    # puntos no-suelo: árboles sobre la vereda y fachadas al fondo
    m = 4000
    sa = RNG.uniform(0, L, m)
    oa = RNG.choice([-1, 1], m) * RNG.uniform(5.5, 7.5, m)
    pa = eje.punto_offset(sa, oa)
    za = superficie(sa, oa)[0] + RNG.uniform(2, 8, m)

    X = np.r_[p[:, 0], pa[:, 0]]
    Y = np.r_[p[:, 1], pa[:, 1]]
    Z = np.r_[z, za]
    hdr = laspy.LasHeader(point_format=1, version="1.4")
    hdr.offsets = [X.min(), Y.min(), Z.min()]
    hdr.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = X, Y, Z
    las.intensity = np.clip(np.r_[inten, RNG.uniform(500, 4000, m)], 0, 65535).astype(np.uint16)
    las.classification = np.r_[clase, np.full(m, 5, np.uint8)]
    las.write(out / "nube.laz")

    doc = ezdxf.new("R2010")
    doc.layers.add("EJES")
    doc.modelspace().add_lwpolyline(xy.tolist(), dxfattribs={"layer": "EJES"})
    doc.saveas(out / "ejes.dxf")

    est = np.arange(0, L, 1.0)
    with open(out / "verdad.csv", "w") as f:
        f.write("estacion,lado,offset,h,rebaje,bocacalle\n")
        for lado, nombre in ((1, "izq"), (-1, "der")):
            d, h, reb, boca = verdad(est, np.full(len(est), lado))
            for i in range(len(est)):
                f.write(f"{est[i]:.1f},{nombre},{d[i]:.4f},{h[i]:.4f},{int(reb[i])},{int(boca[i])}\n")

    cfg_base = Path(__file__).resolve().parents[1] / "config.yaml"
    txt = cfg_base.read_text(encoding="utf-8")
    txt = txt.replace('nube: "datos/nube.laz"', 'nube: "nube.laz"').replace(
        'ejes: "datos/ejes.dxf"', 'ejes: "ejes.dxf"')
    (out / "config.yaml").write_text(txt, encoding="utf-8")
    print(f"Caso sintético: {len(X)} puntos, eje de {L:.1f} m -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ejemplo/datos")
