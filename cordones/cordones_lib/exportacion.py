"""Exportación: DXF (POLYLINE 3D por capa), CSV de control y LandXML."""

from __future__ import annotations

import csv
import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .postproceso import INTERPOLADO, ASUMIDO, MEDIDO

CAPAS = {
    "CUNETA_IZQ": 4,      # cian
    "CUNETA_DER": 5,      # azul
    "CORDON_IZQ": 3,      # verde
    "CORDON_DER": 2,      # amarillo
    "BAJA_CONFIANZA": 1,  # rojo
}

NOMBRE_ESTADO_D = {0: "sin_dato", MEDIDO: "medido", INTERPOLADO: "interpolado"}
NOMBRE_ESTADO_H = {0: "sin_dato", MEDIDO: "medido", INTERPOLADO: "interpolado", ASUMIDO: "asumido"}


def _corridas(mask: np.ndarray):
    """(i0, i1) inclusive de cada corrida de True consecutivos."""
    m = np.r_[False, mask.astype(bool), False]
    dif = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(dif == 1), np.flatnonzero(dif == -1) - 1))


def lineas_3d(resultados_ejes: list[dict]):
    """Genera (capa, nombre, (N,3) array) para todas las polilíneas a exportar.

    Las capas principales llevan la línea completa (medida + interpolada +
    asumida); BAJA_CONFIANZA lleva copias de los tramos dudosos.
    """
    for r in resultados_ejes:
        for lado, suf in (("izq", "IZQ"), ("der", "DER")):
            L = r["lineas"][lado]
            hay = L["estado_d"] > 0
            for i0, i1 in _corridas(hay):
                if i1 - i0 < 1:
                    continue
                sl = slice(i0, i1 + 1)
                yield f"CUNETA_{suf}", f"{r['nombre']}_CUNETA_{suf}_{i0}", L["xyz_cuneta"][sl]
                yield f"CORDON_{suf}", f"{r['nombre']}_CORDON_{suf}_{i0}", L["xyz_cabeza"][sl]
            # tramos dudosos: se extienden un vértice a cada lado para que se vean continuos
            for i0, i1 in _corridas(L["dudoso"] & hay):
                a = max(i0 - 1, 0)
                b = min(i1 + 1, len(hay) - 1)
                while not hay[a]:
                    a += 1
                while not hay[b]:
                    b -= 1
                if b - a < 1:
                    continue
                sl = slice(a, b + 1)
                yield "BAJA_CONFIANZA", f"{r['nombre']}_DUDOSO_CUNETA_{suf}_{a}", L["xyz_cuneta"][sl]
                yield "BAJA_CONFIANZA", f"{r['nombre']}_DUDOSO_CORDON_{suf}_{a}", L["xyz_cabeza"][sl]


def exportar_dxf(ruta: Path, resultados_ejes: list[dict]) -> int:
    import ezdxf

    doc = ezdxf.new("R2010", setup=False)
    doc.units = ezdxf.units.M
    for capa, color in CAPAS.items():
        doc.layers.add(capa, color=color)
    msp = doc.modelspace()
    n = 0
    for capa, _, xyz in lineas_3d(resultados_ejes):
        msp.add_polyline3d([tuple(map(float, p)) for p in xyz], dxfattribs={"layer": capa})
        n += 1
    doc.saveas(ruta)
    return n


CAMPOS_VENTANAS = [
    "eje", "lado", "estacion", "x_eje", "y_eje", "n_puntos", "n_calzada", "n_vereda",
    "offset_geom", "offset_raster", "conf_raster", "ratio_raster", "offset_final", "fuente_offset",
    "z_cuneta", "h", "z_cabeza", "pend_calzada", "pend_vereda", "pend_long",
    "rms", "delta_bic", "se_offset", "se_h", "en_limite", "conf_geom", "conf",
    "f_n", "f_rms", "f_bic", "f_lim", "f_lados", "h_confiable", "motivo",
]


def exportar_csv_ventanas(ruta: Path, filas: list[dict]) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_VENTANAS, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({k: _fmt(v) for k, v in fila.items()})


def exportar_csv_lineas(ruta: Path, resultados_ejes: list[dict]) -> None:
    campos = ["eje", "lado", "estacion", "offset", "x_cuneta", "y_cuneta", "z_cuneta",
              "x_cabeza", "y_cabeza", "z_cabeza", "h", "conf", "estado_offset", "estado_h", "dudoso"]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(campos)
        for r in resultados_ejes:
            for lado in ("izq", "der"):
                L = r["lineas"][lado]
                for i in np.flatnonzero(L["estado_d"] > 0):
                    c, k = L["xyz_cuneta"][i], L["xyz_cabeza"][i]
                    w.writerow([r["nombre"], lado, _fmt(L["s"][i]), _fmt(L["d"][i]),
                                _fmt(c[0]), _fmt(c[1]), _fmt(c[2]), _fmt(k[0]), _fmt(k[1]), _fmt(k[2]),
                                _fmt(L["h"][i]), _fmt(L["conf"][i]),
                                NOMBRE_ESTADO_D[int(L["estado_d"][i])],
                                NOMBRE_ESTADO_H[int(L["estado_h"][i])], int(L["dudoso"][i])])


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.4f}"
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    return v


def exportar_landxml(ruta: Path, resultados_ejes: list[dict], epsg=None, nombre_sup="CORDONES") -> None:
    """LandXML 1.2 con una superficie que solo contiene breaklines (SourceData).

    Al importarlo en Civil 3D se crea una superficie con las breaklines como
    datos de origen. LandXML usa el orden Norte Este Cota (Y X Z).
    """
    ns = "http://www.landxml.org/schema/LandXML-1.2"
    ET.register_namespace("", ns)
    ahora = dt.datetime.now()
    raiz = ET.Element(f"{{{ns}}}LandXML", {
        "version": "1.2", "date": ahora.strftime("%Y-%m-%d"), "time": ahora.strftime("%H:%M:%S"),
    })
    unid = ET.SubElement(raiz, f"{{{ns}}}Units")
    ET.SubElement(unid, f"{{{ns}}}Metric", {
        "areaUnit": "squareMeter", "linearUnit": "meter", "volumeUnit": "cubicMeter",
        "temperatureUnit": "celsius", "pressureUnit": "HPA", "angularUnit": "decimal degrees",
        "directionUnit": "decimal degrees"})
    if epsg:
        ET.SubElement(raiz, f"{{{ns}}}CoordinateSystem", {"epsgCode": str(epsg)})
    ET.SubElement(raiz, f"{{{ns}}}Application", {"name": "cordones.py", "version": "1.0"})
    sups = ET.SubElement(raiz, f"{{{ns}}}Surfaces")
    sup = ET.SubElement(sups, f"{{{ns}}}Surface", {"name": nombre_sup})
    src = ET.SubElement(sup, f"{{{ns}}}SourceData")
    bls = ET.SubElement(src, f"{{{ns}}}Breaklines")
    for capa, nombre, xyz in lineas_3d(resultados_ejes):
        if capa == "BAJA_CONFIANZA":
            continue
        bl = ET.SubElement(bls, f"{{{ns}}}Breakline", {"name": nombre, "brkType": "standard",
                                                       "desc": capa})
        pl = ET.SubElement(bl, f"{{{ns}}}PntList3D")
        pl.text = " ".join(f"{p[1]:.4f} {p[0]:.4f} {p[2]:.4f}" for p in xyz)
    ET.ElementTree(raiz).write(ruta, encoding="utf-8", xml_declaration=True)
