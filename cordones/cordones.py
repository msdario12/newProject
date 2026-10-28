#!/usr/bin/env python3
"""Inferencia de líneas de cuneta y cabeza de cordón a partir de LiDAR aéreo.

Uso:
    python cordones.py --config config.yaml
    python cordones.py --config config.yaml --debug            # un solo eje, gráficos de cada ventana
    python cordones.py --config config.yaml --debug --eje 3    # eje por índice o nombre
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from cordones_lib import ajuste, apilado, exportacion, graficos, lectura, postproceso, raster
from cordones_lib.config import cargar_config, params_geometria
from cordones_lib.eje import Eje

log = logging.getLogger("cordones")
LADOS = (("izq", 1), ("der", -1))


def procesar_eje(eje: Eje, cfg: dict, carpeta: Path, debug: bool) -> tuple[list[dict], dict]:
    t0 = time.time()
    geo = params_geometria(cfg, eje.nombre)
    vcfg, rcfg, gcfg = cfg["ventana"], cfg["raster"], cfg["graficos"]

    d_esp = geo["ancho_calzada"] / 2
    dmin = max(d_esp - geo["tolerancia_offset"], 0.5)
    dmax = d_esp + geo["tolerancia_offset"]
    offset_max = dmax + geo["ancho_vereda"] + geo["margen"]
    log.info("Eje %s: longitud %.1f m, d ∈ [%.2f, %.2f] m, banda ±%.2f m",
             eje.nombre, eje.longitud, dmin, dmax, offset_max)

    # 1) lectura del buffer + suelo
    datos = lectura.leer_puntos_eje(eje, cfg, offset_max + 1.0, carpeta / "cache")

    # 2) rasgo radiométrico y proyección a (s, o)
    usar_raster = rcfg["usar"] != "no"
    rasgo = apilado.rasgo_radiometrico(datos, rcfg["campo"]) if usar_raster else None
    if rcfg["usar"] == "si" and rasgo is None:
        log.warning("  raster.usar = si, pero la nube no tiene intensidad ni RGB")
    nube = apilado.proyectar_nube(eje, datos, offset_max, rasgo)
    log.info("  puntos proyectados en la banda: %d (%.1f pts/m²)", len(nube.s),
             len(nube.s) / max(eje.longitud * 2 * offset_max, 1e-9))
    ras = raster.construir_raster(nube, eje.longitud, offset_max, rcfg["resolucion"]) if rasgo is not None else None

    # 3) ventanas
    estaciones = apilado.estaciones_ventanas(eje.longitud, vcfg["longitud"], vcfg["paso"])
    carpeta_dbg = carpeta / "graficos" / eje.nombre / "ventanas"
    if gcfg["generar"]:
        carpeta_dbg.mkdir(parents=True, exist_ok=True)
    cada = 1 if debug else max(1, int(gcfg["cada_n_ventanas"]))

    filas: list[dict] = []
    por_lado = {lado: {k: [] for k in ("s", "d", "zc", "h", "conf", "h_ok")} for lado, _ in LADOS}
    for iv, s_c in enumerate(estaciones):
        ds, o, z, _ = nube.ventana(s_c, vcfg["longitud"])
        xy = eje.punto(s_c)[0]
        perfil_r = ras.perfil(s_c, vcfg["longitud"]) if ras is not None else None
        info_graf = {}
        for lado, sgn in LADOS:
            u = sgn * o
            m = (u > geo["u_min"]) & (u <= offset_max)
            res = ajuste.ajustar_lado(u[m], ds[m], z[m], dmin, dmax, cfg)

            d_ras, c_ras, ratio = (np.nan, 0.0, 0.0)
            if ras is not None:
                d_ras, c_ras, ratio = raster.detectar_borde(ras, s_c, vcfg["longitud"], sgn,
                                                            dmin, dmax, rcfg)
            d_geo = res.d if res.ok else np.nan
            d_fin, conf, fuente = raster.combinar(d_geo, res.conf if res.ok else 0.0, d_ras, c_ras, rcfg)

            # cotas en el offset final: calzada evaluada en d (centro de ventana)
            zc = res.z0 + res.sc * d_fin if res.ok and np.isfinite(d_fin) else np.nan
            h_ok = ajuste.h_confiable(res, cfg)

            por_lado[lado]["s"].append(s_c)
            por_lado[lado]["d"].append(d_fin)
            por_lado[lado]["zc"].append(zc)
            por_lado[lado]["h"].append(res.h)
            por_lado[lado]["conf"].append(conf)
            por_lado[lado]["h_ok"].append(h_ok)

            cp = res.conf_partes
            filas.append({
                "eje": eje.nombre, "lado": lado, "estacion": s_c, "x_eje": xy[0], "y_eje": xy[1],
                "n_puntos": res.n, "n_calzada": res.n_calzada, "n_vereda": res.n_vereda,
                "offset_geom": d_geo, "offset_raster": d_ras, "conf_raster": c_ras, "ratio_raster": ratio,
                "offset_final": d_fin, "fuente_offset": fuente,
                "z_cuneta": zc, "h": res.h, "z_cabeza": zc + res.h if np.isfinite(zc) else np.nan,
                "pend_calzada": res.sc, "pend_vereda": res.sv, "pend_long": res.g,
                "rms": res.rms, "delta_bic": res.delta_bic, "se_offset": res.se_d, "se_h": res.se_h,
                "en_limite": res.en_limite, "conf_geom": res.conf, "conf": conf,
                "f_n": cp.get("n", np.nan), "f_rms": cp.get("rms", np.nan), "f_bic": cp.get("bic", np.nan),
                "f_lim": cp.get("lim", np.nan), "f_lados": cp.get("lados", np.nan),
                "h_confiable": h_ok, "motivo": res.motivo,
            })
            info_graf[lado] = {"o": o[m], "ds": ds[m], "z": z[m], "res": res, "d_ras": d_ras,
                               "d_final": d_fin, "fuente": fuente, "conf": conf}

        if gcfg["generar"] and iv % cada == 0:
            graficos.grafico_ventana(carpeta_dbg / f"est_{s_c:08.1f}.png", eje.nombre, s_c, info_graf,
                                     perfil_r, ras.o_centros if ras is not None else None,
                                     cfg["ajuste"]["ancho_transicion"], gcfg["dpi"])

    # 4) post-proceso y reconstrucción en X, Y, Z
    lineas = {}
    for lado, sgn in LADOS:
        P = {k: np.asarray(v, dtype=float) for k, v in por_lado[lado].items()}
        L = postproceso.procesar_lado(P["s"], P["d"], P["zc"], P["h"], P["conf"],
                                      P["h_ok"].astype(bool), cfg)
        d = np.nan_to_num(L["d"])
        xy = eje.punto_offset(L["s"], sgn * d)
        L["xyz_cuneta"] = np.column_stack([xy, L["z_cuneta"]])
        L["xyz_cabeza"] = np.column_stack([xy, L["z_cabeza"]])
        lineas[lado] = L
        ok = L["estado_d"] > 0
        log.info("  %s: %.0f %% de estaciones con línea (medidas %d, interpoladas %d, h asumida %d, dudosas %d)",
                 lado, 100 * ok.mean() if len(ok) else 0, (L["estado_d"] == 1).sum(),
                 (L["estado_d"] == 2).sum(), (ok & (L["estado_h"] == 3)).sum(), (ok & L["dudoso"]).sum())

    if gcfg["generar"]:
        carpeta_eje = carpeta / "graficos" / eje.nombre
        graficos.grafico_eje(carpeta_eje / "perfil_longitudinal.png", eje.nombre, filas, lineas, gcfg["dpi"])
        graficos.grafico_raster(carpeta_eje / "raster.png", eje.nombre, ras, lineas, gcfg["dpi"])

    log.info("  eje %s procesado en %.1f s (%d ventanas)", eje.nombre, time.time() - t0, len(estaciones))
    return filas, {"nombre": eje.nombre, "lineas": lineas}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inferencia de cordones y cunetas desde LiDAR aéreo.")
    ap.add_argument("--config", required=True, help="archivo YAML o JSON de parámetros")
    ap.add_argument("--debug", action="store_true",
                    help="procesa un solo eje y guarda el gráfico de control de cada ventana")
    ap.add_argument("--eje", default=None, help="índice o nombre del eje (por defecto debug.eje en modo --debug)")
    ap.add_argument("--sin-graficos", action="store_true", help="no generar gráficos")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ezdxf").setLevel(logging.WARNING)
    cfg = cargar_config(args.config)
    if args.sin_graficos:
        cfg["graficos"]["generar"] = False

    carpeta = Path(cfg["salida"]["carpeta"])
    if args.debug:
        carpeta = carpeta / "debug"
    carpeta.mkdir(parents=True, exist_ok=True)

    ejes = lectura.leer_ejes(cfg)
    selector = args.eje if args.eje is not None else (cfg["debug"]["eje"] if args.debug else None)
    ejes = lectura.seleccionar_ejes(ejes, selector)
    if args.debug:
        ejes = ejes[:1]
        log.info("MODO DEBUG: eje %s, gráfico de cada ventana en %s", ejes[0].nombre, carpeta / "graficos")

    todas_filas, resultados = [], []
    for eje in ejes:
        try:
            filas, res = procesar_eje(eje, cfg, carpeta, args.debug)
        except Exception as exc:
            if args.debug:
                raise
            log.error("Eje %s: error, se omite (%s)", eje.nombre, exc)
            continue
        todas_filas += filas
        resultados.append(res)

    if not resultados:
        log.error("No se procesó ningún eje")
        return 1

    pre = carpeta / cfg["salida"]["prefijo"]
    exportacion.exportar_csv_ventanas(pre.with_name(pre.name + "_ventanas.csv"), todas_filas)
    exportacion.exportar_csv_lineas(pre.with_name(pre.name + "_lineas.csv"), resultados)
    n = exportacion.exportar_dxf(pre.with_suffix(".dxf"), resultados)
    log.info("DXF: %d polilíneas 3D -> %s", n, pre.with_suffix(".dxf"))
    if cfg["salida"]["landxml"]:
        exportacion.exportar_landxml(pre.with_suffix(".xml"), resultados, cfg["entrada"]["epsg"])
        log.info("LandXML -> %s", pre.with_suffix(".xml"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
