"""Lectura de ejes (DXF/SHP) y de la nube LAS/LAZ recortada a un buffer del eje.

Para nubes grandes:
  * se aceptan varios archivos (tiles) mediante un patrón glob;
  * se descartan los tiles cuyo encabezado no intersecta el buffer del eje;
  * cada tile se lee por bloques (chunk_iterator) y solo se retienen los
    puntos a menos de `buffer` del eje (consulta con cKDTree);
  * opcionalmente se guarda el recorte en un .npz para relecturas rápidas
    mientras se calibran parámetros.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .eje import Eje

log = logging.getLogger(__name__)


# =============================================================================
# Ejes
# =============================================================================
def leer_ejes(cfg: dict) -> list[Eje]:
    ruta = Path(cfg["entrada"]["ejes"])
    suf = ruta.suffix.lower()
    if suf == ".dxf":
        ejes = _leer_ejes_dxf(ruta, cfg["entrada"].get("capa_ejes"))
    elif suf == ".shp":
        ejes = _leer_ejes_shp(ruta, cfg["entrada"].get("campo_nombre"))
    else:
        raise ValueError(f"Formato de ejes no soportado: {suf} (usar .dxf o .shp)")
    if not ejes:
        raise ValueError(f"No se encontraron polilíneas de eje en {ruta}")
    log.info("Ejes leídos: %d", len(ejes))
    return ejes


def _leer_ejes_dxf(ruta: Path, capa: str | None) -> list[Eje]:
    import ezdxf
    from ezdxf import path as ezpath

    doc = ezdxf.readfile(ruta)
    ejes = []
    for e in doc.modelspace().query("LWPOLYLINE POLYLINE LINE SPLINE ARC"):
        if capa and e.dxf.layer != capa:
            continue
        # make_path + flattening resuelve arcos (bulges) con tolerancia de 2 cm
        try:
            pts = np.array([(v.x, v.y) for v in ezpath.make_path(e).flattening(0.02)])
        except Exception as exc:  # entidades degeneradas
            log.warning("Entidad %s ignorada: %s", e.dxf.handle, exc)
            continue
        if len(pts) < 2:
            continue
        nombre = f"{e.dxf.layer}_{e.dxf.handle}"
        try:
            ejes.append(Eje(nombre, pts))
        except ValueError as exc:
            log.warning(str(exc))
    return ejes


def _leer_ejes_shp(ruta: Path, campo: str | None) -> list[Eje]:
    import shapefile  # pyshp

    ejes = []
    with shapefile.Reader(str(ruta)) as sf:
        campos = [f[0] for f in sf.fields[1:]]
        for i, sr in enumerate(sf.iterShapeRecords()):
            shp = sr.shape
            if shp.shapeType not in (shapefile.POLYLINE, shapefile.POLYLINEZ, shapefile.POLYLINEM):
                continue
            base = str(sr.record[campos.index(campo)]) if campo in campos else f"eje{i:04d}"
            partes = list(shp.parts) + [len(shp.points)]
            for j in range(len(partes) - 1):
                pts = np.array(shp.points[partes[j]:partes[j + 1]])[:, :2]
                nombre = base if len(partes) == 2 else f"{base}_p{j}"
                try:
                    ejes.append(Eje(nombre, pts))
                except ValueError as exc:
                    log.warning(str(exc))
    return ejes


def seleccionar_ejes(ejes: list[Eje], selector) -> list[Eje]:
    """Selecciona ejes por índice (int o str numérico) o por nombre."""
    if selector is None:
        return ejes
    sel = str(selector)
    por_nombre = [e for e in ejes if e.nombre == sel]
    if por_nombre:
        return por_nombre
    if sel.lstrip("-").isdigit():
        return [ejes[int(sel)]]
    raise ValueError(f"No existe el eje '{selector}'. Disponibles: {[e.nombre for e in ejes]}")


# =============================================================================
# Nube de puntos
# =============================================================================
def _archivos_nube(patron: str) -> list[str]:
    archivos = sorted(glob.glob(patron))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos de nube con: {patron}")
    return archivos


def _clave_cache(archivos, eje: Eje, buffer: float, modo: str) -> str:
    h = hashlib.sha1()
    for a in archivos:
        st = Path(a).stat()
        h.update(f"{a}|{st.st_size}|{st.st_mtime_ns}".encode())
    h.update(eje.xy.tobytes())
    h.update(f"{buffer:.3f}|{modo}".encode())
    return h.hexdigest()[:16]


def leer_puntos_eje(eje: Eje, cfg: dict, buffer: float, carpeta_cache: Path | None = None) -> dict:
    """Devuelve los puntos de SUELO a menos de `buffer` m del eje.

    Diccionario: x, y, z (float64), intensidad (float32 o None), rgb ((N,3) o None).
    """
    ccl = cfg["clasificacion"]
    modo = ccl["modo"]
    extra = ccl["buffer_extra"] if modo in ("smrf", "csf", "auto") else 0.0
    archivos = _archivos_nube(cfg["entrada"]["nube"])

    ruta_cache = None
    if carpeta_cache is not None and cfg["lectura"]["cache"]:
        carpeta_cache.mkdir(parents=True, exist_ok=True)
        ruta_cache = carpeta_cache / f"{eje.nombre}_{_clave_cache(archivos, eje, buffer, modo)}.npz"
        if ruta_cache.exists():
            log.info("  puntos desde caché: %s", ruta_cache.name)
            d = np.load(ruta_cache, allow_pickle=False)
            return {k: (d[k] if d[k].size else None) for k in d.files}

    crudo = _leer_buffer(archivos, eje, buffer + extra, cfg["lectura"]["tamano_bloque"])
    n = len(crudo["x"])
    log.info("  puntos en buffer (todas las clases): %d", n)
    if n == 0:
        raise RuntimeError(f"Eje {eje.nombre}: no hay puntos de la nube en el buffer")

    es_suelo = crudo["clase"] == ccl["clase_suelo"]
    usar_pdal = modo in ("smrf", "csf")
    if modo == "auto":
        frac = es_suelo.mean()
        usar_pdal = frac < ccl["fraccion_min_suelo"]
        log.info("  fracción clase %d: %.1f %% -> %s", ccl["clase_suelo"], 100 * frac,
                 "clasificar con PDAL/SMRF" if usar_pdal else "usar clasificación existente")

    if usar_pdal:
        metodo = "csf" if modo == "csf" else "smrf"
        mask = clasificar_suelo_pdal(crudo, metodo, ccl[metodo])
    else:
        mask = es_suelo

    # recorte final al buffer pedido (el extra solo era contexto para el filtro de suelo)
    if extra > 0:
        mask &= _distancia_eje(eje, crudo["x"], crudo["y"]) <= buffer

    datos = {
        "x": crudo["x"][mask],
        "y": crudo["y"][mask],
        "z": crudo["z"][mask],
        "intensidad": crudo["intensidad"][mask] if crudo["intensidad"] is not None else None,
        "rgb": crudo["rgb"][mask] if crudo["rgb"] is not None else None,
    }
    log.info("  puntos de suelo en buffer: %d", len(datos["x"]))

    if ruta_cache is not None:
        np.savez(ruta_cache, **{k: (v if v is not None else np.empty(0)) for k, v in datos.items()})
    return datos


def _distancia_eje(eje: Eje, x, y, paso: float = 0.5) -> np.ndarray:
    """Distancia aproximada al eje usando sus vértices densificados (por defecto,
    nunca sobreestima: el recorte resulta levemente holgado)."""
    arbol = cKDTree(eje.densificar(paso))
    d, _ = arbol.query(np.column_stack([x, y]), k=1)
    # corrección: un punto a distancia r de la línea queda a ≤ sqrt(r²+(paso/2)²) del vértice
    return np.sqrt(np.maximum(d ** 2 - (paso / 2) ** 2, 0.0))


def _leer_buffer(archivos, eje: Eje, buffer: float, tamano_bloque: int) -> dict:
    import laspy

    xmin, ymin, xmax, ymax = eje.bbox(buffer)
    arbol = cKDTree(eje.densificar(0.5))
    radio = np.hypot(buffer, 0.25)

    partes = {k: [] for k in ("x", "y", "z", "clase", "intensidad", "rgb")}
    tiene_int = tiene_rgb = True
    for ruta in archivos:
        with laspy.open(ruta) as lf:
            hd = lf.header
            if hd.maxs[0] < xmin or hd.mins[0] > xmax or hd.maxs[1] < ymin or hd.mins[1] > ymax:
                continue
            dims = set(hd.point_format.dimension_names)
            tiene_int &= "intensity" in dims
            tiene_rgb &= {"red", "green", "blue"} <= dims
            log.debug("  leyendo %s", ruta)
            for bloque in lf.chunk_iterator(tamano_bloque):
                x = np.asarray(bloque.x, dtype=float)
                y = np.asarray(bloque.y, dtype=float)
                m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                if not m.any():
                    continue
                idx = np.flatnonzero(m)
                d, _ = arbol.query(np.column_stack([x[idx], y[idx]]), k=1,
                                   distance_upper_bound=radio)
                idx = idx[np.isfinite(d)]
                if idx.size == 0:
                    continue
                partes["x"].append(x[idx])
                partes["y"].append(y[idx])
                partes["z"].append(np.asarray(bloque.z, dtype=float)[idx])
                partes["clase"].append(np.asarray(bloque.classification, dtype=np.uint8)[idx])
                if tiene_int:
                    partes["intensidad"].append(np.asarray(bloque.intensity, dtype=np.float32)[idx])
                if tiene_rgb:
                    partes["rgb"].append(np.column_stack([
                        np.asarray(bloque.red)[idx], np.asarray(bloque.green)[idx],
                        np.asarray(bloque.blue)[idx]]).astype(np.float32))

    if not partes["x"]:
        return {"x": np.empty(0), "y": np.empty(0), "z": np.empty(0), "clase": np.empty(0, np.uint8),
                "intensidad": None, "rgb": None}
    n_bloques = len(partes["x"])
    out = {k: np.concatenate(partes[k]) for k in ("x", "y", "z", "clase")}
    out["intensidad"] = (np.concatenate(partes["intensidad"])
                         if tiene_int and len(partes["intensidad"]) == n_bloques else None)
    out["rgb"] = (np.vstack(partes["rgb"])
                  if tiene_rgb and len(partes["rgb"]) == n_bloques else None)
    if out["intensidad"] is not None and np.ptp(out["intensidad"]) == 0:
        out["intensidad"] = None   # campo presente pero vacío
    if out["rgb"] is not None and np.ptp(out["rgb"]) == 0:
        out["rgb"] = None
    return out


# =============================================================================
# Clasificación de suelo con PDAL (solo si la nube no viene clasificada)
# =============================================================================
def _etapas_pdal(metodo: str, prm: dict) -> list[dict]:
    etapas = [
        {"type": "filters.assign", "value": "Classification = 0"},
        {"type": "filters.elm"},   # marca puntos bajos aislados como ruido (clase 7)
        {"type": "filters.outlier", "method": "statistical", "mean_k": 8, "multiplier": 3.0},
    ]
    if metodo == "smrf":
        etapas.append({"type": "filters.smrf", "ignore": "Classification[7:7]", **prm})
    else:
        etapas.append({"type": "filters.csf", "ignore": "Classification[7:7]", **prm})
    return etapas


def clasificar_suelo_pdal(crudo: dict, metodo: str, prm: dict) -> np.ndarray:
    """Clasifica suelo con PDAL (SMRF o CSF). Devuelve una máscara booleana.

    Usa los bindings de Python de PDAL si están instalados; si no, la CLI `pdal`.
    """
    n = len(crudo["x"])
    arr = np.zeros(n, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("ReturnNumber", "u1"),
                             ("NumberOfReturns", "u1"), ("Classification", "u1"),
                             ("OrigId", "u4")])
    arr["X"], arr["Y"], arr["Z"] = crudo["x"], crudo["y"], crudo["z"]
    arr["ReturnNumber"] = 1
    arr["NumberOfReturns"] = 1
    arr["OrigId"] = np.arange(n, dtype=np.uint32)
    etapas = _etapas_pdal(metodo, prm)

    try:
        import pdal  # noqa: F401
        pipe = pdal.Pipeline(json.dumps(etapas), arrays=[arr])
        pipe.execute()
        res = np.concatenate(pipe.arrays)
    except ImportError:
        if shutil.which("pdal") is None:
            raise RuntimeError(
                "La nube no tiene suelo clasificado y PDAL no está disponible. "
                "Instalar PDAL (conda install -c conda-forge pdal python-pdal) o "
                "clasificar la nube antes (clase 2).")
        res = _pdal_cli(arr, etapas)

    mask = np.zeros(n, dtype=bool)
    suelo = res["Classification"] == 2
    mask[res["OrigId"][suelo].astype(np.int64)] = True
    log.info("  PDAL/%s: %d de %d puntos clasificados como suelo", metodo.upper(), mask.sum(), n)
    return mask


def _pdal_cli(arr: np.ndarray, etapas: list[dict]) -> np.ndarray:
    """Alternativa con la CLI de PDAL usando archivos temporales LAS."""
    import laspy

    with tempfile.TemporaryDirectory() as tmp:
        ent = Path(tmp) / "entrada.las"
        sal = Path(tmp) / "salida.las"
        hd = laspy.LasHeader(point_format=6, version="1.4")
        hd.offsets = [arr["X"].min(), arr["Y"].min(), arr["Z"].min()]
        hd.scales = [0.001, 0.001, 0.001]
        hd.add_extra_dim(laspy.ExtraBytesParams(name="OrigId", type=np.uint32))
        las = laspy.LasData(hd)
        las.x, las.y, las.z = arr["X"], arr["Y"], arr["Z"]
        las.return_number = 1
        las.number_of_returns = 1
        las.OrigId = arr["OrigId"]
        las.write(ent)
        pipeline = [str(ent), *etapas,
                    {"type": "writers.las", "filename": str(sal), "extra_dims": "all",
                     "minor_version": 4, "dataformat_id": 6}]
        ruta_json = Path(tmp) / "pipeline.json"
        ruta_json.write_text(json.dumps(pipeline))
        subprocess.run(["pdal", "pipeline", str(ruta_json)], check=True)
        out = laspy.read(sal)
        res = np.zeros(len(out.x), dtype=[("Classification", "u1"), ("OrigId", "u4")])
        res["Classification"] = np.asarray(out.classification)
        res["OrigId"] = np.asarray(out.OrigId)
        return res
