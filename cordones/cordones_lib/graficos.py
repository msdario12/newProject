"""Gráficos de control (matplotlib, sin pantalla)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .ajuste import modelo_escalon, modelo_nulo  # noqa: E402

COLOR = {"izq": "#1f77b4", "der": "#d62728"}


def _medianas_por_bin(x, y, ancho=0.10):
    if len(x) == 0:
        return np.array([]), np.array([])
    b = np.floor(x / ancho).astype(int)
    ub, inv = np.unique(b, return_inverse=True)
    med = np.array([np.median(y[inv == k]) for k in range(len(ub))])
    cnt = np.bincount(inv)
    ok = cnt >= 3
    return (ub[ok] + 0.5) * ancho, med[ok]


def grafico_ventana(ruta: Path, eje: str, s_c: float, lados: dict, perfil_raster=None,
                    o_raster=None, ancho_transicion: float = 0.05, dpi: int = 110):
    """lados[lado] = dict(o, ds, z, res, d_ras, d_final, fuente, conf)."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    titulo = [f"Eje {eje} — estación {s_c:.1f} m"]
    zref = None
    for lado, info in lados.items():
        res = info["res"]
        sgn = 1 if lado == "izq" else -1
        o, ds, z = info["o"], info["ds"], info["z"]
        if len(o) == 0:
            continue
        # quitar la pendiente longitudinal para ver el perfil apilado "plano"
        g = res.g if res.ok else 0.0
        zp = z - g * ds
        if zref is None:
            zref = np.median(zp)
        ax.scatter(o, zp - zref, s=3, alpha=0.25, color=COLOR[lado], lw=0)
        bx, by = _medianas_por_bin(o, zp - zref)
        ax.plot(bx, by, ".", ms=4, color="k", alpha=0.6)
        if res.ok:
            uu = np.linspace(max(np.abs(o).min(), 0), np.abs(o).max(), 400)
            p = [res.z0, res.sc, res.h, res.sv, res.d, 0.0]
            ax.plot(sgn * uu, modelo_escalon(p, uu, 0 * uu, ancho_transicion) - zref,
                    color=COLOR[lado], lw=2, label=f"plantilla {lado}")
            if res.params_nulo is not None:
                pn = np.array(res.params_nulo, dtype=float)
                pn[4] = 0.0
                ax.plot(sgn * uu, modelo_nulo(pn, uu, 0 * uu, ancho_transicion) - zref,
                        color=COLOR[lado], lw=1, ls=":", label=f"sin escalón {lado}")
            ax.axvline(sgn * res.d, color=COLOR[lado], lw=0.8)
            titulo.append(f"{lado.upper()}: d={res.d:.2f} h={res.h:.3f}±{res.se_h:.3f} "
                          f"rms={res.rms:.3f} ΔBIC={res.delta_bic:.1f} conf={info['conf']:.2f} "
                          f"[{info['fuente']}]")
        else:
            titulo.append(f"{lado.upper()}: sin ajuste ({res.motivo})")
        if np.isfinite(info.get("d_ras", np.nan)):
            ax.axvline(sgn * info["d_ras"], color=COLOR[lado], lw=1, ls="--", label=f"borde raster {lado}")
        if np.isfinite(info.get("d_final", np.nan)):
            ax.axvline(sgn * info["d_final"], color="k", lw=1, ls="-.")

    if perfil_raster is not None and o_raster is not None:
        ax2 = ax.twinx()
        ax2.step(o_raster, perfil_raster, where="mid", color="0.5", lw=0.8)
        ax2.set_ylabel("rasgo raster (0–1)", color="0.5")
        ax2.set_ylim(-0.05, 2.0)
    ax.set_xlabel("offset [m] (+ izquierda)")
    ax.set_ylabel("z relativa (sin pendiente longitudinal) [m]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", fontsize=7, ncol=3)
    ax.set_title("\n".join(titulo), fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=dpi)
    plt.close(fig)


def grafico_eje(ruta: Path, eje: str, filas: list[dict], lineas: dict, dpi: int = 110):
    """Offset, h y confianza a lo largo del eje (bruto vs final)."""
    fig, axs = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for lado in ("izq", "der"):
        F = [f for f in filas if f["lado"] == lado]
        s = np.array([f["estacion"] for f in F])
        L = lineas[lado]
        c = COLOR[lado]
        ok = L["estado_d"] > 0
        axs[0].scatter(s, [f["offset_geom"] for f in F], s=6, color=c, alpha=0.35, label=f"geom {lado}")
        axs[0].scatter(s, [f["offset_raster"] for f in F], s=6, marker="x", color=c, alpha=0.35,
                       label=f"raster {lado}")
        axs[0].plot(L["s"], np.where(ok, L["d"], np.nan), color=c, lw=1.8, label=f"final {lado}")
        axs[1].scatter(s, [f["h"] for f in F], s=6, color=c, alpha=0.35)
        axs[1].plot(L["s"], np.where(ok, L["h"], np.nan), color=c, lw=1.8)
        asum = ok & (L["estado_h"] == 3)
        axs[1].plot(L["s"][asum], L["h"][asum], "s", ms=3, mfc="none", color=c, label=f"h asumida {lado}")
        axs[2].scatter(s, [f["z_cuneta"] for f in F], s=6, color=c, alpha=0.35)
        axs[2].plot(L["s"], np.where(ok, L["z_cuneta"], np.nan), color=c, lw=1.5)
        axs[3].plot(s, [f["conf"] for f in F], ".", ms=3, color=c)
        duda = ok & L["dudoso"]
        axs[3].plot(L["s"][duda], np.full(duda.sum(), -0.05 if lado == "izq" else -0.1), "|",
                    color=c, label=f"dudoso {lado}")
    axs[0].set_ylabel("offset cordón [m]")
    axs[1].set_ylabel("h [m]")
    axs[2].set_ylabel("z cuneta [m]")
    axs[3].set_ylabel("confianza")
    axs[3].set_xlabel("estación [m]")
    for a in axs:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7, ncol=6, loc="upper right")
    axs[0].set_title(f"Eje {eje}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=dpi)
    plt.close(fig)


def grafico_raster(ruta: Path, eje: str, raster, lineas: dict, dpi: int = 110):
    """Raster enderezado (estación vs offset) con las líneas finales encima."""
    if raster is None:
        return
    v = raster.valores
    ns, no = v.shape
    ext = [0, ns * raster.res, raster.o_min, raster.o_min + no * raster.res]
    ancho = min(30, max(8, ext[1] / 15))
    fig, ax = plt.subplots(figsize=(ancho, 4))
    ax.imshow(v.T, origin="lower", extent=ext, aspect="auto", cmap="gray", interpolation="nearest")
    for lado, sgn in (("izq", 1), ("der", -1)):
        L = lineas[lado]
        ok = L["estado_d"] > 0
        ax.plot(L["s"], np.where(ok, sgn * L["d"], np.nan), color=COLOR[lado], lw=1)
    ax.set_xlabel("estación [m]")
    ax.set_ylabel("offset [m]")
    ax.set_title(f"Raster radiométrico enderezado — eje {eje}")
    fig.tight_layout()
    fig.savefig(ruta, dpi=dpi)
    plt.close(fig)
