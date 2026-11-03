#!/usr/bin/env python3
"""
TEAM-28-Hexaedernetz mit Luftgebiet und Zonenkennung, direkt mit numpy gebaut.

    python3 tools/gmsh/team28_hex_air.py --scale 1 --plate-z0 11.3 -o air_s1_t3.vtk
    python3 tools/gmsh/team28_hex_air.py --truncation-radial 5 --truncation-axial 10 -o kasten.vtk

Ein einziges, strukturiertes und rotationssymmetrisches Netz aus Leitern und
Luft. Die Luftzellen tragen eine Zonenkennung (NAME.zones.json); tools/strip_air.py
loescht Zonen und liefert so die Stufen der Kopplungsflaechen-Verlegung
(docs/design/reference-harmonic-air.md). Gmsh wird nicht gebraucht.

GEOMETRIE (mm, wie team28_hex.py)
  Aussenspule  r 80..95, z -52..0     component_id 1
  Innenspule   r 27..55, z -52..0     component_id 2
  Platte       r 0..65, Dicke 3, Unterkante --plate-z0   component_id 3
  Luft                                component_id 0, Zone > 0

EBENE (O-Gitter, 4*n_q Segmente umlaufend, Knoten auf Kreisen -> exakte N-Ecke)
  Kernquadrat mit n_q x n_q Zellen, Ecken bei 0,8 * 27 mm. Darum ein oder
  mehrere Ringe bis r = 27 (Uebergang Quadrat -> Kreis, gerade Verbindungslinien
  von Quadratknoten zu Kreisknoten), dann Ringzonen bis 55, 65, 80, 95 und
  geometrisch wachsende Luftringe bis zum Abschneideradius.

HOEHE
  Luft unten (gestuft) bis -52, Spulen -52..0, Spalt 0..z0, Platte z0..z0+3,
  Luft oben (gestuft). Die Schichten nahe den Leitern haengen nicht vom
  Abschneidefaktor ab: ein groesserer Kasten haengt nur aeussere Luftschichten
  an (die letzte Schicht nimmt den Rest). Leiter- und Spaltknoten sind daher
  fuer alle Abschneidefaktoren identisch.

ZONEN DER LUFT (jede Luftzelle genau eine)
  1..S   Schalen ausserhalb des Leiterzylinders r = 95, -52 <= z <= z0+3;
         d = groesserer Abstand (radial oder axial) in Zellschichten, auf S
         Gruppen verteilt. 1 = innerste Gruppe, S = aeusserste.
  10     Spalt: r < 95, 0 <= z <= z0
  11     Kern: r < 27, -52 <= z <= 0
  12     zwischen den Spulen: 55 < r < 80, -52 <= z <= 0
  13     neben der Platte: 65 < r < 95, z0 <= z <= z0+3

AUSGABE
  NAME.vtk         Legacy-VTK (solver_vtk.write_legacy_vtk), genau ein
                   SCALARS component_id, Einheiten mm
  NAME.zones.json  {"version":1, "generator":..., "args":..., "n_cells":...,
                   "zone":[...]} mit einer Zone je Zelle (0 fuer Leiter) in der
                   Zellreihenfolge der VTK

PRUEFUNG NACH DEM SCHREIBEN (Exit-Code 1 bei Fehler)
  meshcheck (nur Hexaeder, konform, Orientierung, Luft umschliesst alles,
  Volumen je Leiter gegen das exakte N-Eck), check_map, Zonen (jede Luftzelle
  genau eine, Zonen 10 bis 13 nicht leer), Sprungfaktor benachbarter
  Zellkanten, Abschneidefaktoren wie im Programm, Randknoten und dichte Algebra.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)

from solver_vtk import orient_to_solver, write_legacy_vtk  # noqa: E402
import meshcheck as mc  # noqa: E402

VTK_HEX = mc.VTK_HEX
R_CORE, R_IC1, R_GAP, R_PL, R_OC0, R_OC1 = 27.0, 55.0, 65.0, 80.0, 80.0, 95.0
# Radiale Zonen (Name, r0, r1); die Ringe zwischen Kernquadrat und 27 mm kommen davor
Z_COIL0, Z_COIL1 = -52.0, 0.0
PLATE_T = 3.0
CORE_CORNER = 0.8 * R_CORE            # Kern-Ecken
NAMES = {1: "Aussenspule", 2: "Innenspule", 3: "Platte"}
ZONE_GAP, ZONE_CORE, ZONE_GROOVE, ZONE_PLATE_SIDE = 10, 11, 12, 13

# radiale Klassen
RK, RIC, RG, RP, ROC, RAIR = range(6)
# axiale Klassen
ZBELOW, ZCOIL, ZGAP, ZPLATE, ZABOVE = range(5)


def scaled(n, s, lo=1):
    return max(lo, int(round(n * s)))


def graded(length, first, growth):
    """Schichtdicken first, first*g, ... bis length; die letzte nimmt den Rest.

    Ist der Rest kleiner als die halbe vorletzte Schicht, wird er dieser
    zugeschlagen. Nur die aeusserste Schicht haengt damit von der Laenge ab;
    alle inneren Schichten sind fuer jede groessere Laenge identisch.
    """
    w, total = [], 0.0
    while total + first * growth ** len(w) < length - 1e-9:
        w.append(first * growth ** len(w))
        total += w[-1]
    rest = length - total
    if rest > 1e-9:
        if w and rest < 0.5 * w[-1]:
            w[-1] += rest
        else:
            w.append(rest)
    return w


def derive_radial(a):
    """Zellzahlen je radialer Zone aus --coil-nr und --plate-nr."""
    h = (R_OC1 - R_OC0) / a.coil_nr         # Zellweite der Aussenspule
    n_ic = scaled((R_IC1 - R_CORE) / h, 1.0)
    n_g = scaled((R_GAP - R_IC1) / h, 1.0)
    n_p = scaled((R_PL - R_GAP) / h, 1.0)
    # Ringe Kernquadrat -> 27 mm: --plate-nr zaehlt die Plattenzellen vom
    # Kern bis zum Rand; mindestens ein Ring, Achsen- und Diagonalzelle
    # hoechstens um den Faktor 2 von der ersten Spulenzelle verschieden
    a_s = CORE_CORNER / math.sqrt(2.0)
    w_axis = R_CORE - a_s
    h_ic = (R_IC1 - R_CORE) / n_ic
    w_diag = R_CORE - CORE_CORNER
    lo = max(1, math.ceil(w_axis / (2.0 * h_ic)))       # Achsenzelle <= 2 * Nachbar
    hi = max(lo, int(math.floor(2.0 * w_diag / h_ic)))   # Diagonalzelle >= Nachbar / 2
    n_ring = min(max(a.plate_nr - n_ic - n_g + 2, lo), hi)
    return dict(n_ring=n_ring, n_ic=n_ic, n_g=n_g, n_p=n_p, n_oc=a.coil_nr, h_ref=h)


class Layout:
    """Radiale und axiale Teilung samt Klassen je Schicht."""

    def __init__(self, a):
        d = derive_radial(a)
        self.d = d
        # radiale Ebenen ab r = 27 mm und die Klasse je Schicht
        self.rad_edges, self.rad_class, self.rad_ir = [R_CORE], [], []
        r = R_CORE
        for cls, r1, n in ((RIC, R_IC1, d["n_ic"]), (RG, R_GAP, d["n_g"]),
                           (RP, R_PL, d["n_p"]), (ROC, R_OC1, d["n_oc"])):
            for i in range(n):
                self.rad_edges.append(r + (r1 - r) * (i + 1) / n)
                self.rad_class.append(cls)
                self.rad_ir.append(0)
            r = r1
        h_oc = (R_OC1 - R_OC0) / d["n_oc"]
        self.r_air = a.truncation_radial * R_OC1
        air_r = graded(self.r_air - R_OC1, h_oc * a.air_growth, a.air_growth)
        for i, w in enumerate(air_r):
            self.rad_edges.append(self.rad_edges[-1] + w)
            self.rad_class.append(RAIR)
            self.rad_ir.append(i + 1)
        self.n_air_r = len(air_r)

        # axiale Ebenen
        zc_hi = a.plate_z0 + PLATE_T
        mid, half = 0.5 * (Z_COIL0 + zc_hi), 0.5 * (zc_hi - Z_COIL0)
        self.z_bot, self.z_top = mid - a.truncation_axial * half, mid + a.truncation_axial * half
        h_coil = (Z_COIL1 - Z_COIL0) / a.coil_nz
        h_plate = PLATE_T / a.plate_nz
        below = graded(Z_COIL0 - self.z_bot, h_coil * a.air_growth, a.air_growth)
        above = graded(self.z_top - zc_hi, h_plate * a.air_growth, a.air_growth)
        self.z_edges, self.z_class, self.z_iz = [self.z_bot], [], []
        for i, w in enumerate(below[::-1]):
            self.z_edges.append(self.z_edges[-1] + w)
            self.z_class.append(ZBELOW)
            self.z_iz.append(len(below) - i)
        for cls, z1, n in ((ZCOIL, Z_COIL1, a.coil_nz), (ZGAP, a.plate_z0, a.gap_nz),
                           (ZPLATE, zc_hi, a.plate_nz)):
            z0 = self.z_edges[-1]
            for i in range(n):
                self.z_edges.append(z0 + (z1 - z0) * (i + 1) / n)
                self.z_class.append(cls)
                self.z_iz.append(0)
        for i, w in enumerate(above):
            self.z_edges.append(self.z_edges[-1] + w)
            self.z_class.append(ZABOVE)
            self.z_iz.append(i + 1)
        self.n_below, self.n_above = len(below), len(above)
        self.D = max(self.n_air_r, self.n_below, self.n_above)


def build_plane(a, lay):
    """Ebenes O-Gitter: Knoten (N,2), Vierecke (M,4) gegen den Uhrzeigersinn,
    radiale Schicht-Index je Viereck (-1 = Kern, 0..n_ring-1 = Ring Quadrat->27
    mm, danach n_ring + Schicht in lay.rad_class)."""
    nq = a.n_quarter
    ns = 4 * nq
    d = lay.d
    a_s = CORE_CORNER / math.sqrt(2.0)
    nc1 = nq + 1
    pts = []
    for j in range(nc1):
        for i in range(nc1):
            pts.append((-a_s + 2 * a_s * i / nq, -a_s + 2 * a_s * j / nq))
    core_id = lambda i, j: j * nc1 + i
    quads, rlayer = [], []
    for j in range(nq):
        for i in range(nq):
            quads.append((core_id(i, j), core_id(i + 1, j), core_id(i + 1, j + 1),
                          core_id(i, j + 1)))
            rlayer.append(-1)
    # Quadratrand gegen den Uhrzeigersinn, Start an der Ecke bei -45 Grad
    ring0 = []
    for u in range(nq):
        ring0.append(core_id(nq, u))                  # Seite +x, y aufwaerts
    for u in range(nq):
        ring0.append(core_id(nq - u, nq))             # Seite +y, x abwaerts
    for u in range(nq):
        ring0.append(core_id(0, nq - u))              # Seite -x, y abwaerts
    for u in range(nq):
        ring0.append(core_id(u, 0))                   # Seite -y, x aufwaerts
    theta = np.array([-math.pi / 4 + k * (math.pi / 2) / nq for k in range(ns)])
    sq = np.array([pts[i] for i in ring0])
    circ = R_CORE * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    levels = [np.array(ring0)]
    nr = d["n_ring"]
    for j in range(1, nr + 1):
        t = j / nr
        # sanfte Verteilung: dicht am Kernquadrat, damit die Zellen an den
        # Diagonalen nicht flach werden
        xy = (1 - t) * sq + t * circ
        base = len(pts)
        pts.extend(map(tuple, xy))
        levels.append(np.arange(base, base + ns))
    for e in lay.rad_edges[1:]:
        xy = e * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        base = len(pts)
        pts.extend(map(tuple, xy))
        levels.append(np.arange(base, base + ns))
    for j in range(len(levels) - 1):
        for k in range(ns):
            k1 = (k + 1) % ns
            quads.append((levels[j][k], levels[j + 1][k], levels[j + 1][k1], levels[j][k1]))
            rlayer.append(j if j < nr else nr + (j - nr))
    return np.array(pts), np.array(quads, dtype=np.int64), np.array(rlayer)


def classify(a, lay, rlayer, kz):
    """Komponente und Zone der Zelle (Quad mit Schicht rlayer, axiale Schicht kz)."""
    nr = lay.d["n_ring"]
    if rlayer < nr:
        rc, ir = RK, 0                                   # Kern und Ringe bis 27 mm
    else:
        rc, ir = lay.rad_class[rlayer - nr], lay.rad_ir[rlayer - nr]
    zc, iz = lay.z_class[kz], lay.z_iz[kz]
    if rc == RAIR or zc in (ZBELOW, ZABOVE):
        d = max(ir, iz)
        return 0, 1 + ((d - 1) * a.shells) // lay.D
    if zc == ZCOIL:
        return {RK: (0, ZONE_CORE), RIC: (2, 0), RG: (0, ZONE_GROOVE),
                RP: (0, ZONE_GROOVE), ROC: (1, 0)}[rc]
    if zc == ZGAP:
        return 0, ZONE_GAP
    return {RK: (3, 0), RIC: (3, 0), RG: (3, 0), RP: (0, ZONE_PLATE_SIDE),
            ROC: (0, ZONE_PLATE_SIDE)}[rc]


def build(a):
    lay = Layout(a)
    pts2, quads, rlayer = build_plane(a, lay)
    n2 = len(pts2)
    nz = len(lay.z_class)
    pts = np.empty(((nz + 1) * n2, 3))
    for k, z in enumerate(lay.z_edges):
        pts[k * n2:(k + 1) * n2, :2] = pts2
        pts[k * n2:(k + 1) * n2, 2] = z
    conn, comp, zone = [], [], []
    for kz in range(nz):
        lo, hi = kz * n2, (kz + 1) * n2
        for q, rl in zip(quads, rlayer):
            conn.append(np.concatenate([q + lo, q + hi]))
            c, zn = classify(a, lay, int(rl), kz)
            comp.append(c)
            zone.append(zn)
    conn = np.array(conn, dtype=np.int64)
    conn, flipped = orient_to_solver(pts, VTK_HEX, conn)
    return lay, pts, conn, np.array(comp), np.array(zone), flipped


def scaled_jacobians(pts, conn):
    """Kleinste skalierte Jacobi-Determinante je Hexaeder ueber die 8 Ecken."""
    best = np.full(len(conn), np.inf)
    for v, b, c, d in mc.HEX_CORNERS:
        e1, e2, e3 = (pts[conn[:, i]] - pts[conn[:, v]] for i in (b, c, d))
        det = np.einsum("ij,ij->i", e1, np.cross(e2, e3))
        den = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)
        best = np.minimum(best, det / den)
    return best


def ngon_volumes(n_seg):
    f = n_seg / (2 * math.pi) * math.sin(2 * math.pi / n_seg)
    ring = lambda r0, r1: math.pi * (r1 ** 2 - r0 ** 2) * (Z_COIL1 - Z_COIL0) * f
    return {1: ring(R_OC0, R_OC1), 2: ring(R_CORE, R_IC1),
            3: math.pi * R_GAP ** 2 * PLATE_T * f}


def truncation_factors(mesh):
    """Wie FemDirectRun.cpp: Ausdehnung Luft / Ausdehnung Leiter, je Achse."""
    def box(mask):
        nodes = np.unique(np.concatenate([mesh.cells[i] for i in np.flatnonzero(mask)]))
        p = mesh.points[nodes]
        return p.max(axis=0) - p.min(axis=0)
    return box(mesh.comp == 0) / box(mesh.comp != 0)


def jump_factors(lay, a):
    """Groesster Quotient benachbarter Zellkanten radial und axial.

    Radial zaehlen auch die Ringzellen zwischen Kernquadrat und 27 mm (an den
    Achsen und an den Diagonalen) gegen die erste Zelle der Innenspule.
    """
    d = lay.d
    w = list(np.diff(lay.rad_edges))
    a_s = CORE_CORNER / math.sqrt(2.0)
    ring_axis = (R_CORE - a_s) / d["n_ring"]
    ring_diag = (R_CORE - CORE_CORNER) / d["n_ring"]
    seqs = [[ring_axis] + w, [ring_diag] + w]
    zr = list(np.diff(lay.z_edges))
    q = lambda v: float(max(max(x / y, y / x) for x, y in zip(v[1:], v[:-1])))
    return max(q(s) for s in seqs), q(zr)


def write_outputs(a, lay, pts, conn, comp, zone):
    write_legacy_vtk(a.out, pts, [(VTK_HEX, conn, comp)],
                     f"FEMBEM team28_hex_air nq={a.n_quarter} coil={a.coil_nr}x{a.coil_nz} "
                     f"plate={a.plate_nr}x{a.plate_nz} gap={a.gap_nz} z0={a.plate_z0} "
                     f"trunc={a.truncation_radial:g}/{a.truncation_axial:g}")
    stem = os.path.splitext(a.out)[0]
    args = {k: v for k, v in vars(a).items() if k != "out"}
    with open(stem + ".zones.json", "w") as f:
        json.dump({"version": 1, "generator": "tools/gmsh/team28_hex_air.py", "args": args,
                   "n_cells": int(len(zone)), "zone": [int(z) for z in zone]}, f)
    return stem + ".zones.json"


def check_zones(mesh, zone_path, shells):
    """Zonen: je Luftzelle genau eine, Leiter 0, Zonen 1..S und 10..13 belegt."""
    try:
        zone = mc.read_zones(zone_path, len(mesh.comp))
    except ValueError as e:
        return [mc.Result("Zonen: Datei passt zur VTK", False, str(e))]
    air = mesh.comp == 0
    ok_air = bool(np.all(zone[air] > 0) and np.all(zone[~air] == 0))
    res = [mc.Result("Zonen: je Luftzelle genau eine", ok_air,
                     f"{int(air.sum())} Luftzellen, {int((~air).sum())} Leiterzellen")]
    missing = [k for k in (ZONE_GAP, ZONE_CORE, ZONE_GROOVE, ZONE_PLATE_SIDE)
               if not (zone == k).any()]
    missing += [k for k in range(1, shells + 1) if not (zone == k).any()]
    res.append(mc.Result("Zonen 1..S und 10..13 nicht leer", not missing,
                         "fehlt: " + str(missing) if missing else "alle belegt"))
    return res


def check(a, lay, zone_path, verbose):
    import contextlib
    import io
    import check_map
    mesh = mc.read_vtk(a.out)
    results = mc.run_checks(mesh, require_hex=True, separate_bodies=False, air_enclosed=True,
                            expected_volumes=ngon_volumes(4 * a.n_quarter),
                            names={k: f"{v} ({4 * a.n_quarter}-Eck)" for k, v in NAMES.items()})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = check_map.main(a.out)
    results.append(mc.Result("check_map (Zuordnung, Masse, Vollscheibe)", rc == 0,
                             "besteht" if rc == 0 else "siehe Ausgabe"))
    results += check_zones(mesh, zone_path, a.shells)
    jr, jz = jump_factors(lay, a)
    results.append(mc.Result("Sprungfaktor radial <= 2", jr <= 2.0 + 1e-9, f"{jr:.2f}"))
    if jz > 2.0 + 1e-9:
        print(f"  Hinweis: axialer Sprungfaktor {jz:.2f} (Spalt gegen Platte; der Auftrag "
              "verlangt die Grenze 2 nur radial)")
    tf = truncation_factors(mesh)
    ok_tf = (abs(tf[0] - a.truncation_radial) / a.truncation_radial < 0.02
             and abs(tf[2] - a.truncation_axial) / a.truncation_axial < 0.02)
    results.append(mc.Result("Abschneidefaktoren (x / y / z)", ok_tf,
                             f"{tf[0]:.3f} / {tf[1]:.3f} / {tf[2]:.3f} (verlangt "
                             f"{a.truncation_radial:g} radial, {a.truncation_axial:g} axial)"))
    ok = mc.print_results(results)
    import reference_series as rs
    n_hex, _, n_bem = rs.boundary_nodes(a.out)
    gib = rs.dense_gib(n_bem)
    print(f"  Randknoten des vollen Netzes {n_bem}, dichte Algebra ca. {gib:.2f} GiB")
    if verbose or rc != 0:
        print(buf.getvalue())
    return ok


def report(a, lay, pts, conn, comp, zone, flipped, t0):
    import time
    sj = scaled_jacobians(pts, conn)
    print(f"{a.out}: {len(conn)} Hexaeder, {len(pts)} Knoten "
          f"({time.time() - t0:.1f} s, {flipped} Zellen gedreht)")
    d = lay.d
    print(f"  radial: Kernquadrat {a.n_quarter}x{a.n_quarter}, {d['n_ring']} Ringe bis 27, "
          f"Innenspule {d['n_ic']}, 55-65 {d['n_g']}, 65-80 {d['n_p']}, Aussenspule "
          f"{d['n_oc']}, Luft {lay.n_air_r} (bis r = {lay.r_air:.1f} mm)")
    print(f"  axial:  unten {lay.n_below}, Spulen {a.coil_nz}, Spalt {a.gap_nz}, Platte "
          f"{a.plate_nz}, oben {lay.n_above} (z = {lay.z_bot:.1f} .. {lay.z_top:.1f} mm); "
          f"Schalen: D = {lay.D} Schichten auf {a.shells} Gruppen")
    cnt = {c: int((comp == c).sum()) for c in (1, 2, 3)}
    print("  Zellen je Komponente: " + ", ".join(f"{NAMES[c]} {n}" for c, n in cnt.items())
          + f", Luft {int((comp == 0).sum())}")
    zn = {int(z): int((zone == z).sum()) for z in np.unique(zone) if z > 0}
    print("  Zellen je Zone: " + ", ".join(f"{k}: {v}" for k, v in sorted(zn.items())))
    print("  min. skalierte Jacobi-Determinante: " +
          ", ".join(f"{NAMES[c]} {sj[comp == c].min():.3f}" for c in (1, 2, 3)) +
          f", Luft {sj[comp == 0].min():.3f}")
    return sj


def main():
    ap = argparse.ArgumentParser(
        description="TEAM-28-Hexaedernetz mit Luftgebiet und Zonenkennung erzeugen und pruefen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-o", "--out", default="team28_hex_air.vtk", help="Ausgabedatei (Legacy-VTK)")
    ap.add_argument("--scale", type=float, default=1.0, help="Faktor auf alle Teilungen")
    ap.add_argument("--n-quarter", type=int, default=12, help="Zellen je Viertelkreis")
    ap.add_argument("--coil-nr", type=int, default=4, help="Zellen radial in der Aussenspule")
    ap.add_argument("--coil-nz", type=int, default=8, help="Spulenzellen in z")
    ap.add_argument("--plate-nr", type=int, default=10, help="Plattenzellen radial, Kern bis Rand")
    ap.add_argument("--plate-nz", type=int, default=3, help="Plattenzellen in z (Dicke 3 mm)")
    ap.add_argument("--gap-nz", type=int, default=3, help="Zellen im Spalt (mindestens 3)")
    ap.add_argument("--plate-z0", type=float, default=3.8, help="Unterkante der Platte in mm")
    ap.add_argument("--truncation-radial", type=float, default=3.0,
                    help="Abschneidefaktor radial: Ausdehnung Luft / Ausdehnung Leiter")
    ap.add_argument("--truncation-axial", type=float, default=3.0,
                    help="Abschneidefaktor axial")
    ap.add_argument("--air-growth", type=float, default=1.3, help="Wachstum der Luftschichten")
    ap.add_argument("--shells", type=int, default=4, help="Zahl S der Schalengruppen")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    for name in ("n_quarter", "coil_nr", "coil_nz", "plate_nr", "plate_nz"):
        setattr(a, name, scaled(getattr(a, name), a.scale))
    a.gap_nz = max(3, scaled(a.gap_nz, a.scale))
    if a.plate_z0 <= 0.0:
        ap.error("--plate-z0 muss > 0 sein (Spulen enden bei z = 0)")
    if a.truncation_radial <= 1.05 or a.truncation_axial <= 1.05:
        ap.error("Abschneidefaktoren muessen > 1,05 sein")
    if not 1.0 < a.air_growth <= 1.35:
        ap.error("--air-growth muss in (1, 1,35] liegen (Sprungfaktor <= 2 mit dem Restschicht-Zuschlag)")
    if a.shells < 1 or a.shells > 9:
        ap.error("--shells muss zwischen 1 und 9 liegen")
    import time
    t0 = time.time()
    lay, pts, conn, comp, zone, flipped = build(a)
    if lay.D < a.shells:
        sys.exit(f"FEHLER: nur {lay.D} Luftschichten, aber {a.shells} Schalengruppen verlangt")
    zp = write_outputs(a, lay, pts, conn, comp, zone)
    sj = report(a, lay, pts, conn, comp, zone, flipped, t0)
    if a.no_check:
        return 0
    ok = check(a, lay, zp, a.verbose)
    if sj.min() < 0.5:
        print(f"  WARNUNG: kleinste skalierte Jacobi-Determinante {sj.min():.3f} < 0,5")
    print("Netz in Ordnung." if ok else "NETZ FEHLERHAFT -- nicht verwenden.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
