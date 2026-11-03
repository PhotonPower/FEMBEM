#!/usr/bin/env python3
"""Netzserie fuer den harmonischen Referenzloeser (--reference --harmonic).

Erzeugt die Netze mit tools/gmsh/team28_hex.py, rechnet jeden Lauf mit
fembem, liest die Logs aus und schreibt eine Tabelle (Markdown und JSON).

    python tools/reference_series.py                       # alles (A bis D)
    python tools/reference_series.py --groups A,B --dry-run
    python tools/reference_series.py --groups E            # 3,8 mm, optional
    python tools/reference_series.py --reuse               # nur Logs auswerten

Gruppen (alle bei 11,3 mm ausser E; Platte starr verschoben mit --plate-z,
--plate-z0 des Generators bleibt 3,8 mm):
  A  uniform: --scale 0.5, 0.75, 1.0, 1.5, 2.0
  B  eine Richtung je Lauf auf --scale 1.0: --n-quarter 18, --coil-nz 16,
     --plate-nz 6, --plate-nr 15
  C  --no-conductor-penalty auf --scale 1.0 und dem feinsten Netz aus A,
     das gelaufen ist
  D  team28.vtk und das grobe Netz gmsh_team28_hex_coarse.vtk
  E  3,8 mm fuer --scale 1.0 und 2.0
  F  Netze mit Luft (tools/gmsh/team28_hex_air.py), Kopplungsflaeche schrittweise
     auf die Leiter verlegt (docs/design/reference-harmonic-air.md):
       trunc  Abschneidestudie bei 11,3 mm: D und B0 auf air_s1_t1p5/t2/t3/t5/t5x10,
              dazu --verify-field (sigma = 0) auf air_s1_t3
       chain  Stufenkette B1 .. B_end auf air_s1_t3 (11,3 mm), jede Stufe aus dem
              vollen Netz mit tools/strip_air.py
       z38    dieselbe Kette bei 3,8 mm: B0, B_S, B_platte, B_end
       smoke  --smoke: scale 0.5, 11,3 mm, nur D, B0, B_end (Rauchtest)
     Netze, Logs und Dumps landen in build/refair, die Tabellen in
     build/refair/summary.md. --groups F waehlt alles ausser smoke;
     --f-parts trunc,chain,z38 waehlt Teile, --f-extra SCALE:Z0:STUFE,STUFE ergaenzt
     weitere Stufen auf einem anderen Netz (z. B. 1.5:11.3:B_S,B_rille).

Vor jedem Lauf werden Randknoten und Speicher der dichten BEM aus dem Netz
bestimmt (Bedarf wie im Loeser: 3,2 * N^2 * 8 Byte). Liegt er ueber
--budget-gib, wird der Lauf nicht gestartet und als "Speicher" vermerkt.

Exit-Code 1, wenn eine Netzpruefung oder ein Lauf scheitert (Abbruch,
fehlendes Ergebnis, FGMRES nicht konvergiert oder Kohomologie != 2).
Das Skript prueft den Exit-Code von fembem (1 bei nicht konvergiertem FGMRES)
und zusaetzlich das Log; bei --reuse liest es den Exit-Code aus NAME.log.exit.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RING = {11.3: 1.00684, 3.8: 3.28965}    # Ringmodell, gmsh-phase9-report.md
FEM_DIRECT = (1.015, 1.017)             # FEM-direkt bei 11,3 mm, ebenda

A_SCALES = (0.5, 0.75, 1.0, 1.5, 2.0)
B_DIRS = (("nq18", ["--n-quarter", "18"], "n-quarter 12 -> 18"),
          ("cnz16", ["--coil-nz", "16"], "coil-nz 8 -> 16"),
          ("pnz6", ["--plate-nz", "6"], "plate-nz 3 -> 6"),
          ("pnr15", ["--plate-nr", "15"], "plate-nr 10 -> 15"))


def tag(s):
    return f"s{s:g}".replace(".", "p")


def plan(groups):
    """Laeufe: name, Gruppe, Netz (gen-Argumente oder feste Datei), Hoehe, extra."""
    runs = []
    if "A" in groups or "B" in groups or "C" in groups:
        for s in A_SCALES:
            if "A" in groups or s == 1.0:
                runs.append(dict(name=f"A_{tag(s)}", group="A", mesh=f"hex_{tag(s)}",
                                 gen=["--scale", str(s)], z=11.3, extra=[],
                                 what=f"--scale {s:g}"))
    if "B" in groups:
        for key, gen, what in B_DIRS:
            runs.append(dict(name=f"B_{key}", group="B", mesh=f"hex_{key}", gen=gen,
                             z=11.3, extra=[], what=what))
    if "C" in groups:
        runs.append(dict(name="C_s1_nopen", group="C", mesh="hex_s1", gen=["--scale", "1.0"],
                         z=11.3, extra=["--no-conductor-penalty"],
                         what="--scale 1, ohne Strafterm im Leiter"))
        runs.append(dict(name="C_finest_nopen", group="C", mesh=None, gen=None, z=11.3,
                         extra=["--no-conductor-penalty"],
                         what="feinstes A, ohne Strafterm im Leiter"))
    if "D" in groups:
        runs.append(dict(name="D_team28", group="D", file="team28.vtk", z=11.3, extra=[],
                         what="team28.vtk (Standardnetz)"))
        runs.append(dict(name="D_coarse", group="D", file="gmsh_team28_hex_coarse.vtk",
                         z=11.3, extra=[], what="grobes Netz"))
    if "E" in groups:
        for s in (1.0, 2.0):
            runs.append(dict(name=f"E_{tag(s)}_3p8", group="E", mesh=f"hex_{tag(s)}",
                             gen=["--scale", str(s)], z=3.8, extra=[],
                             what=f"--scale {s:g}, 3,8 mm"))
    return runs


def boundary_nodes(path):
    """Hexaeder und Randknoten (Knoten auf Flaechen mit genau einer Zelle)."""
    import numpy as np
    import meshcheck as mc
    mesh = mc.read_vtk(path)
    owners = mc._faces_with_owner(mesh)
    nodes = set()
    for face, cells in owners.items():
        if len(cells) == 1:
            nodes.update(face)
    n_hex = int(np.count_nonzero(mesh.types == 12))
    return n_hex, len(mesh.types), len(nodes)


def dense_gib(n_bem):
    return 3.2 * n_bem * n_bem * 8.0 / 1024 ** 3


def generate(run, workdir, log):
    out = os.path.join(workdir, run["mesh"] + ".vtk")
    gen_log = os.path.join(workdir, run["mesh"] + ".gen.log")
    if os.path.exists(out) and os.path.exists(gen_log):
        text = open(gen_log, encoding="utf-8").read()
        if "Netz in Ordnung." in text:
            return out, text, True
    cmd = [sys.executable, os.path.join(HERE, "gmsh", "team28_hex.py"), "-o", out] + run["gen"]
    log(" ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    text = p.stdout + p.stderr
    open(gen_log, "w", encoding="utf-8").write(" ".join(cmd) + "\n" + text)
    return out, text, p.returncode == 0 and "Netz in Ordnung." in text


def parse_gen(text):
    r = {}
    m = re.search(r"Jacobi-Determinante: (.*?)\s+\(", text)
    if m:
        r["min_sj"] = m.group(1).strip()
    m = re.search(r"FEMBEM team28_hex (nq=\S+ coil=\S+ plate=\S+)", text)
    if m:
        r["divisions"] = m.group(1)
    r["meshcheck"] = "ok" if "Netz in Ordnung." in text else "FEHLER"
    m = re.search(r"check_map \(.*?\)\s+(\S+)", text)
    r["check_map"] = m.group(1) if m else "?"
    return r


def num(pat, text, cast=float):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def parse_log(text):
    r = {}
    r["hexes"] = num(r"Cell types: (\d+) hexahedra", text, int)
    r["bem_nodes"] = num(r"\[Referenz\] Rand: (\d+) Knoten", text, int)
    r["dense_gib"] = num(r"Bedarf dichte Algebra ca\. ([\d.]+) GiB", text)
    r["fem_rows"] = num(r"FEM-System: (\d+) Zeilen", text, int)
    r["cohomology"] = num(r"\[Gauge\] (\d+) Kohomologie-Generatoren", text, int)
    r["z_mm"] = num(r"Schwebehoehe .*?= ([\d.]+) mm", text)
    r["penalty_off"] = "--no-conductor-penalty" in text
    m = re.search(r"FGMRES (KONVERGIERT|NICHT KONVERGIERT): (\d+) Iterationen, echtes rel\. "
                  r"Residuum ([\d.eE+-]+), innere Iterationen gesamt (\d+)", text)
    if m:
        r["converged"] = m.group(1) == "KONVERGIERT"
        r["outer"] = int(m.group(2))
        r["residual"] = float(m.group(3))
        r["inner"] = int(m.group(4))
    for order in (3, 2):
        m = re.search(rf"F \(Lorentz, {order}x{order}x{order}\) = \(([^,]+), ([^,]+), ([^)]+)\) N",
                      text)
        if m:
            r[f"fz{order}"] = float(m.group(3))
    r["q_eff"] = num(r"Q_eff = .*?= ([\d.]+)", text)
    r["time_s"] = num(r"Zeit: .*gesamt ([\d.]+) s", text)
    r["power_w"] = num(r"Verlustleistung \(Zyklusmittel\) = ([\d.eE+-]+) W", text)
    # Dirichlet-Zweig (--reference-dirichlet)
    m = re.search(r"\[Ref-Harm\] Dirichlet: A = 0 an (\d+) Randknoten", text)
    r["dirichlet"] = m is not None
    if m:
        r["dirichlet_nodes"] = int(m.group(1))
    r["dirichlet_max_a"] = num(r"max\|A\| auf den Dirichlet-Zeilen = ([\d.eE+-]+)", text)
    # Feldpruefung (--verify-field): groesste und mittlere Abweichung B_z gegen Biot-Savart
    r["vf_worst_pct"] = num(r"Groesste Abweichung im Betrag: ([\d.]+) %", text)
    devs = [float(x.replace(" ", "")) for x in re.findall(
        r"^\s+[-\d.]+\s+[+-][\d.]+e[+-]\d+\s+[+-][\d.]+e[+-]\d+\s+([+-]\s*[\d.]+) %",
        text, re.M)]
    r["vf_devs"] = devs
    return r


def run_one(run, args, log):
    workdir = args.workdir
    rec = dict(name=run["name"], group=run["group"], what=run["what"], z=run["z"],
               status="", extra=run["extra"])
    if "file" in run:
        mesh = os.path.join(args.build, run["file"])
        rec["mesh"] = run["file"]
        rec["gen"] = {}
    else:
        mesh, gen_text, ok = generate(run, workdir, log)
        rec["mesh"] = os.path.basename(mesh)
        rec["gen"] = parse_gen(gen_text)
        rec["gen_cmd"] = "python tools/gmsh/team28_hex.py " + " ".join(run["gen"])
        if not ok:
            rec["status"] = "Netzpruefung gescheitert"
            return rec
    n_hex, n_cells, n_bem = boundary_nodes(mesh)
    rec["hexes_mesh"], rec["bem_nodes_mesh"] = n_hex, n_bem
    rec["dense_gib_est"] = dense_gib(n_bem)
    log(f"{run['name']}: {n_hex} Hexaeder, {n_bem} Randknoten, dichte BEM ca. "
        f"{rec['dense_gib_est']:.2f} GiB (Grenze {args.budget_gib:g} GiB)")
    if n_hex != n_cells:
        rec["status"] = "keine reinen Hexaeder"
        return rec
    if rec["dense_gib_est"] > args.budget_gib:
        rec["status"] = "Speicher"
        return rec
    cmd = [args.fembem, "--reference", "--harmonic", "--fixed-plate",
           f"--plate-z={run['z']:g}", f"--mesh={os.path.relpath(mesh, args.build)}"] + run["extra"]
    rec["cmd"] = "fembem " + " ".join(cmd[1:])
    log_path = os.path.join(workdir, run["name"] + ".log")
    rec["log"] = os.path.relpath(log_path, REPO)
    if args.dry_run:
        rec["status"] = "geplant"
        return rec
    if not (args.reuse and os.path.exists(log_path)):
        log(rec["cmd"])
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            p = subprocess.run(cmd, cwd=args.build, stdout=f, stderr=subprocess.STDOUT)
        rec["exit"] = p.returncode
        rec["wall_s"] = time.time() - t0
        with open(log_path + ".exit", "w") as f:
            f.write(str(p.returncode))
    elif os.path.exists(log_path + ".exit"):
        # --reuse: Exit-Code des frueheren Laufs
        rec["exit"] = int(open(log_path + ".exit").read().strip() or "0")
    text = open(log_path, encoding="utf-8", errors="replace").read()
    rec.update(parse_log(text))
    if rec.get("exit") not in (None, 0):
        rec["status"] = f"Exit-Code {rec['exit']}"
    elif rec.get("fz3") is None:
        rec["status"] = "kein Ergebnis"
    elif not rec.get("converged"):
        rec["status"] = "NICHT KONVERGIERT"
    elif rec.get("cohomology") != 2:
        rec["status"] = f"Kohomologie {rec.get('cohomology')}"
    else:
        rec["status"] = "ok"
    return rec


def observed_order(pts):
    """pts: [(h, F)] mit absteigendem h. Liefert (p, F_inf) oder (None, Grund)."""
    (h1, f1), (h2, f2), (h3, f3) = pts
    d12, d23 = f1 - f2, f2 - f3
    if d12 == 0 or d23 == 0 or (d12 > 0) != (d23 > 0):
        return None, "nicht monoton"
    target = d12 / d23

    def g(p):
        return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - target

    lo, hi = 0.05, 8.0
    if g(lo) * g(hi) > 0:
        return None, f"keine Ordnung in [{lo}, {hi}] (Verhaeltnis {target:.3f})"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    p = 0.5 * (lo + hi)
    c = d23 / (h2 ** p - h3 ** p)
    return p, f3 - c * h3 ** p


def fmt(v, spec, dash="–"):
    return dash if v is None else format(v, spec)


def table(recs):
    head = ("| Lauf | Netz | Hexaeder | Randknoten | Zeilen (n) | F_z O3 [N] | F_z O2 [N] | "
            "F_z/Ring | Q_eff | außen / innen | Zeit [s] | Status |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in recs:
        ring = RING.get(r["z"])
        ratio = r["fz3"] / ring if r.get("fz3") is not None and ring else None
        it = (f"{r['outer']} / {r['inner']}" if r.get("outer") is not None else "–")
        rows.append(f"| {r['name']} | {r['what']} | {fmt(r.get('hexes') or r.get('hexes_mesh'), 'd')} | "
                    f"{fmt(r.get('bem_nodes') or r.get('bem_nodes_mesh'), 'd')} | "
                    f"{fmt(r.get('fem_rows'), 'd')} | {fmt(r.get('fz3'), '.5f')} | "
                    f"{fmt(r.get('fz2'), '.5f')} | {fmt(ratio, '.4f')} | "
                    f"{fmt(r.get('q_eff'), '.4f')} | {it} | {fmt(r.get('time_s'), '.0f')} | "
                    f"{r['status']} |")
    return head + "\n".join(rows) + "\n"


def mesh_table(recs):
    head = ("| Netz | Teilungen | kleinste skal. Jacobi-Det. | meshcheck | check_map |\n"
            "|---|---|---|---|---|\n")
    seen, rows = set(), []
    for r in recs:
        g = r.get("gen") or {}
        if not g or r["mesh"] in seen:
            continue
        seen.add(r["mesh"])
        rows.append(f"| {r['mesh']} | {g.get('divisions', '?')} | {g.get('min_sj', '?')} | "
                    f"{g.get('meshcheck', '?')} | {g.get('check_map', '?')} |")
    return head + "\n".join(rows) + "\n"


def evaluate(recs):
    out = []
    by = {r["name"]: r for r in recs}
    ok = lambda r: r is not None and r.get("status") == "ok"   # noqa: E731
    a = [(s, by.get(f"A_{tag(s)}")) for s in A_SCALES]
    a_ok = [(s, r) for s, r in a if ok(r)]
    if len(a_ok) >= 3:
        fin = a_ok[-3:]
        pts = [(1.0 / s, r["fz3"]) for s, r in fin]
        p, finf = observed_order(pts)
        names = ", ".join(f"{s:g}" for s, _ in fin)
        if p is None:
            out.append(f"A: Ordnung aus scale {names}: {finf}")
        else:
            ring = RING[11.3]
            out.append(f"A: beobachtete Ordnung aus scale {names}: p = {p:.2f}, "
                       f"extrapoliert F_inf = {finf:.5f} N ({100 * (finf / ring - 1):+.2f} % "
                       f"zum Ringmodell, {100 * (finf / FEM_DIRECT[0] - 1):+.2f} bis "
                       f"{100 * (finf / FEM_DIRECT[1] - 1):+.2f} % zu FEM-direkt)")
    base = by.get("A_s1")
    if ok(base):
        for key, _, what in B_DIRS:
            r = by.get(f"B_{key}")
            if ok(r):
                out.append(f"B: {what}: {100 * (r['fz3'] / base['fz3'] - 1):+.2f} %")
        r = by.get("C_s1_nopen")
        if ok(r):
            out.append(f"C: ohne Strafterm, scale 1: {100 * (r['fz3'] / base['fz3'] - 1):+.2f} %")
    r = by.get("C_finest_nopen")
    if ok(r) and ok(by.get(r.get("ref"))):
        out.append(f"C: ohne Strafterm, {r['what']}: "
                   f"{100 * (r['fz3'] / by[r['ref']]['fz3'] - 1):+.2f} %")
    return out



# =============================================================================
# Gruppe F: Netze mit Luft, Kopplungsflaeche schrittweise auf die Leiter verlegt
# =============================================================================
F_TRUNC = (("t1p5", 1.5, 1.5), ("t2", 2.0, 2.0), ("t3", 3.0, 3.0), ("t5", 5.0, 5.0),
           ("t5x10", 5.0, 10.0))
F_STAGES = (("B1", "4"), ("B2", "3-4"), ("B3", "2-4"), ("B_S", "1-4"),
            ("B_rille", "1-4,12"), ("B_kern", "1-4,11,12"),
            ("B_platte", "1-4,11,12,13"), ("B_end", "1-4,10-13"))
F_STAGE_ZONES = dict(F_STAGES)
F_Z38_STAGES = ("B_S", "B_platte", "B_end")
BODY_NAMES = (("Platte", 3), ("Innenspule", 2), ("Aussenspule", 1))


def f_mesh_name(scale, z0, trunc=None):
    s_tag = f"s{scale:g}".replace(".", "p")
    base = f"air_{s_tag}_{trunc or 't3'}"
    return base + ("" if abs(z0 - 11.3) < 1e-9 else f"_z{z0:g}".replace(".", "p"))


def f_gen_args(scale, z0, tr, tz):
    return ["--scale", f"{scale:g}", "--plate-z0", f"{z0:g}",
            "--truncation-radial", f"{tr:g}", "--truncation-axial", f"{tz:g}"]


def plan_f(args):
    """Laeufe der Gruppe F: dicts mit name, mesh (Elternnetz), stage, gen, mode, z0, extra."""
    runs = []

    def add(mesh, gen, stage, mode, z0, extra=(), tag=None, part=""):
        name = f"{mesh}_{tag or (stage if mode == 'B' else 'D')}"
        runs.append(dict(name=name, mesh=mesh, gen=gen, stage=stage, mode=mode, z0=z0,
                         extra=list(extra), part=part))

    if args.smoke:
        gen = f_gen_args(0.5, 11.3, 3, 3)
        mesh = "air_s0p5_t3"
        add(mesh, gen, "B0", "B", 11.3, part="smoke", tag="B0")
        add(mesh, gen, "B0", "D", 11.3, part="smoke", tag="D")
        add(mesh, gen, "B_end", "B", 11.3, part="smoke")
        return runs
    parts = set(args.f_parts.split(","))
    scale = args.f_scale
    t3 = f_mesh_name(scale, 11.3, "t3")
    g3 = f_gen_args(scale, 11.3, 3, 3)
    if "trunc" in parts:
        # Bezug zuerst: B0 auf t3
        order = [x for x in F_TRUNC if x[0] == "t3"] + [x for x in F_TRUNC if x[0] != "t3"]
        for tag, tr, tz in order:
            mesh = f_mesh_name(scale, 11.3, tag)
            gen = f_gen_args(scale, 11.3, tr, tz)
            add(mesh, gen, "B0", "B", 11.3, part="trunc", tag="B0")
            add(mesh, gen, "B0", "D", 11.3, part="trunc", tag="D")
        add(t3, g3, "B0", "B", 11.3, ["--verify-field"], tag="B0_vf", part="trunc")
        add(t3, g3, "B0", "D", 11.3, ["--verify-field"], tag="D_vf", part="trunc")
    if "chain" in parts:
        add(t3, g3, "B0", "B", 11.3, tag="B0", part="chain")
        for stage, _ in F_STAGES:
            add(t3, g3, stage, "B", 11.3, part="chain")
    if "z38" in parts:
        m38 = f_mesh_name(scale, 3.8, "t3")
        g38 = f_gen_args(scale, 3.8, 3, 3)
        add(m38, g38, "B0", "B", 3.8, tag="B0", part="z38")
        for stage in F_Z38_STAGES:
            add(m38, g38, stage, "B", 3.8, part="z38")
    for item in args.f_extra:
        sc, z0, stages = item.split(":")
        sc, z0 = float(sc), float(z0)
        mesh = f_mesh_name(sc, z0, "t3")
        gen = f_gen_args(sc, z0, 3, 3)
        add(mesh, gen, "B0", "B", z0, tag="B0", part="extra")
        for st in stages.split(","):
            add(mesh, gen, st, "B", z0, part="extra")
    # Duplikate (gleicher Name) nur einmal
    seen, out = set(), []
    for r in runs:
        if r["name"] not in seen:
            seen.add(r["name"])
            out.append(r)
    return out


def f_run_cmd(cmd, log_path, cwd=None):
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, open(log_path, encoding="utf-8", errors="replace").read()


def f_ensure_mesh(run, args, log):
    """Volles Luftnetz erzeugen (oder wiederverwenden). Gibt (Pfad, Text, ok) zurueck."""
    out = os.path.join(args.workdir, run["mesh"] + ".vtk")
    gl = os.path.join(args.workdir, run["mesh"] + ".gen.log")
    if os.path.exists(out) and os.path.exists(gl) and os.path.exists(out[:-4] + ".zones.json"):
        text = open(gl, encoding="utf-8").read()
        if "Netz in Ordnung." in text:
            return out, text, True
    cmd = [sys.executable, os.path.join(HERE, "gmsh", "team28_hex_air.py"), "-o", out] + run["gen"]
    log(" ".join(cmd))
    rc, text = f_run_cmd(cmd, gl)
    return out, text, rc == 0 and "Netz in Ordnung." in text


def f_ensure_stage(run, parent, args, log):
    """Stufe mit strip_air.py. Gibt (VTK-Pfad, Text, ok) zurueck; B0 = volles Netz."""
    if run["stage"] == "B0":
        gl = os.path.join(args.workdir, run["mesh"] + ".b0check.log")
        if not (args.reuse and os.path.exists(gl)):
            cmd = [sys.executable, os.path.join(HERE, "strip_air.py"), parent, "--check-only"]
            rc, text = f_run_cmd(cmd, gl)
        else:
            text = open(gl, encoding="utf-8").read()
        return parent, text, "Netz in Ordnung." in text
    out = os.path.join(args.workdir, f"{run['mesh']}_{run['stage']}.vtk")
    gl = out[:-4] + ".strip.log"
    if os.path.exists(out) and os.path.exists(gl):
        text = open(gl, encoding="utf-8").read()
        if "Stufe in Ordnung." in text:
            return out, text, True
    cmd = [sys.executable, os.path.join(HERE, "strip_air.py"), parent,
           "--remove-zones", F_STAGE_ZONES[run["stage"]], "-o", out]
    log(" ".join(cmd[1:]))
    rc, text = f_run_cmd(cmd, gl)
    return out, text, rc == 0 and "Stufe in Ordnung." in text


def parse_surface(text):
    """Randkomponenten, Randknoten, Euler-Charakteristiken aus der Ausgabe von strip_air.py."""
    r = {}
    m = re.search(r"Randkomponenten (\d+), Randknoten (\d+), Randflaechen (\d+), dichte Algebra "
                  r"ca\. ([\d.]+) GiB", text)
    if m:
        r.update(components=int(m.group(1)), bnodes=int(m.group(2)), bfaces=int(m.group(3)),
                 dense_gib=float(m.group(4)))
    r["chi"] = [int(x) for x in re.findall(r"Euler-Charakteristik (-?\d+)", text)]
    return r


def parse_air_gen(text):
    r = {}
    m = re.search(r"Zellen je Komponente: (.*)", text)
    r["cells"] = m.group(1).strip() if m else "?"
    m = re.search(r"Zellen je Zone: (.*)", text)
    r["zones"] = m.group(1).strip() if m else "?"
    m = re.search(r"min\. skalierte Jacobi-Determinante: (.*)", text)
    r["min_sj"] = m.group(1).strip() if m else "?"
    m = re.search(r"Randknoten des vollen Netzes (\d+)", text)
    r["bnodes_full"] = int(m.group(1)) if m else None
    m = re.search(r"Abschneidefaktoren \(x / y / z\)\s+\S+\s+(.*?) \(verlangt", text)
    r["trunc"] = m.group(1).strip() if m else "?"
    return r


def f_check_conductor_nodes(meshes, args):
    """Leiter- und Spaltknoten zweier Netze muessen identisch sein (AP 7, Voraussetzung)."""
    import numpy as np
    import meshcheck as mc

    def nodes(path):
        m = mc.read_vtk(path)
        z = mc.read_zones(path[:-4] + ".zones.json", len(m.comp))
        sel = np.flatnonzero((m.comp > 0) | (z == 10))
        n = np.unique(np.concatenate([m.cells[i] for i in sel]))
        return {tuple(p) for p in m.points[n]}
    ref = nodes(meshes[0])
    bad = []
    for other in meshes[1:]:
        if nodes(other) != ref:
            bad.append(os.path.basename(other))
    return bad


def f_table(recs, ref_name):
    head = ("| Lauf | Randknoten | n (FEM) | dichte Algebra [GiB] | F_z O3 [N] | F_z O2 [N] | "
            "Q_eff | Verlust [W] | außen / innen | echtes Res. | Zeit [s] | Status |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|\n")
    rows = []
    for r in recs:
        it = f"{r['outer']} / {r['inner']}" if r.get("outer") is not None else "–"
        rows.append(f"| {r['name']} | {fmt(r.get('bem_nodes') or r.get('bnodes'), 'd')} | "
                    f"{fmt(r.get('fem_rows'), 'd')} | {fmt(r.get('dense_gib') or r.get('dense_gib_est'), '.2f')} | "
                    f"{fmt(r.get('fz3'), '.5f')} | {fmt(r.get('fz2'), '.5f')} | "
                    f"{fmt(r.get('q_eff'), '.4f')} | {fmt(r.get('power_w'), '.3f')} | {it} | "
                    f"{fmt(r.get('residual'), '.1e')} | {fmt(r.get('time_s') or r.get('wall_s'), '.0f')} | "
                    f"{r['status']} |")
    return head + "\n".join(rows) + "\n"


def f_table_a(recs, ref_name):
    head = (f"| Lauf | F_z / F_z({ref_name}) − 1 | Platte ΔA_r | Platte ΔA_i | Innenspule ΔA_r | "
            "Innenspule ΔA_i | Außenspule ΔA_r | Außenspule ΔA_i | Phase Platte [°] | "
            "max. Knotenabw. Platte (Ort) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    ref = next((r for r in recs if r["name"] == ref_name and r.get("fz3") is not None), None)
    rows = []
    for r in recs:
        c = r.get("cmp")
        rel = (f"{100 * (r['fz3'] / ref['fz3'] - 1):+.3f} %"
               if ref and r.get("fz3") is not None else "–")
        if not c:
            rows.append(f"| {r['name']} | {rel} | – | – | – | – | – | – | – | – |")
            continue
        b = c["bodies"]
        cell = lambda cid, key: fmt(b[cid][key], ".2e") if cid in b else "–"   # noqa: E731
        pl = b.get(3)
        rows.append(f"| {r['name']} | {rel} | {cell(3, 'rel_r')} | {cell(3, 'rel_i')} | "
                    f"{cell(2, 'rel_r')} | {cell(2, 'rel_i')} | {cell(1, 'rel_r')} | "
                    f"{cell(1, 'rel_i')} | {fmt(pl['phase_deg'], '+.3f') if pl else '–'} | "
                    f"{fmt(pl['max_dev_rel'], '.2e') if pl else '–'}"
                    + (f" bei ({pl['where_mm'][0]:.0f}, {pl['where_mm'][1]:.0f}, {pl['where_mm'][2]:.1f}) mm"
                       if pl else "") + " |")
    return head + "\n".join(rows) + "\n"


def f_surface_table(recs):
    head = ("| Lauf | Randkomponenten | Euler-Charakteristik je Komponente | Randflächen | Randknoten |\n"
            "|---|---:|---|---:|---:|\n")
    rows = []
    for r in recs:
        sf = r.get("surface") or {}
        if not sf:
            continue
        rows.append(f"| {r['name']} | {sf.get('components', '–')} | "
                    f"{', '.join(str(x) for x in sf.get('chi', []))} | {sf.get('bfaces', '–')} | "
                    f"{sf.get('bnodes', '–')} |")
    return head + "\n".join(rows) + "\n"


def f_evaluate(recs, scale=1.0):
    """Abnahmepunkte aus AP 7 und 8, nur Zahlen."""
    out = []
    by = {r["name"]: r for r in recs}
    ok = lambda r: r is not None and r.get("status") == "ok" and r.get("fz3") is not None   # noqa: E731
    b0 = [(t, by.get(f_mesh_name(scale, 11.3, t) + "_B0")) for t, _, _ in F_TRUNC]
    b0 = [(t, r) for t, r in b0 if ok(r)]
    if len(b0) >= 2:
        fz = [r["fz3"] for _, r in b0]
        out.append(f"B0 ueber {len(b0)} Abschneidefaktoren: F_z von {min(fz):.5f} bis {max(fz):.5f} N, "
                   f"Streuung {100 * (max(fz) - min(fz)) / (sum(fz) / len(fz)):.3f} % "
                   "(Vorschlag 0,1 %, Abbruch ab 0,5 %)")
    d = [(t, by.get(f_mesh_name(scale, 11.3, t) + "_D")) for t, _, _ in F_TRUNC]
    d = [(t, r) for t, r in d if ok(r)]
    ref = by.get(f_mesh_name(scale, 11.3, "t3") + "_B0")
    if d and ok(ref):
        out.append("D gegen B0(t3), F_z: " + ", ".join(
            f"{t}: {100 * (r['fz3'] / ref['fz3'] - 1):+.2f} %" for t, r in d))
    return out


def f_run_one(run, args, log, parents, ref_dump):
    """Ein Lauf der Gruppe F. Gibt rec zurueck."""
    rec = dict(name=run["name"], group="F", mesh=run["mesh"], stage=run["stage"],
               mode=run["mode"], z=run["z0"], status="", extra=run["extra"], part=run["part"])
    if run["mesh"] not in parents:
        full, text, ok = f_ensure_mesh(run, args, log)
        parents[run["mesh"]] = (full, parse_air_gen(text), ok)
    full, gen, ok = parents[run["mesh"]]
    rec["gen"] = gen
    if not ok:
        rec["status"] = "Netzpruefung gescheitert"
        return rec
    vtk, stext, sok = f_ensure_stage(run, full, args, log)
    rec["surface"] = parse_surface(stext)
    rec["bnodes"] = rec["surface"].get("bnodes")
    rec["dense_gib_est"] = rec["surface"].get("dense_gib")
    if not sok:
        rec["status"] = "Stufenpruefung gescheitert"
        return rec
    if rec["dense_gib_est"] and rec["dense_gib_est"] > args.budget_gib and run["mode"] == "B":
        rec["status"] = "Speicher"
        return rec
    base = os.path.splitext(os.path.basename(vtk))[0]
    tag = "D" if run["mode"] == "D" else run["stage"]
    suffix = "_vf" if "--verify-field" in run["extra"] else ""
    dump_rel = os.path.join(os.path.relpath(args.workdir, args.build),
                            f"{run['mesh']}_{run['name'][len(run['mesh']) + 1:]}.csv")
    log_path = os.path.join(args.workdir, run["name"] + ".log")
    csv_path = os.path.join(args.build, dump_rel)
    mesh_rel = os.path.relpath(vtk, args.build)
    cmd = [args.fembem, "--reference", "--harmonic", "--fixed-plate", f"--mesh={mesh_rel}"]
    if run["mode"] == "D":
        cmd.append("--reference-dirichlet")
    if not run["extra"]:
        cmd.append(f"--reference-dump-a={dump_rel}")
    cmd += run["extra"]
    rec["cmd"] = "fembem " + " ".join(cmd[1:])
    rec["log"] = os.path.relpath(log_path, REPO)
    if args.dry_run:
        rec["status"] = "geplant"
        return rec
    if not (args.reuse and os.path.exists(log_path)):
        log(rec["cmd"])
        t0 = time.time()
        rc, _ = f_run_cmd(cmd, log_path, cwd=args.build)
        rec["exit"] = rc
        rec["wall_s"] = time.time() - t0
        with open(log_path + ".exit", "w") as f:
            f.write(str(rc))
    elif os.path.exists(log_path + ".exit"):
        rec["exit"] = int(open(log_path + ".exit").read().strip() or "0")
    text = open(log_path, encoding="utf-8", errors="replace").read()
    rec.update(parse_log(text))
    if rec.get("exit") not in (None, 0):
        rec["status"] = f"Exit-Code {rec['exit']}"
    elif "--verify-field" in run["extra"]:
        rec["status"] = "ok" if rec.get("vf_worst_pct") is not None and rec.get("converged") else "kein Ergebnis"
    elif rec.get("fz3") is None:
        rec["status"] = "kein Ergebnis"
    elif not rec.get("converged"):
        rec["status"] = "NICHT KONVERGIERT"
    elif run["mode"] == "D" and rec.get("dirichlet_max_a") != 0.0:
        rec["status"] = "Dirichlet verletzt"
    else:
        rec["status"] = "ok"
    # Vergleich von A mit dem Bezug (B0 auf t3 bzw. B0 des gleichen Netzes)
    if rec["status"] == "ok" and os.path.exists(csv_path) and not run["extra"]:
        ref = ref_dump.get(run["mesh"]) or ref_dump.get("t3_" + str(run["z0"]))
        if ref and os.path.exists(ref) and ref != csv_path:
            import compare_a
            try:
                rec["cmp"] = compare_a.compare(ref, csv_path)
            except compare_a.CompareError as e:
                rec["status"] = f"Vergleich: {e}"
        if run["mode"] == "B" and run["stage"] == "B0":
            ref_dump[run["mesh"]] = csv_path
            if abs(run["z0"] - 11.3) < 1e-9 and run["mesh"].endswith("_t3"):
                ref_dump["t3_" + str(run["z0"])] = csv_path
    return rec


def run_group_f(args, log):
    parents, ref_dump, recs = {}, {}, []
    runs = plan_f(args)
    for run in runs:
        rec = f_run_one(run, args, log, parents, ref_dump)
        recs.append(rec)
        log(f"{rec['name']}: {rec['status']}"
            + (f", F_z = {rec['fz3']:.5f} N" if rec.get("fz3") is not None else ""))
    # Voraussetzung AP 7: Leiter- und Spaltknoten gleich ueber die Abschneidefaktoren
    trunc_meshes = [os.path.join(args.workdir, m + ".vtk") for m in
                    [f_mesh_name(args.f_scale, 11.3, t) for t, _, _ in F_TRUNC]
                    if os.path.exists(os.path.join(args.workdir, m + ".vtk"))]
    node_check = []
    if len(trunc_meshes) >= 2 and not args.smoke:
        bad = f_check_conductor_nodes(trunc_meshes, args)
        node_check.append("Leiter- und Spaltknoten ueber die Abschneidefaktoren identisch: "
                          + ("ja" if not bad else "NEIN bei " + ", ".join(bad)))
    return recs, node_check


def f_summary(recs, node_check, args):
    md = ["# Gruppe F: Netze mit Luft", ""]
    md.append("## Laeufe\n\n" + f_table(recs, "")) 
    ref = f_mesh_name(0.5 if args.smoke else args.f_scale, 11.3, "t3") + "_B0"
    md.append(f"\n## A gegen B0(t3) ({ref})\n\n" + f_table_a(recs, ref))
    md.append("\n## Kopplungsflaeche\n\n" + f_surface_table(recs))
    vf = [r for r in recs if r.get("vf_worst_pct") is not None]
    if vf:
        md.append("\n## Feldpruefung s = 0 (B_z auf der Achse gegen Biot-Savart)\n\n"
                  "| Lauf | groesste Abweichung [%] | mittlere Abweichung [%] |\n|---|---:|---:|\n")
        for r in vf:
            devs = r.get("vf_devs") or []
            mean = sum(devs) / len(devs) if devs else None
            md.append(f"| {r['name']} | {r['vf_worst_pct']:.3f} | {fmt(mean, '+.3f')} |\n")
    seen, gm = set(), []
    for r in recs:
        if r["mesh"] in seen or not r.get("gen"):
            continue
        seen.add(r["mesh"])
        g = r["gen"]
        gm.append(f"| {r['mesh']} | {g['cells']} | {g['zones']} | {g['min_sj']} | {g['trunc']} | "
                  f"{g['bnodes_full']} |")
    if gm:
        md.append("\n## Netze\n\n| Netz | Zellen je Komponente | Zellen je Zone | kleinste skal. Jacobi-Det. | "
                  "Abschneidefaktoren | Randknoten (voll) |\n|---|---|---|---|---|---:|\n" + "\n".join(gm) + "\n")
    notes = node_check + f_evaluate(recs, 0.5 if args.smoke else args.f_scale)
    if notes:
        md.append("\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--groups", default="A,B,C,D", help="Gruppen, z. B. A,B,C,D,E")
    ap.add_argument("--build", default=os.path.join(REPO, "build"),
                    help="Verzeichnis, in dem fembem laeuft (Netze team28.vtk usw.)")
    ap.add_argument("--fembem", default=None, help="Pfad zu fembem (Vorgabe: build/Release)")
    ap.add_argument("--workdir", default=None, help="Netze und Logs (Vorgabe: build/refseries)")
    ap.add_argument("--budget-gib", type=float, default=18.0,
                    help="hoechster Bedarf der dichten BEM, darueber kein Lauf")
    ap.add_argument("--reuse", action="store_true", help="vorhandene Logs auswerten statt rechnen")
    ap.add_argument("--dry-run", action="store_true", help="Netze erzeugen und pruefen, Randknoten und Speicher, kein fembem")
    ap.add_argument("--smoke", action="store_true",
                    help="Gruppe F: Rauchtest scale 0.5, 11,3 mm, D, B0 und B_end")
    ap.add_argument("--f-parts", default="trunc,chain,z38",
                    help="Gruppe F: Teile (trunc, chain, z38)")
    ap.add_argument("--f-scale", type=float, default=1.0, help="Gruppe F: --scale der Luftnetze")
    ap.add_argument("--f-extra", action="append", default=[],
                    help="Gruppe F: weitere Stufen, SCALE:Z0:STUFE,STUFE (z. B. 1.5:11.3:B_S,B_rille)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args.build = os.path.abspath(args.build)
    if args.fembem is None:
        exe = "fembem.exe" if os.name == "nt" else "fembem"
        cands = [os.path.join(args.build, "Release", exe), os.path.join(args.build, exe)]
        args.fembem = next((c for c in cands if os.path.exists(c)), cands[0])
    groups = set(g.strip().upper() for g in args.groups.split(","))
    if args.smoke:
        groups = {"F"}
    only_f = groups == {"F"}
    args.workdir = os.path.abspath(args.workdir or os.path.join(
        args.build, "refair" if only_f else "refseries"))
    if args.smoke:
        args.workdir = os.path.join(args.workdir, "smoke") if not args.workdir.endswith("smoke") else args.workdir
    os.makedirs(args.workdir, exist_ok=True)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if "F" in groups:
        recs_f, node_check = run_group_f(args, log)
        md_f = f_summary(recs_f, node_check, args)
        open(os.path.join(args.workdir, "summary.md"), "w", encoding="utf-8").write(md_f)
        json.dump(recs_f, open(os.path.join(args.workdir, "results.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print()
        print(md_f)
        bad = [r for r in recs_f if r["status"] not in ("ok", "geplant", "Speicher")]
        if bad:
            print("FEHLGESCHLAGEN: " + ", ".join(f"{r['name']} ({r['status']})" for r in bad))
        groups.discard("F")
        if not groups:
            return 1 if bad else 0

    recs = []
    for run in plan(groups):
        if run["name"] == "C_finest_nopen":
            done = [r for r in recs if r["group"] == "A" and r["status"] in ("ok", "geplant")]
            if not done:
                recs.append(dict(name=run["name"], group="C", what=run["what"], z=run["z"],
                                 status="kein Netz aus A", mesh="–"))
                continue
            src = done[-1]
            run = dict(run, mesh=src["mesh"][:-4], gen=["--scale", src["what"].split()[-1]],
                       what=f"{src['what']}, ohne Strafterm im Leiter")
            run["ref"] = src["name"]
        rec = run_one(run, args, log)
        if "ref" in run:
            rec["ref"] = run["ref"]
        recs.append(rec)
        log(f"{rec['name']}: {rec['status']}"
            + (f", F_z = {rec['fz3']:.5f} N" if rec.get("fz3") is not None else ""))

    md = table(recs) + "\n" + mesh_table(recs) + "\n" + "\n".join(f"- {e}" for e in evaluate(recs)) + "\n"
    open(os.path.join(args.workdir, "summary.md"), "w", encoding="utf-8").write(md)
    json.dump(recs, open(os.path.join(args.workdir, "results.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print()
    print(md)
    bad = [r for r in recs if r["status"] not in ("ok", "geplant", "Speicher")]
    if bad:
        print("FEHLGESCHLAGEN: " + ", ".join(f"{r['name']} ({r['status']})" for r in bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
