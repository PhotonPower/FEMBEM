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
    args.workdir = os.path.abspath(args.workdir or os.path.join(args.build, "refseries"))
    os.makedirs(args.workdir, exist_ok=True)
    groups = set(g.strip().upper() for g in args.groups.split(","))

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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
