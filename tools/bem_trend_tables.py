#!/usr/bin/env python3
"""Tabellen fuer docs/design/bem-near-field.md (Konvergenztrend des BEM-FEM-Pfads).

Liest aus einem Verzeichnis je Skalierung SC (0.5, 0.75, 1.0, ...):
    bf_sSC.log / bf_sSC.csv     BEM-FEM, 50 Hz (--harmonic-inner-direct, --harmonic-field-dump)
    ref_sSC.log / ref_sSC.csv   Referenzpfad auf demselben Netz
    row_sSC.log                 --bem-row-check bei 2 Hz
    hex_sSC.vtk                 das Netz (Panelkanten werden gemessen)
und schreibt Markdown-Tabellen: F_z je Phase und Mittel, Iterationen, F_z / F_ref,
Projektion auf Biot-Savart und Vergleich gegen den Referenzpfad (compare_fields.py),
Zeilenresiduen, Plattenresiduum ueber Panelkante / Plattendicke, F_z nach beiden Schaetzern
(Maxwell-Spannungstensor und Lorentz-Volumenkraft; letztere ist der maßgebliche Schaetzer,
siehe docs/design/force-estimator-check.md). Mit c_bf_sSC.log (--bem-near-subdiv) als zusaetzliche Zeilen.

    python3 tools/bem_trend_tables.py DIR [--prefix-bf bf] [SC ...]
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bem_panel_sizes  # noqa: E402
import numpy as np  # noqa: E402


def read(path):
    return open(path, encoding="utf-8", errors="replace").read()


def parse_force(log):
    ph = [float(m.group(1)) for m in re.finditer(r"Phase \d \(\d+ Grad\): F_z = ([-+0-9.eE]+) N", log)]
    mg = re.search(r"F_z / \(m g\) = ([-+0-9.eE]+)", log)
    it = re.search(r"(\d+) aeussere Iterationen, rel\. Residuum ([0-9.eE+-]+)", log)
    inner = re.search(r"innere \(W\+T\)-Loesungen: (\d+) mit zusammen (\d+) Iterationen", log)
    f_mean = float(mg.group(1)) * 1.0497 if mg else float("nan")   # m g = 1.0497 N
    return ph[:4], f_mean, (int(it.group(1)), it.group(2)) if it else None, inner.groups() if inner else None


def parse_lorentz_z(log):
    """F_z der Zeile 'LORENTZ (Volumen)' (Volumenintegral <J x B>, nur Platte)."""
    m = re.search(r"LORENTZ \(Volumen\)\s*: \(([-+0-9.eE]+), ([-+0-9.eE]+), ([-+0-9.eE]+)\) N", log)
    return float(m.group(3)) if m else float("nan")


def parse_maxwell_z(log):
    """F_z der Zeile 'ZEITGEMITTELTE KRAFT' (8-Phasen-Mittel des Maxwell-Spannungstensors)."""
    m = re.search(r"ZEITGEMITTELTE KRAFT\s*: \(([-+0-9.eE]+), ([-+0-9.eE]+), ([-+0-9.eE]+)\) N", log)
    return float(m.group(3)) if m else float("nan")


def estimator_table(d, scs, prefixes):
    """Tabelle: beide Kraftschaetzer des BEM-FEM-Pfads gegen den Referenzpfad (Lorentz).

    prefixes: Liste (Praefix, Beschriftung), z. B. [("bf", ""), ("c_bf", " (--bem-near-subdiv)")].
    Der Referenzpfad gibt nur die Lorentzkraft aus; eine Spalte Maxwell-Referenz gibt es nicht.
    """
    print("### F_z nach Schaetzer (50 Hz, 11,3 mm)\n")
    print("| Skalierung | F_z Maxwell BEM-FEM [N] | F_z Lorentz BEM-FEM [N] | F_z Lorentz Referenz [N] | Maxwell / Ref | Lorentz / Ref |")
    print("|---|---:|---:|---:|---:|---:|")
    for pre, label in prefixes:
        for s in scs:
            f = os.path.join(d, f"{pre}_s{s}.log")
            r = os.path.join(d, f"ref_s{s}.log")
            if not (os.path.exists(f) and os.path.exists(r)):
                continue
            log = read(f)
            mx, lo, rf = parse_maxwell_z(log), parse_lorentz_z(log), parse_ref_force(read(r))
            print(f"| {s}{label} | {mx:.4g} | {lo:.4g} | {rf:.5g} | {mx / rf:.3f} | {lo / rf:.3f} |")
    print()


def parse_ref_force(log):
    m = re.search(r"F_z = ([-+0-9.eE]+) N", log)
    return float(m.group(1))


def compare(test_csv, ref_csv, mesh, biot, freq=50.0, z0=11.3):
    cmd = [sys.executable, os.path.join(HERE, "compare_fields.py"), test_csv, ref_csv, "--mesh", mesh, "--markdown"]
    if biot:
        cmd += ["--biot-savart", "--freq", str(freq), "--plate-z0", str(z0)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.split("\n")
    heads = [i for i, l in enumerate(out) if l.startswith("==")]
    sec = 1 if biot else 0
    rows = []
    for l in out[heads[sec]:(heads[sec + 1] if sec + 1 < len(heads) else len(out))]:
        c = [x.strip() for x in l.strip("|").split("|")]
        if len(c) >= 9 and c[1] == "alle" and c[0] in ("Platte", "Innenspule", "Aussenspule"):
            rows.append(c)
    return rows


def row_numbers(log):
    """Zeile 1 (+K1, +K2) und Zeile 2 (+K1, +K2) je Koerper (V0, Zeile 'alle') und gesamt."""
    sections = {}
    key = None
    for l in log.split("\n"):
        if l.startswith("Zeile 1 (FEM auf Randkanten), V0"):
            key = "z1"
        elif l.startswith("Zeile 2 (BEM), V0"):
            key = "z2"
        elif l.startswith("Zeile") or l.startswith("T-Test") or l.startswith("||k1") or l.startswith("Randspur"):
            key = None
        elif key and l.startswith("|"):
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) >= 7 and c[1] == "alle":
                sections.setdefault(key, {})[c[0]] = c[3:]
            elif len(c) >= 7 and c[0] == "alle":
                sections.setdefault(key, {})["alle"] = c[3:]
    return sections


def plate_row2_by_class(log):
    """Plattenresiduum Zeile 2 (+K1) fuer 'eben' und 'alle'."""
    out = {}
    key = None
    for l in log.split("\n"):
        if l.startswith("Zeile 2 (BEM), V0"):
            key = True
        elif l.startswith("Zeile") or l.startswith("T-Test"):
            key = None
        elif key and l.startswith("| Platte |"):
            c = [x.strip() for x in l.strip("|").split("|")]
            out[c[1]] = c[3]
    return out


def main():
    d = sys.argv[1]
    scs = sys.argv[2:] or ["0.5", "0.75", "1.0", "1.25"]
    scs = [s for s in scs if os.path.exists(os.path.join(d, f"bf_s{s}.log"))]
    estimator_table(d, scs, [("bf", ""), ("c_bf", " (--bem-near-subdiv)")])
    print("### F_z und Iterationen (50 Hz, 11,3 mm)\n")
    print("| Skalierung | F_z Phasen 0/45/90/135 [N] | F_z Mittel [N] | F_z Referenzpfad [N] | F_z / F_ref | aeussere / innere Iterationen |")
    print("|---|---|---:|---:|---:|---|")
    data = {}
    for s in scs:
        ph, f, it, inner = parse_force(read(os.path.join(d, f"bf_s{s}.log")))
        fr = parse_ref_force(read(os.path.join(d, f"ref_s{s}.log")))
        data[s] = (f, fr)
        its = f"{it[0]} ({it[1]}) / {inner[1]}" if it and inner else "?"
        print(f"| {s} | " + " / ".join(f"{x:.3g}" for x in ph) + f" | {f:.4f} | {fr:.5f} | {f / fr:.3f} | {its} |")
    print("\n### Panelkanten aus dem Netz (Platte, Deckel/Boden)\n")
    print("| Skalierung | Plattendicke [mm] | Deckelkante mittel [mm] | sqrt(Flaeche) [mm] | Kante/Dicke |")
    print("|---|---:|---:|---:|---:|")
    ratios = {}
    for s in scs:
        out, t = bem_panel_sizes.analyse(os.path.join(d, f"hex_s{s}.vtk"))
        n, emin, emean, emax, sq = out[(3, "Deckel/Boden")]
        ratios[s] = emean / t
        print(f"| {s} | {t:.3f} | {emean:.3f} | {sq:.3f} | {emean / t:.3f} |")
    print("\n### Zeilenresiduen bei 2 Hz (V0 = Operator; Zeile 1 mit +K2, Zeile 2 mit +K1)\n")
    print("| Skalierung | Z1 Platte | Z1 Innenspule | Z1 Aussenspule | Z1 gesamt | Z2 Platte (alle) | Z2 Platte (eben) | Z2 Innenspule | Z2 Aussenspule | Z2 gesamt | Kante/Dicke |")
    print("|---|---|---|---|---|---|---|---|---|---|---:|")
    for s in scs:
        log = read(os.path.join(d, f"row_s{s}.log"))
        r = row_numbers(log)
        pc = plate_row2_by_class(log)
        z1, z2 = r["z1"], r["z2"]
        print(f"| {s} | {z1['Platte'][2]} | {z1['Innenspule'][2]} | {z1['Aussenspule'][2]} | {z1['alle'][2]} | {z2['Platte'][0]} | {pc.get('eben','?')} | {z2['Innenspule'][0]} | {z2['Aussenspule'][0]} | {z2['alle'][0]} | {ratios[s]:.2f} |")
    print("\n### Felder: Projektion auf Biot-Savart (+Ringmodell) und Vergleich gegen den Referenzpfad (Zeile \"alle\")\n")
    for s in scs:
        mesh = os.path.join(d, f"hex_s{s}.vtk")
        bs = compare(os.path.join(d, f"bf_s{s}.csv"), os.path.join(d, f"bf_s{s}.csv"), mesh, True)
        rf = compare(os.path.join(d, f"bf_s{s}.csv"), os.path.join(d, f"ref_s{s}.csv"), mesh, False)
        print(f"Skalierung {s}:\n")
        print("| Koerper | Phase | proj Biot-Savart | L2 Biot-Savart | B/Bref Biot-Savart | proj Referenzpfad | L2 Referenzpfad | B/Bref Referenzpfad | B_phi/B |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for a, b in zip(bs, rf):
            print(f"| {a[0]} | {a[2]} | {a[4]} | {a[5]} | {a[6]} | {b[4]} | {b[5]} | {b[6]} | {a[7]} |")
        print()


if __name__ == "__main__":
    main()
