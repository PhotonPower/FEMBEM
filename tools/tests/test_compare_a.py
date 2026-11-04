#!/usr/bin/env python3
"""
Tests fuer tools/compare_a.py: identische Dateien ergeben 0, skaliertes A den
Faktor, vertauschte Knotenreihenfolge aendert nichts; Gegenprobe: eine doppelte
Koordinate wird als Fehler gemeldet (ebenso fehlende Knoten).

    python3 tools/tests/test_compare_a.py        # oder: pytest tools/tests
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import compare_a as ca  # noqa: E402


def make_data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-0.1, 0.1, size=(n, 3))
    comp = np.array([3] * 20 + [2] * 25 + [1] * 15)
    A = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
    return xyz, comp, A


def write(path, xyz, comp, A, order=None):
    idx = np.arange(len(xyz)) if order is None else order
    with open(path, "w") as f:
        f.write(",".join(ca.COLUMNS) + "\n")
        for i in idx:
            f.write(",".join("%.17g" % v for v in xyz[i]) + f",{comp[i]},"
                    + ",".join("%.17g" % v for v in A[i].real) + ","
                    + ",".join("%.17g" % v for v in A[i].imag) + "\n")


def tmp(name):
    d = getattr(tmp, "d", None)
    if d is None:
        tmp.d = d = tempfile.mkdtemp(prefix="cmpa_")
    return os.path.join(d, name)


def test_identical_files_give_zero():
    xyz, comp, A = make_data()
    write(tmp("a.csv"), xyz, comp, A)
    write(tmp("b.csv"), xyz, comp, A)
    res = ca.compare(tmp("a.csv"), tmp("b.csv"))
    assert res["n_matched"] == len(xyz) and not res["only_ref"] and not res["only_test"]
    for b in res["bodies"].values():
        assert b["rel_r"] == 0.0 and b["rel_i"] == 0.0 and b["max_dev_rel"] == 0.0
        assert abs(b["phase_deg"]) < 1e-12 and abs(b["amp_ratio"] - 1.0) < 1e-12
    assert ca.main([tmp("a.csv"), tmp("b.csv")]) == 0


def test_scaled_A_gives_the_factor():
    xyz, comp, A = make_data(1)
    write(tmp("a.csv"), xyz, comp, A)
    f = 1.25
    write(tmp("c.csv"), xyz, comp, f * A)
    res = ca.compare(tmp("a.csv"), tmp("c.csv"))
    for b in res["bodies"].values():
        assert abs(b["rel_r"] - (f - 1.0)) < 1e-12 and abs(b["rel_i"] - (f - 1.0)) < 1e-12
        assert abs(b["amp_ratio"] - f) < 1e-12 and abs(b["phase_deg"]) < 1e-9
    # Phase: Multiplikation mit exp(j*0.3) dreht die Phase um 0,3 rad
    write(tmp("d.csv"), xyz, comp, A * np.exp(0.3j))
    res = ca.compare(tmp("a.csv"), tmp("d.csv"))
    for b in res["bodies"].values():
        assert abs(b["phase_deg"] - np.degrees(0.3)) < 1e-9 and abs(b["amp_ratio"] - 1.0) < 1e-12


def test_permuted_node_order_changes_nothing():
    xyz, comp, A = make_data(2)
    write(tmp("a.csv"), xyz, comp, A)
    perm = np.random.default_rng(5).permutation(len(xyz))
    write(tmp("p.csv"), xyz, comp, A, order=perm)
    res = ca.compare(tmp("a.csv"), tmp("p.csv"))
    for b in res["bodies"].values():
        assert b["rel_r"] == 0.0 and b["rel_i"] == 0.0 and b["max_dev_rel"] == 0.0


def test_max_deviation_and_location():
    xyz, comp, A = make_data(3)
    B = A.copy()
    B[7] += 0.5 + 0.0j                                  # ein Knoten gestoert
    write(tmp("a.csv"), xyz, comp, A)
    write(tmp("e.csv"), xyz, comp, B)
    res = ca.compare(tmp("a.csv"), tmp("e.csv"))
    b = res["bodies"][3]
    mags = np.sqrt(np.sum(np.abs(A[comp == 3]) ** 2, axis=1))
    assert abs(b["max_dev_rel"] - np.sqrt(3 * 0.25) / mags.max()) < 1e-12
    assert np.allclose(b["where_mm"], xyz[7] * 1e3)


def test_counterprobe_duplicate_coordinate_is_an_error():
    xyz, comp, A = make_data(4)
    write(tmp("a.csv"), xyz, comp, A)
    xyz2 = xyz.copy()
    xyz2[10] = xyz2[11]                                  # zwei Knoten, eine Koordinate
    write(tmp("dup.csv"), xyz2, comp, A)
    try:
        ca.compare(tmp("a.csv"), tmp("dup.csv"))
        raise AssertionError("doppelte Koordinate wurde nicht gemeldet")
    except ca.CompareError:
        pass
    assert ca.main([tmp("a.csv"), tmp("dup.csv")]) == 1
    try:
        ca.compare(tmp("dup.csv"), tmp("a.csv"))
        raise AssertionError("doppelte Koordinate in REF wurde nicht gemeldet")
    except ca.CompareError:
        pass


def test_counterprobe_missing_node_reported_and_fails():
    xyz, comp, A = make_data(5)
    write(tmp("a.csv"), xyz, comp, A)
    keep = np.arange(len(xyz) - 1)
    write(tmp("f.csv"), xyz[keep], comp[keep], A[keep])
    res = ca.compare(tmp("a.csv"), tmp("f.csv"))
    assert res["only_ref"] == 1 and res["only_test"] == 0
    assert ca.main([tmp("a.csv"), tmp("f.csv")]) == 1


def test_wrong_header_rejected():
    with open(tmp("h.csv"), "w") as f:
        f.write("a,b,c\n1,2,3\n")
    try:
        ca.read_dump(tmp("h.csv"))
        raise AssertionError("falsche Kopfzeile akzeptiert")
    except ca.CompareError:
        pass


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok             {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FEHLGESCHLAGEN {name}: {e!r}")
    print(f"{len(tests) - fails} von {len(tests)} Tests bestanden")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
