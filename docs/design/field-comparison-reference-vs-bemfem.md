# Feldvergleich: Referenzpfad gegen BEM-FEM harmonisch

Stand 2026-11-03. Auftrag: auf identischem Netz die Felder beider Löser je
Körper vergleichen und lokalisieren, wo die Kopplung der Kanten-Elemente an
die BEM abweicht. **Der Fehler wird nicht behoben**; produktiver Pfad und
Löser sind unverändert. Neu sind nur der Dump-Schreiber im Referenzpfad
(`include/ReferenceFieldDump.cuh`, `src/ReferenceFieldDump.cu`) und die
Werkzeuge (`tools/compare_fields.py`).

## Ergebnis in Kürze

**Gemessen**

- Auf identischem Netz (`--scale 1.0`, 2 Hz) ist das BEM-FEM-Feld in der
  Platte praktisch unkorreliert mit dem Referenzfeld (Projektion +0,035) und
  hat den 3,7-fachen Betrag; in der Innenspule +0,87 bei 2,9-fachem Betrag,
  in der Außenspule +0,52 bei 3,3-fachem Betrag. Der relative L2-Fehler liegt
  überall über 1.
- Der Fehler sitzt auch im Inneren der Spulen (Außenspule innen: Projektion
  0,30, L2 1,05), wächst aber im Betrag zum Rand hin (Betragsverhältnis
  Innenspule 3,04 am Rand gegen 1,55 innen, Außenspule 3,72 gegen 0,84).
- B_φ/B beträgt im BEM-FEM-Feld 0,13 bis 0,55 (Referenz: höchstens 4e-4) und
  ist nicht auf den Rand beschränkt.
- Das BEM-FEM-Feld hängt von der Frequenz ab, das Referenzfeld nicht:
  Realteil bei 50 Hz gegen 2 Hz, Projektion 0,51 / 0,41 / 0,74 (Platte /
  Innen- / Außenspule); beim Referenzpfad ändert sich der rms von |B_r| in der
  Platte von 58,4 auf 56,6 mT.
- `--scale 1.5` wurde für BEM-FEM nicht gerechnet (Speicher, siehe
  Aufgabe 4).

**Nicht belastbar:** alles unter „Interpretation“. Zwei Netze, ein
Plattenabstand (11,3 mm).

## Werkzeuge (Aufgaben 1 bis 3)

**Dump im Referenzpfad.** `fembem --reference --harmonic[=HZ] --fixed-plate
--harmonic-field-dump=DATEI` schreibt je Hexaeder eine Zeile im Format des
BEM-FEM-Dumps (`elem,body,x,y,z,Arx..Aiz,Brx..Biz`, SI-Einheiten, Körper 1
Platte, 2 Innen-, 3 Außenspule). Eigener Schreiber ohne Code aus dem
Nédélec-Pfad. **Definition:** A ist das Q1-Knotenfeld A(x) = Σ N_i(x) A_i. Am
Schwerpunkt (Referenzmittelpunkt; der Bildpunkt ist der Mittelwert der acht
Knoten) ist A_c = (1/8) Σ A_i und **B = rot A = Σ grad N_i × A_i**,
elementweise und ungeglättet, mit grad N_i aus `evaluate_nodal_gradients`.
Das ist dieselbe Auswertung wie in der Lorentzkraft des Referenzpfads. Der
FEM-System-Teil (Kanten, W-Tripel) entfällt. Die Option wurde in
`Backend.cu` für `--reference --harmonic` freigegeben; alle übrigen
Nédélec-Optionen werden weiter abgewiesen.

**Vergleich.** `tools/compare_fields.py TEST.csv REF.csv --mesh NETZ.vtk
[--biot-savart --freq HZ --plate-z0 MM] [--markdown]`. Je Körper, Phase
(Re/Im getrennt) und Schicht: Projektion ⟨B_test, B_ref⟩/⟨B_ref, B_ref⟩ über
alle drei Komponenten, relativer L2-Fehler, Betragsverhältnis der L2-Normen,
B_φ-Anteil beider Dumps. Schicht aus dem Netz: Randtiefe = kleinste Zahl von
Flächenübergängen im Körper bis zu einem Element mit freier Fläche (Tiefe 0 =
Randschicht, „Innen“ = Tiefe ≥ 1). Das Werkzeug bricht ab, wenn die Dumps
oder Netz und Dump nicht zusammenpassen.

**Tests.** `test_reference_field_dump` (Host; Rundlauf mit linearem Feld auf
zwei verzerrten Hexaedern, 1,3e-10; gegen Gegenproben: vertauschte Phasen und
vertauschte Körperkennungen werden erkannt) und
`tools/tests/test_compare_fields.py` (synthetische Felder mit bekannter
Schichtstruktur; Gegenproben: vertauschte Phasen, vertauschte Körper, Dump
eines anderen Netzes). `ctest -L host`: 36 von 36 grün, `ctest -L tools`:
8 von 8 grün. (In dieser Umgebung braucht ctest `LD_LIBRARY_PATH` auf cudart
und cublas des HPC-SDK; das ist keine Codeänderung.)

**Werkzeugprüfung (Aufgabe 3), gemessen.**

1. Referenz gegen Biot-Savart bei 2 Hz auf `--scale 1.0` (Tabelle unten):
   Re-Projektion Platte 0,994, Innenspule 1,026, Außenspule 0,999. In der
   Innenspule liegen Rand bei 1,048 und innen bei 0,901. Das Q1-Feld am
   Schwerpunkt der Wicklungselemente und die Fadenmodell-Referenz in der
   Wicklung stimmen also nur auf etwa 10 % überein; Aussagen für Werte nahe 1
   sind nicht genauer als das.
2. Auf `team28.vtk` bei 2 Hz (BEM-FEM, 37 min) liefert das Werkzeug Platte
   +0,0016, Innenspule +0,2765, Außenspule +0,0411 und B_φ/B 0,133 / 0,237 /
   0,172. Das reproduziert die Zahlen aus `bemfem-field-check.md` (+0,002 /
   +0,28 / +0,04, B_φ/B 0,13 / 0,24 / 0,17).

## Läufe (Aufgabe 4)

Gleiches Netz für beide Löser, Platte bei 11,3 mm. Netze erzeugt mit
`python3 tools/gmsh/team28_hex.py --scale S -o hex_sS.vtk` (S = 1.0 ist
byteweise `data/team28/team28_hex.vtk`).

| Lauf | Hexaeder | Referenz | BEM-FEM (äußere Iterationen, rel. Residuum) | Zeit BEM-FEM |
|---|---|---|---|---|
| 0,5; 2 Hz | 696 | konvergiert (4 FGMRES-It.) | 8, 9,05e-07 | 4 min |
| 0,5; 50 Hz | 696 | F_z = 0,85333 N | 22, 8,35e-07 | 4 min |
| 1,0; 2 Hz | 4 944 | konvergiert (4 It.) | 9, 2,08e-07 | 27 min |
| 1,0; 50 Hz | 4 944 | F_z = 0,96649 N | 24, 7,71e-07 | 14 min |
| `team28.vtk`; 2 Hz (nur Werkzeugprüfung) | 4 805 | – | 7, 4,43e-07 | 38 min |

(Zeiten mit parallel laufenden Läufen auf derselben GPU, nur Größenordnung.)

`--scale 1.5` für BEM-FEM **nicht gerechnet**: Der Pfad mit
`--harmonic-inner-direct` speichert (W+T) dicht, nd² · 8 B. Aus dem Netz
(Kanten + 4 · Randflächen; auf `--scale 1.0` exakt die bekannten 33 435)
folgt für `--scale 1.5` nd = 56 416 + 4 · 8 280 = 89 536, das sind 59,7 GiB
gegen 48 GiB der RTX 6000 Ada, ohne Arbeitsspeicher der Zerlegung. Es passt
nicht; nötig wäre die Schur-Variante aus `bem-block-preconditioning.md`.

Die Referenz-Kraft auf `--scale 1.0` liegt 4,0 % unter dem Ringmodell
(0,9665 gegen 1,0068 N). Der Vergleich trägt deshalb nur für Abweichungen
weit darüber. Alle hier gemessenen Abweichungen sind es (Projektionen 0,03
bis 0,95, Normverhältnisse 0,7 bis 1700); die 4 % ändern keine Aussage.

Befehle (aus `build/`, S = 0.5 oder 1.0, F = 2 oder 50):

```
./fembem --reference --harmonic=F --fixed-plate --plate-z=11.3 \
    --mesh=hex_sS.vtk --harmonic-field-dump=ref_sS_FHz.csv
./fembem --harmonic=F --harmonic-inner-direct --fixed-plate --plate-z=11.3 \
    --mesh=hex_sS.vtk --harmonic-field-dump=bf_sS_FHz.csv
python3 ../tools/compare_fields.py bf_sS_FHz.csv ref_sS_FHz.csv \
    --mesh hex_sS.vtk --markdown
python3 ../tools/compare_fields.py ref_s1.0_2Hz.csv ref_s1.0_2Hz.csv \
    --mesh hex_s1.0.vtk --biot-savart --freq 2
```

Der BEM-FEM-Lauf schreibt zusätzlich `.edges.csv` und `.W.csv`.

## Tabellen

Spalten: Projektion proj (Soll 1), relativer L2-Fehler, Betragsverhältnis
|B|/|B_ref|, B_φ-Anteil von TEST und Referenz. Schicht „Rand“ = Tiefe 0,
„Innen“ = Tiefe ≥ 1. Auf `--scale 0.5` hat kein Körper ein Inneres (Spule
2 × 4, Platte 5 × 2 Elemente im Querschnitt), dort gibt es nur „Rand“. Die
Zeilen „Tiefe 1“ sind weggelassen (mit `compare_fields.py` abrufbar).

### BEM-FEM gegen Referenz, `--scale 1.0`, 2 Hz

| Körper | Schicht | Phase | n | proj | L2 | ‖B‖/‖Bref‖ | Bphi test | Bphi ref |
|---|---|---|---|---|---|---|---|---|
| Platte | Rand | Re | 1296 | 0.0301 | 4.4465 | 4.3395 | 0.1932 | 0.0004 |
| Platte | Rand | Im | 1296 | 0.0389 | 37.4272 | 37.4148 | 0.2431 | 0.0005 |
| Platte | Innen | Re | 576 | 0.0451 | 1.9804 | 1.7356 | 0.2910 | 0.0004 |
| Platte | Innen | Im | 576 | 0.0541 | 19.4753 | 19.4524 | 0.3327 | 0.0000 |
| Platte | alle | Re | 1872 | 0.0347 | 3.8674 | 3.7451 | 0.2011 | 0.0004 |
| Platte | alle | Im | 1872 | 0.0433 | 33.3089 | 33.2952 | 0.2532 | 0.0004 |
| Innenspule | Rand | Re | 960 | 0.8562 | 2.9201 | 3.0396 | 0.5503 | 0.0000 |
| Innenspule | Rand | Im | 960 | 0.1250 | 262.0888 | 262.0874 | 0.6001 | 0.0001 |
| Innenspule | Innen | Re | 576 | 0.9521 | 1.2215 | 1.5480 | 0.1740 | 0.0000 |
| Innenspule | Innen | Im | 576 | -0.0326 | 71.1498 | 71.1423 | 0.2003 | 0.0001 |
| Innenspule | alle | Re | 1536 | 0.8673 | 2.7765 | 2.9058 | 0.5420 | 0.0000 |
| Innenspule | alle | Im | 1536 | 0.0876 | 231.4750 | 231.4732 | 0.5941 | 0.0001 |
| Aussenspule | Rand | Re | 960 | 0.5823 | 3.6998 | 3.7220 | 0.4924 | 0.0000 |
| Aussenspule | Rand | Im | 960 | 0.2907 | 1735.4959 | 1735.4958 | 0.8334 | 0.0001 |
| Aussenspule | Innen | Re | 576 | 0.3012 | 1.0468 | 0.8356 | 0.3926 | 0.0000 |
| Aussenspule | Innen | Im | 576 | 1.8903 | 161.7779 | 161.7865 | 0.6569 | 0.0000 |
| Aussenspule | alle | Re | 1536 | 0.5230 | 3.3213 | 3.3282 | 0.4912 | 0.0000 |
| Aussenspule | alle | Im | 1536 | 0.8959 | 1371.9941 | 1371.9944 | 0.8326 | 0.0001 |

### BEM-FEM gegen Referenz, `--scale 1.0`, 50 Hz

| Körper | Schicht | Phase | n | proj | L2 | ‖B‖/‖Bref‖ | Bphi test | Bphi ref |
|---|---|---|---|---|---|---|---|---|
| Platte | Rand | Re | 1296 | 0.0165 | 2.6904 | 2.5043 | 0.1653 | 0.0004 |
| Platte | Rand | Im | 1296 | 0.0229 | 7.9935 | 7.9336 | 0.1785 | 0.0005 |
| Platte | Innen | Re | 576 | 0.0264 | 1.3181 | 0.8890 | 0.2931 | 0.0004 |
| Platte | Innen | Im | 576 | 0.0303 | 3.0777 | 2.9211 | 0.3067 | 0.0000 |
| Platte | alle | Re | 1872 | 0.0195 | 2.3620 | 2.1490 | 0.1742 | 0.0004 |
| Platte | alle | Im | 1872 | 0.0250 | 6.9565 | 6.8878 | 0.1872 | 0.0004 |
| Innenspule | Rand | Re | 960 | 0.8559 | 1.2240 | 1.4866 | 0.3692 | 0.0000 |
| Innenspule | Rand | Im | 960 | 0.0697 | 140.2548 | 140.2517 | 0.6019 | 0.0001 |
| Innenspule | Innen | Re | 576 | 0.9500 | 0.5414 | 1.0923 | 0.1260 | 0.0000 |
| Innenspule | Innen | Im | 576 | -0.0168 | 39.0359 | 39.0226 | 0.2002 | 0.0001 |
| Innenspule | alle | Re | 1536 | 0.8669 | 1.1653 | 1.4463 | 0.3582 | 0.0000 |
| Innenspule | alle | Im | 1536 | 0.0500 | 124.6461 | 124.6424 | 0.5959 | 0.0001 |
| Aussenspule | Rand | Re | 960 | 0.5831 | 2.8321 | 2.8613 | 0.3183 | 0.0000 |
| Aussenspule | Rand | Im | 960 | 0.1846 | 1129.8579 | 1129.8577 | 0.8437 | 0.0001 |
| Aussenspule | Innen | Re | 576 | 0.3007 | 0.9653 | 0.7303 | 0.3332 | 0.0000 |
| Aussenspule | Innen | Im | 576 | 1.5452 | 104.6140 | 104.6240 | 0.6678 | 0.0000 |
| Aussenspule | alle | Re | 1536 | 0.5235 | 2.5545 | 2.5637 | 0.3186 | 0.0000 |
| Aussenspule | alle | Im | 1536 | 0.7006 | 892.5371 | 892.5373 | 0.8429 | 0.0001 |

### BEM-FEM gegen Referenz, `--scale 0.5`, 2 Hz

| Körper | Schicht | Phase | n | proj | L2 | ‖B‖/‖Bref‖ | Bphi test | Bphi ref |
|---|---|---|---|---|---|---|---|---|
| Platte | Rand | Re | 312 | -0.0628 | 2.3385 | 2.0840 | 0.3795 | 0.0017 |
| Platte | Rand | Im | 312 | 0.0259 | 99.2244 | 99.2196 | 0.3795 | 0.0015 |
| Innenspule | Rand | Re | 192 | 0.5376 | 1.5450 | 1.5692 | 0.4772 | 0.0000 |
| Innenspule | Rand | Im | 192 | 0.2617 | 24.2568 | 24.2470 | 0.7781 | 0.0004 |
| Aussenspule | Rand | Re | 192 | 0.3393 | 1.5510 | 1.4437 | 0.8468 | 0.0000 |
| Aussenspule | Rand | Im | 192 | 0.9057 | 245.9735 | 245.9752 | 0.9399 | 0.0002 |

### BEM-FEM gegen Referenz, `--scale 0.5`, 50 Hz

| Körper | Schicht | Phase | n | proj | L2 | ‖B‖/‖Bref‖ | Bphi test | Bphi ref |
|---|---|---|---|---|---|---|---|---|
| Platte | Rand | Re | 312 | 0.0071 | 1.0520 | 0.3475 | 0.3530 | 0.0017 |
| Platte | Rand | Im | 312 | 0.0096 | 1.7480 | 1.4403 | 0.4743 | 0.0015 |
| Innenspule | Rand | Re | 192 | 0.5383 | 1.5301 | 1.5549 | 0.4699 | 0.0000 |
| Innenspule | Rand | Im | 192 | 0.0156 | 1.0483 | 0.3606 | 0.5863 | 0.0004 |
| Aussenspule | Rand | Re | 192 | 0.3395 | 1.5394 | 1.4313 | 0.8441 | 0.0000 |
| Aussenspule | Rand | Im | 192 | 0.0241 | 4.7474 | 4.6461 | 0.9366 | 0.0001 |

### Referenz gegen Biot-Savart (2 Hz, `--scale 1.0`)

Der Imaginärteil der analytischen Referenz stammt aus dem Ringmodell.

| Körper | Schicht | Phase | n | proj | L2 | ‖B‖/‖Bref‖ | Bphi test | Bphi ref |
|---|---|---|---|---|---|---|---|---|
| Platte | Rand | Re | 1296 | 0.9936 | 0.0148 | 0.9937 | 0.0004 | 0.0000 |
| Platte | Rand | Im | 1296 | 0.9806 | 0.0392 | 0.9812 | 0.0005 | 0.0000 |
| Platte | Innen | Re | 576 | 0.9934 | 0.0134 | 0.9935 | 0.0004 | 0.0000 |
| Platte | Innen | Im | 576 | 0.9827 | 0.0287 | 0.9829 | 0.0000 | 0.0000 |
| Platte | alle | Re | 1872 | 0.9936 | 0.0144 | 0.9936 | 0.0004 | 0.0000 |
| Platte | alle | Im | 1872 | 0.9812 | 0.0365 | 0.9817 | 0.0004 | 0.0000 |
| Innenspule | Rand | Re | 960 | 1.0483 | 0.0925 | 1.0512 | 0.0000 | 0.0000 |
| Innenspule | Rand | Im | 960 | 0.9601 | 0.0446 | 0.9603 | 0.0001 | 0.0000 |
| Innenspule | Innen | Re | 576 | 0.9007 | 0.1499 | 0.9077 | 0.0000 | 0.0000 |
| Innenspule | Innen | Im | 576 | 0.9587 | 0.0432 | 0.9588 | 0.0001 | 0.0000 |
| Innenspule | alle | Re | 1536 | 1.0261 | 0.1032 | 1.0310 | 0.0000 | 0.0000 |
| Innenspule | alle | Im | 1536 | 0.9597 | 0.0443 | 0.9599 | 0.0001 | 0.0000 |
| Aussenspule | Rand | Re | 960 | 1.0038 | 0.0363 | 1.0044 | 0.0000 | 0.0000 |
| Aussenspule | Rand | Im | 960 | 0.9394 | 0.0667 | 0.9398 | 0.0001 | 0.0000 |
| Aussenspule | Innen | Re | 576 | 0.9825 | 0.0498 | 0.9836 | 0.0000 | 0.0000 |
| Aussenspule | Innen | Im | 576 | 0.9445 | 0.0585 | 0.9447 | 0.0000 | 0.0000 |
| Aussenspule | alle | Re | 1536 | 0.9991 | 0.0397 | 0.9999 | 0.0000 | 0.0000 |
| Aussenspule | alle | Im | 1536 | 0.9413 | 0.0637 | 0.9417 | 0.0001 | 0.0000 |

### BEM-FEM gegen Biot-Savart (2 Hz, `--scale 1.0`)

| Körper | Schicht | Phase | n | proj | L2 | ‖B‖/‖Bref‖ | Bphi test | Bphi ref |
|---|---|---|---|---|---|---|---|---|
| Platte | Rand | Re | 1296 | 0.0299 | 4.4199 | 4.3122 | 0.1932 | 0.0000 |
| Platte | Rand | Im | 1296 | 0.0260 | 36.7234 | 36.7105 | 0.2431 | 0.0000 |
| Platte | Innen | Re | 576 | 0.0450 | 1.9706 | 1.7243 | 0.2910 | 0.0000 |
| Platte | Innen | Im | 576 | 0.0462 | 19.1438 | 19.1201 | 0.3327 | 0.0000 |
| Platte | alle | Re | 1872 | 0.0345 | 3.8444 | 3.7213 | 0.2011 | 0.0000 |
| Platte | alle | Im | 1872 | 0.0318 | 32.6992 | 32.6849 | 0.2532 | 0.0000 |
| Innenspule | Rand | Re | 960 | 0.8984 | 3.0681 | 3.1953 | 0.5503 | 0.0000 |
| Innenspule | Rand | Im | 960 | 0.1425 | 251.6728 | 251.6713 | 0.6001 | 0.0000 |
| Innenspule | Innen | Re | 576 | 0.8614 | 1.1188 | 1.4051 | 0.1740 | 0.0000 |
| Innenspule | Innen | Im | 576 | -0.0323 | 68.2161 | 68.2083 | 0.2003 | 0.0000 |
| Innenspule | alle | Re | 1536 | 0.8928 | 2.8617 | 2.9958 | 0.5420 | 0.0000 |
| Innenspule | alle | Im | 1536 | 0.1009 | 222.1931 | 222.1913 | 0.5941 | 0.0000 |
| Aussenspule | Rand | Re | 960 | 0.5844 | 3.7158 | 3.7384 | 0.4924 | 0.0000 |
| Aussenspule | Rand | Im | 960 | -0.1488 | 1631.0918 | 1631.0914 | 0.8334 | 0.0000 |
| Aussenspule | Innen | Re | 576 | 0.2924 | 1.0443 | 0.8219 | 0.3926 | 0.0000 |
| Aussenspule | Innen | Im | 576 | 1.7063 | 152.8279 | 152.8358 | 0.6569 | 0.0000 |
| Aussenspule | alle | Re | 1536 | 0.5207 | 3.3218 | 3.3280 | 0.4912 | 0.0000 |
| Aussenspule | alle | Im | 1536 | 0.5485 | 1291.9547 | 1291.9547 | 0.8326 | 0.0000 |

## Antworten a bis e

Jeweils zuerst die **Messung**, dann gesondert die *Interpretation*.

**a) Sitzt der Fehler schon im Inneren der Spulen oder nur am Rand?**

**Messung** (`--scale 1.0`, Re, 2 Hz): Auch im Inneren (Tiefe ≥ 1) ist das
Feld falsch: Innenspule Projektion 0,95, aber Betragsverhältnis 1,55 und
L2-Fehler 1,22; Außenspule Projektion 0,30, Betrag 0,84, L2 1,05. Bei 50 Hz
entsprechend (Innenspule 0,95 / 1,09 / 0,54, Außenspule 0,30 / 0,73 / 0,97 als
Projektion / Betrag / L2). Das Referenzfeld stimmt dort auf etwa 10 % mit
Biot-Savart (Tabelle oben).

*Interpretation:* Der Fehler ist kein reiner Randeffekt. Die Spulen haben
keinen Leitwert; die BEM wirkt dort nur über die Randzeilen. Ein Fehler dieser
Größe im Inneren passt zu einem global wirkenden Fehler der Kopplung; bewiesen
ist das nicht. Bei der Innenspule (Projektion 0,95) kann der Fehler im Inneren
auch fast nur Betrag und Rest-Richtung betreffen.

**b) Wächst er zum Rand hin?**

**Messung:** Ja, in Betrag und L2-Fehler. Betragsverhältnis Rand / innen,
2 Hz: Platte 4,34 / 1,74, Innenspule 3,04 / 1,55, Außenspule 3,72 / 0,84; L2
Platte 4,45 / 1,98, Innenspule 2,92 / 1,22, Außenspule 3,70 / 1,05. Bei 50 Hz
gleiches Bild (Platte Betrag 2,50 / 0,89). Die Projektion ist am Rand dagegen
nicht durchweg schlechter: Innenspule 0,86 gegen 0,95, Außenspule 0,58 gegen
0,30, Platte 0,030 gegen 0,045.

*Interpretation:* Der Amplitudenfehler konzentriert sich am Rand, die
Richtungsübereinstimmung nicht. Das deutet auf einen Fehler am Randterm, der
sich ins Innere fortsetzt; mehr geben die Daten nicht her.

**c) Ändert er sich mit dem Netz (0,5 gegen 1,0)?**

**Messung** (Re, 2 Hz, alle Elemente): Projektion Platte −0,063 → +0,035,
Innenspule 0,538 → 0,867, Außenspule 0,339 → 0,523; Betragsverhältnis Platte
2,08 → 3,75, Innenspule 1,57 → 2,91, Außenspule 1,44 → 3,33. B_φ-Anteil
Platte 0,38 → 0,20, Innenspule 0,48 → 0,54, Außenspule 0,85 → 0,49.

*Interpretation:* Mit feinerem Netz verbessert sich die Richtung in den
Spulen, der Betragsfehler wird aber größer, nicht kleiner. In diesem Bereich
konvergiert der Fehler nicht gegen null. Mit zwei Netzen ist keine Ordnung
bestimmbar; `--scale 1.5` ließ sich nicht rechnen.

**d) Ist der B_φ-Anteil ein Randeffekt?**

**Messung** (BEM-FEM, 2 Hz, `--scale 1.0`; Referenz höchstens 4e-4): Platte
Rand 0,19, innen 0,29; Innenspule Rand 0,55, innen 0,17; Außenspule Rand 0,49,
innen 0,39. Bei 50 Hz: Platte 0,17 / 0,29, Innenspule 0,37 / 0,13, Außenspule
0,32 / 0,33.

*Interpretation:* Nicht überwiegend. In der Innenspule fällt B_φ zum Inneren
hin ab, in Platte und Außenspule ist es innen gleich groß oder größer. Der
B_φ-Anteil ist damit nicht nur an den Rand gebunden. Das widerspricht der
Vermutung aus `bemfem-field-check.md` (Fehler in der Formulierung, Eichung,
∇(A·n)) nicht, bestätigt sie aber auch nicht.

**e) Zeigen Realteil bei 2 Hz und Imaginärteil bei 50 Hz dasselbe Muster?**

**Messung** (`--scale 1.0`, alle Elemente): Projektion Re (2 Hz) / Im (50 Hz):
Platte 0,035 / 0,025, Innenspule 0,867 / 0,050, Außenspule 0,523 / 0,701.
Betragsverhältnis Re (2 Hz) / Im (50 Hz): 3,7 / 6,9, 2,9 / 125, 3,3 / 893. Im
Spulenkörper beträgt Im/Re des Betrags bei 50 Hz im BEM-FEM-Feld 1,46
(Innenspule) bzw. 0,52 (Außenspule), im Referenzfeld 0,017 bzw. 0,0015. Schon
bei 2 Hz hat das BEM-FEM-Feld in den Spulen Im/Re = 0,064 / 0,036 gegen 0,0008 /
0,00009 beim Referenzfeld.

*Interpretation:* Nein, kein gemeinsames Muster. Der Imaginärteil im
Spulenkörper ist im BEM-FEM-Feld um zwei bis drei Größenordnungen zu groß; er
dürfte dort nur von der kleinen Rückwirkung der Platte kommen. Das BEM-FEM-Feld
ist außerdem frequenzabhängig, wo das Referenzfeld es kaum ist (Re bei 50
gegen 2 Hz: Projektion 0,51 / 0,41 / 0,74). Das spricht dafür, dass die
Kopplung Real- und Imaginärteil vermischt oder die Plattenrückwirkung weit
über der Physik liegt. Nicht untersucht, wie viel davon schon ohne Leitwert
(κ = 0) vorliegt.

## Offene Punkte

- `--scale 1.5` für BEM-FEM nicht gerechnet (59,7 GiB gegen 48 GiB). Ohne
  Schur-Variante keine dritte Netzstufe, also keine Konvergenzordnung.
- Auf `--scale 0.5` gibt es keine Innenschicht; die Schichtfragen (a, b, d)
  beantwortet nur `--scale 1.0`.
- Ein einziger Plattenabstand (11,3 mm) und die Amplitude 20 A. Die
  Phasenlage des Ringmodell-Imaginärteils ist nur über den Vergleich mit dem
  Referenzpfad gestützt (Projektion 0,98 in der Platte).
- Das Referenzfeld ist im Spulenkörper nur auf etwa 10 % gegen die
  Fadenmodell-Referenz geprüft; Projektionen nahe 1 (Innenspule innen 0,95)
  lösen feinere Unterschiede nicht auf.
- Die Gegenproben aus `bemfem-field-check.md` (exaktes (A, K) in die
  BEM-Zeilen, Maxwell-BEM) stehen weiter aus. Für eine Änderung am BEM-FEM-Pfad
  wäre eine Rückfrage nötig; es wurde nichts am Pfad geändert.
