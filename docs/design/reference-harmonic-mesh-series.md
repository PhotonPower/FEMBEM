# Netzserie für den harmonischen Referenzlöser

Stand: 03.11.2026. Gerechnet auf einer NVIDIA GeForce RTX 3090 (sm_86,
24 GB), Windows 11. Programm aus `main` a7f6cf2, ohne Änderung am Löser.
Netze aus `tools/gmsh/team28_hex.py` (Gmsh 4.15.2), Läufe und Tabelle aus
`tools/reference_series.py`.

## Frage

Der harmonische Referenzpfad (`--reference --harmonic --fixed-plate`)
lieferte bei 11,3 mm auf team28.vtk F_z = 0,9436 N, 6,3 % unter dem
Ringmodell (1,00684 N, [gmsh-phase9-report.md](gmsh-phase9-report.md)).
FEM-direkt liegt hochgerechnet bei 1,015 bis 1,017 N, +0,8 bis +1 % über
dem Ringmodell. Woher kommt die Lücke?

1. Netzauflösung,
2. der dokumentierte Kantenfehler der BEM: stetige Neumann-Spur an
   Kanten, siehe `ReferenceSolver.cuh`,
3. der div-div-Strafterm in der Platte.

## Läufe

Alle Läufe: `fembem --reference --harmonic --fixed-plate --plate-z=H
--mesh=DATEI`, 50 Hz. Die Netze haben die Platte bei `--plate-z0 3.8`.
`--plate-z` verschiebt sie starr.

| Gruppe | Inhalt |
|---|---|
| A | uniform, `--scale` 0.5 / 0.75 / 1.0 / 1.5 / 2.0 |
| B | eine Richtung je Lauf auf `--scale 1.0`: `--n-quarter 18`, `--coil-nz 16`, `--plate-nz 6`, `--plate-nr 15` |
| C | `--no-conductor-penalty` auf `--scale 1.0` und `--scale 2.0` (feinstes Netz aus A) |
| D | team28.vtk und das grobe Netz `gmsh_team28_hex_coarse.vtk` |
| E | 3,8 mm auf `--scale 1.0` und `--scale 2.0` |

Ringmodell: 1,00684 N (11,3 mm), 3,28965 N (3,8 mm). Ring = F_z /
Ringmodell bei derselben Höhe. „Zeilen“ ist n, die Größe des reellen
FEM-BEM-Systems je Phasenanteil; der Löser arbeitet auf 2n Unbekannten.
Alle Läufe erreichen die FGMRES-Toleranz 1e-8, das echte relative Residuum
am Ende liegt zwischen 4,6e-9 und 9,4e-9. Alle Läufe finden 2
Kohomologie-Generatoren.

| Lauf | Netz | Hexaeder | Randknoten | Zeilen n | F_z (Ordnung 3) | F_z (Ordnung 2) | Ring | Q_eff | außen / innen | Zeit |
|---|---|---|---|---|---|---|---|---|---|---|
| A_s0p5 | `--scale 0.5` | 696 | 938 | 3 681 | 0,85333 N | 0,85333 N | 0,8475 | 0,8069 | 9 / 288 | 5 s |
| A_s0p75 | `--scale 0.75` | 2 034 | 2 108 | 9 540 | 0,93495 N | 0,93495 N | 0,9286 | 0,7668 | 11 / 428 | 25 s |
| A_s1 | `--scale 1.0` | 4 944 | 3 698 | 20 748 | 0,96649 N | 0,96649 N | 0,9599 | 0,7594 | 10 / 450 | 85 s |
| A_s1p5 | `--scale 1.5` | 15 984 | 8 282 | 60 927 | 0,98852 N | 0,98852 N | 0,9818 | 0,7530 | 10 / 600 | 303 s |
| A_s2 | `--scale 2.0` | 39 552 | 14 786 | 141 573 | 0,99689 N | 0,99689 N | 0,9901 | 0,7512 | 11 / 834 | 1 519 s |
| B_nq18 | `--n-quarter 18` | 7 740 | 5 762 | 32 412 | 0,97777 N | 0,97777 N | 0,9711 | 0,7576 | 9 / 467 | 161 s |
| B_cnz16 | `--coil-nz 16` | 8 016 | 5 234 | 32 268 | 0,95878 N | 0,95878 N | 0,9523 | 0,7630 | 11 / 590 | 197 s |
| B_pnz6 | `--plate-nz 6` | 6 816 | 3 842 | 26 589 | 0,96981 N | 0,96981 N | 0,9632 | 0,7607 | 10 / 526 | 119 s |
| B_pnr15 | `--plate-nr 15` | 5 664 | 4 178 | 23 628 | 0,97167 N | 0,97167 N | 0,9651 | 0,7561 | 10 / 471 | 94 s |
| C_s1_nopen | `--scale 1.0`, ohne Strafterm | 4 944 | 3 698 | 20 748 | 0,96682 N | 0,96682 N | 0,9603 | 0,7594 | 28 / 2 400 | 502 s |
| C_finest_nopen | `--scale 2.0`, ohne Strafterm | 39 552 | 14 786 | 141 573 | *läuft noch* | | | | | > 4 h |
| D_team28 | team28.vtk | 4 805 | 4 138 | 21 039 | 0,94356 N | 0,94356 N | 0,9372 | 0,7836 | 11 / 518 | 87 s |
| D_coarse | `gmsh_team28_hex_coarse.vtk` | 696 | 938 | 3 681 | 0,85333 N | 0,85333 N | 0,8475 | 0,8069 | 9 / 288 | 5 s |
| E_s1_3p8 | `--scale 1.0`, 3,8 mm | 4 944 | 3 698 | 20 748 | 3,17886 N | 3,17886 N | 0,9663 | 0,5818 | 10 / 541 | 105 s |
| E_s2_3p8 | `--scale 2.0`, 3,8 mm | 39 552 | 14 786 | 141 573 | 3,26241 N | 3,26241 N | 0,9917 | 0,5824 | 11 / 837 | 1 521 s |

Die Kraft mit Ordnung 2 und Ordnung 3 stimmt in jedem Lauf auf alle
ausgegebenen Stellen überein. Randknoten und Speicherbedarf hat das Skript
vor jedem Lauf aus dem Netz bestimmt. Das größte Netz (`--scale 2.0`)
braucht etwa 5,2 GiB dichte Algebra und liegt weit unter der Grenze von
18 GiB. Kein Lauf wurde aus Speichergründen ausgelassen.

### Netze

Alle Netze bestehen meshcheck (Zelltypen, doppelte und verwaiste Knoten,
getrennte Körper, Konformität, Jacobi-Determinante, Volumen) und check_map.
Spalte „Teilungen“: Zellen je Viertelkreis, dann Spule radial × z, dann
Platte radial × z, jeweils nach der Rundung im Generator.

| Netz | Teilungen | kleinste skalierte Jacobi-Det. (Außenspule / Innenspule / Platte) |
|---|---|---|
| `--scale 0.5` | nq 6, Spule 2×4, Platte 5×2 | 0,955 / 0,794 / 0,678 |
| `--scale 0.75` | nq 9, Spule 3×6, Platte 8×**2** | 0,970 / 0,853 / 0,761 |
| `--scale 1.0` | nq 12, Spule 4×8, Platte 10×3 | 0,977 / 0,885 / 0,802 |
| `--scale 1.5` | nq 18, Spule 6×12, Platte 15×**4** | 0,985 / 0,920 / 0,857 |
| `--scale 2.0` | nq 24, Spule 8×16, Platte 20×6 | 0,988 / 0,939 / 0,888 |
| `--n-quarter 18` | nq 18, Spule 4×8, Platte 10×3 | 0,977 / 0,885 / 0,823 |
| `--coil-nz 16` | nq 12, Spule 4×16, Platte 10×3 | 0,977 / 0,885 / 0,802 |
| `--plate-nz 6` | nq 12, Spule 4×8, Platte 10×6 | 0,977 / 0,885 / 0,802 |
| `--plate-nr 15` | nq 12, Spule 4×8, Platte 15×3 | 0,977 / 0,885 / 0,834 |

**Die Serie A ist nicht streng uniform.** Bei `--scale 0.75` und `1.5`
rundet der Generator die Plattenschichten. Uniform wären 2,25 und 4,5
Schichten, es werden 2 und 4 (Python rundet 4,5 auf die gerade Zahl).
Diese beiden Stufen haben die Platte in z also etwas gröber als die übrigen.
Wie stark das wiegt, zeigt B: Die Plattenschichten verdoppeln (3 → 6) bringt
+0,34 %.

**Das grobe Netz gehört zur Serie.** `gmsh_team28_hex_coarse.vtk` ist
byteweise gleich der Ausgabe von `team28_hex.py --scale 0.5`, und D_coarse
liefert denselben Wert wie A_s0p5. Die Annahme im Auftrag, es stamme von
einem anderen Generator, trifft nicht zu.

## Auswertung

### A: Konvergenz uniform

Die Folge steigt monoton: 0,853 / 0,935 / 0,966 / 0,989 / 0,997 N. Als
Netzweite gilt h = 1/scale. Die beobachtete Ordnung p und der Grenzwert
F_∞ folgen aus F(h) = F_∞ + c·h^p durch drei Punkte; das Skript löst nach p
auf.

| Tripel (scale) | Verhältnis der Differenzen | p | F_∞ | gegen Ringmodell |
|---|---|---|---|---|
| **1.0 / 1.5 / 2.0** (die drei feinsten) | 2,632 | **1,77** | **1,00948 N** | **+0,26 %** |
| 0.75 / 1.0 / 1.5 | 1,432 | 2,07 | 1,00531 N | −0,15 % |
| 0.5 / 0.75 / 1.0 | 2,588 | 1,72 | 1,01560 N | +0,87 % |

Die drei Tripel liefern eine Ordnung zwischen 1,7 und 2,1 und einen
Grenzwert zwischen 1,005 und 1,016 N. Das maßgebliche Tripel sind die drei
feinsten Netze: F_∞ = 1,0095 N. Das liegt +0,26 % über dem Ringmodell und
0,55 bis 0,74 % unter FEM-direkt (1,015 bis 1,017 N).

Der extrapolierte Wert ist eine Rechnung, keine Messung. Er hängt an der
Ordnung. Wird die Ordnung vorgegeben statt beobachtet, ergibt Richardson
aus 1.5 / 2.0:

| angenommene Ordnung | F_∞ |
|---|---|
| p = 1 | 1,0220 N |
| p = 1,77 (beobachtet) | 1,0095 N |
| p = 2 | 1,0077 N |

Die Stufen 0.75 und 1.5 sind in den Plattenschichten nicht uniform (siehe
oben). Gemessen ist das nicht, nach B aber klein: Eine halbe zusätzliche
Plattenschicht bei 1.5 dürfte weniger als 0,1 % ausmachen (Schätzung).

### B: Beitrag je Richtung

Bezug: A_s1 = 0,96649 N.

| Richtung | Schritt | F_z | Änderung |
|---|---|---|---|
| Umfang `--n-quarter` | 12 → 18 (×1,5) | 0,97777 N | **+1,17 %** |
| Spule axial `--coil-nz` | 8 → 16 (×2) | 0,95878 N | −0,80 % |
| Platte radial `--plate-nr` | 10 → 15 (×1,5) | 0,97167 N | +0,54 % |
| Platte axial `--plate-nz` | 3 → 6 (×2) | 0,96981 N | +0,34 % |

**Die Umfangsrichtung dominiert**, obwohl sie nur um den Faktor 1,5
verfeinert wurde. Ein Grund ist wohl der Rand: Mit 4·nq Segmenten ist er
ein Vieleck, und mehr Segmente verkleinern mit dem Umfang auch die Fläche
der BEM-Kanten je Segment. Das ist eine Interpretation, nicht gemessen.

Die axiale Spulenteilung senkt F_z wie bei FEM-direkt (Phase 7: −2,6 bis
−4,5 %), hier aber nur um 0,8 %.

Die Summe der vier Einzelschritte ist +1,25 %. Sie lässt sich nicht direkt
mit A vergleichen, weil die Schritte verschieden groß sind und `--coil-nr`
nicht einzeln verfeinert wurde. Der uniforme Schritt 1.0 → 1.5 bringt
+2,28 %, 1.0 → 2.0 bringt +3,15 %.

### C: Strafterm

`--no-conductor-penalty` nimmt den div-div-Strafterm aus den leitenden
Elementen.

| Netz | mit Strafterm | ohne | Änderung |
|---|---|---|---|
| `--scale 1.0` | 0,96649 N | 0,96682 N | **+0,03 %** |
| `--scale 2.0` | 0,99689 N | *läuft noch* | – |

Auf `--scale 1.0` ändert der Strafterm die Kraft kaum. Der Lauf auf
`--scale 2.0` steht noch aus; ein Gegencheck auf dem feinsten Netz fehlt
daher. Ohne ihn konvergiert FGMRES schlechter:
28 statt 10 äußere und 2 400 statt 450 innere Iterationen bei `--scale 1.0`,
das Sechsfache an Zeit.

### D: Einordnung der Standardnetze

- **team28.vtk:** 0,94356 N, wie im vorigen Bericht (0,9436 N). Das
  Standardnetz hat etwa so viele Zellen wie `--scale 1.0` (4 805 gegen
  4 944 Hexaeder, 4 138 gegen 3 698 Randknoten), liegt aber 2,4 % darunter.
  Q_eff ist dort 0,784 gegen 0,759. Die −6,3 % des Standardnetzes sind also
  zu einem Teil eine Eigenschaft dieses Netzes, nicht der Zellzahl. Woran es
  liegt, ist nicht untersucht.
- **Kontrolle `--scale 1.0` gegen 0,9436 N:** Die beiden Werte stimmen
  nicht überein (0,96649 gegen 0,94356 N), weil es zwei verschiedene Netze
  sind. Bitgleich reproduziert ist dagegen A_s1: Gruppe E hat den Lauf ein
  zweites Mal gerechnet, F_z = 9.66492e-01 N in beiden Logs.
- **Grobes Netz:** gleich A_s0p5, siehe oben.

### E: 3,8 mm

| Netz | F_z | gegen Ringmodell (3,28965 N) |
|---|---|---|
| `--scale 1.0` | 3,17886 N | −3,37 % |
| `--scale 2.0` | 3,26241 N | −0,83 % |
| Richardson mit p = 1,77 aus A (**Schätzung**) | 3,2971 N | +0,23 % |
| zum Vergleich mit p = 2 (**Schätzung**) | 3,2903 N | +0,02 % |

Mit zwei Netzen ist keine eigene Ordnung bestimmbar. Mit der Ordnung aus A
liegt der Grenzwert bei 3,8 mm ebenso knapp über dem Ringmodell wie bei
11,3 mm.

## Entscheidungshilfe (Interpretation)

**A läuft gegen etwa 1,005 bis 1,016 N (maßgeblich 1,0095 N) und erreicht
damit den Bereich 1,00 bis 1,02 N. Die Netzauflösung erklärt die Lücke.**
Bei 11,3 mm flacht A nicht mehrere Prozent unter dem Ringmodell ab.

Zur Trennung der drei Ursachen:

- **Auflösung:** Sie erklärt die −6,3 % auf team28.vtk vollständig. Davon
  sind rund 2,4 Prozentpunkte die Wahl des Netzes; bei gleicher Zellzahl ist
  das Gmsh-Netz besser. Der Rest konvergiert mit Ordnung 1,7 bis 2,1 heraus.
- **Strafterm:** vernachlässigbar auf `--scale 1.0` (C, +0,03 %). Auf dem
  feinsten Netz steht der Lauf noch aus.
- **Kantenfehler der BEM:** Er ist kein eigener Lauf. Er bleibt aber nicht
  als Sockel stehen: Er nimmt mit dem Netz ab und steckt in der beobachteten
  Ordnung mit drin. Dazu passt, dass die Umfangsrichtung dominiert (B). Ein
  Fehler, der beim Verfeinern nicht verschwindet, würde A unterhalb des
  Ringmodells abflachen lassen; das ist nicht zu sehen.

Der extrapolierte Referenzwert liegt bei 11,3 mm und bei 3,8 mm knapp über
dem Ringmodell (+0,26 % bzw. geschätzt +0,23 %). FEM-direkt liegt etwa
+0,8 bis +1 % darüber. Ein Abstand von rund 0,6 % zwischen Referenzpfad und
FEM-direkt bleibt also; er liegt in der Größenordnung der Unsicherheit der
Extrapolation (Tabelle oben: 1,005 bis 1,016 N).

## Laufzeiten

Gesamtzeit nach Lauf: siehe Tabelle. Den Aufwand macht FGMRES, mit der
ILU-Substitution auf dem Host:

- `--scale 2.0`: 1 452 s von 1 519 s, LU von V 60 s.
- ohne Strafterm, `--scale 1.0`: 502 s statt 85 s.
- C auf `--scale 2.0` ohne Strafterm: nach 4 Stunden (Start 01:42) noch
  ohne Ergebnis. Wegen der gepufferten Ausgabe ist kein Zwischenstand
  sichtbar.

Alle übrigen Läufe zusammen: rund 1,3 Stunden Rechenzeit (Summe der
Tabelle, mit E).

## Offene Punkte

- C_finest_nopen (`--scale 2.0` ohne Strafterm) läuft noch. Sobald er
  fertig ist, schreibt `python tools/reference_series.py --groups A,B,C,D,E
  --reuse` die Tabelle in `build/refseries/summary.md`.
- Ein dritter Punkt bei 3,8 mm (`--scale 1.5`) würde die Ordnung dort
  messen statt aus A zu übernehmen (etwa 5 min).
- Die Stufen 0.75 und 1.5 sind in den Plattenschichten nicht uniform. Für
  eine strenge Serie bräuchte `team28_hex.py` eine andere Rundung oder die
  Basis `--plate-nz 4` (dann 2 / 3 / 4 / 6 / 8 Schichten).
- Warum team28.vtk bei gleicher Zellzahl 2,4 % unter dem Gmsh-Netz liegt,
  ist nicht untersucht (Zellverteilung, Kern der Platte, Kantenwinkel).
- Der Kantenfehler der BEM ist nicht isoliert. Dafür bräuchte es eine
  unstetige Neumann-Spur im Löser (Doppelknoten an Kanten), also eine
  Änderung am Löser. Sie ist hier ausgeschlossen.
- `--scale 3` (rund 33 000 Randknoten, etwa 26 GiB dichte Algebra) liegt
  über der Grenze von 18 GiB. Eine weitere Stufe von A geht mit der dichten
  BEM nicht.
- fembem endet auch bei nicht konvergiertem FGMRES mit Exit-Code 0
  (`main.cpp` gibt den Rückgabewert von `run_reference_harmonic` nicht
  weiter). Das Skript erkennt den Fall am Log; in dieser Serie kam er nicht
  vor.
- Ein Lauf länger als 2 Stunden wird im Werkzeug, das die Läufe gestartet
  hat, nach 2 Stunden beendet. Der erste Versuch von C_finest_nopen ging
  dabei ohne Ausgabe verloren, weil fembem seine Ausgabe puffert. Er wurde
  losgelöst neu gerechnet.

## Befehle

    # Netze erzeugen und prüfen, Randknoten und Speicher, ohne fembem
    python tools/reference_series.py --groups A,B,C,D,E --dry-run
    # Läufe (Netze und Logs in build/refseries, nicht eingecheckt)
    python tools/reference_series.py --groups A,B,C,D
    python tools/reference_series.py --groups E
    # Tabelle aus vorhandenen Logs
    python tools/reference_series.py --groups A,B,C,D,E --reuse

Einzelbefehle, aus `build/`:

    python ../tools/gmsh/team28_hex.py --scale 2.0 -o refseries/hex_s2.vtk
    Release/fembem --reference --harmonic --fixed-plate --plate-z=11.3 --mesh=refseries/hex_s2.vtk
    Release/fembem --reference --harmonic --fixed-plate --plate-z=11.3 --mesh=refseries/hex_s2.vtk --no-conductor-penalty
    Release/fembem --reference --harmonic --fixed-plate --plate-z=3.8 --mesh=refseries/hex_s2.vtk

Generator-Argumente der übrigen Netze: `--scale 0.5 | 0.75 | 1.0 | 1.5`,
`--n-quarter 18`, `--coil-nz 16`, `--plate-nz 6`, `--plate-nr 15`.
