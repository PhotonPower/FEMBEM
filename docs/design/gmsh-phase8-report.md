# Gmsh-Vernetzung: Bericht zu Phase 8 (Absicherung und unabhängige Kraftreferenz)

Stand: 31.10.2026. Gerechnet auf einer NVIDIA GeForce RTX 3090 (sm_86, 24 GB),
Windows 11, CUDA 13.4, cuDSS 0.8.0. Programm aus `main` f4fc5a6. Netze aus
`tools/gmsh/team28_fem_air.py`, Lauf `fembem --fem-direct --mesh=…`, wie in
[gmsh-phase7-report.md](gmsh-phase7-report.md).

## Fragen

1. Hält die Hochrechnung aus Phase 7? Offen waren die Addition der
   Einzeleffekte, die Ordnung im Spalt und die Ordnung der Netzweite in der
   Ebene.
2. Woher kommt die Lücke von gut 3 % zum Sollwert 1,0497 N (Gewicht der
   Platte bei der gemessenen Schwebehöhe 11,3 mm)?

## Messreihe

Basis wie in Phase 7: `--plate-z0 11.3 --scale 0.75 --coil-nz 6`. `--coil-nz 6`
steht ausdrücklich da, weil die Vorgabe seit Phase 7 bei 12 liegt. Die
Generator-Argumente wirken vor `--scale 0.75`, die Schichtzahlen in Klammern
sind die tatsächlichen. Alle Läufe: relatives Residuum ≤ 7e-14.

| Lauf | Änderung | nnz(LU) | F_z | gegen Basis |
|---|---|---|---|---|
| Basis | – | 1,48e9 | 1,04469 N | – |
| kombiniert | `--coil-nz 22 --plate-nz 8 --gap-nz 6` (16 / 6 / 4) | 2,96e9 | 1,00706 N | −3,60 % |
| Spalt | `--gap-nz 11` (8) | 1,96e9 | 1,05180 N | +0,68 % |
| Netzweite | `--h 2.5` (3,33 mm) | 4,44e9 | 1,05708 N | +1,19 % |

Die Basis trifft wieder 1,04469 N; das Programm aus `main` rechnet also wie
das aus Phase 7. Der Lauf mit 4,44e9 nnz(LU) passte noch in die 24 GB.

### Die Einzeleffekte addieren sich

Aus den Einzelläufen in Phase 7 (Spule 4 → 16: −0,046477 N, Platte 3 → 6:
+0,00344 N, Spalt 2 → 4: +0,00574 N) folgen für das kombinierte Netz
1,00739 N. Gemessen: 1,00706 N, **0,03 % daneben**. Die Annahme der
Hochrechnung ist damit an einem Punkt geprüft, an dem alle drei großen
Schichtbeiträge zugleich wirken.

### Spalt: Ordnung 2 bestätigt

2 → 4 → 8 Schichten: Schritte +0,00574 und +0,00137 N, Verhältnis 4,19,
Ordnung 2,07. Grenzwert 1,05223 N, **+0,72 %** gegen die Basis. Phase 7 hatte
mit angenommener Ordnung 2 +0,73 % eingesetzt.

### Netzweite in der Ebene: +1,30 … +1,45 %

h = 6,67 / 5 / 4 / 3,33 mm. Je drei Punkte ergeben:

| Punkte | Ordnung | Grenzwert gegen Basis |
|---|---|---|
| 6,67 / 5 / 4 (Phase 7) | 4,2 | +1,22 % |
| 6,67 / 4 / 3,33 | 3,5 | +1,30 % |
| 5 / 4 / 3,33 | 2,8 | +1,34 % |
| 5 / 4 / 3,33, Ordnung 2 angenommen | 2 | +1,45 % |

Die Ordnung sinkt, je feiner die Punkte; das gröbste Netz liegt noch nicht im
asymptotischen Bereich. Der Grenzwert ist enger eingegrenzt als in Phase 7
(dort +1,2 … +1,5 %).

### Hochrechnung

Wie in Phase 7, mit den neuen Werten für Spalt und Netzweite:

    1,04469 N · (1 − 4,75 % + 0,44 % + 0,72 % + 1,30…1,45 % + 0,2 %)
      ≈ 1,0229 … 1,0245 N    (radial 5 / axial 10)

Mit −0,7 % für das ausgeschöpfte Luftgebiet radial (Phase 6):

**F_z ≈ 1,016 … 1,017 N** bei voller Auflösung und ausgeschöpftem
Luftgebiet. Phase 7 hatte 1,015 … 1,018 N.

## Unabhängige Referenz: Ringmodell

TEAM 28 ist rotationssymmetrisch, alle Ströme fließen in Umfangsrichtung.
`tools/team28_axisymmetric.py` zerlegt die Platte in koaxiale Ringe und die
Spulen in Stromfäden. Die Gegeninduktivität koaxialer Kreisringe ist exakt
bekannt (elliptische Integrale), die Ringströme folgen aus einem kleinen
dichten System, die Kraft aus ½·Re(I_p·I_s*·∂M/∂z). Das Modell teilt mit dem
Löser nur die Eingabedaten (`src/fem.cu`); es hat kein abgeschnittenes
Luftgebiet und keine Elemente.

| Gitter (Platte radial × axial) | F_z bei 11,3 mm |
|---|---|
| 65 × 3 | 1,00349 N |
| 130 × 6 | 1,00602 N |
| 260 × 12 | 1,00611 N |

Die Näherung für die Selbstinduktivität eines Rings ist der einzige nicht
exakte Baustein. Um ±30 % verstellt, ändert sie F_z auf dem feinen Gitter um
±1e-4 N. Geprüft in `tools/tests/test_team28_axisymmetric.py`: elliptische
Integrale gegen Tabellenwerte, Gegeninduktivität gegen das Neumann-Integral
(9e-15), Zusammenbau und Zeitmittelung gegen die geschlossene Lösung für
einen Ring (mit Gegenprobe: Faktor 1 statt ½ wird erkannt), F ∝ σ² bei
kleiner Leitfähigkeit.

**Mit den Daten von TEAM 28 ist F_z bei 11,3 mm gleich 1,006 N, nicht
1,0497 N.** Das Gewicht trägt die Platte bei **11,05 mm**
(`team28_axisymmetric.py --equilibrium`).

### Die Lücke zum Sollwert liegt in den Daten, nicht im Löser

Die Lücke von 4,1 % zwischen 1,006 N und dem Gewicht hat mit der
Diskretisierung nichts zu tun. Sie steckt zwischen den Eingabedaten der
Spezifikation und der gemessenen Schwebehöhe. Kleine, für eine Messung
übliche Abweichungen erklären sie jeweils allein (aus dem Ringmodell):

| Größe | schließt die Lücke bei |
|---|---|
| Höhe | 11,05 statt 11,3 mm (dF/dz ≈ −0,17 N/mm) |
| Leitfähigkeit | 3,49e7 statt 3,40e7 S/m (+2,5 %; F ∝ σ^1,7) |
| Strom | 20,43 statt 20 A Scheitelwert (+2,1 %) |

Dazu ist die Masse nur auf drei Stellen angegeben (±0,5 %). Die Daten im
Repo stimmen mit der Spezifikation überein, soweit prüfbar: σ = 34 S/mm,
960 / 576 Windungen bei 20 A Scheitelwert nennt auch die LS-DYNA-Eingabe zu
TEAM 28 (Amperewindungen 19 200 und −11 520). Die Verifikationstabelle in
`BUILD.md` nennt für den Hauptpfad ebenfalls eine tiefere Ruhelage (10,90 mm).

Eine Prüfung des Codes ohne GPU (Daten, Stromdichte J = w·I/A über den
r-z-Querschnitt, Zeitmittelung, Lorentz-Kraft nur auf die Platte, Piola-
Transformation, A*-Formulierung ohne Skalarpotential) fand keinen Fehler.

### FEM-direkt liegt 1 % über der Referenz

Die Hochrechnung 1,016 … 1,017 N liegt **1,0 … 1,1 % über** dem Ringmodell.
Offen ist, wo dieses Prozent herkommt. Die Netzrichtungen sind jetzt alle
gemessen und ihre Addition geprüft; unsicher sind die beiden Beiträge des
Luftgebiets:

- **Luft fein** (+0,2 %): nur ein Schritt, Ordnung angenommen.
- **Abschneiden radial** (−0,7 %): in Phase 6 bei 4 Spulenschichten
  gemessen. Ob sich dieser Beitrag mit den Schichtbeiträgen addiert, ist
  nicht geprüft; A × n = 0 am Rand erhöht die Kraft, ein zu kleiner Abzug
  ließe F_z zu hoch erscheinen.

## Folgerungen

- **Die Lücke zum Sollwert ist kein Fehler des Lösers.** Wer F_z mit TEAM 28
  vergleicht, vergleicht mit dem Ringmodell (1,006 N bei 11,3 mm) und nicht
  mit dem Gewicht der Platte.
- **`tools/team28_axisymmetric.py` ist die Kraftreferenz**, die der
  Maxwell-Flächenpfad nicht liefert (er verfehlt einen bekannten Nullwert
  um 0,2 N). Eine Volumenformulierung des Maxwell-Tensors ist damit für den
  Abgleich nicht mehr nötig.
- Die Hochrechnung des FEM-direkt-Pfads ist bis auf das Luftgebiet gemessen.

**Nachtrag 31.10.2026:** Die Messungen unten sind gemacht, siehe
[gmsh-phase9-report.md](gmsh-phase9-report.md). Das Ringmodell rechnet jetzt
mit feineren Spulenfäden: 1,00684 N bei 11,3 mm statt 1,00611 N (+0,07 %).
Der Rand trägt −0,77 % und erklärt den Abstand nicht; FEM-direkt liegt
+0,8 … +1,0 % (11,3 mm) und +0,5 … +0,6 % (3,8 mm) über dem Ringmodell. Der
BEM-FEM-Lauf konvergierte nicht.

## Nächste Messungen

1. Abschneiden radial 5 → 10 an einem Netz mit 12 Spulenschichten
   (`--scale 0.75 --coil-nz 16`), um den Abzug von −0,7 % bei der heutigen
   Vorgabe zu prüfen.
2. Ringmodell und FEM-direkt bei einer zweiten Höhe (Startlage 3,8 mm)
   vergleichen: Ist der Abstand von 1 % fest oder hängt er von der Höhe ab?
3. BEM-FEM-Pfad (Hexaeder, `--harmonic --fixed-plate`) bei 11,3 mm gegen das
   Ringmodell.
