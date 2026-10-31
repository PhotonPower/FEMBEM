# Gmsh-Vernetzung: Bericht zu Phase 7 (Auflösung je Richtung, FEM-direkt)

Stand: 31.10.2026. Gerechnet auf einer NVIDIA GeForce RTX 3090 (sm_86, 24 GB),
Windows 11, CUDA 13.4, cuDSS 0.8.0, Stand `main` 14f69c3. Netze aus
`tools/gmsh/team28_fem_air.py`, Lauf `fembem --fem-direct --mesh=…`
(ε = 1e-8, 50 Hz, Quelle auf der Amplitude), wie in
[gmsh-phase6-report.md](gmsh-phase6-report.md).

Sollwert im Schwebegleichgewicht (Unterkante 11,3 mm): 0,107 kg · 9,81 =
**1,0497 N**.

## Frage

Phase 6 endete mit einer offenen Frage: `--scale` senkt F_z um rund 1,1 % je
Stufe, aber welcher Teil der Auflösung trägt das? `--scale` verfeinert alles
zugleich (Netzweite in der Ebene, Schichten in Spule, Spalt und Platte, das
Luftgebiet), und `--scale 1.25` passt nicht mehr in den Speicher.

Hier wird **jeweils genau eine** Auflösung verändert, bei sonst gleichem Netz.

## Basis

`team28_fem_air.py --plate-z0 11.3 --scale 0.75` mit den Vorgaben radial 5 /
axial 10. Nach `--scale 0.75` (Schichtzahlen gerundet): h = 6,67 mm in der
Ebene, 4 Schichten über die Spulenhöhe (52 mm), 2 im Spalt, 3 durch die
Platte (3 mm).

F_z = **1,04469 N**, nnz(LU) = 1,48e9, 217 s. Das ist **bitgleich der Wert
des Servers** aus Phase 6 (radial 5 / axial 10, `--scale 0.75`), auf anderer
Hardware, anderem Betriebssystem und anderer CUDA-Version.

Alle Läufe: relatives Residuum ≤ 5e-13, keine Pivotierungen. Die
Generator-Argumente unten wirken **vor** `--scale 0.75`, die Schichtzahlen
in Klammern sind die tatsächlichen.

## Ergebnis

| Lauf | Änderung | nnz(LU) | F_z | gegen Basis |
|---|---|---|---|---|
| Basis | – | 1,48e9 | 1,04469 N | – |
| `--coil-nz 8` | Spule 4 → 6 Schichten | 1,65e9 | 1,01714 N | −2,64 % |
| `--coil-nz 11` | Spule → 8 | 1,84e9 | 1,00750 N | −3,56 % |
| `--coil-nz 16` | Spule → 12 | 2,14e9 | 1,00062 N | −4,22 % |
| `--coil-nz 22` | Spule → 16 | 2,50e9 | 0,998213 N | −4,45 % |
| `--plate-nz 8` | Platte 3 → 6 Schichten | 1,75e9 | 1,04813 N | +0,33 % |
| `--plate-nz 16` | Platte → 12 | 2,27e9 | 1,04899 N | +0,41 % |
| `--gap-nz 6` | Spalt 2 → 4 Schichten | 1,63e9 | 1,05043 N | +0,55 % |
| `--h 3.75` | h 6,67 → 5 mm | 2,42e9 | 1,05371 N | +0,86 % |
| `--h 3.0` | h → 4 mm | 3,24e9 | 1,05602 N | +1,08 % |
| `--h-air 26.25 --air-first 2.25` | Luft wie bei `--scale 1.0` | 2,26e9 | 1,04580 N | +0,11 % |

Zum letzten Lauf: Er macht die Luftschichten feiner und **vermehrt** sie
dabei, das Netz wächst um 38 %. F_z steigt um 0,11 %.

### Die Spulenschichten bestimmen den Fehler

**Die axiale Auflösung der Spulen ist der mit Abstand größte Beitrag, und sie
senkt F_z.** Die vier Punkte liegen auf

    F_z(n) = F∞ + C / n²,   F∞ = 0,99511 N,   C = 0,793 N

mit einer größten Abweichung von 4,5e-6 N; die beste freie Ordnung ist
p = 2,000. Der Punkt n = 16 war **vorhergesagt** (0,99821 N), bevor er
gerechnet wurde; gemessen 0,998213 N.

Fehler gegen den Grenzwert: 4 Schichten +5,0 %, 6 Schichten (Vorgabe des
Generators bei `--scale 1`) **+2,2 %**, 12 Schichten +0,55 %, 16 Schichten
+0,31 %.

Warum gerade die Spule: Die Prismen sind senkrecht extrudiert, jede Schicht
reicht über das ganze Gebiet. Über die Spulenhöhe liegen bei 4 Schichten
13 mm hohe Elemente, über der Platte (Spalt, Platte) sind sie 1–6 mm hoch.
Quellstromdichte und Projektion sind in allen Läufen gleich
(J_innen = 1,32e7, J_aussen = −1,48e7 A/m², ‖Gᵀb‖/‖b‖ nach der Projektion
1,5e-16), die Spulenvolumina exakt; der Fehler sitzt also in der
Darstellung des Feldes über die Spulenhöhe, nicht in der Quelle.

### Platte, Spalt und Netzweite erhöhen F_z

- **Platte:** 3 → 6 → 12 Schichten: Schritte +0,00344 und +0,00086 N,
  Verhältnis genau 4, also Ordnung 2. Grenzwert 1,04928 N (+0,44 %).
- **Spalt:** nur ein Schritt (2 → 4). Mit Ordnung 2 angenommen, wie bei der
  Platte: +0,73 % bis zum Grenzwert.
- **Netzweite in der Ebene:** h = 6,67 / 5 / 4 mm. Die drei Punkte ergeben
  formal Ordnung 4,2 und einen Grenzwert von +1,22 %; mit Ordnung 2
  angenommen +1,48 %. Die Ordnung ist auf drei unstrukturierten Netzen nicht
  belastbar; die Spanne 1,2–1,5 % ist die Unsicherheit.
- **Luft:** Ein Schritt feiner ändert F_z um +0,11 %. Mit Ordnung 2
  angenommen wie bei der Platte, kämen bis zum Grenzwert noch einmal etwa
  0,1 % hinzu. Aus einem Schritt ist die Ordnung nicht bestimmbar; der
  Beitrag liegt in jedem Fall unter dem der übrigen Richtungen.

### Gegenprobe: setzen sich die Einzeleffekte zusammen?

Ein Schritt `--scale 0.75 → 1.0` ändert zugleich: Spule 4 → 6 (−2,64 %),
h 6,67 → 5 (+0,86 %), Spalt 2 → 3 (+0,41 % aus der Ordnung-2-Anpassung),
Platte 3 → 4 (+0,19 %), Luft (+0,11 %). Summe **−1,07 %**. Phase 6 hat für
genau diesen Schritt −1,1 % gemessen (radial 3 / axial 10 und radial 7 /
axial 10; ein Lauf bei radial 5 fehlt). Die Effekte addieren sich also
näherungsweise. Damit ist auch erklärt, warum F_z bei `--scale` fällt,
obwohl die meisten Verfeinerungen die Kraft erhöhen: der Spulenanteil
überwiegt.

## Hochrechnung

Unter der Annahme, dass sich die Beiträge addieren (siehe Gegenprobe), voll
aufgelöst bei radial 5 / axial 10:

    1,04469 N · (1 − 4,75 % + 0,44 % + 0,73 % + 1,2…1,5 % + 0,2 %)
      ≈ 1,022 … 1,025 N

Dazu das Luftgebiet: radial 5 liegt nach Phase 6 um 0,7 % über dem
ausgeschöpften Wert (radial 10), axial 10 ist ausgeschöpft. Ergebnis:

**F_z ≈ 1,015 … 1,018 N bei voller Auflösung und ausgeschöpftem
Luftgebiet, 3,1 … 3,3 % unter dem Sollwert 1,0497 N.**

Die Spanne kommt aus der Netzweite in der Ebene. Einzeln gemessen sind die
Spulen- und die Plattenreihe; Spalt und Luft beruhen auf je einem Schritt mit
angenommener Ordnung, die Addition auf einer Gegenprobe mit einem Punkt. Der
Sollwert selbst ist auf etwa 0,5 % genau (Masse auf drei Stellen).

Phase 6 hatte aus zwei `--scale`-Punkten 0,99 bis 1,01 N hochgerechnet. Der
Abfall aus der Spule ist aber nur ein Teil davon, und andere Verfeinerungen
wirken in die Gegenrichtung. Die richtungsweise Messung liegt deshalb höher.

## Folgerungen

- **Die Vorgabe `--coil-nz 6` ist zu grob.** Sie überschätzt F_z um 2,2 %,
  mehr als jeder andere Auflösungsfehler. Mit 12 Schichten sind es 0,55 %, mit
  16 Schichten 0,31 %. Der Preis ist mäßig: bei `--scale 0.75` kosten 4 → 12
  Schichten +45 % nnz(LU), weniger als ein Schritt `--scale` (1,48e9 →
  3,9e9 geschätzt aus dem Verhältnis bei radial 7). **Vorschlag: Vorgabe auf 12 anheben** (nicht umgesetzt,
  Entscheidung offen).
- **Netzserien über `--scale` verdecken die Physik.** Gegenläufige Beiträge
  heben sich teilweise auf, und eine Ordnung aus zwei `--scale`-Punkten ist
  nicht deutbar. Für Konvergenzaussagen die Richtungen einzeln verfeinern.
- **Die Restabweichung von gut 3 % zum Sollwert bleibt offen.** Sie liegt
  über der Unsicherheit der Hochrechnung und des Sollwerts.

## Nächste Messungen

1. Gegenprobe der Addition an einem kombinierten Netz: `--coil-nz 16` mit
   `--plate-nz 8 --gap-nz 6` (bei `--scale 0.75` etwa 2,9e9 nnz(LU)).
2. Netzweite in der Ebene mit einem vierten Punkt auf dem Server (h = 3,3 mm),
   um die Ordnung festzulegen.
3. Spalt mit einem zweiten Schritt (→ 8 Schichten).

## Nebenbefund: Maxwell-Gegenprobe

Auf Branch `fix/piola-helpers` ist ein Fehler in `harmonic_maxwell_body`
behoben: Vierecksflächen zählten nur zu einem Viertel. Auf dem Basisnetz
ändert das die Maxwell-Kraft von 0,168 N auf 0,2 N; F_z (Lorentz) bleibt bei
1,04469 N. Die Oberflächenkraft trägt hier weiterhin nicht. Das Programm
weist das selbst aus: Die Nullprobe aus A_r verfehlt 0 N um 0,213 N, weil die
Kraft als Differenz von Beiträgen um 200 N entsteht. Die Viertelung war ein
echter Fehler, aber nicht die Ursache dieser Abweichung.
