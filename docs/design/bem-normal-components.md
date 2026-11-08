# Gegenprobe Normalanteile in Zeile 2

Stand: 08.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), 32 CPU-Kerne, Linux. Basis: `main` 3db24f2
(Merge von #27), Zweig `bem-normal-components`. Netze `near/hex_s0.5.vtk` und `near/hex_s1.0.vtk` (936 bzw. 3696
Randquads), Platte bei 11,3 mm, **dichter Aufbau** (`--bem-eta=1e-9`), Zeilenprüfung bei 2 Hz, Schritt 3 bei 50 Hz und
0,01 Hz. Logs in `build/trace/` (nicht im Repo). Am Operator, Lösungspfad und der Eichung wurde nichts geändert: neu sind
Ausgaben in `--bem-row-check`, `include/BemNormalPart.h`, ein Host-Test und ein zusätzlicher Dump
(`<dump>.bquads.csv`, nur bei `--harmonic-field-dump`, ändert die Lösung nicht).

## Ergebnis in Kürze

**Hypothese: für zulässige Eichungen (harmonisch, abklingend) bestätigt, mit einer Einschränkung: die vollständige
Zeile 2 ist nur für Coulomb-Klasse-Felder eichinvariant.**

**Gemessen** (Skala 1,0, gesamt, Residuum Zeile 2):

| Fall | tangential (op, +K1') | vollständig (op) | tangential (ex) | vollständig (ex) |
|---|---:|---:|---:|---:|
| 0 (keine Verschiebung) | 0,77 % | 0,74 % | 0,55 % | 0,55 % |
| 3 (Dipol-psi, harmonisch, abklingend) | 36,03 % | 3,99 % | 35,38 % | 0,44 % |
| 4 (nichtharmonisch, abklingend) | – | 61,8 % | – | – |

(Fälle wie in `bem-trace-consistency.md`: 0 ohne Verschiebung; 1 psi = c·x·z (harmonisch, wächst); 2 psi = c(x²+y²)z
(nichtharmonisch, wächst); 3 Dipol (harmonisch, abklingend); 4 psi = c·z·exp(−r²/s²) (nichtharmonisch, abklingend).)

- Die vollständige Zeile 2 senkt das Residuum im Eichfall 3 von 36 % auf 4 % (op) bzw. 0,44 % (ex). Der Normalanteil
  erklärt damit rund 32 Prozentpunkte. Rest (op) ist Interpolationsanteil der Kantenbasis (3,6 % bei 1,0; ~9 % bei 0,5,
  etwa h^1,7).
- Fall 0 wird nur wenig kleiner (0,77 → 0,74 %; Normalanteil 0,13 % des Operanden), wie erwartet kleiner als 0,77 %,
  und zwar um 0,03 Prozentpunkte.
- Fall 4 (nichtharmonisches psi): vollständig bleibt bei 61,8 % (1,0) bzw. 62,6 % (0,5). Die vollständige Zeile ist also
  **nicht** für beliebige Eichung invariant, sondern nur im harmonischen Fall.
- Gegenproben bestanden: falsches Vorzeichen, nur D, nur G verfehlen am Würfel um Faktor ≥ 80–290.
- Schritt 3, ‖A·n‖/‖A_tan‖ auf den Randquads (Flächen-L2, gesamt über alle Körper/Klassen; je Körper s. u.):

| Lauf | Platte (eben) | Platte (Seite) | Innenspule (alle) | Außenspule (alle) | alle |
|---|---:|---:|---:|---:|---:|
| Lösung 0,01 Hz, mit Eichung | **0,479** | 0,071 | 0,046 | 0,041 | 0,087 |
| Lösung 0,01 Hz, `--nogauge` | 0,016 | 0,038 | divergiert (~1e7) | divergiert | – |
| Lösung 50 Hz, mit Eichung | 0,038 | 0,074 | 0,046 | 0,041 | 0,043 |
| Lösung 50 Hz, `--nogauge` | 0,004 | 0,038 | divergiert (~1e9) | divergiert | – |
| exaktes Feld (2 Hz-Geometrie) | 0,000 | 0,038 | 0,038 (Seite) | 0,038 (Seite) | 0,034 |

  Das exakte Feld hat auf den ebenen Flächen (n = ±z) kein A·n (A zirkulär in xy); die 0,038 auf Seiten/Kanten sind
  Facettierung des Netzes. In den Spulen ohne Eichung ist A nur bis auf einen Gradienten bestimmt, ihr Normalanteil ist
  dort nicht aussagekräftig (Nullraum der Kurl-Kurl-Matrix); die Platte (Nullraum durch Wirbelstromterm bei 50 Hz
  fixiert, bei 0,01 Hz fast singulär) ist es.

**Interpretation (nicht geprüft)**

- Bei 0,01 Hz mit Eichung ist der Normalanteil in der Platte auf den ebenen Flächen mit 0,48 von der Größe des
  Tangentialanteils, bei 50 Hz nur 0,038. Das ist die einzige Stelle, an der A·n stark vom exakten Wert (0) abweicht,
  und sie fällt zeitlich mit dem Quellfeldfehler bei 0,01 Hz (+28 %) zusammen. Ein quantitativer Zusammenhang wurde nicht
  nachgewiesen (keine Feldkorrektur, nur Größenordnung). Dass die Abweichung ohne Eichung verschwindet (0,016), spricht
  dafür, dass sie vom Eichterm stammt, nicht vom Operator.
- Schritt 1 zeigt, dass Zeile 2 mit nur tangentialen Daten für eine Eichverschiebung A+∇ψ die Normalanteile (A·n)n
  und ((∂A/∂n)·n)n vernachlässigt; diese werden mit der Eichung geändert.

## Herleitung der vollständigen Zeile 2

Für ein außerhalb harmonisches, abklingendes u (Normale in den Außenraum): ½u − D u + G ∂ₙu = 0 mit Kernen
D = (x−y)·nᵧ/(4π|x−y|³) und G = 1/(4π|x−y|). Gilt komponentenweise für A und ∂A/∂n (alle drei kartesischen
Komponenten). Zeile 2 testet mit den Rooftop-Funktionen W_i (tangential). Quellenanteile:
tangential (A_tan, (∂A/∂n)_tan) und normal ((A·n)n, ((∂A/∂n)·n)n). Der Identitätsterm ½∫W_i·(A·n)n verschwindet auf
ebenen Quads, weil W_i tangential ist; die Knickbeiträge liegen in D und G. Implementiert in
`include/BemNormalPart.h` (Kopfkommentar enthält die Herleitung).

## Schritt 2: Varianten (i) Kanten-DOF a' und (ii) exaktes Feld A_ex+∇ψ

Siehe Tabelle oben (op = (i), ex = (ii)). Fall 3, Skala 1,0, je Körper, vollständig (ex): Platte 1,15 %,
Innenspule 0,20 %, Außenspule 0,61 %; vollständig (op): 3,00 / 5,19 / 0,53 %. Skala 0,5 gesamt: tangential op 43,73 %,
vollständig op 12,88 %, tangential ex 40,52 %, vollständig ex 3,87 %. Die Differenz op − ex ist der Interpolationsanteil.
Dipolstelle (Fall 3): 13,63 mm (1,0) bzw. 13,49 mm (0,5) vom nächsten Randpunkt; nächste Quadkanten 6,50–7,19 mm bzw.
13,00–14,36 mm (lokale Elementgröße).

## Schritt 4: Einordnung

- **Bestätigt**: Die Eichabhängigkeit der Zeile 2 ist zu einem großen Teil (32 von 36 Prozentpunkten in Fall 3) auf die
  fehlenden Normalanteile zurückzuführen; mit vollständigen Daten ist das Residuum im harmonischen, abklingenden Fall
  eichinvariant bis auf Interpolationsfehler.
- **Einschränkung**: Fall 4 (nichtharmonisch) bleibt bei 62 %. Die Komponentenweise-Darstellungsformel setzt
  Coulomb-Klasse voraus.
- **Folge für Option (a)** aus `bem-trace-consistency.md` (Übersetzungsterme): Eine Übersetzung K1' → K1 bräuchte die
  Kopplung an A·n; A·n ist selbst eichabhängig (0,48 bei 0,01 Hz mit Eichung gegenüber 0,016 ohne). Option (a) allein
  ist damit nach diesen Messungen nicht hinreichend, solange A·n nicht zusätzlich kontrolliert wird.
- Eine Empfehlung zwischen den Optionen (b) und (c) wird nicht gegeben; die Entscheidung liegt nicht beim Agenten.

## Abweichungen vom Auftrag

1. Die Erwartung, der Operand (½T−D)a ändere sich bei vollständiger Zeile nur im Quadraturfehler, ist falsch:
   ½∇ψ − D∇ψ = −G∂ₙ∇ψ, die Kompensation steckt im verschobenen k. Invariant ist das Residuum (Änderung des Operanden
   ~92 % in Fall 3, auch vollständig).
2. Der Selbstpaar-Term für G auf Tangentialdaten braucht polare Integration (1/r-Singularität); ohne ihn ergibt der
   Würfeltest 16 % Fehler. Implementiert als `add_self_G_tangential`.
3. Fälle 1 und 2 (wachsende psi) wurden mitgerechnet, aber nicht ausgewertet (verletzen
   Abklingen; ~100–134 %).
4. Die Genauigkeitsgrenze der unabhängigen Quadratur liegt am Würfel bei ~1e-4 (innen) bis 3e-3 (Kantenquads).
5. `tools/bem_normal_ratio.py` bricht bei der Zusammenfassungszeile der `--nogauge`-Läufe mit „math domain error"
   ab (divergierende Spulenwerte); die Tabellen je Körper werden vorher ausgegeben.
6. Der Vergleichswert „exaktes Feld" wurde in einem eigenen Row-Check-Lauf (Skala 1,0, 2 Hz) nachgeholt, da die
   früheren Logs den Block noch nicht enthielten.

## Befehle

```
cd build
./fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=near/hex_s{0.5,1.0}.vtk --bem-eta=1e-9 --bem-row-check
./fembem --harmonic={0.01,50} --harmonic-inner-direct --fixed-plate --plate-z=11.3 --mesh=near/hex_s1.0.vtk \
  --bem-eta=1e-9 [--nogauge] --harmonic-field-dump=trace/s3_{g|ng}_f<f>.csv
python3 ../tools/bem_normal_ratio.py trace/s3_*.csv.bquads.csv
ctest -L host ; ctest
```

## Tests

`ctest -L host`: vorher 42/42, nachher 43/43. Volles `ctest`: vorher 75/75, nachher 76/76 (alle bestanden,
mit `LD_LIBRARY_PATH` auf die CUDA-Bibliotheken). Neu: `test_bem_normal_components` (Würfel, n = 2, 4, 8; Gegenproben).
