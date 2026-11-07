# Spurkonsistenz der FEM-BEM-Kopplung (K1 gegen K2, Eichung)

Stand: 07.11.2026. Gerechnet auf einer NVIDIA RTX 6000 Ada (sm_89, 48 GB), Linux. Basis: `main` a5484a8
(Merge von #25 und #26), Zweig `bem-trace-consistency`. Netze `near/hex_s0.5.vtk` und `near/hex_s1.0.vtk` (936
bzw. 3696 Randquads, 3744 bzw. 14784 BEM-Freiheitsgrade, 3086 bzw. 18651 FEM-Kanten), Platte bei 11,3 mm, 2 Hz, **dichter Aufbau** (`--bem-eta=1e-9`). Logs in `build/trace/` (nicht im Repo). Am
Lösungspfad wurde nichts geändert: neu sind nur Ausgaben in `--bem-row-check`, Hilfsfunktionen in
`include/BemRowCheck.h` und ein Host-Test.

## Ergebnis in Kürze

**Hypothese: teilweise bestätigt, aber die vorgeschlagene Korrektur über Übersetzungsterme reicht nach
diesen Messungen nicht aus.**

**Gemessen**

1. **Die Identität stimmt** (Schritt 1). Mit n × rot A = ∇(A·n) − (∇n)A − (n·∇)A gilt am exakten Feld
   d := μ₀(K2 − K1) = −∇_Γ(A·n) + W·A_tan (W Weingarten-Abbildung). Auf allen 3696 (936 bei 0,5) Randquads
   stimmt das bis auf Rundung: ‖d − Vorhersage‖/‖d‖ ≈ 5e-16. Auf ebenen Quads ist d null (1e-16), auf den
   Spulenmänteln ist ‖d‖/‖K2‖ = **25 bis 27 %**, an Spulenkanten 24 bis 28 %. Das ist die Größe des
   K1/K2-Abstands aus #23 (27 bis 42 %, dort gröber gemessen). Zusatzbefund: auch die Plattenseiten und
   -kanten tragen d = 12,9 %.
2. **Der K-Widerspruch ist damit quantitativ erklärt.** Wo d null ist (ebene Quads), sind +K1 und +K2 in
   Zeile 1 gleich (z. B. Platte eben 0,60 % beide). Wo d ≠ 0 ist (Spulenmantel), unterscheiden sie sich
   (Innenspule Seite 35,6 % gegen 6,5 %).
3. **Zeile 1 ist eichinvariant, Zeile 2 nicht** (Schritt 2). Verschiebung a_ex → a_ex + ∇ψ in der Kantenbasis:
   - Zeile 1 in der K-Form (reine curl-curl) ändert sich um 1e-11 in allen Körpern, auch in der Platte. K2 ist
     invariant (5e-16).
   - Zeile 2 ändert sich stark: der Operand (0,5 T − D)a' weicht um 96 bis 490 % vom unverschobenen ab. Das
     gilt auch für das **zulässige** ψ (harmonisch, abklingend, Fall 3): gesamt 96 %.
4. **Das konsistente Paar ist nur im unverschobenen Fall klein, und dort per Konstruktion** (Schritt 3).
   k_BEM = K2 − d_Vorhersage ist gleich K1' (4e-16), also ist Zeile 2 mit k_BEM identisch zu Zeile 2 mit +K1
   (0,77 % bei 1,0, 1,94 % bei 0,5). **Unter Eichverschiebung bleibt das Paar nicht klein**: Zeile 2 mit
   k_BEM wird 36 % (Fall 3, 1,0; 44 % bei 0,5) bis 100 %. Zeile 1 mit K2 bleibt unverändert.

**Folgerung (Interpretation, nicht geprüft):** K1 und K2 unterscheiden sich um genau d, und das erklärt, warum
keine einzelne Unbekannte k beide Zeilen bedient. Aber Zeile 2 ist zusätzlich nicht eichinvariant, auch wenn
K1 mitgeführt wird. Naheliegend: D und G in Zeile 2 sehen nur den Tangentialanteil (A_tan, (∂A/∂n)_tan). Beim
exakten A_ex ist A·n an den Facetten klein, nach einer Verschiebung nicht mehr. Dann würde ein reiner
K-Übersetzungsterm (Option a unten) das Problem nur halb lösen. Das ist nicht gemessen (siehe „Offen").

## Gemessen

### Schritt 1: Identität am exakten Feld

**Herleitung.** Für festes n gilt n × (∇ × A) = (∇A)n − (n·∇)A mit ((∇A)n)ᵢ = nⱼ∂ᵢAⱼ. Also
μ₀K2 = −n × B = (n·∇)A − (∇A)n, und mit μ₀K1 = tang (n·∇)A folgt d = −tang[(∇A)n]. Auf der Fläche x(u,v)
mit Einheitsnormale n(u,v) und f := A·n ist ∂ₐf = (xₐ·∇)A·n + A·∂ₐn und tₐ·(∇A)n = (xₐ·∇)A·n. Daraus

    d = −∇_Γ(A·n) + C,      C = g^{ab} (A·∂_b n) xₐ = W·A_tan.

Der Krümmungsterm C kommt aus der Änderung der Flächennormale, der Gradiententerm aus A·n.
Sonderfälle: Zylinder (n = +r̂, A = A_φ φ̂): A·n = 0, d = +A_φ/r φ̂ (für n = −r̂ das umgekehrte Vorzeichen).
Ebene n = ±ẑ: beide Terme null, d = 0. Das stimmt mit dem Auftrag überein (K2 − K1 = ±A_φ/(μ₀r) φ̂).
(v·∇)A wird analytisch aus A_φ(r,z) gebildet (∂ᵣA_φ = B_z − A_φ/r, ∂_zA_φ = −B_r).

**Host-Test `test_bem_trace_identity`** (Marke host, reine Host-Rechnung):

| Fläche | ‖d − Vorhersage‖/‖d‖ |
|---|---:|
| Zylinder n = +r̂ (R = 60 mm) | 4,6e-15 |
| Zylinder n = −r̂ | 4,4e-15 |
| Kugel (R = 20 mm) | 4,0e-16 |
| gekippte Ebene | 3,3e-16 |
| Ebene z = const | d = 0 exakt |
| Zylinder: d = ±A_φ/R φ̂ (geschlossene Formel) | 4,5e-15 |
| Kugel: d = A_tan/R (geschlossene Formel) | 2,9e-16 |

Gegenproben (müssen durchfallen, tun es): falsches Vorzeichen des Krümmungsterms (Zylinder und Kugel:
Abweichung 2,0), fehlendes ∇_Γ(A·n) (gekippte Ebene: 1,0), falsches Vorzeichen von ∇_Γ(A·n) (2,0).

**Konvergenz unter Verfeinerung.** Auf dem facettierten Zylinder (ebene Quads, W = 0) gilt die Identität
exakt, die Krümmung steckt dann in ∇_Γ(A·n_f). Die Flächen-L²-Abweichung gegen die glatte Formel
d = A_φ/R φ̂ sinkt etwa linear:

| Facetten | 12 | 24 | 48 | 96 | 192 |
|---|---:|---:|---:|---:|---:|
| L²-Abweichung | 61,7 % | 25,5 % | 9,4 % | 3,9 % | 1,8 % |

Im Koeffizientenraum der Rooftops stagniert sie bei etwa 4,9 %. Grund: die langen z-Kanten tragen
d_z ∝ B_r·δ mit der Kantenlänge als Gewicht, das skaliert wie die kurzen φ-Kanten. Deshalb ist L² das Maß für
die Konvergenz und der Koeffizientenwert nur Information.

**Am echten Netz** (`--bem-row-check`, Galerkin-Projektion auf die Rooftops, 16-Punkt-Gauss):

| Körper | Klasse | n (Koeff., 1,0) | ‖d‖/‖K2‖ (0,5) | ‖d‖/‖K2‖ (1,0) | ‖d − Vorh.‖/‖d‖ (1,0) | ohne Krümmungsterm | ohne grad-Term |
|---|---|---:|---:|---:|---:|---:|---:|
| Platte | eben | 4992 | 2e-17 | 4e-17 | – | – | – |
| Platte | Seite | 192 | – | 12,95 % | 7e-16 | 0 % | 100 % |
| Platte | Kante | 384 | 12,94 % | 12,92 % | 8e-16 | 0 % | 100 % |
| Innenspule | eben | 1536 | 7e-17 | 8e-17 | – | – | – |
| Innenspule | Seite | 2304 | 26,06 % | 25,20 % | 5e-16 | 0 % | 100 % |
| Innenspule | Kante | 768 | 25,26 % | 23,71 % | 5e-16 | 0 % | 100 % |
| Außenspule | eben | 1536 | 6e-17 | 1e-16 | – | – | – |
| Außenspule | Seite | 2304 | 26,66 % | 26,41 % | 5e-16 | 0 % | 100 % |
| Außenspule | Kante | 768 | 27,20 % | 27,57 % | 5e-16 | 0 % | 100 % |
| alle | alle | 14784 | 22,53 % | 22,29 % | 5e-16 | 0 % | 100 % |

Die Quads am Netz sind eben, der Krümmungsterm trägt dort nichts bei (Fehler ohne ihn 0 %); der ganze
Unterschied steckt in ∇_Γ(A·n) (Fehler ohne ihn 100 %). Die Krümmung des 48-Ecks steckt in den
Facettenknicken, nicht im Krümmungsterm.

### Schritt 2: Eichverschiebung in der Zeilenmessung

**Durchführung.** a' = a_ex + c·(ψ(Endknoten) − ψ(Anfangsknoten)) je Kante (exakt für ∇ψ). B und K2 ändern sich
nicht. K1' = K1 + tang(H_ψ n) mit analytischer Hesse-Matrix (im Host-Test gegen zentrale Differenzen und
gegen die Vorhersage geprüft: 1e-16 bis 1e-14), in der Zeilenmessung als Galerkin-Projektion mitgeführt.
Amplitude c so, dass die Randkanten-2-Norm ‖∇ψ‖ gleich ‖a_ex‖ ist (beide 1,04e-3 bei 1,0; 1,02e-3 bei
0,5), also ∇ψ von der Größenordnung A_ex. Vier ψ:

| Fall | ψ | harmonisch | im Unendlichen |
|---|---|---|---|
| 1 | c·x·z | ja | wächst |
| 2 | c·(x²+y²)·z | nein (Δψ = 4cz) | wächst |
| 3 | c·d_z/|d|³, Dipol bei (41,3; 0,7; −26,3) mm in der Innenspule | ja (außerhalb des Leiters) | klingt ab |
| 4 | c·z·exp(−r²/s²), s = 150 mm | nein | klingt ab |

Fall 1 und 2 sind die im Auftrag genannten. **Sie verletzen die Abklingbedingung der
Außenraum-Darstellungsformel**; ihre Zeile-2-Werte sind deshalb mit dem Verhalten im Unendlichen vermengt
und sagen allein nichts über die Hypothese. Fall 3 und 4 sind eine Ergänzung. Fall 3 ist der einzige Fall,
in dem A' = A_ex + ∇ψ im gesamten Luftraum exakt komponentenweise harmonisch ist und abklingt (die
Dipolstelle liegt im Leiter), also die Voraussetzung der Darstellung erfüllt. Ein erster Lauf mit der
Dipolstelle auf einem Gitterknoten und Normierung über alle Kanten lieferte c = 1e-30 (die Norm wurde von
einer Kante an der Singularität dominiert) und wurde verworfen.

**Zeile 1: Messung gegen Vorhersage.** Meine erste Vorhersage (Zusatzterm −ωC·g in der Platte) war für die
verwendete Form falsch: K = W − ωC ist die reine curl-curl-Form ohne Wirbelstromterm, K·g = 0 in allen
Körpern. Den Wirbelstromterm trägt nur die W-Form: W·g = ωC·g. Gemessen (Skalierung 1,0, Randkanten):

| Fall | Körper | ‖K g‖ (K-Form) | ‖W g‖ | ‖ωC g‖ | ‖W g − ωC g‖ | Grundpegel ‖K a_ex − b‖ |
|---|---|---:|---:|---:|---:|---:|
| 1 | Platte | 6,4e-11 | 262,2 | 262,2 | 6,4e-11 | 4840 |
| 1 | Innenspule | 2,9e-11 | 2,9e-11 | 0 | 2,9e-11 | 27330 |
| 3 | Platte | 1,8e-11 | 63,4 | 63,4 | 1,8e-11 | 4840 |
| 4 | Platte | 3,5e-11 | 419,4 | 419,4 | 3,5e-11 | 4840 |

Die Spulen haben C = 0 (σ = 0, gemessen: ‖C g‖ = 0). In der K-Form ist Zeile 1 in allen Körpern invariant, die
Spalte +K2 bleibt in allen fünf Fällen bitgleich die des Falls 0. Die Spalte +K1' ändert sich, weil sich K1
ändert (Innenspule Seite, Spalte +K1': 35,6 % im Fall 0, dann 42,3 / 49,2 / 527,8 / 37,0 % in den Fällen 1 bis 4).

Grundpegel von Zeile 1 an den Spulen (innere Zeilen, Fadenmodell gegen Volumenstrom): 24,4 % / 24,3 %
(Innen-/Außenspule, 1,0) und 69,0 % / 68,8 % (0,5). Die Randzeilen-Werte unten sind relativ zu
‖K a − b‖ auf den Randkanten, nicht zu diesem Pegel.

**Zeile 2** (Skalierung 1,0; Residuum ‖(0,5 T − D)a' + μ₀G k‖ / ‖(0,5 T − D)a'‖, gesamt über alle Quads):

| Fall | +K1' | +K2 | Änderung des Operanden (0,5 T − D)a gegen Fall 0 |
|---|---:|---:|---:|
| 0 (keine) | 0,77 % | 16,89 % | 0 |
| 1 (xz) | 99,9 % | 87,7 % | 179 % |
| 2 ((x²+y²)z) | 142,9 % | 93,2 % | 252 % |
| 3 (Dipol) | 36,0 % | 70,3 % | 96,1 % |
| 4 (Gauß-ψ) | 70,8 % | 86,1 % | 165,6 % |

Skalierung 0,5, gesamt, +K1': 1,94 % / 100,0 % / 142,8 % / 43,7 % / 71,1 %, Operandenänderung
0 / 179 / 255 / 96,4 / 168 %.

Zeile 2 ist also **nachweislich eichabhängig**, auch im zulässigen Fall 3. Die Änderung des Operanden ist
dort gleich groß wie der Operand selbst (96 %). Im Fall 3 ist die Streuung zwischen den Körpern groß (1,0,
+K1', Seite): Innenspule 25,2 %, Außenspule 43,0 %, Platte Seite/Kante 323/328 %, Platte eben 65,8 %. Auffällig:
in der Außenspule ist dort +K2 (13,6 %) besser als +K1' (43,0 %). Das habe ich nicht untersucht.

### Schritt 3: konsistentes Paar

**Wahl von A·n.** A·n und ∇_Γ(A·n) werden aus dem exakten Feld A_ex (in den verschobenen Fällen
A_ex + ∇ψ) am Quadraturpunkt gebildet, nicht aus dem FEM-Feld. Begründung: der Nédélec-Interpolant des
Nachbarelements hat nur tangentiale Stetigkeit; seine Normalkomponente an der Randfläche ist unstetig und
O(h)-genau, ihre Oberflächenableitung nicht konvergent, und sie hängt von Kanten im Elementinneren ab, die
nicht zu den BEM-Randfreiheitsgraden gehören. Das ist **eine gewollte Vereinfachung**: dadurch ist k_BEM
nach Schritt 1 bis auf Rundung gleich K1 (4e-16), und Zeile 2 mit k_BEM ist identisch zu Zeile 2 mit +K1.
Die Variante prüft damit nur, ob das Paar die Eichverschiebung übersteht, **nicht**, ob das FEM-Feld A·n
hinreichend genau liefern kann.

**Ergebnis** (Skalierung 1,0, gesamt; Zeile 1 mit k = K2, Zeile 2 mit k = k_BEM):

| Fall | Zeile 1 (K2) | Zeile 2 (k_BEM) | ‖k_BEM − K1'‖/‖K1'‖ |
|---|---:|---:|---:|
| 0 | 7,44 % | 0,77 % | 4e-16 |
| 1 | 7,44 % | 99,9 % | 5e-16 |
| 2 | 7,44 % | 142,9 % | 5e-16 |
| 3 | 7,44 % | 36,0 % | 4e-16 |
| 4 | 7,44 % | 70,8 % | 4e-16 |

Zeile 1 (K2): Platte eben 0,60 %, Platte Kante 0,36 %, Innenspule Seite 6,5 %, Außenspule Seite 5,5 %;
bei 0,5 gesamt 16,2 %. Der Grundpegel von Zeile 1 an den Spulen (siehe oben) liegt höher. Im
unverschobenen Fall erfüllt das Paar die Erwartung (beide Zeilen klein), im verschobenen nicht.

Hinweis zur Lesart von Zeile 2: das Residuum auf einer ebenen Platten-Quad (d = 0) ist trotzdem nicht klein
für +K2 (22,5 %), weil G k die k-Werte aller Quads mitnimmt, auch die der Spulenmäntel. Zeile 2 ist nicht
lokal; nur Zeile 1 erlaubt eine Zuordnung zu einer Klasse.

## Interpretation (nicht geprüft)

1. **K1 ≠ K2 ist durch d vollständig beschrieben.** Das ist keine Interpretation mehr (Schritt 1), aber
   dass d die *einzige* Ursache des Kraftüberschusses ist, ist nicht gemessen. Die Messungen hier betreffen
   nur Zeilenresiduen am exakten Feld, nicht das gelöste Feld oder die Kraft.
2. **Gradiententerm statt Krümmung.** Auf dem facettierten Netz steckt die gesamte Differenz in
   ∇_Γ(A·n_f). Die Hypothese nannte A_φ/(μ₀r) auf glatten Zylindern; auf dem 48-Eck ist das dasselbe Objekt,
   nur anders verteilt (Schritt 1, Konvergenzzeile).
3. **Warum Zeile 2 eichabhängig bleibt.** Naheliegend ist, dass D und G in Zeile 2 mit den
   Tangentialanteilen von A und ∂A/∂n arbeiten, die Darstellungsformel für ein komponentenweise harmonisches
   Vektorfeld aber alle drei Komponenten braucht. Das passt zu den Daten (Fall 3: Operand ändert sich um
   96 %), ist aber nicht getestet. Eine Gegenprobe wäre, die fehlenden Normalanteile direkt in die
   Zeile-2-Auswertung aufzunehmen und zu sehen, ob das Residuum in Fall 3 auf das Niveau von Fall 0 fällt.
4. **Die Interpolationsgenauigkeit der Kantenvariablen ist nicht getrennt.** Die verschobene Kantenbelegung
   a' stellt ∇ψ nur im Whitney-Sinn dar. Beim Dipol (Dipolstelle etwa 13 mm vom Rand, Elementgröße nicht nachgemessen) könnte das einen Teil der Zeile-2-Abweichung ausmachen. Dagegen spricht, dass das Residuum
   von 0,5 nach 1,0 nur um etwa ein Fünftel fällt (43,7 % → 36,0 %); beweisen kann ich es nicht.

## Schritt 4: Bewertung und Optionen (keine Umsetzung)

**Bewertung.** Schritt 1 bestätigt die Identität, ihre Größe und den Ort (Spulenmäntel, zusätzlich
Plattenrand). Schritt 2 bestätigt die Eichabhängigkeit der BEM-Zeile. Schritt 3 zeigt aber, dass das mit
K1 übersetzte Paar die Eichverschiebung nicht übersteht. Die Hypothese ist also für den K-Widerspruch
bestätigt und für die Eichabhängigkeit bestätigt, aber ihre implizite Folgerung („Übersetzungsterm
genügt“) wird von Schritt 3 nicht gestützt. Die Entscheidung trifft nicht der Agent.

**(a) Übersetzungsterme im bestehenden Operator** (k_BEM = k − d(A·n) in Zeile 2).
- *Bleibt:* alles am Operator (T, G, D, ACA, Block-Vorkonditionierer), Unbekannte k = K2.
- *Neu:* eine Kopplung von Zeile 2 an A·n und ∇_Γ(A·n). A·n steckt nicht in den Randkanten-DOF, sondern in den
  Elementfunktionen (Kanten senkrecht zur Randfläche, also innere Kanten der Randhexaeder). Die Matrixstruktur
  bekommt damit Einträge zwischen BEM-Zeilen und inneren FEM-Kanten; ∇_Γ(A·n) ist schwach über Randterme der
  Rooftops zu bilden und braucht die Normalkomponente an Facettenkanten, die für das FEM-Feld unstetig ist.
- *Eichabhängigkeit:* verschwindet **nicht**. A·n ist selbst eichabhängig und die komponentenweise
  Darstellung verlangt Coulomb-Eichung im Außenraum. Nach Schritt 3 allein genügt der Term nicht.
- *Kohomologie/Ringspulen:* strukturell unberührt, solange nur Zeile 2 geändert wird.

**(b) Skalares Potential im Luftraum (A-φ-Kopplung).**
- *Bleibt:* FEM-Teil im Leiter (Baum-Kobaum-Eichung, Kantenelemente), G und D als skalare Operatoren (die
  vorhandenen Laplace-Kerne), ACA.
- *Neu:* Luftraum B = −μ₀∇φ, Randgrößen φ und ∂φ/∂n; der Fluss n·B = n·rot A ist aus der
  Tangentialspur von A bildbar (Oberflächenrotation), A·n tritt nicht auf. Die Kopplungsblöcke ändern sich.
- *Eichabhängigkeit:* im Luftraum entfällt A und damit seine Eichung. Im Leiter bleibt Baum-Kobaum.
- *Kohomologie/Ringspulen:* hier der heikle Punkt. Der Luftraum ist wegen der Ringspulen nicht einfach
  zusammenhängend, φ ist mehrdeutig; man braucht Schnittflächen oder ein Quellfeld (Biot-Savart des
  Spulenstroms plus reduziertes φ). Das hängt eng mit den vorhandenen Kohomologiegeneratoren zusammen,
  müsste aber für den Luftraum neu gebaut werden.

**(c) BEM in Maxwell-Spuren (n × A, n × rot A).**
- *Bleibt:* FEM-Teil und die Randdarstellung der Spur n × A (Rooftops), die Kopplung ist natürlich, weil
  FEM und BEM dieselben Spuren benutzen; ACA und Blockstruktur.
- *Neu:* elektrischer und magnetischer Operator der Maxwell-Randintegralgleichungen (neue Kerne, mit
  divergenzkonformen Testfunktionen; die vorhandenen Rooftops sind die curl-konforme Spur).
- *Eichabhängigkeit:* verschwindet, die Größen sind E und H bzw. B.
- *Kohomologie/Ringspulen:* verträglich, ohne zusätzliche Schnittflächen im Luftraum. Wegen der
  quasistatischen Frequenzen (0,01 bis 50 Hz) ist mit dem bekannten Niederfrequenz-Zusammenbruch dieser
  Gleichungen zu rechnen (Schleifen-Stern-Zerlegung oder Calderón-Vorkonditionierung).

**Aufwand grob (Einschätzung, nicht gemessen):** (a) am kleinsten, aber nach Schritt 3 vermutlich nicht
ausreichend; (b) mittel bis groß, wegen des mehrfach zusammenhängenden Luftraums; (c) am größten, am
saubersten bezüglich Eichung.

## Offen

- Wirkt ein Entfernen von d auf Feld und Kraft? Nicht gemessen; diese Arbeit betrifft nur Zeilenresiduen am
  exakten Feld.
- Interpretation 3: Normalanteile von A und ∂A/∂n in Zeile 2 nachrüsten und prüfen, ob Fall 3 auf das Niveau
  von Fall 0 fällt (Messung, kein Eingriff in den Operator).
- A·n aus dem FEM-Feld statt aus dem exakten Feld: nicht implementiert; ohne sie bleibt unbekannt, wie
  genau ein Übersetzungsterm im gelösten System wäre.
- Interpolationsfehler der verschobenen Kantenbelegung (Interpretation 4) nicht von der Eichabhängigkeit
  getrennt.
- Außenspule Fall 3: +K2 besser als +K1' (13,6 % gegen 43,0 %), nicht untersucht.
- Skalierung 1,25, Imaginärteil bei kleiner Frequenz, transienter Pfad: wie im Auftrag nicht Teil.

## Abweichungen vom Auftrag

- Die Eich-Funktionen aus dem Auftrag (Fall 1 und 2) wachsen im Unendlichen; ergänzt um Fall 3 und 4
  (abklingend), weil sonst Eichabhängigkeit und Abklingbedingung nicht zu trennen sind.
- Die Amplitude wurde über Randkanten normiert, nicht über alle Kanten (siehe Schritt 2).
- Meine erste Vorhersage für Zeile 1 in der Platte war falsch (K-Form, nicht W-Form) und ist korrigiert.
- A·n in Schritt 3 aus dem exakten Feld statt aus dem FEM-Feld (begründet, gewollte Vereinfachung).
- Schritt 1: ‖d‖/‖K2‖ am Netz 25 bis 27 % auf den Spulenmänteln, am unteren Rand der Erwartung (27 bis 42 %).
- Der Auftrag erwartete die Platte fast nur eben; sie hat Seiten und Kanten mit d = 12,9 %.

## Befehle

```sh
# Bauen und Test (LD_LIBRARY_PATH wie in BUILD.md)
cmake --build build --target test_bem_trace_identity fembem
./build/test_bem_trace_identity

# Zeilenmessung: Schritt 1 bis 3 (dicht, 2 Hz); Skalierung 0.5 und 1.0
for sc in 0.5 1.0; do
  ./build/fembem --harmonic=2 --fixed-plate --plate-z=11.3 --mesh=near/hex_s$sc.vtk \
      --bem-eta=1e-9 --bem-row-check > trace/row23c_s$sc.log
done

# Tests
cd build && ctest -L host
cd build && ctest
```

Die Netze `near/hex_sS.vtk` stammen aus `tools/gmsh/team28_hex.py` (wie in `bem-near-field.md`).

## Tests

`ctest -L host` und vollständiges `ctest` vorher (Basis `main` a5484a8) und nachher:

| | vorher (Basis `main` a5484a8) | nachher |
|---|---:|---:|
| `ctest -L host` | 41/41 | 42/42 |
| vollständiges `ctest` | 74/74 | 75/75 |

Neu ist `test_bem_trace_identity` (Marke host). Die Vorher-Zahlen stammen aus den Berichten zu #25 und #26, nicht
aus einem erneuten Lauf auf a5484a8.
