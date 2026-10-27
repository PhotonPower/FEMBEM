# Reines FEM mit sparse Direktlöser — Plan

> **Stand der Umsetzung.**
>
> | Teil | Stand |
> |---|---|
> | AP1 BDF-Import (`tools/import_bdf.py`) | fertig, an der echten Datei erprobt |
> | AP3 Elemente (Tetraeder, Prisma) | fertig, geprüft |
> | Gemischte Typen in Netz, Reader, Topologie, Assemblierung | fertig, baut mit nvcc, `ctest -L host` grün |
> | AP4 Dirichlet und Regularisierung (`FemDirectBC`) | fertig, geprüft |
> | AP7 Feldauswertung und Kraftpfad (`NedelecDispatch`) | fertig, geprüft |
> | AP5 Reelle 2n-Blockform (`HarmonicBlockSystem`) | fertig, geprüft |
> | AP6 cuDSS-Anbindung (`SparseDirectSolver`) | fertig, gegen 0.8.0 **auf einer RTX 5060 (sm_120) erprobt**: Zerlegung und Lösung laufen, relatives Residuum 1,23e-10 |
> | AP8 Verdrahtung (`--fem-direct`) | fertig, **übersetzt und gelaufen** |
> | AP2 Netzanforderung (Luftkasten, Abschneidung) | offen, hängt am Netzgenerator |
> | Kraftauswertung anschließen | **fertig**, aber nicht konvergiert: $F_z$ = 0,842 N bzw. 1,250 N je nach Netzstufe, siehe unten |
> | Uniforme Verfeinerung `--refine=N` | **fertig für Hexaeder UND Tetraeder**, mit Abbruch bei gemischtem Netz |
> | Iterativ statt direkt: `--fem-direct-iterative` | **fertig und gemessen**: mit Projektion keine Stagnation mehr, aber ~1000 Iterationen je Dekade. Ohne AMS kein Ersatz für cuDSS |
> | Gradientenprojektion des Quellterms | **fertig und bestätigt**: M2 erfüllt (5,4e-16), Scheingradient weg, Residuum 4,3e-15, Kraft unverändert |
>
> **Erste Kraftzahl** (BDF-Netz, $\varepsilon$ = 1e-8, Startlage):
>
> ```
> F = (-0,0186, 0,008, 1,97) N
> M = (-0,00197, -0,00146, -0,000145) Nm um den Plattenschwerpunkt
> ```
>
> Einordnung. Die Platte hat bei $r$ = 10,7…65 mm und 3,0 mm Dicke ein Volumen
> von 3,87e-05 m³, also 0,105 kg und **1,026 N Gewicht** — das ist die Kraft im
> Schwebegleichgewicht. Der Kopf von `HarmonicLorentz.cuh` nennt als Sollwert
> **1,0497 N bei 11,3 mm**. Das Netz ist aber die **Startlage**, nicht das
> Gleichgewicht: Schwerpunkt bei $z$ = 5,3 mm, also 6 mm näher an den Spulen,
> und dort muss die Kraft größer sein. Gemessen 1,97 N, Verhältnis 1,88.
>
> Querkraft 0,020 N (1,0 % von $F_z$) und ein Kippmoment, das einem Hebelarm
> von 1,24 mm bei 65 mm Plattenradius entspricht. Beides passt zur Asymmetrie
> des Netzes — der Plattenschwerpunkt liegt schon 0,1 mm neben der Achse.
>
> **Korrektur einer Abschätzung von mir.** Ich hatte die Scheinkraft aus dem
> regularisierten Luftgebiet auf 1,9 N geschätzt und daraus geschlossen, sie
> sei größer als das Ergebnis. Gemessen sind es **-3,83e-05 N**, also fünf
> Größenordnungen weniger. Der Fehler war die Formel $F \approx |J||B|V$: sie
> unterstellt kohärente Aufaddierung über das Volumen. Für ein Gradientenfeld
> im Kreuzprodukt gilt aber
> $\int \nabla\varphi \times \mathbf{B}\,dV
>  = \oint \mathbf{n}\times\varphi\mathbf{B}\,dS
>  - \int \varphi\,(\nabla\times\mathbf{B})\,dV$,
> und in der stromfreien Luft ist $\nabla\times\mathbf{B} = 0$ — der
> Volumenanteil verschwindet identisch, es bleibt ein Randterm. Das ist
> strukturell, nicht zufällig.
>
> Die Maske bleibt trotzdem: sie macht den Ausschluss ausdrücklich statt
> zufällig, und die zweite Zahl im Protokoll hat den Irrtum aufgedeckt statt
> ihn zu konservieren.
>
> **Zweiter Messpunkt: Gleichgewichtslage** (`team28_simple_air_113.bdf`,
> 8216 Tetraeder, Plattenunterkante 11,3 mm, Schwerpunkt 12,8 mm).
>
> | | Startlage | Gleichgewicht |
> |---|---|---|
> | $F_z$ | 1,97 N | **0,842 N** |
> | max$\|B_r\|$ Platte | 0,136 T | 0,086 T |
> | Plattengewicht | — | 1,026 N |
> | Sollwert `HarmonicLorentz.cuh` | — | 1,0497 N |
>
> **-17,9 % gegen das Gewicht, -19,8 % gegen den Sollwert.** Die erste Zahl
> des Projekts mit einem Referenzwert daneben.
>
> Zwei Beobachtungen sprechen für einen Auflösungsfehler und gegen einen
> Formelfehler:
>
> * **Die beiden Netze sind untereinander konsistent.** $B$ fällt um Faktor
>   1,58, die Kraft geht mit $B^2$, erwartet also 2,50 — gemessen 2,34. Zwei
>   unabhängig vernetzte Aufbauten ergeben eine physikalisch sinnvolle
>   Kraft-Weg-Kurve.
> * **Der Fehler des Oberflächenpfades skaliert mit dem Ergebnis:**
>   9,04/1,97 = 4,59 gegen 4,06/0,842 = 4,82. Fast identisch — passend dazu,
>   dass er von der Plattenoberfläche bestimmt wird, und die ist in beiden
>   Netzen dieselbe.
>
> **Was dieses Netz NICHT verbessert hat**, und das ist der eigentliche
> Befund:
>
> | | Netz 1 | Netz 2 | Faktor |
> |---|---|---|---|
> | Elemente gesamt | 3692 | 8216 | 2,23 |
> | Luft | 2665 | 7189 | 2,70 |
> | **Platte** | **259** | **259** | **1,00** |
> | Plattenkanten | 468 | 468 | 1,00 |
> | Grenzflächen | 208 | 208 | 1,00 |
> | Abschneidefaktor radial | 1,742 | 1,745 | — |
> | Abschneidefaktor axial | 1,729 | 3,068 | — |
>
> Die Verfeinerung ging vollständig in die Luft, radial ist der Luftkasten
> unverändert knapp. Die Platte ist Element für Element dieselbe — dort sitzen
> die Wirbelströme und dort wird integriert. Beide Ursachen, die für die 18 %
> in Frage kommen, hat dieses Netz nicht angefasst.
>
> **AP2 braucht deshalb zwei Achsen, nicht eine:** den Luftkasten radial über
> Faktor 2 hinaus UND die Platte feiner. Ein Netz, das nur eines von beidem
> tut, verschiebt die Zahl nicht nachweisbar — dieser Lauf ist der Beleg.

> **Dritter Messpunkt: dasselbe Netz mit `--refine=1`** (65 728 Tetraeder,
> Plattenauflösung von 259 auf 2072 Elemente, Luftkasten unverändert).
>
> | | grob | fein | Änderung |
> |---|---|---|---|
> | Plattenelemente | 259 | 2072 | ×8 |
> | **$F_z$** | **0,842 N** | **1,250 N** | **+48,5 %** |
> | gegen Gewicht 1,026 N | -17,9 % | +21,8 % | — |
> | max$\|B_r\|$ Platte | 0,0860 T | 0,0874 T | +1,6 % |
> | max$\|A_r\|$ Platte | 1,41e-05 | 8,73e-06 | Faktor 0,619 |
>
> **Die Kraft hat den Sollwert übersprungen.** Damit ist keiner der beiden
> Werte konvergiert; dass der feine zufällig auf der anderen Seite landet,
> macht ihn nicht belastbarer als den groben.
>
> **Entscheidend ist aber, was sich NICHT geändert hat.** $B$ in der Platte
> bleibt auf 1,6 % stehen, und max$|A_r|$ fällt um Faktor 0,619 — Kanten­
> zirkulationen skalieren mit $h$, ein unverändertes Feld gäbe also 0,5. Das
> Feld ist im Rahmen der Erwartung konvergiert.
>
> Das Feld stimmt, die Kraft springt um 48 %. Nicht das Feld ist also
> unteraufgelöst, sondern **das Kraftintegral in der Platte**. 259 Tetraeder
> in einer 3 mm dünnen Scheibe von 65 mm Radius sind eine einzige Lage; der
> Wirbelstrom hat über die Dicke ein Profil, und eine Lage kann davon nichts
> darstellen. Nach der Verfeinerung sind es zwei — immer noch wenig.
>
> **Die Maxwell-Gegenprobe konvergiert**, aus zu großer Entfernung:
>
> | | Nullprobe (exakt 0 N) | relativ zu $F_z$ |
> |---|---|---|
> | grob | -4,06 N | 4,82 × |
> | fein | -1,18 N | 0,94 × |
>
> Faktor 3,44 bei halbierter Netzweite, also etwa $h^{1,8}$ — sauber zweiter
> Ordnung. Der Pfad ist richtig umgesetzt, er startet nur bei einem Fehler,
> der das Ergebnis um das Fünffache übertraf. Zwei weitere Stufen brächten ihn
> auf rund 9 % und damit erstmals in den brauchbaren Bereich.
>
> **Hier endet die gleichmäßige Verfeinerung.** Die Faktorisierung braucht
> bereits 214 Mio. Nichtnullen (Füllfaktor 41,9, 1,7 GB allein für die Werte)
> und läuft nur noch mit `FEMBEM_CUDSS_HYBRID=1`. Die nächste Stufe wären
> 1,3 Mio. Zeilen bei weiter steigender Füllung — zweistellige Gigabyte.
>
> Und der Aufwand geht an die falsche Stelle: **57 512 der 65 728 Elemente
> sind Luft.** Die gleichmäßige Verfeinerung steckt 87 % der Rechenzeit
> dorthin, wo weder Wirbelströme sitzen noch integriert wird.
>
> **Was AP2 daher braucht, in dieser Reihenfolge:**
>
> 1. **Die Platte feiner, vor allem über die Dicke** — vier bis sechs Lagen
>    statt zwei, bei gleichbleibendem Luftgebiet. Lokal verfeinern geht nicht
>    ohne hängende Knoten an der Materialgrenze, das muss aus dem
>    Netzgenerator kommen.
> 2. **Den Luftkasten radial über Faktor 2.** Er steht in allen drei
>    Messpunkten unverändert bei 1,745, der systematische Anteil ist also in
>    jeder Zahl oben enthalten und in seinem Vorzeichen unbestimmt.
>
> Zwei bis drei solche Netze bei festem Luftkasten wären die Reihe, die $F_z$
> festnagelt. Eine einzelne weitere Zahl bringt nichts: mit 0,842 und 1,250
> liegen zwei Punkte vor, die den Sollwert einschließen, ohne ihn zu belegen.

> **Vierter Messpunkt: Platte in der Ebene fein** (eigenes Netz aus HyperMesh,
> 133 899 Tetraeder, 19 812 Plattenelemente in zwei Lagen, Luftkasten und
> Spulen unverändert).
>
> | Netz | Platte | Lagen | $F_z$ | gegen Gewicht 1,026 N |
> |---|---|---|---|---|
> | grob | 259 | 1 | 0,842 N | -17,9 % |
> | `--refine=1` | 2072 | 2 | 1,250 N | +21,8 % |
> | fein in der Ebene | 19 812 | 2 | **1,020 N** | **-0,6 %** |
>
> **Die erste Zahl des Projekts, die auf einen unabhängig bekannten Wert
> trifft.** Das Plattengewicht folgt aus Geometrie, Dichte und $g$, gerechnet
> aus dem Netz selbst — es ist nichts angepasst.
>
> Zwei Vorbehalte, und beide sind ernst:
>
> * **Die Reihe schwingt** (0,842 → 1,250 → 1,020). Das ist keine
>   nachgewiesene Konvergenz; 0,6 % können ein Zufall auf dem Weg sein.
> * **Die Ebene ist nicht ausgereizt.** Zwischen Netz 2 und 3 änderte sich nur
>   die Auflösung in der Ebene, beide haben zwei Lagen — und $F_z$ bewegte
>   sich um -18,4 %. Zwei Lagen sind also nicht die ganze Geschichte, wie ich
>   oben vermutet hatte.
>
> Der Abschneidefaktor steht in allen drei Werten unverändert bei 1,745. Der
> systematische Anteil daraus ist in den 0,6 % enthalten, mit unbekanntem
> Vorzeichen.
>
> **Bestätigte Vorhersage zu M2.** Die Spulen sind in allen Netzen dieselben
> 517 und 251 Elemente, und entsprechend liefert dieser Lauf
> ‖$G^Tb$‖/‖$b$‖ = 6,139e-03, ‖$b$‖ = 8,1895e+02 und 1433 Einträge —
> identisch mit dem groben Netz. Damit ist belegt, dass die Verschlechterung
> auf 1,098e-02 unter `--refine=1` von der Verfeinerung **der Spulen** kam und
> nicht vom Netz insgesamt.
>
> **Der Oberflächenpfad flacht ab.**
>
> | Grenzflächen | Nullprobe (exakt 0 N) | relativ zu $F_z$ | Ordnung |
> |---|---|---|---|
> | 208 | 4,060 N | 4,82 × | — |
> | 832 | 1,180 N | 0,94 × | $h^{1,78}$ |
> | 13 274 | 0,492 N | 0,48 × | $h^{0,63}$ |
>
> Die vierfache Auflösung senkt den Fehler nur um 2,4. Die zweite Ordnung von
> der ersten Stufe ist weg. Vermutete Ursache: das diskrete $B$ springt über
> Elementflächen (Nédélec sichert nur die Tangentialstetigkeit von $A$ zu,
> also die NORMALkomponente von $B$), und dieser Anteil konvergiert nicht mit
> der Ordnung des Quadraturfehlers. Die 124 Splitterelemente unter Formgüte
> 0,05 könnten mitspielen. Belegt ist keins von beidem.
>
> **Was ein fünfter Punkt klären würde:** zwei Lagen, Ebene noch feiner, alles
> andere unverändert. Bleibt $F_z$ bei etwa 1,02 N, ist die Übereinstimmung
> mit dem Gewicht belegt statt vermutet.


> **GMRES stagniert — und liefert trotzdem dieselbe Kraft.**
> `--fem-direct-iterative` auf dem feinen Netz: GMRES(100) mit ILU(0) auf
> bitgenau derselben 2n-Matrix, die cuDSS bekommt.
>
> ```
> iter   10   Rel. Res: 0,0941
> iter   30   Rel. Res: 0,0159
> iter  100   Rel. Res: 0,00432
> iter  400   Rel. Res: 0,00260
> iter 4000   Rel. Res: 0,00258     311 s, nicht konvergiert
> ```
>
> Von Iteration 400 bis 4000 bewegt sich die letzte Stelle nicht. Harte
> Stagnation, nicht langsame Konvergenz — die `[Gauge]`-Warnung („ILU(0)
> konvergiert auf dem ungeeichten System nicht") gilt also auch mit der
> σ-Regularisierung.
>
> | | cuDSS (2,8e-10) | GMRES (2,6e-03) |
> |---|---|---|
> | global max A_r | 6,65e-05 | **0,00117** |
> | global max A_i | **6,88** | 2,53e-06 |
> | Platte max A_r | 2,42e-06 | 2,42e-06 |
> | Platte max A_i | 7,50e-07 | 7,55e-07 |
> | B_r Platte | 0,0719 T | 0,0720 T |
> | **F_z** | **1,02 N** | **1,02 N** |
>
> Global sind die Lösungen völlig verschieden, **in der Platte stimmen sie auf
> unter ein Prozent überein, und die Kraft ist identisch.**
>
> **Gemessen, nicht geschlossen:**
>
> ```
> ||G^T r|| / ||r||  =  2,38
> ||G^T b|| / ||b||  =  0,00614      Verhältnis 387
> ```
>
> Der Anteil, an dem GMRES stagniert, ist praktisch reiner Gradient. Zweite
> Bestätigung von anderer Seite: max A_r global wächst von 2,21e-04 bei 500
> Iterationen auf 1,17e-03 bei 4000, während die Plattenwerte auf allen
> Stellen stehenbleiben. Die beiden Anteile sind entkoppelt.
>
> Erklärung: der Imaginärteil von 6,88 ist der Scheingradient, den der direkte
> Löser exakt findet. GMRES baut ihn gar nicht auf, und genau das ist die
> Stagnation. Dieser Anteil ist physikalisch träge — rot grad = 0, und die
> Platte erreicht er nicht.
>
> 500 Iterationen genügen und kosten 38 s. Für ein System, das cuDSS nur mit
> Auslagerung und 445 Mio. Nichtnullen im Faktor bewältigt, ist das ein
> Ausweg — **aber einer ohne verwendbares Abbruchkriterium.** Das gemeldete
> Residuum bedeutet nichts, wenn es von einem trägen Unterraum beherrscht
> wird.
>
> ### Eine Maßnahme, die beide offenen Punkte trifft
>
> Zwei Befunde dieses Dokuments gehören zusammen, was hier lange getrennt
> stand:
>
> * Der Quellterm ist nicht divergenzfrei (6,1e-03, facettierte
>   Spulenoberfläche, durch Verfeinerung nicht behebbar).
> * Genau dieser Anteil wird um 1/ε verstärkt, erzeugt den Scheingradienten in
>   A_i, und ist das Einzige, woran GMRES scheitert.
>
> **Den Gradientenanteil aus b herausprojizieren trifft beides**, und es ist
> billiger als die Formulierung über rot T: ein skalares Poisson-Problem
> (G^T G) φ = G^T b, dann b ← b − G φ. Der diskrete Laplace-Operator auf
> Knoten ist gut konditioniert — ein CG-Lauf von wenigen Sekunden.
>
> Prüfbare Vorhersage, dreiteilig:
>
> 1. ‖G^T b‖/‖b‖ wird maschinengenau null. **M2 erfüllt, nicht
>    umgeschrieben.**
> 2. Der Scheingradient in A_i verschwindet, weil ihm die Quelle fehlt.
>    max A_i fällt auf die Größenordnung von A_r und wird ε-unabhängig.
> 3. GMRES konvergiert, weil keine Nullraumkomponente mehr aufzubauen ist.
>
> Dass die Lösung sich ändert, ist unbedenklich: sie ändert sich genau um den
> Gradientenanteil, und der ist über vier ε-Dekaden gemessen ohne Wirkung auf
> B und auf die Kraft.
>
> Scheitern kann die Vorhersage an jedem Teil. Punkt 3 ist der unsicherste:
> die Nullraumrichtungen des OPERATORS existieren unabhängig von b, und ob ein
> Krylow-Verfahren sie meidet, nur weil die rechte Seite dort nichts hat,
> hängt an der Rundung. Scheitert Punkt 3, bleibt AMS.

> **Die Projektion greift. Alle drei Teile der Vorhersage sind eingetroffen,
> und ein vierter dazu.**
>
> | | ohne Projektion | mit Projektion |
> |---|---|---|
> | ‖G^T b‖/‖b‖ | 6,139e-03 | **5,44e-16** |
> | global max A_i | 6,88 | **2,38e-06** |
> | global max A_r | 6,65e-05 | 6,65e-05 |
> | Residuum (cuDSS) | 2,76e-10 | **4,28e-15** |
> | Platte max A_r | 2,42e-06 | 2,42e-06 |
> | Platte max A_i | 7,50e-07 | 7,48e-07 |
> | B_r Platte | 0,0719 T | 0,0719 T |
> | **F_z** | **1,02 N** | **1,02 N** |
>
> **M2 ist erfüllt**, maschinengenau, mit 300 CG-Iterationen auf dem
> Knoten-Laplace. Nicht umgeschrieben — erfüllt.
>
> **Der Scheingradient ist restlos weg**, Faktor 2,9 Millionen. Der globale
> max A_i liegt jetzt bei 3× dem Plattenwert statt bei 9 Millionen mal.
>
> **Die Kontrolle ist bestanden.** Alle physikalischen Größen sind unverändert:
> F_z, A_r und A_i in der Platte, B_r, B_i. Die Projektion hat genau nichts
> Physikalisches angefasst. Damit ist das Argument bestätigt und nicht nur
> plausibel: der entfernte Anteil war ein Artefakt.
>
> **Das Unerwartete: das Residuum fiel um fünf Größenordnungen**, von 2,76e-10
> auf 4,28e-15. Erklärbar, und die Erklärung stand bereits in der
> ε-Reihe oben — dort wuchs das Residuum selbst mit 1/ε, was auf den
> Dynamikumfang zurückzuführen war: eine Lösung von 6,65e-05 bis 6,88 spannt
> fünf Dekaden und lässt sich in doppelter Genauigkeit nicht auf 1e-15
> auflösen. Nach der Projektion spannt sie das nicht mehr.
>
> Daraus folgte eine prüfbare Zusage, und sie ist **eingetroffen**:
>
> | ε | Residuum | max A_r | max A_i | F_z |
> |---|---|---|---|---|
> | 1e-6 | 4,36e-15 | 6,65e-05 | 2,38e-06 | 1,02 N |
> | 1e-8 | 4,31e-15 | 6,65e-05 | 2,38e-06 | 1,02 N |
> | 1e-10 | 4,33e-15 | 6,65e-05 | 2,38e-06 | 1,02 N |
>
> Vorher wuchs das Residuum über diese Spanne um vier Dekaden und max A_i um
> Faktor 10 000. Jetzt bewegt sich keine Ziffer. **Die Abnahme aus Abschnitt 1
> ist damit nicht nur an B erfüllt, sondern an jeder Größe des Laufs**, und
> die Erklärung über den Dynamikumfang ist belegt.
>
> Nebenbefund, vorsichtig: die Querkräfte und das Kippmoment sind kleiner
> geworden (M_y von -3,59e-04 auf -2,78e-05). Ein Teil des Rauschens dort kam
> offenbar aus dem Artefakt. Ein Datenpunkt, kein Beleg.
>
> **Teil 3 der Vorhersage steht noch aus**: konvergiert GMRES jetzt? Das ist
> der unsicherste Teil, weil die Nullraumrichtungen des OPERATORS unabhängig
> von $b$ existieren.

> **Teil 3: die Stagnation ist weg, die Konvergenz bleibt langsam.**
>
> ```
> iter   100   3,45e-03      (ohne Projektion: 4,32e-03)
> iter   200   9,55e-04      (ohne Projektion: 2,76e-03)
> iter  1000   4,36e-05      (ohne Projektion: eingefroren bei 2,58e-03)
> iter  2000   4,39e-06      152 s, nicht konvergiert
> ```
>
> Die harte Stagnation ist beseitigt — das war die Vorhersage und sie stimmt.
> Aber es sind rund 1000 Iterationen je Dekade; von 4,4e-06 bis 1e-10 wären
> noch etwa 4000 weitere. GMRES ist damit kein Ersatz für cuDSS, nur nicht
> mehr wertlos.
>
> **Und der Grund stand im Log.** ‖G^T r‖/‖r‖ = 2,26 — das Restresiduum ist
> weiterhin gradientendominiert, und daneben:
>
> ```
> ||G^T b||/||b|| = 1,61e-05     die Projektion selbst meldete 5,45e-16
> ```
>
> Die beiden messen nicht dasselbe $b$. Die Projektion arbeitet auf dem
> FEM-Kantenraum, die Diagnose auf der 2n-Blockform **nach der
> Randbehandlung** — und dazwischen geht die Divergenzfreiheit teilweise
> verloren.
>
> Ursache, eine Nebenwirkung der Projektion selbst: $b$ war vorher nur auf
> 1433 Kanten im Spulenbereich besetzt und auf dem Rand des Luftkastens exakt
> null. $\varphi$ löst aber ein GLOBALES Poisson-Problem, also ist
> $G\varphi$ überall von null verschieden — nach $b \leftarrow b - G\varphi$
> tragen die Randkanten einen kleinen Wert, und
> `apply_homogeneous_dirichlet()` setzt ihn zurück auf null. Genau das bringt
> die Divergenz an den Randknoten wieder ein.
>
> **Behoben, indem die Projektion nur über die FREIEN Kanten läuft:**
> $(G_F^TG_F)\varphi = G_F^Tb$. Auf gesperrten Kanten bleibt $b'$ null per
> Konstruktion, und $G^Tb'$ verschwindet exakt, weil die gesperrten Kanten
> nichts beitragen. Der Dirichlet-Kantensatz wird dafür einen Schritt früher
> gesammelt und an beide Stellen gegeben — würden sie je auseinanderlaufen,
> wäre genau dieser Fehler zurück.
>
> Knoten, die nur an gesperrten Kanten hängen, fallen aus dem Teilgraphen
> heraus und werden aus dem CG ausgeschlossen; sonst bringt die
> Konstantenbereinigung ihnen einen Wert ein, den keine Gleichung mehr
> korrigiert.
>
> `test_source_projection` prüft jetzt beides zusammen: $b$ ist null auf den
> gesperrten Kanten UND $G^Tb$ ist null an allen Knoten, gerechnet über alle
> Kanten. Einzeln sagt keine der zwei Zahlen etwas.
>
> **Die Maske behebt die Inkonsistenz — und macht GMRES nicht schneller.**
> Gemessen nach der Umstellung:
>
> ```
> ||G^T b||/||b||   6e-16     (vorher 1,61e-05)   -> Erklärung bestätigt
> GMRES 2000 Iter   4,39e-06, 152 s               -> identisch mit vorher
> ||G^T r||/||r||   2,26                          -> unverändert
> ```
>
> Die Erklärung für die 1,61e-05 war richtig, die Folgerung daraus falsch. Ich
> hatte vorhergesagt, die Konvergenz werde schneller; sie ist auf die Stelle
> dieselbe.
>
> **Damit ist der Mechanismus geklärt, und zwar gegen diese Vorhersage: die
> Gradientenkomponente im Residuum kommt nicht aus $b$, sondern aus dem
> OPERATOR.** Die rechte Seite hat dort jetzt buchstäblich nichts (Verhältnis
> 3,8e+15), und das Restresiduum ist weiterhin reiner Gradient. Es entsteht
> aus Rundung und wächst mit den winzigen Eigenwerten von $\omega C$ auf den
> Gradientenfeldern — Faktor 350 unter dem Rest, ablesbar an
> max$|\omega C|$/max$|K|$ = 0,00285. ILU(0) hat dafür keinen Mechanismus.
>
> Der Vorbehalt stand zweimal in diesem Dokument („die Nullraumrichtungen des
> Operators existieren unabhängig von $b$") und wurde trotzdem nur als
> unsicherster Teil einer Vorhersage geführt, die er widerlegt.
>
> **Die Maske bleibt richtig**, aber aus einem anderen Grund als behauptet: M2
> gilt jetzt auf dem System, das TATSÄCHLICH gelöst wird, nicht nur auf dem
> $b$ davor. Für den direkten Löser ändert sie nichts, der steht bei 4,3e-15.
>
> **Was für die Iteration bleibt, ist AMS.** Die Projektion beseitigt die
> Quelle im rechten Term, nicht die schlechten Eigenwerte des Operators. Offen
> ist, ob `AmsPreconditioner.cuh` sich auf die 2n-Blockform übertragen lässt —
> er ist für den $n$-großen FEM-Operator geschrieben, und die Blockform
> verdoppelt die Dimension mit einer Kopplung über $\omega C$.

> diesem Netz nicht.** Gemessen, nicht argumentiert.
>
> | | $F_z$ | Sollwert |
> |---|---|---|
> | Lorentz, Volumenintegral | 1,97 N | — |
> | Maxwell, 208 Grenzflächen | -9,62 N | 1,97 N |
> | Maxwell nur aus $A_r$ | **-9,04 N** | **exakt 0 N** |
> | Maxwell nur aus $A_i$ | -0,58 N | 1,97 N |
>
> Die dritte Zeile ist der Beleg. Mit $A_i = 0$ verschwindet der Realteil der
> Stromdichte in der Platte ($\mathrm{Re}(J) = \sigma\omega A_i$), und ohne
> Strom innerhalb der Hülle muss das Spannungsintegral bei $\mu_r$ = 1 exakt
> null ergeben. Es ergibt -9,04 N, also **das 4,6-fache des gesuchten
> Ergebnisses**. Der Oberflächenpfad verfehlt hier einen analytisch bekannten
> Nullwert um mehr als die Größe, die er messen soll.
>
> Zwei Gründe, beide auf diesem Netz verschärft:
>
> * **Auslöschung.** Die Einzelbeiträge liegen bei
>   $|B|^2/(2\mu_0)$ mal Hüllfläche, also rund 200 N bei gesuchten 1,97 N.
>   Ein Verhältnis von 1:100 — 5 % Fehler in den Einzelbeiträgen ergeben
>   10 N. Der Kopf von `HarmonicLorentz.cuh` nennt denselben Mechanismus für
>   das alte Hexaedernetz mit -0,122 N.
> * **Das diskrete $B$ ist über Elementflächen nicht stetig.** Nédélec sichert
>   die Tangentialstetigkeit von $A$ zu, daraus folgt die Stetigkeit der
>   NORMALkomponente von $B$ — die Tangentialkomponenten springen. Der
>   Spannungstensor braucht beide. Der Kommentar in `NodalMaxwell.cuh`
>   argumentiert nur mit $\mu_r$ = 1 und der Stetigkeit von $B$ in der
>   EXAKTEN Lösung; für das diskrete Feld gilt das so nicht. Bei einer 3 mm
>   dünnen Platte aus flachen Tetraedern ist der Sprung besonders groß.
>
> **Folge.** Die Gegenprobe ist damit nicht bestanden, aber sie ist auch nicht
> gescheitert — sie ist auf diesem Netz nicht aussagekräftig, und das ist
> belegt statt behauptet. Das Volumenintegral bleibt unverdächtigt: es wertet
> $J \times B$ dort aus, wo beide von null verschieden sind, ohne Auslöschung
> und ohne Feldspur über eine Materialgrenze. Eine unabhängige Bestätigung der
> 1,97 N steht damit weiter aus; der nächste ernsthafte Weg dahin ist die
> Netzverfeinerung (AP2), nicht ein dritter Kraftpfad.
>
> `harmonic_maxwell_body` selbst ist geprüft: `test_patch_assembly` Prüfung 6
> zeigt, dass es bei homogenem $B$ auf einer geschlossenen Hülle 1,09e-16
> relativ liefert. Der Kernel ist richtig, die Aufgabe ist ihm zu schwer.


> 2 (AP2, hängt am Netzgenerator), und M2 bleibt bei 6,1e-03 — der
> Scheingradient erreicht die Platte nicht, aber der Quellterm ist nicht
> divergenzfrei, und wie stark das $F_z$ selbst verschiebt, ist ungeprüft.
>
> **Offen aus dem ersten gelungenen Lauf**, in dieser Reihenfolge:
>
> **Gemessene $\varepsilon$-Reihe** (BDF-Netz, 1 Schritt, cuDSS 0.8, RTX 5060):
>
> | $\varepsilon$ | Residuum | max$\|A_r\|$ | max$\|A_i\|$ |
> |---|---|---|---|
> | 1e-6 | 1,26e-12 | 6,71e-05 | 0,0671 |
> | 1e-7 | 1,20e-11 | 6,71e-05 | 0,671 |
> | 1e-8 | 1,25e-10 | 6,71e-05 | 6,71 |
> | 1e-9 | 1,26e-09 | 7,08e-05 | 67,1 |
> | 1e-10 | 1,21e-08 | 1,69e-03 | 671 |
>
> max$|A_i|$ geht exakt mit $1/\varepsilon$ über vier Dekaden. Der
> Scheingradient ist damit belegt: auf Gradientenfeldern ist $K$ exakt null,
> die Blockform reduziert sich dort auf $x_r = 0$, $x_i = -b_g/(\omega C)$,
> und der Gradientenanteil von $b$ landet vollständig im Imaginärteil.
>
> **Die Abnahme aus Abschnitt 1 ist ERFÜLLT — an der richtigen Größe gemessen.**
>
> Eine frühere Fassung dieses Absatzes behauptete das Gegenteil, gestützt auf
> den Drift von max$|A_r|$. Das ist die falsche Messgröße: sie wird global
> genommen und vom Luftgebiet beherrscht, wo der Scheingradient sitzt.
> „Das Ergebnis" im Sinne der Abnahme ist $B$, und $B$ ist gegen einen
> Gradienten immun.
>
> Gemessen über die Regionenaufteilung (Schritt 8b in `fem.cu`), fünf Läufe:
>
> | $\varepsilon$ | Residuum | max$\|A_i\|$ global | max$\|A_i\|$ Platte | max$\|B_r\|$ Platte | max$\|B_i\|$ Platte |
> |---|---|---|---|---|---|
> | 1e-6 | 1,29e-12 | 0,0671 | 7,53e-06 | 0,136 T | 0,0399 T |
> | 1e-7 | 1,42e-11 | 0,671 | 7,53e-06 | 0,136 T | 0,0399 T |
> | 1e-8 | 1,77e-10 | 6,71 | 7,53e-06 | 0,136 T | 0,0399 T |
> | 1e-9 | 1,24e-09 | 67,1 | 7,53e-06 | 0,136 T | 0,0399 T |
> | 1e-10 | 1,11e-08 | 671 | 7,53e-06 | 0,136 T | 0,0399 T |
>
> Der Scheingradient wächst global um vier Dekaden und erreicht die Platte
> nicht: dort ist $\sigma$ = 3,4e7 statt $\varepsilon\cdot$3,4e7, die
> Verstärkung $1/(\omega\sigma)$ also um $1/\varepsilon$ kleiner. max$|A_i|$
> in der Platte ist über alle fünf Läufe identisch, und der Wert wird an einer
> Kante angenommen, die **nur** zu Plattenelementen gehört — über die
> Oberfläche dringt nichts ein.
>
> Bemerkenswert ist die letzte Zeile: bei $\varepsilon$ = 1e-10 ist das
> Residuum auf 1,11e-08 gestiegen und max$|A_r|$ global auf 1,48e-03
> weggelaufen, und trotzdem sind max$|A_r|$ und $B$ in der Platte unverändert.
> Auch der Genauigkeitsverlust bleibt im Luftgebiet.
>
> **Folgen.**
>
> * **AP7 kann laufen.** Der Lorentzpfad bildet $J = -\sigma j\omega A$ aus
>   $A$ selbst — aber aus $A$ in der Platte, und das ist sauber. Der
>   Quellterm als $b = \int W\cdot\mathrm{rot}\,T$ ist damit nicht die
>   Voraussetzung für die Kraft, sondern eine Verbesserung, die man später
>   machen kann.
> * **M2 bleibt formal unerfüllt** (6,1e-03 gegen 1e-4), aber die praktische
>   Folge ist auf das Luftgebiet beschränkt. Das Kriterium misst eine
>   Eigenschaft des Quellterms, nicht des Ergebnisses.
> * **$\varepsilon$ = 1e-6 ist die bessere Vorgabe als 1e-8.** Das Residuum
>   ist mit 1,29e-12 zwei Dekaden besser, die Plattengrößen sind identisch,
>   und $\sigma_{\text{Luft}}$ = 34 S/m erzeugt dort Ströme um 1 A/m² gegen
>   1,3e7 in der Spule — physikalisch belanglos. Nicht geändert, weil die
>   Vorgabe eine Entscheidung ist und keine Messung.
>
> **Folge für die Entwurfsentscheidung.** Der Satz in Abschnitt 1, die
> $\sigma$-Regularisierung entferne beide Nullraumanteile „auf einen Schlag,
> kostet drei Zeilen", gilt nur unter einer Bedingung, die dort nicht steht:
> $b$ muss hinreichend divergenzfrei sein. Der Verstärkungsfaktor ist
> $1/\varepsilon$, angegriffen wird ‖$G^T b$‖/‖$b$‖, und das steht bei
> 6,1e-03 statt 1e-4. M2 ist damit keine Kosmetik, sondern die Vorbedingung
> des ganzen Ansatzes: bei 1e-4 wäre $A_i$ um Faktor 60 kleiner, das kaufte
> knapp zwei Dekaden $\varepsilon$ und machte die Abnahme erreichbar.
>
> **Woher die 6,1e-03 kommen — entschieden durch `test_patch_assembly`.**
> Der Test läuft über ein Netz aus Kuhn-Tetraedern, sammelt die Koeffizienten
> aus den globalen Freiheitsgraden und nutzt `TopologyExtractor` und
> `NedelecDispatch` selbst, nicht Nachbauten.
>
> | N | Tets | h | J_s überall | Faktor | J_s im Kreisring | Faktor |
> |---|---|---|---|---|---|---|
> | 4 | 384 | 0,25 | 6,90e-06 | — | 1,58e-01 | — |
> | 8 | 3072 | 0,125 | 1,04e-06 | 6,61 | 1,22e-01 | 1,30 |
> | 16 | 24576 | 0,0625 | 1,50e-07 | 6,94 | 9,42e-02 | 1,30 |
>
> Drei Aussagen daraus:
>
> * **Die Assemblierung ist exakt.** Konstantes $J_s$ ergibt
>   ‖$G^T b$‖/‖$b$‖ = 1,25e-16 an inneren Knoten. Kantennummerierung,
>   Orientierungsvorzeichen, Streuung und beide Piola-Transformationen sind
>   zusammen richtig. Ein Vorzeichenfehler ist ausgeschlossen.
> * **Der Quadraturfehler konvergiert**, mit etwa $h^{2,8}$.
> * **Mit Materialgrenze konvergiert er nicht** — Faktor 1,30 je Halbierung,
>   also $h^{0,38}$. Auf einem unverzerrten Netz sogar 1,33 und 1,13, dort
>   stagniert es fast. Es ist also nicht die Elementverzerrung.
>
> Der Grund ist die elementweise Materialzuordnung: `assign_materials_by_`
> `component_id` steckt ein Element ganz in die Spule oder gar nicht, die
> Ersatzfläche ist deshalb gezackt, ihre Normalen stehen nicht senkrecht auf
> $\hat\varphi$, und damit ist der diskrete Spulenstrom **nicht geschlossen**.
> Das ist eine Eigenschaft der Quellterm-Vorgabe, keine der Netzweite.
>
> **Damit ist ein Weg widerlegt, der oben noch als Möglichkeit stand:** ein
> feineres Netz bringt die 6,1e-03 nicht auf 1e-4. 64-mal so viele Elemente
> gewinnen Faktor 1,7.
>
> **Am echten Netz bestätigt, und zwar schärfer.** Der Lauf mit `--refine=1`
> auf `team28_113.vtk`:
>
> ```
> ohne Verfeinerung   ||G^T b||/||b|| = 6,139e-03
> mit  --refine=1     ||G^T b||/||b|| = 1,098e-02    -> NICHT divergenzfrei
> ```
>
> **Es wird schlechter, um Faktor 1,79.** Und hier ist die Geometrie
> nachweislich unverändert: die Verfeinerung unterteilt jede Facette der
> Spulenoberfläche in vier koplanare Dreiecke. Facettierung, Normalen und
> Winkelfehler gegen $\hat\varphi$ sind identisch, nur die Feldauflösung
> steigt. Wäre die Nichtdivergenzfreiheit ein Quadratur- oder
> Auflösungsfehler, müsste sie fallen.
>
> Warum sie *steigt*, ist offen. Eine Abschätzung über die Skalierung
> ($b_e \sim h^2$, Kantenzahl $\sim h^{-3}$, Randknotenzahl $\sim h^{-2}$)
> liefert einen Abfall um etwa 0,7; der gemessene Anstieg um 1,79 liegt nicht
> in deren Fehlergrenzen. Da fehlt ein Stück des Mechanismus.

>
> Es bleiben drei, in aufsteigendem Aufwand:
>
> * $\varepsilon$ auf 1e-6 bis 1e-8 beschränken und die Abnahme entsprechend
>   umschreiben. Billig, aber sie prüft dann weniger als beabsichtigt, und der
>   Scheingradient bleibt in $A_i$ stehen.
> * Den Gradientenanteil aus $b$ herausprojizieren, bevor gelöst wird. Genau
>   das tat `TreeCotreeGauge`, und genau das sollte dieser Pfad einsparen.
>   Wäre die Rücknahme der zentralen Vereinfachung — aber `--no-gauge`
>   existiert bereits, der Apparat ist also nicht weg.
> * Den Quellterm divergenzfrei **vorgeben** statt ihn punktweise zu
>   integrieren: $b = \int W\cdot\mathrm{rot}\,T$ mit einem Kantenfeld $T$.
>   Dann folgt $G^T b = 0$ aus der diskreten exakten Sequenz
>   (grad → rot → div) und nicht aus Quadraturglück — unabhängig von Netz und
>   Elementform. Das ist der übliche Weg in Kantenelement-Wirbelstromcodes und
>   der einzige der drei, der M2 wirklich erfüllt.
>
> 1. ~~Der Imaginärteil trägt einen Scheingradienten.~~ Belegt, siehe Tabelle
>    oben. **B ist nicht betroffen** (rot grad = 0), der Lorentz-Kraftpfad
>    schon, weil er $J = -\sigma j\omega A$ aus $A$ selbst bildet. Das ist als
>    erstes zu prüfen, wenn die Kraft angeschlossen wird.
> 2. ~~$\varepsilon$ steht als Literal im Aufruf.~~ **Behoben:**
>    `--fem-direct-eps=X`, Standard 1e-8 unverändert. Reihe gemessen, siehe
>    oben.
> 3. ~~Die rechte Seite ist eine Zeitprobe, kein Zeiger.~~ **Behoben:** der
>    Quellterm wird jetzt bei $t = T/4$ ausgewertet, also im Maximum. Das ist
>    die Konvention $i(t) = I\cos(\omega t)$ mit reeller rechter Seite; für die
>    zeitgemittelte Kraft gleichgültig, weil sie unter einer gemeinsamen
>    Phasendrehung invariant ist. ‖b‖ muss dadurch von 819 auf etwa
>    $819/0{,}0314 \approx 2{,}6\text{e}{+}04$ steigen.
> 4. M2 ist nicht erfüllt: ‖$G^T b$‖/‖$b$‖ = 6,1e-03 gegen verlangte 1e-4.
>    Auf dem Tetraedernetz sind die Spulen mit 251 und 517 Elementen
>    aufgelöst, gegen 1400 und 1200 Hexaeder vorher. Ob das Faktor 60
>    hergibt, entscheidet ein feineres Netz: fällt die Zahl mit der
>    Elementgröße, ist es Diskretisierung; bleibt sie, sitzt in der
>    Tetraeder-Assemblierung ein Fehler — und der Patchtest aus AP3 ist auf
>    einem reinen Tetraedernetz noch nicht gelaufen.
>
> Der Maxwell-Spannungstensor bleibt absichtlich hexaederonly und bricht auf
> einem gemischten Netz mit Begründung ab. Kraftpfad ist das
> Lorentz-Volumenintegral, das keine Feldspur auf der Luftseite braucht.

## 0 Das gelieferte Netz

`team28_simple_air.bdf` (HyperMesh 5.0), gemessen mit `tools/import_bdf.py`:

| | |
|---|---|
| Elemente | 3692 CTETRA, **keine** CPENTA |
| Knoten | 766 |
| Kanten (= Freiheitsgrade) | 4709, davon 756 auf dem Rand |
| Unbekannte nach Dirichlet | 3953, in reeller 2n-Form 7906 |
| Flächen | 7132 innen, 504 am Rand, keine mit mehr als zwei Nachbarn |
| Formgüte (1 = regelmäßig) | Median 0,342, 5 %-Quantil 0,125, min 0,046 |
| PID | 1 coil1 (517), 2 coil2 (251), 3 plate (259), 4 air (2665) |

Zwei Punkte, die den Plan berühren:

**Die Geometrie ist nicht die von `team28.vtk`.** Gemessen gegen
`tools/make_coarse_team28.py`:

| Bauteil | `team28.vtk` | `team28_simple_air.bdf` |
|---|---|---|
| Platte | z 4,3…6,3 (2,0 mm), r ≤ 63 | z 3,8…6,8 (3,0 mm), r ≤ 65 |
| Innenspule | z −49,4…−2,6, r 30…51 | z −52…0, r 27…55 |
| Außenspule | z −49,4…−2,6, r 84…91 | z −52…0, r 80…95 |

Jede Abmessung ist nach außen versetzt. Solange das so bleibt, vergleichen M3
und M4 zwei verschiedene Aufgaben, und der Abgleich mit den TEAM-28-Messwerten
ist nicht möglich.

**Der Luftkasten ist knapp.** ±165 mm gegen ±95 mm Leiterausdehnung ist Faktor
1,74; nach oben und unten bleiben nur rund 21 mm. Die Abschneidestudie M5
(2×, 3×, 5×) fängt damit unterhalb ihres kleinsten Werts an.

Für die Inbetriebnahme (M0 bis M2) ist das Netz gut: 3953 Unbekannte lösen sich
direkt in Sekundenbruchteilen. Für eine belastbare Kraft ist es zu grob — die
Platte hat 259 Tetraeder.

Vorhaben: eine dritte Option neben dem transienten und dem harmonischen
BEM-FEM-Pfad. Das Außenraumproblem wird nicht mehr durch BEM erledigt, sondern
durch ein mitvernetztes Luftgebiet mit abgeschnittenem Rand, und das entstehende
dünn besetzte System wird direkt zerlegt statt iterativ gelöst.

Zweck in zwei Stufen: **zuerst Gegenprobe** zum BEM-FEM-Ergebnis, danach
möglicher Produktivpfad.

Getroffene Vorentscheidungen:

* Netz kommt von außen (HyperMesh v5, Export als Nastran BDF), kein eigener
  Generator.
* Nur `CPENTA` mit 6 Knoten. Damit ist genau **ein** neues Element zu bauen.
* cuDSS ist als Abhängigkeit zulässig.

---

## 1 Warum der Direktlöser die Sache vereinfacht

Der Befund aus `docs/reports/convergence-findings.tex` ist, dass $\kappa_2 \approx 10^6$ ein Sockel
aus der Struktur des Problems ist: Curl-Curl-Nullraum plus Leiter/Luft-Übergang.
Für ein Krylov-Verfahren ist das der Engpass. Für eine Zerlegung ist es
gleichgültig.

Daraus folgt die zentrale Entwurfsentscheidung dieses Pfades. Der Nullraum von
$K$ — Gradienten in den Gebieten mit $\sigma = 0$, dazu die harmonischen Moden
bei $b_1 \neq 0$ — wird **nicht** mit Baum-Kobaum-Eichung und
Kohomologie-Projektion behandelt, sondern durch eine kleine Leitfähigkeit in der
Luft:

$$\sigma_{\text{Luft}} = 10^{-8}\,\sigma_{\text{Al}}$$

Das entfernt beide Nullraumanteile auf einen Schlag, kostet drei Zeilen und
verschlechtert die Kondition um Größenordnungen, die einer Zerlegung nichts
ausmachen. Der gesamte Apparat aus AMS, `TreeCotreeGauge` und `CohomologyGauge`
entfällt auf diesem Pfad. Er bleibt als unabhängige Gegenprobe verfügbar
(`--no-gauge` existiert bereits).

**Abnahme:** das Ergebnis muss zwischen $\varepsilon = 10^{-6}$ und $10^{-10}$
relativ stehenbleiben. Tut es das nicht, ist die Regularisierung keine
Regularisierung, sondern Physik.

Wichtig ist dabei: Eichung und Randbedingung müssen **in die CSR-Werte hinein**.
Der Operator behandelt die Eichung bisher als Projektion
$P A P + (I-P)$ — das ist für GMRES richtig und für cuDSS wertlos, weil dort
eine Matrix übergeben wird, keine `apply()`-Funktion.

---

## 2 Arbeitspakete Phase 1 (Gegenprobe)

### AP1 — BDF-Import

Kein BDF-Parser in C++. Das Format hat zu viele Fallen für den Aufwand:
Small/Large/Free Field nebeneinander, Fortsetzungszeilen, und vor allem das
Feld `CP` auf `GRID`, mit dem Knotenkoordinaten in einem lokalen System
(`CORD2R`, `CORD2C`) stehen können. Wer das übersieht, bekommt ein Netz, das an
einer Stelle verdreht ist, ohne dass irgendetwas abstürzt.

Werkzeug: `tools/import_bdf.py` auf Basis von **pyNastran** (löst die
Koordinatensysteme korrekt auf) oder meshio, falls sicher alles im Basissystem
steht. Ausgabe ist Legacy-VTK ASCII, weil das gemischte Zelltypen kann
(10 Tetraeder, 12 Hexaeder, 13 Penta, 14 Pyramide) und ParaView es direkt
anzeigt.

Der Konverter prüft und bricht laut ab bei:

| Prüfung | Warum |
|---|---|
| Legacy-Layout **vor** VTK 5.1 | Neuere meshio-Versionen schreiben `CELLS` mit `OFFSETS`/`CONNECTIVITY`. `VtkReader` erwartet `CELLS n_cells total_ints` mit Knotenzahl je Zelle. |
| genau **ein** `SCALARS`-Array in `CELL_DATA` | `VtkReader.cpp:151` nimmt das erste und ignoriert den Namen. Mehrere Zellarrays heißt stillschweigend falsche Materialien. |
| `CPENTA` mit 15 Knoten | Löser ist erster Ordnung. Mittenknoten wegzuwerfen ist bei gekrümmten Rändern (r = 30/51/63/84/91) eine Geometrieänderung. |
| Knotenreihenfolge, positive Jacobi-Determinante | Für gelesene Netze prüft das bisher niemand. |
| Konformität: keine doppelten, keine hängenden Knoten | Erzeugt keinen Fehler, sondern ein falsches Feld. |
| `PID` → `component_id` | Zuordnung 1/2/3/0 gehört in eine sichtbare Tabelle, nicht in eine Konvention im Kopf. |
| Materialgebiete gegen die vermessene Geometrie | Radien 30/51/63/84/91, z-Schnitte −49,4/−2,6/4,3/6,3, Einheit mm. |

`CTRIA3`/`CQUAD4` tragen keine Volumen-Freiheitsgrade, werden aber **getrennt
mitgeführt**: wenn HyperMesh damit Außenfläche und Materialgrenzen markiert,
ist das die zuverlässigste Quelle für den Dirichlet-Rand — deutlich besser, als
beides aus der Topologie abzuleiten.

### AP2 — Netzanforderung

Ein reines Prismennetz ist ein extrudiertes Netz: Dreiecksvernetzung in x-y,
Schichtung in z. Das passt zu TEAM 28 gut. Die x-y-Triangulierung muss die
Radien 30/51/63/84/91 als Kantenzüge auflösen, die z-Schichtung die Grenzen
−49,4 / −2,6 / 4,3 / 6,3. Der schmale Luftspalt ist eine z-Schicht mit mehr
Unterteilungen.

Der Luftkasten wird zum Zylinder (Sweep in z). Für die Abschneidestudie werden
drei Netze mit 2-, 3- und 5-facher Leiterausdehnung gebraucht. `--refine=N`
hilft hier nicht, `MeshRefinement` ist hexaederonly — die Netzreihe muss aus
HyperMesh kommen.

**Nachgetragen, gemessen: die Extrusionsrichtung muss je Element konstant sein.**
Nédélec-Elemente niedrigster Ordnung reproduzieren ein konstantes $B$ nur auf
affin abgebildeten Elementen exakt. Beim Prisma heißt affin: die Deckfläche ist
eine reine Parallelverschiebung der Bodenfläche, also
$x_3 - x_0 = x_4 - x_1 = x_5 - x_2$. Ist das verletzt, bleibt ein Fehler, der
**linear** mit der Verzerrungsamplitude $t$ geht:

```
t = 1.0000   Fehler = 1.339e-01
t = 0.5000   Fehler = 6.568e-02   Verhaeltnis = 2.04
t = 0.2500   Fehler = 3.249e-02   Verhaeltnis = 2.02
t = 0.1250   Fehler = 1.615e-02   Verhaeltnis = 2.01
t = 0.0625   Fehler = 8.051e-03   Verhaeltnis = 2.01
t = 0        Fehler = 2.220e-16
```

Das ist keine Eigenheit des neuen Elements, sondern der Familie. Das vorhandene
Hexaederelement verhält sich unter demselben Test identisch (Quader 1,1e-16;
Parallelepiped 3,3e-16; verzerrt 1,0e-01). Es ist also kein Fehler und kein
Grund, das Element zu ändern — aber es ist ein Grund, im Netzgenerator auf
schichtweise konstante Extrusion zu achten. `tools/import_bdf.py` misst den
Defekt und gibt Median, 95 %-Quantil und Maximum aus.

### AP3 — Prismenelement

Das einzige neue Element. Nédélec erster Art auf dem Prisma, 9 Kanten:

* 6 horizontale Kanten: Whitney-1-Form des Dreiecks, linear in $\zeta$
* 3 vertikale Kanten: baryzentrische Koordinate des Dreiecks, konstant in
  $\zeta$, Richtung $e_\zeta$

Kovariante Piola-Transformation und Quadraturschleife werden aus
`NedelecHexahedron.cuh` übernommen; ausgetauscht wird nur die Auswertung von
Basis und Rotation. Das Prisma ist im Allgemeinen nicht affin abgebildet, also
Quadratur, keine geschlossene Formel.

Hexaeder und Prismen sind konform mischbar — die gemeinsame Fläche ist
beidseitig ein Viereck, und die Tangentialspuren passen. `team28.vtk` bleibt
damit lauffähig.

**Die Datenstrukturen passen bereits, ohne eine einzige Formänderung:**

| Struktur | Fasst | Prisma braucht |
|---|---|---|
| `struct Hexahedron { int v[8]; }` | 8 Knoten | 6 |
| `hex_edges`, `hex_edge_signs`, Schrittweite 12 | 12 Kanten | 9 |
| `SOLVER_LOCAL_MATRIX_SIZE = 144` (12×12) | 12 Freiheitsgrade | 9 |

Also: ein zusätzliches Array `elem_type` je Element, ungenutzte Kantenplätze auf
$-1$, ungenutzte Zeilen und Spalten der Elementmatrix auf null. Die
Assemblierung verarbeitet nur Einträge mit Index $\geq 0$. `build_csr_structure`,
`assemble_csr_matrix_gpu`, das gesamte CSR und cuDSS bleiben **unangetastet**.

Elemente nach Typ sortiert ablegen und pro Typ einen eigenen Kernelstart über
den zugehörigen Bereich — das vermeidet Divergenz und hält
`assemble_local_matrices_kernel` einfach.

**Abnahmetests, jeder als eigenes CTest-Ziel mit der Marke `host`:**

1. Gegenstück zu `test_edge_sign_convention` für das Prisma. Vorzeichenfehler in
   der Kantenorientierung sind die wahrscheinlichste Fehlerursache und
   produzieren keinen Absturz, sondern eine plausibel aussehende falsche Kraft.
2. Gegenstück zu `test_curl_gradient_nullspace`: $K \cdot G\varphi = 0$ exakt.
3. **Patchtest auf einem gemischten Hex-Penta-Netz**: ein homogenes Feld $B$ muss
   exakt reproduziert werden. Das ist der Test, der Konformitätsfehler an der
   Typgrenze wirklich findet.

### AP4 — Rand und Regularisierung, materialisiert

Dirichlet $A \times n = 0$ auf allen Kanten der Außenfläche, umgesetzt als Zeile
und Spalte streichen, 1 auf die Diagonale. Kantensatz vorzugsweise aus den
Schalenelementen des BDF (AP1), ersatzweise aus `extract_boundary_faces` — das
liefert jetzt die Außenfläche des Luftkastens statt der BEM-Kopplungsfläche.

Dazu $\sigma_{\text{Luft}}$ nach Abschnitt 1 über `MaterialProperties.kappa`.

**Fehler, der bisher nicht auffallen konnte:** `d_materials` wird in `fem.cu` nur
für die Hexaeder von Platte und Spulen beschrieben. `DeviceArray` legt seinen
Speicher per `cudaMallocManaged` **ohne Konstruktoraufruf** an. Ohne Luftgebiet
war das folgenlos, mit Luftgebiet stünde dort Müll. Vor jeder Zuweisung das
gesamte Array explizit mit `air.get_properties()` füllen.

### AP5 — Systemform

Start mit der reellen $2n$-Form und vollem CSR als `CUDSS_MTYPE_GENERAL`, also
LU. Nicht weil es billig ist, sondern weil es keine neue Assemblierung braucht:

$$H = \begin{bmatrix} K & -\omega C \\ \omega C & K \end{bmatrix}$$

$C$ wird wie im harmonischen Pfad bereits vorhanden aus zwei Assemblierungen
gewonnen (`assemble_at(1e30)` für $K$, `assemble_at(1/\omega)` für $K + \omega C$,
Differenz eintragsweise auf den CSR-Werten bei identischem Muster).

Zwei Verbesserungen, beide **Phase 2**, nicht jetzt:

* Zeilentausch auf die real symmetrische indefinite Form
  $\bigl[\begin{smallmatrix} K & -\omega C \\ -\omega C & -K\end{smallmatrix}\bigr]$
  mit rechter Seite $[b_r;\, -b_i]$, dann `CUDSS_MTYPE_SYMMETRIC` und nur ein
  Dreieck: halbe Faktorisierungsarbeit, aber eine neue Assemblierung nötig.
* Die komplexe $n$-Form ($K + j\omega C$, komplex-symmetrisch, `CUDA_C_64F`):
  nochmals etwa halber Aufwand.

Für M0 bis M3 zählt Korrektheit, nicht Faktorisierungskosten.

### AP6 — cuDSS-Anbindung

`find_package(cudss REQUIRED)`, Ziel `cudss`. Ablauf: `cudssCreate` →
`cudssConfigCreate` → `cudssMatrixCreateCsr` direkt auf die vorhandenen
Unified-Memory-Zeiger → Phasen `ANALYSIS`, `FACTORIZATION`, `SOLVE`.

Zwei Dinge, die man vorher wissen sollte:

* `CUDSS_CONFIG_HYBRID_MODE` gibt es für den Fall, dass der Faktor nicht in den
  VRAM passt. Bei einem ernsthaften Luftkasten ist das der Normalfall, nicht die
  Ausnahme.
* **Es gibt kein cuDSS für den CPU-Pfad.** `--fem-direct` muss `--cpu` mit klarer
  Meldung ablehnen. Das kostet die GPU-Freiheit, auf die das Projekt sonst
  achtet, und ist der Preis dieser Entscheidung. Ein eigener LDLᵀ als
  Host-Rückfallebene bleibt möglich, gehört aber nicht in Phase 1.

### AP7 — Kraftpfad

`Mesh::QuadFace` kennt nur Vierecke, Randflächen eines Prismennetzes sind
teilweise Dreiecke. Damit ist das **Lorentz-Volumenintegral** in der Platte
(`harmonic_lorentz`, bereits vorhanden) nicht mehr nur der sauberere, sondern der
einzig gangbare Weg. Der Maxwell-Spannungstensor über
`extract_component_surface_faces` entfällt auf diesem Netz — er müsste ohnehin
die Luftseite auswerten und wäre eine stille Fehlerquelle.

`harmonic_phase_sample` und die Mittelung über 8 Phasen bleiben unverändert.
`harmonic_lorentz` braucht die Typkennung aus AP3.

### AP8 — Verdrahtung

* `--fem-direct`, impliziert `nobem`, unverträglich mit `--cpu`,
  `--harmonic-checks` und `--pc-study` (dort läuft kein iterativer Löser mehr).
* Hilfetext in `Backend.cu`, Option in `Backend.cuh` nach dem Muster von
  `g_harmonic_*`.
* `VtkExporter` und `TopologyExtractor` auf die Typkennung ziehen.
* `VtkReader`: Zelltypen 10/12/13/14 annehmen, alles andere ablehnen;
  `SCALARS`-Namen auf `component_id` prüfen statt das erste Array zu nehmen.

---

## 3 Meilensteine

| M | Inhalt | Kriterium |
|---|---|---|
| M0 | cuDSS gegen die dichte Zerlegung (`--direct`), Kleinstnetz ohne Luft | rel. Residuum $\leq 10^{-13}$ |
| M1 | Prismenelement: Vorzeichen, Nullraum, Patchtest | alle drei Tests grün |
| M2 | BDF-Netz mit Luft eingelesen, Quellterm geprüft | $\|G^T b\|/\|b\| \approx 10^{-4}$ wie am Originalnetz |
| M3 | B-Feld FEM-direkt gegen BEM-FEM harmonisch | wenige Prozent an Sondenpunkten |
| M4 | Zeitgemittelte Kraft und Schwebehöhe | gegen BEM-FEM **und** die TEAM-28-Messwerte |
| M5 | Abschneidestudie über die Kastengröße (2×, 3×, 5×) | Konvergenzreihe, nicht ein Einzelwert |

**Zu M3:** der Vergleich läuft auf zwei verschiedenen Netzen — BEM-FEM auf
`team28.vtk` (Hexaeder, Leiter allein), FEM-direkt auf dem BDF-Netz (Prismen,
mit Luft). Der BEM-Pfad ist auf einem Prismennetz strukturell ausgeschlossen,
Sauter-Schwab arbeitet auf Vierecks-Panels. Verglichen werden also
physikalische Größen, nicht Freiheitsgrade. Das BDF-Netz muss die
Leitergeometrie dafür exakt treffen.

---

## 4 Phase 2 (Produktivpfad)

Der stärkste Hebel steckt nicht im harmonischen Fall, sondern im transienten:
bei ruhender Platte und konstantem $\Delta t$ ist $K + C/\Delta t$ über alle
Zeitschritte **dieselbe** Matrix. Eine Faktorisierung, dann 1000 billige
Rücksubstitutionen statt 1000 GMRES-Läufen bei $\kappa_2 \approx 10^6$.

Dazu die symmetrische und die komplexe Form aus AP5, `HYBRID_MODE`, und die
Netzreihe für die Konvergenz.

**Der Punkt, der Phase 2 begrenzt:** ein körperangepasstes Luftnetz und
`enable_free_motion` vertragen sich nicht. Sobald `apply_newtons_law` die Platte
verschiebt, werden die Luftelemente im schmalen Spalt verzerrt und irgendwann
invertiert. Für die harmonische Aufgabe ist das gleichgültig, dort ruht die
Platte ohnehin: die Schwebehöhe ist dann die Nullstelle von $F(z) - mg$ über
einen Sweep, ein Netz je Höhe. Ein transienter Produktivpfad mit bewegter Platte
bräuchte Neuvernetzung oder eine Gleitschicht — ein eigenes Projekt.
