# Dokumentation

Alle LaTeX-Dokumente lassen sich mit

    make -C docs            # alle, PDFs landen in docs/_build/
    make -C docs clean

übersetzen (benötigt `latexmk` oder `pdflatex`; für die deutsche Silbentrennung
das Paket `babel-german`, ohne es bleibt die Trennung englisch).

## Überblick

| Dokument | Inhalt |
|---|---|
| [architecture.md](architecture.md) | Module, Datenfluss, wo was im Quelltext liegt |
| **Theorie** | |
| [theory/solver-overview.tex](theory/solver-overview.tex) | Überblick über das TEAM-28-Modell und die numerische Pipeline |
| [theory/numerics-and-gpu.tex](theory/numerics-and-gpu.tex) | Theorie, Numerik und GPU-Beschleunigung |
| [theory/formulation.tex](theory/formulation.tex) | BEM-FEM-Kopplung für Wirbelstromprobleme: Formeln |
| **Berichte** (Stand zum Zeitpunkt der Messung) | |
| [reports/convergence-findings.tex](reports/convergence-findings.tex) | Konvergenz des gekoppelten Lösers, geprüfte Hypothesen, behobene Fehler |
| [reports/harmonic-fem.tex](reports/harmonic-fem.tex) | Harmonische FEM für TEAM 28: Formulierung, Löser, Befunde |
| **Entwurf und Arbeitsnotizen** | |
| [design/fem-direct-plan.md](design/fem-direct-plan.md) | Plan und Messreihen für den FEM-direkt-Pfad |
| [design/bem-block-preconditioning.md](design/bem-block-preconditioning.md) | Vorkonditionierung des BEM-Blocks |
| [design/nan-and-thresholds.md](design/nan-and-thresholds.md) | Vergleiche, die NaN nicht sehen: Muster und Durchsicht des Codes |

Die Berichte und Arbeitsnotizen halten fest, was zu einem bestimmten Zeitpunkt
gemessen wurde. Zahlen darin gelten für den damaligen Code und das damalige
Netz; der Quelltext ist maßgeblich. Sie sind in der Ich-Form eines
Arbeitsprotokolls geschrieben und enthalten bewusst auch Irrtümer und ihre
Korrektur.
