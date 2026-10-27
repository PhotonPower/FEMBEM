# LaTeX-Dokumente uebersetzen. PDFs und Nebenprodukte landen in _build/,
# das Quellverzeichnis bleibt sauber.
#
#   make            alle Dokumente
#   make clean      _build/ loeschen
#
# latexmk erledigt die mehrfachen Durchlaeufe (Inhaltsverzeichnis,
# Querverweise) selbst; ohne latexmk wird dreimal pdflatex aufgerufen.

DOCS := theory/solver-overview theory/numerics-and-gpu theory/formulation \
        reports/convergence-findings reports/harmonic-fem
PDFS := $(addsuffix .pdf,$(addprefix _build/,$(DOCS)))

all: $(PDFS)

_build/%.pdf: %.tex
	@mkdir -p $(dir $@)
	@if command -v latexmk >/dev/null 2>&1; then \
	  latexmk -pdf -interaction=nonstopmode -halt-on-error \
	          -outdir=$(dir $@) $<; \
	else \
	  for i in 1 2 3; do \
	    pdflatex -interaction=nonstopmode -halt-on-error \
	             -output-directory=$(dir $@) $< || exit 1; \
	  done; \
	fi

clean:
	rm -rf _build

.PHONY: all clean
