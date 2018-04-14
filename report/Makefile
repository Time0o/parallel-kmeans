AUX_DIR=aux
PDF_DIR=pdf

define run_pdflatex
	-@pdflatex --halt-on-error -output-directory $(AUX_DIR) $< \
	1> /dev/null 2> $(AUX_DIR)/err.log
endef

report.pdf: report.tex
	$(call run_pdflatex)
	$(call run_pdflatex)
	-@mv $(AUX_DIR)/$@ $(PDF_DIR)

.PHONY: clean
clean:
	-@rm $(AUX_DIR)/* || true
	-@rm $(PDF_DIR)/* || true
