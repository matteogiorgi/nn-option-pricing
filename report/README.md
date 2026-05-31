# LaTeX Report

The main report file is `main.tex`.

Recommended compilation command:

```bash
cd report
latexmk -xelatex main.tex
```

Temporary file cleanup:

```bash
cd report
latexmk -c main.tex
```

The report uses `fontspec` and Cascadia Code for monospaced code blocks, so it
must be compiled with XeLaTeX rather than pdfLaTeX.

Final diagnostic figures are loaded from `../results/final/figures/`.
Report-specific comparison figures are stored in `figures/`.
If any expected figure is missing, the PDF remains compilable and shows a
placeholder instead of the missing image.

The report-specific figures can be regenerated from saved JSON results from the
repository root with:

```bash
.venv/bin/python scripts/generate_report_figures.py
```
