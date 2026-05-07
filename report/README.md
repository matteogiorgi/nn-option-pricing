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

Final figures are loaded from `../results/final/figures/`.
If the final results have not been generated yet, the PDF remains compilable and
shows placeholders instead of figures.
