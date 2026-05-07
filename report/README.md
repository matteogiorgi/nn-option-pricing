# LaTeX Report

The main report file is `main.tex`.

Recommended compilation command:

```bash
cd report
latexmk -pdf main.tex
```

Temporary file cleanup:

```bash
cd report
latexmk -c main.tex
```

Figures are loaded from `../outputs/figures/`.
If the experiments have not been run yet, the PDF remains compilable and shows placeholders instead of figures.
