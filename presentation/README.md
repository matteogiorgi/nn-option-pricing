# Project Presentation

This directory contains the Beamer slides for the oral presentation of the
project.

The file [`speaker_notes.md`](speaker_notes.md) contains the proposed division
of the slides among the four speakers, together with the corresponding report
sections and short speaking notes for each slide.

Build the presentation with XeLaTeX:

```bash
cd presentation
latexmk -xelatex main.tex
```

Clean temporary LaTeX artifacts with:

```bash
cd presentation
latexmk -c main.tex
```

The slides use `fontspec` and Cascadia Code, so they should be compiled with
XeLaTeX rather than pdfLaTeX.
