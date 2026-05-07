# Relazione LaTeX

La relazione principale e' `main.tex`.

Compilazione consigliata:

```bash
cd report
latexmk -pdf main.tex
```

Pulizia dei file temporanei:

```bash
cd report
latexmk -c main.tex
```

Le figure vengono lette da `../outputs/figures/`.
Se gli esperimenti non sono ancora stati eseguiti, il PDF resta compilabile e mostra placeholder al posto delle figure.

