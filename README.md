# Neural Network Option Pricing

Project for the Machine Learning for Finance course.

The goal is to test whether a feed-forward neural network can accurately
approximate the Black-Scholes pricing function for European call options.

## Core Question

Can a feed-forward neural network approximate the Black-Scholes European call
pricing function with high accuracy, and how does it compare with analytical
Black-Scholes prices and Monte Carlo estimates?

## Project Structure

```text
.
├── docs/
│   └── source/
├── presentation/
│   ├── main.pdf
│   └── main.tex
├── report/
│   ├── main.pdf
│   ├── main.tex
│   └── sections/
├── results/
│   └── final/
├── scripts/
│   └── run_experiment.py
├── src/
│   └── nn_option_pricing/
├── tests/
├── ISTRUZIONI.md
├── README.md
├── pyproject.toml
├── requirements-docs.txt
└── requirements.txt
```

Generated experiment artifacts are written to `data/` and `outputs/`.
These directories are ignored by Git because they can be reproduced from the
code and configuration. The report and presentation PDFs are tracked because
they are final project deliverables. Selected final metrics and figures are
tracked in `results/final/`.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

Run the complete experiment:

```bash
python scripts/run_experiment.py
```

Use `--data-dir` and `--output-dir` to keep different runs separate:

```bash
python scripts/run_experiment.py \
  --data-dir data/intermediate \
  --output-dir outputs/intermediate
```

The baseline uses the primitive Black-Scholes inputs
`(s0, k, t, r, sigma)` and ReLU activations. Experimental variants can be
selected from the command line:

```bash
python scripts/run_experiment.py \
  --feature-set with_moneyness \
  --activation silu \
  --data-dir data/experiments/moneyness_silu \
  --output-dir outputs/experiments/moneyness_silu
```

For a fast smoke test:

```bash
python scripts/run_experiment.py \
  --n-samples 10000 \
  --max-epochs 40 \
  --batch-size 1024 \
  --mc-n-paths 5000 \
  --mc-evaluation-samples 128
```

The run generates:

- synthetic option data in `data/`;
- trained model checkpoint in `outputs/`;
- experiment configuration snapshot in `outputs/experiment_config.json`;
- metrics in `outputs/metrics/`;
- figures in `outputs/figures/`.

Benchmark pricing runtimes:

```bash
python scripts/benchmark_runtime.py \
  --feature-set with_moneyness \
  --activation silu \
  --output-dir outputs/runtime_benchmark
```

## Testing

Run the automated test suite:

```bash
pytest
```

The tests cover analytical Black-Scholes pricing, synthetic dataset generation,
Monte Carlo pricing, and a small end-to-end pipeline smoke test.

## Documentation

Build the Sphinx documentation:

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs/source docs/build/html
```

For a clean environment, install the runtime requirements and the local package
before building the API documentation:

```bash
pip install -r requirements.txt
pip install -r requirements-docs.txt
pip install -e . --no-build-isolation
```

Open `docs/build/html/index.html` in a browser to read the generated
documentation.

## Report and Presentation

The technical report is written in LaTeX:

```bash
cd report
latexmk -xelatex main.tex
```

The oral presentation is written with Beamer:

```bash
cd presentation
latexmk -xelatex main.tex
```

Both documents use XeLaTeX.



## ToDo / Possible Extensions

Questi punti sono promemoria di lavoro per eventuali estensioni. Il progetto
principale rimane focalizzato sulla domanda: una rete neurale feed-forward
riesce ad approssimare accuratamente la funzione di pricing Black-Scholes?

### Core Extensions

1. **Moneyness as engineered feature**
   Valutare se includere `moneyness = s0 / k` anche tra gli input della rete,
   oltre che usarla come variabile diagnostica per l'analisi degli errori.
   La variabile non aggiunge informazione nuova rispetto a `s0` e `k`, ma rende
   esplicita una relazione finanziariamente rilevante.

2. **Alternative activation functions**
   Confrontare ReLU con una o due funzioni di attivazione alternative, ad
   esempio `Tanh`, `LeakyReLU`, `SiLU` o `GELU`, mantenendo invariata la pipeline
   sperimentale.

3. **Runtime comparison**
   Confrontare i tempi di pricing della formula Black-Scholes analitica, della
   neural network e del benchmark Monte Carlo. Il confronto dovrebbe distinguere
   tra training time e inference time: la rete ha un costo iniziale di
   addestramento, ma una volta addestrata puo' prezzare molte opzioni tramite
   una semplice forward pass. Monte Carlo, invece, richiede simulazioni per ogni
   nuova valutazione.

### Additional Machine Learning Baselines

4. **SVR benchmark**
   Aggiungere una Support Vector Regression come baseline classica di machine
   learning. Poiche' SVR puo' essere computazionalmente costosa su dataset
   grandi, l'esperimento potrebbe essere eseguito su un sottoinsieme del dataset
   o con una configurazione dedicata.

### Future / Robustness Work

5. **Noisy Black-Scholes targets**
   Studiare un dataset sintetico in cui al prezzo Black-Scholes viene aggiunto
   rumore controllato, per analizzare la robustezza della rete al variare del
   signal-to-noise ratio. Questo esperimento cambia leggermente il problema: da
   approssimazione di una funzione deterministica a regressione con target
   rumorosi.

6. **Pseudo-real or exchange-traded option data**
   Valutare, come estensione futura, dati pseudo-reali o exchange-traded
   options. Questa direzione richiede cautela perche' i prezzi di mercato
   incorporano bid-ask spread, liquidita', dividendi, microstruttura e volatilita'
   implicita; quindi la rete non approssimerebbe piu' soltanto la funzione
   Black-Scholes.
