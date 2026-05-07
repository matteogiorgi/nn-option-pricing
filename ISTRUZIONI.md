# Neural Network-Based European Option Pricing under Black-Scholes SDE Dynamics

## Titolo del progetto

Titolo principale:

**Neural Network-Based European Option Pricing under Black-Scholes SDE Dynamics**

Titolo alternativo:

**Option Pricing with Neural Networks: A Comparison with Black-Scholes and Monte Carlo Methods**

## Domanda di progetto

Una feed-forward neural network riesce ad approssimare accuratamente la funzione di pricing di opzioni europee generate sotto il modello Black-Scholes?

In particolare, il progetto vuole studiare se una rete neurale supervisionata puo' essere usata come pricing surrogate, cioe' come approssimatore veloce della funzione che associa i parametri di mercato e contrattuali al prezzo teorico di una call europea.

Il confronto principale sara' tra:

- formula analitica di Black-Scholes;
- simulazione Monte Carlo;
- neural network feed-forward.

## Obiettivo del progetto

L'obiettivo non e' costruire un sistema di trading reale, ma realizzare un progetto accademico ben strutturato per il corso di Machine Learning for Finance.

Il progetto deve combinare:

- stochastic differential equations;
- option pricing;
- Monte Carlo simulation;
- supervised deep learning;
- confronto con metodi quantitativi classici.

La prima implementazione deve rimanere semplice e controllabile:

**Black-Scholes + Monte Carlo + Feed-Forward Neural Network.**

## Framework matematico

Il modello di partenza e' la SDE di Black-Scholes:

```text
dS_t = r S_t dt + sigma S_t dW_t
```

dove:

- `S_t` e' il prezzo dell'asset al tempo `t`;
- `r` e' il tasso risk-free;
- `sigma` e' la volatilita';
- `W_t` e' un moto browniano.

Sotto queste ipotesi, il prezzo di una call europea puo' essere calcolato tramite:

- formula analitica di Black-Scholes;
- simulazione Monte Carlo del processo del sottostante;
- approssimazione tramite rete neurale.

## Prima scaletta del progetto

1. Background teorico
   - Introdurre le equazioni differenziali stocastiche.
   - Spiegare il geometric Brownian motion.
   - Presentare il modello Black-Scholes.
   - Definire le opzioni europee call.
   - Spiegare il pricing risk-neutral.
   - Introdurre la simulazione Monte Carlo.
   - Presentare le neural networks come function approximators.

2. Generazione del dataset
   - Generare dati sintetici usando la formula Black-Scholes.
   - Usare come input:
     - prezzo iniziale `S0`;
     - strike `K`;
     - maturity `T`;
     - tasso risk-free `r`;
     - volatilita' `sigma`.
   - Usare come output:
     - prezzo teorico della call europea.
   - Range iniziali suggeriti:
     - `S0` in `[50, 150]`;
     - `K` in `[50, 150]`;
     - `T` in `[0.1, 2]`;
     - `r` in `[0, 0.05]`;
     - `sigma` in `[0.1, 0.6]`.
   - Generare un numero sufficiente di osservazioni, ad esempio da 50.000 a 200.000.

3. Implementazione dei benchmark
   - Implementare la formula analitica Black-Scholes per call europee.
   - Implementare un metodo Monte Carlo per simulare il payoff scontato.
   - Verificare che Monte Carlo converga verso Black-Scholes aumentando il numero di simulazioni.

4. Neural network
   - Usare una feed-forward neural network in PyTorch.
   - Architettura iniziale suggerita:

```text
Input(5)
-> Dense(64, ReLU)
-> Dense(64, ReLU)
-> Dense(32, ReLU)
-> Dense(1)
```

   - Loss:
     - Mean Squared Error.
   - Optimizer:
     - Adam.
   - Applicare scaling/normalization agli input.

5. Training
   - Dividere i dati in training set, validation set e test set.
   - Fissare un random seed per la riproducibilita'.
   - Usare early stopping.
   - Monitorare training loss e validation loss.
   - Salvare metriche e grafici principali.

6. Valutazione e confronto
   - Confrontare il prezzo predetto dalla rete con il prezzo Black-Scholes.
   - Confrontare eventualmente Monte Carlo con Black-Scholes.
   - Confrontare eventualmente Monte Carlo con Neural Network.
   - Metriche suggerite:
     - MAE;
     - RMSE;
     - R^2;
     - MAPE, se non crea problemi vicino a prezzi molto piccoli.

7. Visualizzazioni
   - Grafico true price vs predicted price.
   - Training loss e validation loss.
   - Distribuzione degli errori.
   - Errore rispetto alla moneyness `S0 / K`.
   - Errore rispetto alla maturity `T`.
   - Errore rispetto alla volatilita' `sigma`.

8. Discussione dei risultati
   - Analizzare quando la rete funziona bene.
   - Identificare dove gli errori aumentano.
   - Discutere se la rete puo' essere usata come pricing surrogate.
   - Confrontare vantaggi e limiti rispetto alla formula analitica.
   - Confrontare vantaggi e limiti rispetto alla simulazione Monte Carlo.
   - Discutere il rischio di overfitting.
   - Commentare l'interpretabilita' del modello.

9. Conclusione
   - Rispondere alla domanda di progetto.
   - Riassumere i risultati principali.
   - Chiarire che il caso Black-Scholes e' un ambiente controllato, utile per validare il metodo.
   - Proporre sviluppi futuri.

## Estensioni future opzionali

Possibili estensioni, da citare nella relazione o implementare solo se resta tempo:

- modello di Heston;
- volatilita' stocastica;
- Physics-Informed Neural Networks;
- Deep BSDE methods;
- basket options;
- opzioni americane;
- stima delle Greeks;
- pricing sotto dinamiche piu' realistiche rispetto a Black-Scholes.

## Stack consigliato

Linguaggio:

- Python.

Librerie:

- NumPy;
- pandas;
- matplotlib;
- scikit-learn;
- PyTorch.

## Organizzazione possibile della directory

```text
project/
├── data/
│   └── synthetic_options.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── black_scholes.py
│   ├── monte_carlo.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── plots.py
├── outputs/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

## Implementazione minimale funzionante

La prima versione del progetto dovrebbe includere:

1. funzione di pricing Black-Scholes per call europee;
2. funzione di pricing Monte Carlo;
3. generazione di un dataset sintetico;
4. training di una rete feed-forward;
5. valutazione con MAE, RMSE e R^2;
6. grafici principali;
7. breve discussione dei risultati.

Solo dopo questa prima versione conviene aggiungere miglioramenti progressivi.


#### Note implementative

Questo progetto deve avere una architettura professionale e deve essere computazionalmente e algoritmicamente efficiente.
