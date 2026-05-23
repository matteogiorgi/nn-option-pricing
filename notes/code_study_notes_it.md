# Appunti di studio sul codice

Questi appunti servono per capire e spiegare il progetto **Neural Network Option Pricing** al gruppo. Sono scritti in italiano e possono essere usati come quaderno condiviso: si possono aggiungere domande, dubbi e richieste di approfondimento direttamente nei punti interessati.

Formato consigliato per le annotazioni:

```md
> DOMANDA:
> ...

> DA APPROFONDIRE:
> ...

> RISPOSTA:
> ...
```

## 1. Idea generale del progetto

Il progetto studia se una rete neurale feed-forward riesce ad approssimare la funzione di pricing Black-Scholes per opzioni call europee.

La domanda non e': "possiamo prevedere i prezzi reali di mercato?", ma:

> Una neural network riesce a imparare la mappa matematica
> `(s0, k, t, r, sigma) -> call_price`
> generata dalla formula analitica di Black-Scholes?

Quindi il dataset e' sintetico per scelta metodologica: vogliamo un problema controllato, riproducibile, con ground truth nota. Il target `call_price` e' calcolato dalla formula Black-Scholes, non osservato dal mercato.

## 2. Flusso completo dell'esperimento

Il file principale da lanciare e':

```bash
.venv/bin/python scripts/run_experiment.py
```

Per la run finale, il comando usato e' stato:

```bash
.venv/bin/python scripts/run_experiment.py \
  --n-samples 100000 \
  --max-epochs 200 \
  --batch-size 1024 \
  --mc-n-paths 50000 \
  --mc-evaluation-samples 512 \
  --data-dir data/final \
  --output-dir outputs/final
```

Il flusso logico e':

1. costruzione della configurazione sperimentale;
2. generazione del dataset sintetico;
3. calcolo dei prezzi target con Black-Scholes;
4. split train/validation/test;
5. scaling degli input e del target;
6. training della neural network;
7. predizione sul test set;
8. calcolo metriche;
9. benchmark Monte Carlo su un sottoinsieme del test set;
10. salvataggio di dataset, modello, scaler, metriche e grafici.

Diagramma principale:

```mermaid
flowchart TD
    A["scripts/run_experiment.py<br/>CLI entry point"] --> B["ExperimentConfig<br/>dataset + training + Monte Carlo + paths"]
    B --> C["pipeline.run_experiment"]
    C --> D["generate_synthetic_dataset"]
    D --> E["Black-Scholes analytical labels"]
    E --> F["train_model"]
    F --> G["PricingMLP"]
    G --> H["predict on test set"]
    H --> I["regression_metrics"]
    C --> J["Monte Carlo benchmark<br/>test subset"]
    C --> K["plots"]
    C --> L["saved artifacts<br/>CSV, model, scalers, metrics, figures"]
```

Lo stesso diagramma e' salvato anche in `notes/assets/pipeline_flow.mmd`.

## 3. Struttura principale del repository

La struttura concettuale e':

```text
.
├── scripts/
│   └── run_experiment.py
├── src/
│   └── nn_option_pricing/
│       ├── black_scholes.py
│       ├── config.py
│       ├── dataset.py
│       ├── evaluation.py
│       ├── model.py
│       ├── monte_carlo.py
│       ├── pipeline.py
│       ├── plots.py
│       └── training.py
├── tests/
├── docs/
├── report/
├── presentation/
├── results/
└── notes/
```

I sorgenti veri stanno in `src/nn_option_pricing/`. Lo script in `scripts/` e' solo l'interfaccia da terminale. Questo e' un buon pattern professionale: il codice riusabile sta nel package, mentre gli script chiamano le funzioni del package.

Mappa dei moduli:

```mermaid
flowchart LR
    CFG["config.py<br/>configuration dataclasses"]
    BS["black_scholes.py<br/>analytical formula + payoff"]
    DATA["dataset.py<br/>synthetic data generation"]
    MODEL["model.py<br/>PyTorch MLP"]
    TRAIN["training.py<br/>split, scaling, train, predict"]
    MC["monte_carlo.py<br/>GBM terminal simulation"]
    EVAL["evaluation.py<br/>MAE, RMSE, R2, MAPE"]
    PLOTS["plots.py<br/>diagnostic figures"]
    PIPE["pipeline.py<br/>end-to-end orchestration"]
    CLI["scripts/run_experiment.py<br/>command-line interface"]

    CLI --> CFG
    CLI --> PIPE
    PIPE --> CFG
    PIPE --> DATA
    PIPE --> TRAIN
    PIPE --> MC
    PIPE --> EVAL
    PIPE --> PLOTS
    DATA --> BS
    MC --> BS
    TRAIN --> MODEL
```

Lo stesso diagramma e' salvato anche in `notes/assets/module_map.mmd`.

## 4. Entry point: `scripts/run_experiment.py`

Questo e' il file da eseguire. Non contiene la logica scientifica principale: legge gli argomenti da terminale, costruisce la configurazione e chiama `run_experiment`.

Punti importanti:

- usa `argparse` per esporre parametri da CLI;
- permette di modificare numero di campioni, epoche, batch size, learning rate, numero di path Monte Carlo;
- costruisce un oggetto `ExperimentConfig`;
- chiama `nn_option_pricing.pipeline.run_experiment(config)`;
- stampa le metriche finali in formato JSON.

Frammento chiave:

```python
config = ExperimentConfig(
    dataset=DatasetConfig(n_samples=args.n_samples, seed=args.seed),
    training=TrainingConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    ),
    monte_carlo=MonteCarloConfig(
        n_paths=args.mc_n_paths,
        evaluation_samples=args.mc_evaluation_samples,
    ),
    paths=make_path_config(data_dir=args.data_dir, output_dir=args.output_dir),
)
metrics = run_experiment(config)
```

Da spiegare oralmente:

> Lo script e' l'interfaccia operativa. La pipeline vera e' nel package, cosi' il codice resta testabile e riutilizzabile.

## 5. Configurazione: `config.py`

`config.py` contiene dataclass immutabili che descrivono tutti i parametri dell'esperimento.

Le classi principali sono:

- `DatasetConfig`: dimensione dataset, seed, range di `s0`, `k`, `t`, `r`, `sigma`;
- `TrainingConfig`: split, batch size, epoche, patience, learning rate, architettura hidden layers;
- `MonteCarloConfig`: numero di path, batch per opzioni e path, numero di esempi usati nel benchmark;
- `PathConfig`: path per dataset, output, figure, metriche, modello e scaler;
- `ExperimentConfig`: contenitore unico che raggruppa tutto.

Perche' e' una buona scelta:

- evita costanti sparse nel codice;
- rende gli esperimenti riproducibili;
- permette di salvare la configurazione completa in JSON;
- separa chiaramente parametri scientifici, training e filesystem.

Esempio:

```python
@dataclass(frozen=True)
class DatasetConfig:
    n_samples: int = 100_000
    seed: int = 42
    s0_min: float = 50.0
    s0_max: float = 150.0
    k_min: float = 50.0
    k_max: float = 150.0
    t_min: float = 0.1
    t_max: float = 2.0
    r_min: float = 0.0
    r_max: float = 0.05
    sigma_min: float = 0.1
    sigma_max: float = 0.6
```

## 6. Formula analitica: `black_scholes.py`

Questo modulo contiene la formula Black-Scholes per il prezzo di una call europea e la payoff function.

La SDE di partenza e':

```text
dS_t = r S_t dt + sigma S_t dW_t
```

Sotto questa dinamica, il prezzo analitico della call europea e':

```text
C = S0 N(d1) - K exp(-rT) N(d2)
```

con:

```text
d1 = [ln(S0/K) + (r + 0.5 sigma^2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

Nel codice:

```python
d1 = (np.log(s0_arr / k_arr) + (r_arr + 0.5 * safe_sigma**2) * safe_t) / (
    safe_sigma * sqrt_t
)
d2 = d1 - safe_sigma * sqrt_t

price = s0_arr * norm.cdf(d1) - k_arr * np.exp(-r_arr * safe_t) * norm.cdf(d2)
return np.maximum(price, 0.0)
```

Aspetti tecnici importanti:

- accetta sia scalari sia array NumPy;
- usa broadcasting NumPy, quindi e' efficiente sulla generazione di molte osservazioni;
- usa `eps` per evitare divisioni per zero in casi limite;
- forza il prezzo a essere non negativo con `np.maximum(price, 0.0)`.

Da spiegare oralmente:

> Black-Scholes e' il generatore del ground truth. La rete non impara dai mercati, ma da una funzione matematica nota.

## 7. Dataset sintetico: `dataset.py`

`dataset.py` genera fisicamente il dataset.

La funzione principale e':

```python
generate_synthetic_dataset(config: DatasetConfig) -> pd.DataFrame
```

Come viene generato:

1. viene creato un generatore random NumPy con seed fissato:

```python
rng = np.random.default_rng(config.seed)
```

2. vengono campionati uniformemente i parametri:

```python
data = {
    "s0": rng.uniform(config.s0_min, config.s0_max, size=n),
    "k": rng.uniform(config.k_min, config.k_max, size=n),
    "t": rng.uniform(config.t_min, config.t_max, size=n),
    "r": rng.uniform(config.r_min, config.r_max, size=n),
    "sigma": rng.uniform(config.sigma_min, config.sigma_max, size=n),
}
```

3. viene calcolato il target con Black-Scholes:

```python
data[TARGET_COLUMN] = call_price(
    data["s0"],
    data["k"],
    data["t"],
    data["r"],
    data["sigma"],
)
```

4. viene aggiunta la variabile diagnostica:

```python
data["moneyness"] = data["s0"] / data["k"]
```

Le colonne sono:

| Colonna | Ruolo | Usata dalla NN? |
|---|---|---|
| `s0` | prezzo iniziale sottostante | si |
| `k` | strike | si |
| `t` | maturity in anni | si |
| `r` | risk-free rate | si |
| `sigma` | volatilita' | si |
| `call_price` | target Black-Scholes | target |
| `moneyness` | `s0/k`, analisi errori | no |

La lista delle feature e' definita esplicitamente:

```python
FEATURE_COLUMNS = ["s0", "k", "t", "r", "sigma"]
TARGET_COLUMN = "call_price"
```

Nota importante:

> `moneyness` non viene data come input alla rete. Serve solo dopo, per capire se gli errori cambiano tra opzioni out-of-the-money, at-the-money e in-the-money.

## 8. Modello neurale: `model.py`

Il modello e' una rete feed-forward fully connected, chiamata `PricingMLP`.

> DA APPROFONDIRE:
> spiegare cos'è e come funziona una MLP (Multi-Layer Perceptron) in generale, e perché è adatta a questo problema.

> RISPOSTA:
> Una MLP, o Multi-Layer Perceptron, e' una rete neurale feed-forward composta da una sequenza di layer completamente connessi. "Feed-forward" significa che l'informazione scorre in una sola direzione: dagli input verso l'output, senza cicli o memoria interna. "Completamente connesso" significa che ogni neurone di un layer riceve in ingresso tutti gli output del layer precedente.
>
> In termini matematici, ogni layer lineare calcola una trasformazione del tipo:
>
> ```text
> z = W x + b
> ```
>
> dove `x` e' il vettore in ingresso, `W` e' la matrice dei pesi, `b` e' il bias e `z` e' l'output lineare del layer. Dopo i layer nascosti viene applicata una funzione non lineare, nel nostro caso ReLU:
>
> ```text
> ReLU(z) = max(z, 0)
> ```
>
> La non linearita' e' fondamentale: senza funzioni di attivazione, una sequenza di layer lineari sarebbe equivalente a un solo layer lineare. Con le ReLU, invece, la rete puo' approssimare funzioni non lineari complesse.
>
> Una MLP e' adatta a questo progetto perche' il nostro problema e' una regressione tabellare: abbiamo cinque variabili numeriche di input (`s0`, `k`, `t`, `r`, `sigma`) e vogliamo predire un singolo valore continuo (`call_price`). Non abbiamo immagini, testi o sequenze temporali da modellare direttamente; quindi non serve una CNN, una RNN o un Transformer. La funzione Black-Scholes e' una funzione deterministica e non lineare di cinque variabili, e una MLP e' una scelta naturale per approssimare questo tipo di mappa.

> DA APPROFONDIRE:
> aggiungi degli schemi / immagini che aiutino a capire

> RISPOSTA:
> Uno schema utile e' vedere la rete come una catena di trasformazioni. I cinque parametri finanziari entrano nel primo layer; ogni layer nascosto combina le informazioni del layer precedente, applica una trasformazione lineare e poi una ReLU; l'ultimo layer restituisce un solo numero, cioe' il prezzo stimato della call.
>
> ```mermaid
> flowchart LR
>     subgraph INPUT["Input features"]
>         S0["s0"]
>         K["k"]
>         T["t"]
>         R["r"]
>         SIG["sigma"]
>     end
>
>     subgraph H1["Hidden layer 1"]
>         H1A["64 neurons"]
>         H1R["ReLU"]
>     end
>
>     subgraph H2["Hidden layer 2"]
>         H2A["64 neurons"]
>         H2R["ReLU"]
>     end
>
>     subgraph H3["Hidden layer 3"]
>         H3A["32 neurons"]
>         H3R["ReLU"]
>     end
>
>     OUT["Predicted call price"]
>
>     INPUT --> H1A --> H1R --> H2A --> H2R --> H3A --> H3R --> OUT
> ```
>
> Un secondo modo di pensarla e' come composizione di funzioni:
>
> ```text
> x = (s0, k, t, r, sigma)
>
> h1 = ReLU(W1 x  + b1)
> h2 = ReLU(W2 h1 + b2)
> h3 = ReLU(W3 h2 + b3)
> y  = W4 h3 + b4
> ```
>
> dove `y` e' il prezzo predetto. Il diagramma e' salvato anche in `notes/assets/mlp_architecture.mmd`, cosi' si puo' riutilizzare o convertire in immagine per slide/appunti.

> DOMANDA:
> puoi descrivere i diversi layer?
> Cosa rappresentano e cosa fanno?

> RISPOSTA:
> Nella nostra rete ci sono tre tipi concettuali di layer: input layer, hidden layers e output layer.
>
> L'input layer non e' un layer che "impara" parametri nel nostro codice: rappresenta semplicemente il vettore delle cinque feature finanziarie:
>
> ```text
> x = (s0, k, t, r, sigma)
> ```
>
> Dopo lo scaling, questi cinque numeri entrano nel primo layer lineare. Lo scaling e' importante perche' le variabili hanno scale diverse: ad esempio `s0` e `k` sono nell'ordine delle decine/centinaia, mentre `r` e' tra 0 e 0.05.
>
> I hidden layers sono i layer interni della rete. Nel nostro caso sono:
>
> ```text
> Linear(5, 64)  + ReLU
> Linear(64, 64) + ReLU
> Linear(64, 32) + ReLU
> ```
>
> Ogni layer lineare costruisce nuove combinazioni delle informazioni ricevute. Per esempio, la rete puo' imparare relazioni implicite tra prezzo iniziale e strike, tra maturity e volatilita', oppure tra tasso risk-free e sconto del payoff. Non dobbiamo interpretare ogni singolo neurone come una variabile finanziaria precisa: i neuroni imparano rappresentazioni intermedie utili a ridurre l'errore di pricing.
>
> Le ReLU introducono non linearita'. Questo e' essenziale perche' la formula Black-Scholes non e' lineare rispetto agli input: contiene logaritmi, radici quadrate, esponenziali e distribuzione normale cumulativa.
>
> L'output layer e':
>
> ```text
> Linear(32, 1)
> ```
>
> Questo layer prende le 32 rappresentazioni finali e produce un solo numero: il prezzo stimato della call. Non usiamo ReLU sull'ultimo layer, perche' in regressione si lascia solitamente l'output libero. Nel nostro codice, pero', dopo la predizione applichiamo un clipping a zero per evitare piccoli prezzi negativi non validi finanziariamente.

> DOMANDA:
> che alternative ci sarebbero state, oltre alla MLP?
> Perchè abbiamo scelto questa? (vantaggi / svantaggi)

> RISPOSTA:
> Le alternative principali sarebbero state:
>
> - modelli lineari, come linear regression o ridge regression;
> - modelli ad alberi, come random forest o gradient boosting;
> - Gaussian processes;
> - reti piu' complesse, come ResNet tabellari o architetture specializzate;
> - Physics-Informed Neural Networks;
> - Deep BSDE methods.
>
> Un modello lineare sarebbe molto semplice e interpretabile, ma probabilmente troppo rigido: Black-Scholes e' fortemente non lineare negli input. Potrebbe funzionare solo come baseline debole.
>
> Random forest e gradient boosting sarebbero alternative forti per dati tabellari. Possono approssimare bene funzioni non lineari e spesso richiedono meno tuning. Pero' sono meno coerenti con l'obiettivo del progetto, che e' studiare esplicitamente una neural network come pricing surrogate. Inoltre, una rete neurale si integra meglio con possibili estensioni future come PINNs, Deep BSDE o modelli differenziabili per il calcolo delle Greeks.
>
> I Gaussian processes sono interessanti perche' danno anche incertezza predittiva, ma scalano male con dataset grandi. Con 100000 osservazioni diventano poco pratici senza approssimazioni.
>
> Architetture neurali piu' sofisticate sarebbero possibili, ma rischierebbero di essere eccessive per un primo esperimento. Il nostro input e' un vettore di cinque feature, non una serie temporale o un'immagine; quindi una MLP e' gia' una scelta naturale e sufficientemente espressiva.
>
> In sintesi, abbiamo scelto la MLP perche':
>
> - e' adatta a problemi di regressione tabellare;
> - e' semplice da spiegare e implementare;
> - e' abbastanza potente da approssimare funzioni non lineari;
> - e' computazionalmente efficiente;
> - e' coerente con la research question del progetto.
>
> Lo svantaggio principale e' che la MLP non incorpora direttamente conoscenza finanziaria: impara dai dati. Per questo la valutiamo contro Black-Scholes e Monte Carlo, e manteniamo la discussione prudente.

Architettura di default:

```text
Input(5)
-> Linear(5, 64) + ReLU
-> Linear(64, 64) + ReLU
-> Linear(64, 32) + ReLU
-> Linear(32, 1)
```

Nel codice l'architettura e' costruita dinamicamente:

```python
for hidden_dim in hidden_layers:
    layers.append(nn.Linear(previous_dim, hidden_dim))
    layers.append(nn.ReLU())
    previous_dim = hidden_dim

layers.append(nn.Linear(previous_dim, 1))
self.net = nn.Sequential(*layers)
```

Il metodo `forward` restituisce un prezzo per ogni riga:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x).squeeze(-1)
```

Da spiegare oralmente:

> La rete e' usata come function approximator: prende cinque parametri finanziari e restituisce una stima del prezzo Black-Scholes.

## 9. Training: `training.py`

`training.py` e' uno dei moduli piu' importanti. Contiene:

- gestione dei seed;
- split train/validation/test;
- scaling;
- DataLoader PyTorch;
- training loop;
- early stopping;
- predizione finale.

### 9.1 Split dei dati

Prima viene separato il test set:

```python
x_train_val, x_test, y_train_val, y_test = train_test_split(
    x,
    y,
    test_size=config.test_size,
    random_state=config.seed,
)
```

Poi il blocco rimanente viene diviso in train e validation:

```python
validation_fraction = config.validation_size / (1.0 - config.test_size)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_val,
    y_train_val,
    test_size=validation_fraction,
    random_state=config.seed,
)
```

Perche' si fa cosi':

- il test set resta completamente separato fino alla valutazione finale;
- la validation serve per early stopping e scelta del miglior modello;
- lo split e' riproducibile grazie al seed.

### 9.2 Scaling

Gli input vengono standardizzati:

```python
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)
```

Anche il target viene standardizzato:

```python
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
```

Perche' scalare il target:

- i prezzi possono avere scale diverse dagli input;
- la loss MSE diventa piu' stabile;
- l'ottimizzazione con Adam e' meglio condizionata;
- in predizione si torna alle unita' monetarie originali con `inverse_transform`.

### 9.3 Training loop

Il training usa:

- `MSELoss`;
- `Adam`;
- `weight_decay`;
- batch training con `DataLoader`;
- early stopping sulla validation loss.

Frammento centrale:

```python
optimizer.zero_grad(set_to_none=True)
predictions = model(xb)
loss = criterion(predictions, yb)
loss.backward()
optimizer.step()
```

Il miglior modello viene salvato in memoria:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    epochs_without_improvement = 0
```

Se la validation loss non migliora per `patience` epoche, il training si ferma:

```python
if epochs_without_improvement >= config.patience:
    break
```

### 9.4 Predizione

La funzione `predict`:

- processa i dati in batch;
- usa `torch.no_grad()`;
- converte le predizioni dal target scalato al prezzo originale;
- forza i prezzi a essere non negativi.

```python
if target_scaler is not None:
    y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
if clip_nonnegative:
    y_pred = np.maximum(y_pred, 0.0)
```

Da spiegare oralmente:

> La rete produce output non vincolati, quindi il clipping evita piccoli prezzi negativi non finanziariamente validi.

## 10. Monte Carlo benchmark: `monte_carlo.py`

Il Monte Carlo serve come benchmark numerico classico.

La funzione principale e':

```python
call_price_mc(...)
```

Per una call europea sotto Black-Scholes non serve simulare tutta la traiettoria giorno per giorno. Basta simulare direttamente la distribuzione terminale del geometric Brownian motion:

```text
S_T = S_0 exp((r - 0.5 sigma^2)T + sigma sqrt(T) Z)
```

Nel codice:

```python
drift = (r[sl, None] - 0.5 * sigma[sl, None] ** 2) * t[sl, None]
diffusion = sigma[sl, None] * np.sqrt(t[sl, None]) * z
terminal_price = s0[sl, None] * np.exp(drift + diffusion)
payoff_sum += call_payoff(terminal_price, k[sl, None]).sum(axis=1)
```

Poi si sconta il payoff medio:

```python
estimates[sl] = np.exp(-r[sl] * t[sl]) * payoff_sum / sampled_paths
```

Aspetto importante di efficienza:

- il codice usa batching sulle opzioni (`option_batch_size`);
- usa batching anche sui path Monte Carlo (`path_batch_size`);
- evita di costruire una matrice enorme `n_options x n_paths` tutta in memoria.

Da spiegare oralmente:

> Monte Carlo e' corretto ma rumoroso e computazionalmente piu' costoso. La neural network, una volta addestrata, produce prezzi con una semplice forward pass.

## 11. Metriche: `evaluation.py`

Le metriche usate sono:

- `MAE`: errore assoluto medio;
- `RMSE`: penalizza di piu' gli errori grandi;
- `R2`: quota di varianza spiegata;
- `MAPE`, calcolato solo su prezzi con valore assoluto maggiore di 1.

Il MAPE viene filtrato:

```python
mask = np.abs(y_true) > 1.0
```

Motivo:

> Per opzioni con prezzo vicino a zero, un errore piccolo in valore assoluto puo' generare un errore percentuale enorme e poco interpretabile.

## 12. Grafici: `plots.py`

`plots.py` genera figure diagnostiche salvate su disco.

Figure principali:

- `loss.png`: train loss e validation loss;
- `true_vs_predicted.png`: prezzo Black-Scholes vs prezzo NN;
- `error_distribution.png`: distribuzione degli errori;
- `error_vs_moneyness.png`: errore rispetto a `s0/k`;
- `error_vs_maturity.png`: errore rispetto alla maturity;
- `error_vs_volatility.png`: errore rispetto alla volatilita'.

Esempio:

```python
plot_true_vs_predicted(
    artifacts.y_test,
    y_pred,
    config.paths.figure_dir / "true_vs_predicted.png",
)
```

Questi grafici sono utili per la relazione e per le slide, perche' non mostrano solo "quanto" sbaglia il modello, ma anche "dove" sbaglia.

## 13. Pipeline: `pipeline.py`

`pipeline.py` coordina tutto. E' il cuore operativo del progetto.

La funzione principale e':

```python
run_experiment(config: ExperimentConfig) -> dict[str, float]
```

Sequenza semplificata:

```python
ensure_directories(config)
save_experiment_config(config, config.paths.output_dir / "experiment_config.json")

df = generate_synthetic_dataset(config.dataset)
save_dataset(df, config.paths.dataset_path)

artifacts = train_model(df, config.training)
y_pred = predict(...)
metrics = regression_metrics(artifacts.y_test, y_pred)
save_metrics(metrics, config.paths.metrics_dir / "nn_metrics.json")

mc_prices = call_price_mc(...)
mc_metrics = regression_metrics(...)
save_metrics(mc_metrics, ...)

plot_loss(...)
plot_true_vs_predicted(...)
plot_error_distribution(...)
plot_error_against_feature(...)
```

Cosa salva:

- dataset CSV;
- configurazione JSON;
- modello PyTorch `.pt`;
- scaler input `scaler.joblib`;
- scaler target `target_scaler.joblib`;
- metriche JSON;
- figure PNG.

Punto importante:

> La pipeline non e' solo training. E' l'esperimento completo, dalla generazione dati alla produzione degli artifact finali.

## 14. Test

I test stanno in `tests/`.

File principali:

- `test_black_scholes.py`: controlla proprieta' della formula e casi noti;
- `test_dataset.py`: controlla colonne, range e forma del dataset;
- `test_monte_carlo.py`: controlla che Monte Carlo converga ragionevolmente verso Black-Scholes;
- `test_pipeline.py`: smoke test della pipeline end-to-end.

Il comando e':

```bash
.venv/bin/python -m pytest
```

Per l'esposizione:

> I test non servono a dimostrare la teoria finanziaria, ma a evitare regressioni nel codice e a verificare che i pezzi principali funzionino assieme.

## 15. Risultati finali

I risultati finali selezionati stanno in:

```text
results/final/
├── README.md
├── experiment_config.json
├── figures/
└── metrics/
```

Metriche finali neural network:

```json
{
  "mae": 0.0619787834584713,
  "rmse": 0.08218503059459481,
  "r2": 0.9999874234199524,
  "mape_percent_price_gt_1": 0.6746048331260681
}
```

Metriche Monte Carlo vs Black-Scholes:

```json
{
  "mae": 0.088821478006634,
  "rmse": 0.15790186868176595,
  "r2": 0.9999513355216346,
  "mape_percent_price_gt_1": 0.6483135955335606
}
```

Interpretazione prudente:

- la rete approssima molto bene la funzione Black-Scholes nel dominio simulato;
- Monte Carlo e' anch'esso vicino, ma resta affetto da errore statistico;
- la rete non sostituisce Black-Scholes come formula teorica, ma funziona come surrogate model nel dominio di training;
- non bisogna concludere che il modello prezza correttamente opzioni reali di mercato.

## 16. Cosa dire se chiedono "perche' dataset sintetico?"

Risposta breve:

> Perche' la research question riguarda l'approssimazione della funzione Black-Scholes, non il fitting dei prezzi di mercato. Con dati sintetici abbiamo un ground truth analitico noto, controllo sui parametri e riproducibilita'. Con opzioni exchange-traded reali, i prezzi incorporano bid-ask spread, liquidita', dividendi, microstruttura e volatilita' implicita; quindi la rete imparerebbe una funzione di mercato rumorosa, non la formula Black-Scholes.

## 17. Cosa dire se chiedono "perche' Monte Carlo se abbiamo la formula chiusa?"

Risposta breve:

> In Black-Scholes la formula chiusa e' il benchmark analitico principale. Monte Carlo viene incluso come benchmark numerico classico: mostra come un metodo quantitativo simulativo approssima lo stesso prezzo e permette un confronto con la neural network in termini di accuratezza e costo computazionale.

## 18. Cosa dire se chiedono "cosa rende efficiente il codice?"

Punti principali:

- Black-Scholes e dataset generation sono vettorizzati con NumPy;
- il training usa mini-batch PyTorch;
- l'inferenza usa batch e `torch.no_grad()`;
- Monte Carlo simula direttamente la distribuzione terminale, non tutta la traiettoria;
- Monte Carlo usa batching su opzioni e path per limitare la memoria;
- configurazione e artifact sono separati dal codice;
- test e documentazione rendono il progetto manutenibile.

## 19. Domande aperte da discutere nel gruppo

Possibili domande da aggiungere durante lo studio:

```md
> DOMANDA:
> Perche' usiamo una MLP e non una rete piu' complessa?

> DOMANDA:
> Quanto dipendono i risultati dai range scelti per s0, k, t, r, sigma?

> DOMANDA:
> La rete generalizza fuori dal dominio simulato?

> DOMANDA:
> Il confronto con Monte Carlo e' completamente equo dal punto di vista computazionale?
```

Prime risposte sintetiche:

- MLP: perche' il problema e' regressione tabellare su cinque input, non sequenziale.
- Range: i risultati valgono soprattutto nel dominio campionato.
- Extrapolation: non e' garantita; andrebbe testata separatamente.
- Monte Carlo: non e' un confronto di training, ma un benchmark numerico sul prezzo.

## 20. Mini scaletta per spiegare il codice al gruppo

Ordine consigliato:

1. partire dalla research question;
2. mostrare `scripts/run_experiment.py` come entry point;
3. mostrare `pipeline.py` come regia dell'esperimento;
4. spiegare `dataset.py` e `black_scholes.py`;
5. spiegare `model.py` e `training.py`;
6. spiegare Monte Carlo;
7. spiegare metriche e grafici;
8. mostrare `results/final/`;
9. chiudere con limiti e prossimi passi.

Frase conclusiva utile:

> Il progetto e' costruito come una pipeline sperimentale riproducibile: fissiamo il modello matematico, generiamo dati controllati, addestriamo una rete come surrogate model, confrontiamo con benchmark classici e analizziamo dove l'approssimazione funziona meglio o peggio.
