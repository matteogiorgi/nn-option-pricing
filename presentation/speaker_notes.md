# Speaker Notes

This file divides the 24 content slides into four blocks of six slides, one
for each speaker. The opening title slide is common and is not assigned to a
specific speaker.

The notes are written in English so that they match the language of the Beamer
presentation. Mathematical expressions use GitHub-flavored Markdown math
delimiters such as `$S_0/K$` and `$C_{BS}$`.

The expected presentation time is about 20 minutes, followed by about 10
minutes of questions. Each speaker should therefore aim for about 5 minutes in
total, roughly 45-50 seconds per slide. These notes are rehearsal prompts, not
a script to read word for word.

## Overall Coordination

The presentation follows the same logical order as the report:

| Speaker | Slides | Main responsibility | Handoff |
| --- | --- | --- | --- |
| Speaker 1 | 2-7 | Explain the research question, the financial setting, and the mathematical pricing benchmarks. | Move from theory to the controlled dataset used in the experiment. |
| Speaker 2 | 8-13 | Explain how the supervised learning problem is built: dataset, synthetic labels, moneyness, neural-network setup, and training metrics. | Move from methodology to the actual software implementation and experiments. |
| Speaker 3 | 14-19 | Explain how the project is implemented and how the main experiments, runtime benchmark, and SVR baseline are organized. | Move from experimental design to the final empirical evidence. |
| Speaker 4 | 20-25 | Present the final results, diagnostics, robustness analysis, limitations, and conclusion. | Close the presentation and prepare for questions. |

The title slide can be introduced very briefly by the first speaker or by the
group before the timed content starts. The key message to preserve throughout
the presentation is that the project studies a neural network as a pricing
surrogate for the Black-Scholes function in a controlled synthetic setting,
not as a direct market-pricing model.

## Speaker 1 - Motivation and Mathematical Setup

**Target time:** about 5 minutes total, roughly 45-50 seconds per slide.  
**Main goal:** make clear what problem is being studied and why Black-Scholes
gives a clean mathematical benchmark.

### Slides

| Slide | Title | Report reference |
| --- | --- | --- |
| 2 | Research Question | Chapter 1, Sections 1.2-1.3 |
| 3 | Why This Problem Is Meaningful | Chapter 1, Section 1.1; Chapter 3, Section 3.5 |
| 4 | Black-Scholes Dynamics | Chapter 2, Sections 2.2-2.4 |
| 5 | European Call Option | Chapter 2, Section 2.1 |
| 6 | Black-Scholes Formula | Chapter 2, Section 2.5 |
| 7 | Monte Carlo Benchmark | Chapter 2, Section 2.6; Chapter 5, Section 5.2 |

### Suggested Speech

**Slide 2 - Research Question.**  
Start by stating the central question of the project: can a feed-forward neural
network accurately approximate the Black-Scholes pricing function for European
call options? Emphasize that the goal is not to predict noisy market prices,
but to approximate a known mathematical function. The map
$(S_0,K,T,r,\sigma) \mapsto C_{BS}$ is the core supervised-learning problem.

**Slide 3 - Why This Problem Is Meaningful.**  
Explain why the problem is relevant. In finance, pricing often has to be
repeated across many contracts, parameter configurations, or risk scenarios.
A closed-form formula is extremely fast, but it is available only in special
cases. Monte Carlo is flexible, but it can be computationally expensive when it
must be run many times. The neural network is therefore studied as a surrogate
model: after training, it can produce prices through a single forward pass.

**Slide 4 - Black-Scholes Dynamics.**  
Introduce the Black-Scholes model under the risk-neutral measure. Briefly
explain the role of the parameters: $r$ is the risk-free rate, $\sigma$ is the
volatility, and $W_t$ is a Brownian motion. Highlight that the explicit
solution for $S_T$ allows Monte Carlo to sample the terminal price directly,
without simulating every intermediate time step.

**Slide 5 - European Call Option.**  
Recall the payoff $(S_T-K)^+$. Explain that the theoretical price is the
discounted risk-neutral expectation of this payoff. This step connects the
stochastic model for the underlying asset with the financial quantity that the
project wants to approximate.

**Slide 6 - Black-Scholes Formula.**  
Present the closed-form Black-Scholes price. State that this formula is used
as the ground truth of the project: every target in the dataset is computed as
$C_{BS}$. This makes the neural-network error directly measurable, because the
correct value of the pricing function is known.

**Slide 7 - Monte Carlo Benchmark.**  
Introduce Monte Carlo as the classical numerical benchmark. It estimates the
same price by averaging simulated discounted payoffs. The key point is the
trade-off: increasing the number of paths reduces statistical error, but also
increases computational cost. End by saying that the next step is to construct
a controlled dataset where the analytical Black-Scholes price is known for
every observation.

## Speaker 2 - Dataset and Machine-Learning Methodology

**Target time:** about 5 minutes total, roughly 45-50 seconds per slide.  
**Main goal:** show how the mathematical pricing problem becomes a supervised
regression problem.

### Slides

| Slide | Title | Report reference |
| --- | --- | --- |
| 8 | Synthetic Dataset | Chapter 4, Sections 4.1-4.2 |
| 9 | Why Synthetic Data? | Chapter 4, Section 4.1; Chapter 8, Section 8.7 |
| 10 | Feature Engineering: Moneyness | Chapter 4, Section 4.3; Chapter 8, Section 8.3 |
| 11 | Supervised Learning Setup | Chapter 3, Section 3.1; Chapter 5, Section 5.1 |
| 12 | Neural Network Architecture | Chapter 3, Sections 3.2-3.3; Chapter 5, Section 5.3 |
| 13 | Training and Metrics | Chapter 5, Sections 5.4-5.5 |

### Suggested Speech

**Slide 8 - Synthetic Dataset.**  
Explain how the dataset is built. We sample option and market parameters
within fixed ranges, then compute the European call price using the
Black-Scholes formula. Each row represents one European call option. The table
clarifies which variables enter the model.

**Slide 9 - Why Synthetic Data?**  
Motivate the synthetic-data choice. The project studies the neural network as
an approximation of the Black-Scholes pricing function, so synthetic data are
the most coherent setting. With real option quotes, the problem would change:
prices would also reflect bid-ask spreads, liquidity, dividends, implied
volatility patterns, and market microstructure effects.

**Slide 10 - Feature Engineering: Moneyness.**  
Introduce moneyness, defined as $S_0/K$. It does not add new information in a
strict mathematical sense because it is derived from two existing features,
but it exposes a financially meaningful ratio directly to the model. It helps
distinguish out-of-the-money, at-the-money, and in-the-money regions.

**Slide 11 - Supervised Learning Setup.**  
Describe the problem as supervised regression: the input is $x_i$, the target
is $y_i=C_{BS,i}$, and the model learns $f_\theta(x_i)$. Emphasize that inputs
and targets are standardized during training, while final metrics are reported
in original price units.

**Slide 12 - Neural Network Architecture.**  
Present the neural network as a multilayer perceptron. This is a reasonable
choice because the data are tabular, the Black-Scholes pricing function is
smooth, and feed-forward neural networks are suitable for nonlinear function
approximation. Highlight the final use of the SiLU activation function.

**Slide 13 - Training and Metrics.**  
Summarize the training procedure: train-validation-test split, MSE loss, Adam
optimizer, and early stopping. Then explain the metrics: MAE and RMSE are
errors in price units, $R^2$ measures overall goodness of fit, and MAPE is
computed only for prices greater than 1 to avoid unstable relative errors near
zero. End by saying that the next speaker will show how this methodology is
implemented in the actual codebase and experimental pipeline.

## Speaker 3 - Implementation and Experiment Design

**Target time:** about 5 minutes total, roughly 45-50 seconds per slide.  
**Main goal:** demonstrate that the project is reproducible, modular, and
experimentally controlled.

### Slides

| Slide | Title | Report reference |
| --- | --- | --- |
| 14 | Software Architecture | Chapter 6, Sections 6.1-6.2 |
| 15 | Experiment Pipeline | Chapter 5, Section 5.1; Chapter 6, Section 6.4 |
| 16 | Final Experiment Configuration | Chapter 5, Sections 5.3-5.7; Chapter 7, Section 7.1 |
| 17 | Model Selection Experiments | Chapter 7, Section 7.3 |
| 18 | Runtime Benchmark | Chapter 5, Section 5.6; Chapter 7, Section 7.4 |
| 19 | SVR Baseline | Chapter 3, Section 3.4; Chapter 5, Section 5.7; Chapter 7, Section 7.5 |

### Suggested Speech

**Slide 14 - Software Architecture.**  
Show that the codebase is organized into modules with separate
responsibilities. This is important for reproducibility and professional
structure: analytical pricing, dataset generation, model definition, training,
Monte Carlo benchmarking, and SVR benchmarking are not mixed together in a
single script.

**Slide 15 - Experiment Pipeline.**  
Describe the end-to-end workflow: data generation, analytical price
computation, split and scaling, training, evaluation, and artifact saving.
Emphasize that the experiment is reproducible from the command line and saves
metrics, figures, configuration files, and outputs.

**Slide 16 - Final Experiment Configuration.**  
Present the final setup: 100,000 synthetic contracts, the feature set with
moneyness, SiLU activation, batch size 1024, up to 200 epochs, and a Monte
Carlo benchmark with 50,000 paths. This slide fixes the main experimental
configuration.

**Slide 17 - Model Selection Experiments.**  
Explain that intermediate experiments were run before the final experiment to
compare moneyness and activation functions. The moneyness + SiLU configuration
was selected because it gave the best MAE and RMSE in the intermediate setup.
Clarify that this is a motivated model-selection step, not an exhaustive
hyperparameter search.

**Slide 18 - Runtime Benchmark.**  
Explain the logic of the runtime benchmark: pricing time is measured after the
neural network has been trained. The analytical Black-Scholes formula remains
the fastest method, but the relevant surrogate-model comparison is between the
neural network and Monte Carlo. In this setup, neural inference is about 40
times faster than Monte Carlo for 1,000 pricing queries.

**Slide 19 - SVR Baseline.**  
Introduce SVR as a classical machine-learning baseline. Clarify that it is run
on reduced datasets because RBF kernel methods scale less favorably with the
number of samples. The result shows that a classical method can also
approximate the function well, but it does not replace the final neural-network
pipeline. End by saying that, after defining the pipeline and baselines, the
last part of the presentation focuses on the final results and their
interpretation.

## Speaker 4 - Results, Discussion, and Conclusions

**Target time:** about 5 minutes total, roughly 45-50 seconds per slide.  
**Main goal:** answer the research question using the empirical results, while
being precise about the limits of the conclusion.

### Slides

| Slide | Title | Report reference |
| --- | --- | --- |
| 20 | Final Neural Network Results | Chapter 7, Section 7.1 |
| 21 | Predicted vs Analytical Prices | Chapter 7, Section 7.1 |
| 22 | Error Diagnostics | Chapter 7, Section 7.7; Chapter 8, Section 8.4 |
| 23 | Noisy Target Robustness | Chapter 4, Section 4.5; Chapter 7, Section 7.6; Chapter 8, Section 8.5 |
| 24 | Interpretation and Limits | Chapter 8, Sections 8.1-8.7 |
| 25 | Final Takeaway | Chapter 9 |

### Suggested Speech

**Slide 20 - Final Neural Network Results.**  
Present the main quantitative result. The final neural network achieves
MAE = 0.04290, RMSE = 0.06208, and $R^2$ almost equal to 1. Explain that these
metrics are computed on the held-out test set and are evaluated against
analytical Black-Scholes prices.

**Slide 21 - Predicted vs Analytical Prices.**  
Comment on the true-vs-predicted plot. The points lie close to the identity
line, meaning that the neural-network predictions closely match analytical
prices. This slide makes visually explicit what the previous metric table
showed numerically.

**Slide 22 - Error Diagnostics.**  
Explain why aggregate metrics are not enough. The error distribution checks
whether low average error hides systematic deviations or problematic tails.
Connect this idea to the additional diagnostic plots in the report against
moneyness, maturity, and volatility.

**Slide 23 - Noisy Target Robustness.**  
Present the noisy-target experiment. The models are trained on perturbed
prices but evaluated against clean Black-Scholes prices. As expected, the
error increases as noise increases, but the neural network remains reasonably
stable under mild noise. The SVR comparison helps interpret the same effect
outside the neural-network model.

**Slide 24 - Interpretation and Limits.**  
Clearly define the scope of the conclusions. The result is strong inside the
sampled synthetic Black-Scholes domain, but it does not automatically imply
that the model is valid for real market option prices. Also mention that the
network does not explicitly enforce arbitrage constraints beyond what is
implicitly learned from the training data.

**Slide 25 - Final Takeaway.**  
Close by answering the research question: yes, a feed-forward neural network
can accurately approximate the Black-Scholes pricing function in a controlled
synthetic setting. Conclude with possible extensions: Greeks, stochastic
volatility, path-dependent or American options, and real market data as a
separate research problem. The final sentence should be conservative: the
project validates the surrogate-model idea inside the Black-Scholes setting,
while richer market settings remain future work.
