# Noisy Black-Scholes Targets Experiment

This experiment evaluates the robustness of the neural-network pricing
surrogate when the training targets are perturbed by controlled Gaussian noise.

The clean Black-Scholes target is preserved and remains the evaluation
reference. The noisy target is used only for training. This keeps the experiment
aligned with the original research question: we still evaluate approximation of
the analytical Black-Scholes pricing function.

## Command

```bash
.venv/bin/python scripts/run_noisy_targets_experiment.py \
  --n-samples 50000 \
  --noise-levels 0.0 0.01 0.05 \
  --max-epochs 100 \
  --batch-size 1024 \
  --feature-set with_moneyness \
  --activation silu \
  --seed 42 \
  --noise-seed 123 \
  --data-dir data/experiments/noisy_targets \
  --output-dir outputs/experiments/noisy_targets \
  --results-dir results/experiments/noisy_targets
```

## Noise Model

For each clean Black-Scholes price `C_BS`, the noisy target is generated as:

```text
noise ~ Normal(0, noise_level * max(C_BS, price_floor))
noisy_call_price = max(C_BS + noise, 0)
```

The experiment uses `price_floor = 1.0`, so near-zero option prices still
receive a controlled amount of perturbation. Negative noisy prices are clipped
to zero to respect the non-negativity constraint of European call prices.

## Evaluation Protocol

- Training target: noisy Black-Scholes price.
- Evaluation target: clean analytical Black-Scholes price.
- Dataset size: `50000` synthetic contracts per noise level.
- Feature set: `with_moneyness`.
- Activation: `silu`.
- Noise levels: `0%`, `1%`, `5%`.

## Results Against Clean Black-Scholes Prices

| Noise level | MAE | RMSE | R2 | MAPE, price > 1 |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 0.1025449345 | 0.1516892342 | 0.9999580021 | 1.3213085809% |
| 1% | 0.1108372179 | 0.1610538331 | 0.9999526565 | 1.3960243887% |
| 5% | 0.1782652291 | 0.2416657391 | 0.9998934021 | 2.0943253084% |

## Interpretation

The network remains accurate when trained with moderate target noise, although
the error against the clean Black-Scholes function increases as the noise level
grows. This supports a cautious robustness interpretation: the neural network
can partially smooth noisy labels, but noisy supervision still degrades
approximation quality.

This experiment does not model real market option prices. It is a controlled
label-noise robustness test with a known analytical ground truth.

## Files

- `noisy_targets_metrics.json`: configuration and metrics for all noise levels.

## Verification

After adding the noisy-target experiment, the following checks were run
successfully:

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/sphinx-build -W -b html docs/source docs/build/html
```

```bash
cd report
latexmk -g -xelatex main.tex
```

```bash
cd presentation
latexmk -g -xelatex main.tex
```
