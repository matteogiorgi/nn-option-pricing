Reproducibility
===============

The project is designed so that the reported results can be reproduced from
the source code, configuration files, and tracked experiment summaries.

Environment
-----------

Create a Python virtual environment and install the project dependencies from
the repository root:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e . --no-build-isolation

The Sphinx documentation has a small additional dependency set:

.. code-block:: bash

   pip install -r requirements-docs.txt

Final Experiment
----------------

The final reported neural-network configuration can be reproduced with:

.. code-block:: bash

   python scripts/run_experiment.py \
     --n-samples 100000 \
     --max-epochs 200 \
     --batch-size 1024 \
     --mc-n-paths 50000 \
     --mc-evaluation-samples 512 \
     --feature-set with_moneyness \
     --activation silu \
     --seed 42 \
     --data-dir data/final_improved \
     --output-dir outputs/final_improved

The compact tracked summary of this run is stored in ``results/final/``. The
larger generated artifacts remain reproducible but untracked:

* ``data/final_improved/`` contains the generated dataset;
* ``outputs/final_improved/`` contains model checkpoints, scalers, and full
  run outputs.

Tracked Results
---------------

The following tracked directories contain the selected experiment summaries
used by the report:

* ``results/final/`` for the final neural-network experiment;
* ``results/experiments/moneyness_feature/`` for the moneyness comparison;
* ``results/experiments/activation_functions/`` for the activation-function
  comparison;
* ``results/experiments/combined_feature_activation/`` for the combined
  moneyness and SiLU experiment;
* ``results/experiments/runtime_benchmark/`` for the pricing-time comparison;
* ``results/experiments/svr_benchmark/`` for the clean-target SVR baseline;
* ``results/experiments/noisy_targets/`` for the noisy-target neural-network
  robustness experiment;
* ``results/experiments/noisy_svr_benchmark/`` for the noisy-target SVR
  robustness benchmark.

Validation Commands
-------------------

Run the automated tests with:

.. code-block:: bash

   pytest

Build the documentation in strict mode with:

.. code-block:: bash

   sphinx-build -W -b html docs/source docs/build/html

The report is built with XeLaTeX:

.. code-block:: bash

   cd report
   latexmk -xelatex main.tex

The printable report variant is generated with:

.. code-block:: bash

   cd report
   latexmk -xelatex -jobname=main_print \
     -usepretex='\def\printlinks{1}' main.tex

The Beamer presentation is also built with XeLaTeX:

.. code-block:: bash

   cd presentation
   latexmk -xelatex main.tex

Reproducibility Notes
---------------------

Random seeds are fixed in the final experiment and in the reduced-scale
benchmarks. Monte Carlo evaluation is performed on a deterministic subset of
test options, which makes the reported comparison reproducible while keeping
the benchmark computationally bounded.

Generated datasets and checkpoints are intentionally not tracked by Git. They
are outputs of the experiment pipeline, while the source code, commands,
configuration snapshots, metrics, figures, and verification notes are tracked.
