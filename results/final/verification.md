# Final Verification

This file records the validation checks executed after the final experiment
artifacts were generated.

## Repository State

- Verification commit: `0f58408`
- Final experiment artifacts directory: `results/final/`
- Generated but untracked reproducible artifacts:
  - `data/final/`
  - `outputs/final/`

## Test Suite

Command:

```bash
.venv/bin/pytest
```

Result:

```text
10 passed
```

## Documentation Build

Command:

```bash
.venv/bin/sphinx-build -W -b html docs/source docs/build/html
```

Result:

```text
build succeeded
```
