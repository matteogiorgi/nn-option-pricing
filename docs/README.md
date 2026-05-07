# Project Documentation

The documentation is built with Sphinx.

Install the project and documentation dependencies from the repository root:

```bash
pip install -r requirements.txt
pip install -r requirements-docs.txt
pip install -e . --no-build-isolation
```

Build the HTML documentation from the repository root:

```bash
sphinx-build -b html docs/source docs/build/html
```

During development, it is useful to treat warnings as errors:

```bash
sphinx-build -W -b html docs/source docs/build/html
```

The generated site will be available at:

```text
docs/build/html/index.html
```
