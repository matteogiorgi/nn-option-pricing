# Project Documentation

The documentation is built with Sphinx.

Install documentation dependencies:

```bash
pip install -r requirements-docs.txt
```

Build the HTML documentation:

```bash
sphinx-build -b html docs/source docs/build/html
```

The generated site will be available at:

```text
docs/build/html/index.html
```

