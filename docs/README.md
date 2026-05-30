# Project Documentation

The Python and project workflow documentation is built with Sphinx.
The generated HTML site is not tracked in Git; it is either built locally or
published automatically through GitHub Pages.

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

The online documentation is published at:

```text
https://matteogiorgi.github.io/nn-option-pricing
```

The GitHub Pages deployment workflow is stored in:

```text
.github/workflows/docs.yml
```
