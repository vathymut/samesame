# Contributing

This closely follows
[Docling's guidelines](https://github.com/docling-project/docling/blob/main/CONTRIBUTING.md).

We welcome external contributions. If you have an itch, please feel
free to scratch it.

## Developing

### Usage of uv

We use [uv](https://docs.astral.sh/uv/) as package and project manager.

#### Installation

To install `uv`, check the documentation on [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).

#### Create an environment and sync it

You can use the `uv sync` to create a project virtual environment (if it does not already exist) and sync
the project's dependencies with the environment.

```bash
git clone https://github.com/vathymut/samesame
cd samesame
uv sync --all-extras
```

#### Use a specific Python version (optional)

If you need to work with a specific version of Python, you can create a new virtual environment for that version
and run the sync command:

```bash
uv venv --python 3.12
uv sync --all-extras
```

More detailed options are described on the [Using Python environments](https://docs.astral.sh/uv/pip/environments/) documentation.

#### Add a new dependency

Simply use the `uv add` command. The `pyproject.toml` and `uv.lock` files will be updated.

```bash
uv add [OPTIONS] <PACKAGES|--requirements <REQUIREMENTS>>
```

## Coding Style Guidelines

We use the following tools to enforce code style:

- [Ruff](https://docs.astral.sh/ruff/), as linter and code formatter

## Tests

When submitting a new feature or fix, please consider adding a short test for it.

```sh
uv run pytest
```

## Documentation

We use [MkDocs](https://www.mkdocs.org/) to write documentation.

To run the documentation server, run:

```bash
mkdocs serve
```

The server will be available at [http://localhost:8000](http://localhost:8000).

### Pushing Documentation to GitHub Pages

Run the following:

```bash
mkdocs gh-deploy
```

### Updating Package Version

```bash
python -m uv version --bump
```

### Building Python Package

```bash
python -m uv build
```

### Publishing to PyPI

Following [these instructions](https://github.com/astral-sh/uv/issues/10878#issuecomment-3473401901),
we first log in securely as follows:

```bash
$ python -m uv auth login upload.pypi.org                      
username: __token__
password: 
```

The latter assumes that the requisite credentials have been generated and
potentially, saved in a config file (e.g. see `.pypirc` file). Only then, we
publish the new version of the package using:

```bash
python -m uv publish --username __token__
```
