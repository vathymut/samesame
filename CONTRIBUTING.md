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

Use `uv sync` to create a project virtual environment, if needed, and sync the project's dependencies.

```bash
git clone https://github.com/vathymut/samesame
cd samesame
uv sync --all-extras
```

#### Use a specific Python version (optional)

If you need a specific Python version, create a virtual environment for it and run sync:

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
uv run mkdocs serve
```

The server will be available at [http://localhost:8000](http://localhost:8000).

### Pushing Documentation to GitHub Pages

Run the following:

```bash
uv run mkdocs gh-deploy
```

### Updating the Package Version

```bash
uv version --bump   # or: uv version --bump patch|minor|major  |  uv version 0.5.0
```

This updates `pyproject.toml:3` (`version`) and `uv.lock`. Keep `mkdocs.yml:83`
(`extra.version`) in sync — bump it to the same `<new>` version in the same commit.

```bash
# example: bump to 0.4.1 and sync docs version
uv version --bump patch
# then edit mkdocs.yml: extra.version: 0.4.1
git add pyproject.toml uv.lock mkdocs.yml
git commit -m "chore(release): 0.4.1"
git push origin main  # or: origin develop -> merge to main before tagging
```

Verify locally before tagging:

```bash
uv run pytest
uv run ruff check .
```

### Creating a GitHub Release

> Must be executed **before** publishing to PyPI. Previous releases `v0.3.2` (`d59f4a1`)
> and `v0.4.0` (`20bd868`) were created this way: annotated tag on `main` → GitHub Release
> → `uv build` → `uv publish`.

```bash
# 1. Ensure main is up to date and version is bumped (above)
git checkout main && git pull
git tag -a v<new> -m "v<new>"
git push origin v<new>

# 2. Create the GitHub Release (requires gh auth login / GH_TOKEN)
gh release create v<new> --title v<new> --target main --generate-notes
# custom body (mirrors v0.4.0 pattern):
# gh release create v<new> --title v<new> --target main --notes "v<new> — summary
#
# Highlights:
# - ...
# - Bumps version from <old> (<sha>) to <new> (<sha>).
# - Site version explicit (mkdocs.yml:83 extra.version: <new>).
#
# PyPI: https://pypi.org/project/samesame/<new>/
# Docs: https://vathymut.github.io/samesame/
# Repo: https://github.com/vathymut/samesame/releases/tag/v<new>"

# 3. Verify
gh release view v<new>
curl -s https://api.github.com/repos/vathymut/samesame/releases/tags/v<new> | head -n 20
```

Alternatively create via UI: `https://github.com/vathymut/samesame/releases/new` → select tag `v<new>` → target `main` → generate notes → Publish.

### Building Python Package

```bash
uv build
```

### Publishing to PyPI

Following [these instructions](https://github.com/astral-sh/uv/issues/10878#issuecomment-3473401901),
we first log in securely as follows:

```bash
uv auth login upload.pypi.org
username: __token__
password: 
```

The latter assumes that the requisite credentials have been generated and
potentially, saved in a config file (e.g. see `.pypirc` file). Only then, we
publish the new version of the package using:

```bash
uv publish --username __token__
```

### Yanking Old Releases

PyPI does **not** support automatic retention ("keep last N") and does **not**
expose a token-based API for yanking (`pypi/warehouse#12708`). Yanking is
UI-only and must be done by a project **Owner** at:

`https://pypi.org/manage/project/samesame/releases/`

See <https://docs.pypi.org/project-management/yanking/> (PEP 592).

**When to yank vs delete**

* **Yank** (preferred, reversible): hides a release from `pip install samesame`
  but leaves it installable via `pip install samesame==0.1.0` with a warning.
  Does **not** free storage.
* **Delete** (irreversible, avoid): permanently removes files, breaks pinned
  installs, and the filename can never be reused. Only use to free project
  storage quota.

**Policy for `samesame`**: yank releases `< 0.2.0` (i.e. `0.1.0`, `0.1.1`,
`0.1.2`, `0.1.3`); keep `>= 0.2.1` installable.

**Manual steps**

1. Go to `https://pypi.org/manage/project/samesame/releases/`.
2. For each version `< 0.2.0`, click `Options` next to the release → `Yank`.
3. Set reason (shown in `pip` warnings and JSON/Simple APIs):

   ```
   Retired: yanked in favor of newer releases, please upgrade to samesame>=0.3.0
   ```

4. Confirm. Repeat for remaining old versions. Only **Owners** can yank
   (Maintainers cannot).

**Verify**

```bash
# JSON API: each file has "yanked": true/false and "yanked_reason"
curl -s https://pypi.org/pypi/samesame/json | python -c "
import urllib.request, json
with urllib.request.urlopen('https://pypi.org/pypi/samesame/json') as r:
    d=json.load(r)
    for v, files in sorted(d['releases'].items()):
        print(v, any(f.get('yanked') for f in files), next((f.get('yanked_reason') for f in files if f.get('yanked_reason')), ''))
"

# Simple API: yanked releases carry data-yanked="..."
curl -s https://pypi.org/simple/samesame/ | grep -i data-yanked

# Resolver: should skip yanked versions
pip index versions samesame
uv pip compile --dry-run -c 'samesame>=0.1.0' 2>&1 | head
```

Yanking is reversible via `Un-yank` in the same UI.
