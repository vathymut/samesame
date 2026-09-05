# Local TeX Setup

The manuscript scaffold now builds locally on this machine with BasicTeX plus a small user-level package overlay.

## Recommended local setup on macOS

Use Homebrew BasicTeX as the smallest practical starting point:

```bash
brew install --cask basictex
sudo installer -pkg /opt/homebrew/Caskroom/basictex/*/mactex-basictex-*.pkg -target /
```

If Homebrew reports the cask as installed but `pdflatex` is still missing, that usually means the `.pkg` was downloaded but not yet run. The manual `installer -pkg ... -target /` step above addresses that case.

After the package install, compile from `research/papers/dw/` with:

```bash
make
```

The `research/papers/dw/Makefile` prefers `/Library/TeX/texbin` automatically on macOS, so a shell restart is not required for `make` to find `pdflatex` and `bibtex`.

## Package-level fixes that worked here

The ICML template required at least one extra package beyond the BasicTeX core:

```bash
/Library/TeX/texbin/tlmgr init-usertree
/Library/TeX/texbin/tlmgr --usermode install forloop
```

`cleveref.sty` was also needed. If `tlmgr --usermode install cleveref` works in your environment, use that. If it does not, copying `cleveref.sty` into `~/Library/texmf/tex/latex/cleveref/` is a workable fallback.

`latexmk` is optional for this repository because the Makefile already runs the explicit `pdflatex -> bibtex -> pdflatex -> pdflatex` sequence.

## Why this route

- `basictex` is much smaller than full MacTeX.
- The paper workspace already vendors the ICML style files locally.
- The remaining missing pieces are small package-level additions rather than a second large TeX distribution.

## Alternative

If you prefer a fully managed online build, upload the `research/papers/dw/` directory to Overleaf and compile there.
