"""Experiment 4.2: subpopulation prevalence shift (Cobb et al. 2022, S4.2).

Null: pi1 in {0.2, 0.8} only (prevalence change, no conditional change).
Alt: same pi1 + mean shift +[0.6, 0] on component 2.
S = X[:, 0] (worse="higher"); C = P(comp1|X) from per-run 2-comp GMM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import samesame as ss
from common import (
    REWEIGHTS,
    SHRINKAGES,
    CLASSIFIERS,
    PRIMARY,
    delta_s,
    domain_probs,
    gen_42,
    rope_posterior,
    s_value,
)

PI1_SETTINGS = (0.2, 0.8)


def single_run(pi1, alt, i, n_source, n_target, n_resamples, seed):
    s = seed + i
    rng = np.random.default_rng(s)
    ss_s, tg_s, ss_c, tg_c, _, _ = gen_42(rng, n_source, n_target, pi1, alt)
    ru = ss.test_harm(
        ss_s, tg_s, worse="higher", n_resamples=n_resamples,
        rng=np.random.default_rng(s))
    row = {"run": i, "pi1": pi1, "alt": int(alt),
           "p_unweighted": float(ru.pvalue),
           "s_unweighted": float(s_value(ru.pvalue))}
    for clf in CLASSIFIERS:
        sp, tp, auc = domain_probs(ss_c, tg_c, clf, s)
        row[f"auc_{clf}"] = auc
        for rw in REWEIGHTS:
            for lam in SHRINKAGES:
                w = ss.domain_weights(source=sp, target=tp,
                                      reweight=rw, shrinkage=lam)
                ess = w.effective_sample_size()
                rr = ss.test_harm(
                    ss_s, tg_s, worse="higher", weights=w,
                    n_resamples=n_resamples,
                    rng=np.random.default_rng(s))
                k = f"{clf}/{rw}/{lam}"
                row[f"p_{k}"] = float(rr.pvalue)
                row[f"ds_{k}"] = delta_s(float(rr.pvalue),
                                         float(ru.pvalue))
                row[f"ess_src_{k}"] = float(ess.source)
                row[f"ess_tgt_{k}"] = float(ess.target)
    return row


def run_setting(pi1, alt, n_runs, n_source, n_target, n_resamples, seed,
                outpath: Path | None = None, overwrite: bool = False):
    """Paired runs with per-run checkpointing; resume skips completed runs."""
    done: set[int] = set()
    if outpath is not None and outpath.exists() and not overwrite:
        try:
            done = set(pd.read_csv(outpath, usecols=["run"])["run"].tolist())
        except Exception:
            done = set()
    elif outpath is not None and overwrite and outpath.exists():
        outpath.unlink()
    rows = []
    if outpath is not None and outpath.exists() and not overwrite:
        rows = pd.read_csv(outpath).to_dict("records")
    for i in range(n_runs):
        if i in done:
            continue
        row = single_run(pi1, alt, i, n_source, n_target, n_resamples, seed)
        rows.append(row)
        if outpath is not None:
            pd.DataFrame([row]).to_csv(
                outpath, mode="a", header=not outpath.exists(), index=False)
        print(f"[pi={pi1}/{'alt' if alt else 'null'}] run {i + 1}/{n_runs} done",
              flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=500)
    ap.add_argument("--n-alt", type=int, default=500)
    ap.add_argument("--n-resamples", type=int, default=999)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-source", type=int, default=2000)
    ap.add_argument("--n-target", type=int, default=500)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--overwrite", action="store_true",
                    help="Ignore existing per-setting CSVs and start from run 0.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    key = f"{PRIMARY['classifier']}/{PRIMARY['reweight']}/{PRIMARY['shrinkage']}"
    summary = {"primary": key, "settings": {}}
    frames = []
    for pi1 in PI1_SETTINGS:
        for alt, n_runs in (("null", args.n_null), ("alt", args.n_alt)):
            if n_runs == 0:
                continue
            outpath = outdir / f"exp42_pi{pi1}_{alt}.csv"
            df = run_setting(pi1, alt == "alt", n_runs, args.n_source,
                             args.n_target, args.n_resamples, args.seed,
                             outpath=outpath, overwrite=args.overwrite)
            frames.append(df)
            post = rope_posterior(df[f"ds_{key}"].to_numpy())
            summary["settings"][f"pi{pi1}/{alt}"] = {
                "median_p_unweighted": float(df["p_unweighted"].median()),
                "median_p_primary": float(df[f"p_{key}"].median()),
                "median_ds_primary": post["median_delta_s"],
                "equiv_rate_primary": post["equiv_rate"],
                "rope_posterior": post,
            }
            print(f"pi={pi1}/{alt}: med p_u={df['p_unweighted'].median():.4g} "
                  f"med p_w={df[f'p_{key}'].median():.4g} "
                  f"med ds={post['median_delta_s']:.2f}")
    pd.concat(frames).to_csv(outdir / "exp42_all.csv", index=False)
    (outdir / "exp42_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {outdir}/exp42_summary.json")


if __name__ == "__main__":
    main()
