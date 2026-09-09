"""Experiment B1: benign private mass in target-only territory (1-D overlap DGP).

Port of draw_overlap_dataset + `synthetic calibration` from the earlier
manuscript suite, re-run on the current test_harm/domain_weights API.

Null (effect=0.0): target-private lump at +3 with no shared shift -> the
unweighted test should false-alarm as severity grows, while common-support
weighting (both) stays silent. severity=0.0 is the calibration anchor
(identical distributions -> all modes silent).

Outputs CSV (one row per paired run) + summary JSON with reject rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    REWEIGHTS,
    SHRINKAGES,
    CLASSIFIERS,
    PRIMARY,
    delta_s,
    domain_probs,
    gen_overlap,
    s_value,
)

import samesame as ss

ALPHA = 0.05
SEVERITY_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)


def single_run(severity, effect, i, n_source, n_target, n_resamples, seed):
    s = seed + i
    rng = np.random.default_rng(s)
    ss_s, tg_s, ss_c, tg_c = gen_overlap(
        rng, n_source, n_target, severity=severity, effect=effect)
    ru = ss.test_harm(
        ss_s, tg_s, worse="higher", n_resamples=n_resamples,
        rng=np.random.default_rng(s),
    )
    row = {
        "run": i, "severity": severity, "effect": effect,
        "p_unweighted": float(ru.pvalue),
        "s_unweighted": float(s_value(ru.pvalue)),
    }
    for clf in CLASSIFIERS:
        sp, tp, auc = domain_probs(ss_c, tg_c, clf, s)
        row[f"auc_{clf}"] = auc
        for rw in REWEIGHTS:
            for lam in SHRINKAGES:
                w = ss.domain_weights(
                    source=sp, target=tp, reweight=rw, shrinkage=lam)
                ess = w.effective_sample_size()
                rr = ss.test_harm(
                    ss_s, tg_s, worse="higher", weights=w,
                    n_resamples=n_resamples,
                    rng=np.random.default_rng(s))
                key = f"{clf}/{rw}/{lam}"
                row[f"p_{key}"] = float(rr.pvalue)
                row[f"ds_{key}"] = delta_s(float(rr.pvalue),
                                           float(ru.pvalue))
                row[f"ess_src_{key}"] = float(ess.source)
                row[f"ess_tgt_{key}"] = float(ess.target)
    return row


def run_setting(severity, effect, n_runs, n_source, n_target, n_resamples,
                seed, outpath: Path | None = None, overwrite: bool = False):
    """Run paired runs, checkpointing one row at a time (resumable)."""
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
        row = single_run(severity, effect, i, n_source, n_target,
                         n_resamples, seed)
        rows.append(row)
        if outpath is not None:
            pd.DataFrame([row]).to_csv(
                outpath, mode="a", header=not outpath.exists(), index=False)
        print(f"[sev={severity}/eff={effect}] run {i + 1}/{n_runs} done",
              flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--n-resamples", type=int, default=999)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-source", type=int, default=180)
    ap.add_argument("--n-target", type=int, default=180)
    ap.add_argument("--effect", type=float, default=0.0)
    ap.add_argument("--outdir", type=str, default="outputs_b1")
    ap.add_argument("--severity", type=float, nargs="*",
                    default=list(SEVERITY_GRID))
    ap.add_argument("--overwrite", action="store_true",
                    help="Ignore existing per-setting CSVs and start from run 0.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    key = f"{PRIMARY['classifier']}/{PRIMARY['reweight']}/{PRIMARY['shrinkage']}"
    summary = {"primary": key, "effect": args.effect, "settings": {}}
    all_frames = []
    for sev in args.severity:
        if args.n_runs == 0:
            continue
        tag = f"{sev}".replace(".", "p")
        outpath = outdir / f"expb1_sev{tag}.csv"
        df = run_setting(sev, args.effect, args.n_runs, args.n_source,
                         args.n_target, args.n_resamples, args.seed,
                         outpath=outpath, overwrite=args.overwrite)
        all_frames.append(df)
        summary["settings"][f"{sev}"] = {
            "reject_unweighted": float((df["p_unweighted"] < ALPHA).mean()),
            "reject_primary": float((df[f"p_{key}"] < ALPHA).mean()),
            "median_p_unweighted": float(df["p_unweighted"].median()),
            "median_p_primary": float(df[f"p_{key}"].median()),
            "median_ds_primary": float(df[f"ds_{key}"].median()),
            "median_ess_src": float(df[f"ess_src_{key}"].median()),
            "median_ess_tgt": float(df[f"ess_tgt_{key}"].median()),
        }
        st = summary["settings"][f"{sev}"]
        print(f"sev={sev}: rej_u={st['reject_unweighted']:.2f} "
              f"rej_w={st['reject_primary']:.2f} "
              f"med p_u={st['median_p_unweighted']:.4g} "
              f"med p_w={st['median_p_primary']:.4g}")
    pd.concat(all_frames).to_csv(outdir / "expb1_all.csv", index=False)
    (outdir / "expb1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {outdir}/expb1_summary.json")


if __name__ == "__main__":
    main()
