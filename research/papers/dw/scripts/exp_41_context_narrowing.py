"""Experiment 4.1: domain-context narrowing (Cobb et al. 2022, S4.1) via samesame.

Null: C1 narrowing (sigma) or K=2 mixture, S1|C unchanged -> weights should
  suppress the unweighted false alarm (smaller s after weighting).
Alt: same C1 + mean shift eps=0.5 in deployed mode -> harm should persist.

Outputs CSV (one row per paired run) + summary JSON with ROPE stats.
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
    append_row_locked,
    delta_s,
    domain_probs,
    gen_41,
    parse_shard,
    rope_posterior,
    s_value,
)

import samesame as ss

C1_SETTINGS = ("sigma_0.25", "sigma_0.5", "sigma_1.0", "k2")


def single_run(c1, alt, i, n_source, n_target, n_resamples, seed):
    s = seed + i
    eps = 0.5 if alt else 0.0
    rng = np.random.default_rng(s)
    ss_s, tg_s, ss_c, tg_c = gen_41(rng, n_source, n_target, c1, eps)
    ru = ss.test_harm(
        ss_s, tg_s, worse="higher", n_resamples=n_resamples,
        rng=np.random.default_rng(s),
    )
    row = {
        "run": i, "c1": c1, "alt": int(alt),
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


def summarize(full: pd.DataFrame, key: str) -> dict:
    """Per-setting ROPE summary over the primary cell from merged rows."""
    summary: dict = {"primary": key, "settings": {}}
    for (c1, alt), df in full.groupby(["c1", "alt"]):
        tag = f"{c1}/{'alt' if int(alt) else 'null'}"
        post = rope_posterior(df[f"ds_{key}"].to_numpy())
        summary["settings"][tag] = {
            "median_p_unweighted": float(df["p_unweighted"].median()),
            "median_p_primary": float(df[f"p_{key}"].median()),
            "median_ds_primary": post["median_delta_s"],
            "equiv_rate_primary": post["equiv_rate"],
            "rope_posterior": post,
            "median_ess_src": float(df[f"ess_src_{key}"].median()),
            "median_ess_tgt": float(df[f"ess_tgt_{key}"].median()),
        }
        print(f"{tag}: med p_u={df['p_unweighted'].median():.4g} "
              f"med p_w={df[f'p_{key}'].median():.4g} "
              f"med ds={post['median_delta_s']:.2f} "
              f"equiv={post['equiv_rate']:.2f}")
    return summary


def run_setting(c1, alt, n_runs, n_source, n_target, n_resamples, seed,
                 outpath: Path | None = None, overwrite: bool = False,
                 shard: tuple[int, int] = (0, 1)):
    """Run paired runs, checkpointing one row at a time.

    If outpath exists and overwrite=False, completed run ids are loaded
    and skipped, so an aborted run resumes where it left off. Each
    completed run is appended + flushed immediately (lock-guarded, so
    --shard workers can share one CSV). A worker with shard=(I, M)
    owns run ids i with i % M == I.
    """
    worker, n_workers = shard
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
        if i % n_workers != worker:
            continue
        if i in done:
            continue
        row = single_run(c1, alt, i, n_source, n_target, n_resamples, seed)
        rows.append(row)
        if outpath is not None:
            append_row_locked(outpath, row)
        print(f"[{c1}/{'alt' if alt else 'null'}] run {i + 1}/{n_runs} done",
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
    ap.add_argument("--c1", type=str, nargs="*", default=list(C1_SETTINGS))
    ap.add_argument("--overwrite", action="store_true",
                    help="Ignore existing per-setting CSVs and start from run 0.")
    ap.add_argument("--shard", type=str, default="0/1",
                    help="Shard spec 'I/M': this worker owns run ids i "
                         "with i %% M == I. Workers share one CSV safely.")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip running; rebuild exp41_all.csv + summary "
                         "from existing per-setting CSVs.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    key = f"{PRIMARY['classifier']}/{PRIMARY['reweight']}/{PRIMARY['shrinkage']}"
    shard = parse_shard(args.shard)
    if args.merge_only:
        all_frames = []
        for c1 in args.c1:
            for alt in ("null", "alt"):
                p = outdir / f"exp41_{c1}_{alt}.csv"
                if p.exists():
                    all_frames.append(pd.read_csv(p))
        if not all_frames:
            raise SystemExit(f"nothing to merge in {outdir}")
        full = pd.concat(all_frames)
        full.to_csv(outdir / "exp41_all.csv", index=False)
        (outdir / "exp41_summary.json").write_text(
            json.dumps(summarize(full, key), indent=2))
        print(f"wrote {outdir}/exp41_summary.json from {len(full)} rows")
        return
    for c1 in args.c1:
        for alt, n_runs in (("null", args.n_null), ("alt", args.n_alt)):
            if n_runs == 0:
                continue
            outpath = outdir / f"exp41_{c1}_{alt}.csv"
            run_setting(c1, alt == "alt", n_runs, args.n_source,
                        args.n_target, args.n_resamples, args.seed,
                        outpath=outpath, overwrite=args.overwrite,
                        shard=shard)
    if shard[1] == 1:
        all_frames = []
        for c1 in args.c1:
            for alt in ("null", "alt"):
                p = outdir / f"exp41_{c1}_{alt}.csv"
                if p.exists():
                    all_frames.append(pd.read_csv(p))
        full = pd.concat(all_frames)
        full.to_csv(outdir / "exp41_all.csv", index=False)
        (outdir / "exp41_summary.json").write_text(
            json.dumps(summarize(full, key), indent=2))
        print(f"wrote {outdir}/exp41_summary.json")
    else:
        print(f"shard {args.shard} done; rerun with --merge-only "
              f"after all workers finish")


if __name__ == "__main__":
    main()
