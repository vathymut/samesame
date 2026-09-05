# Monitor a credit model

An alarm should tell you whether the score moved toward worse outcomes, not
just that it moved. A lender trained on its safest book shows why.

A lender trains a default-risk model on its safest book: applicants the
bureau already smiles on. Then the book changes: a new partner channel, a
bolder marketing push, a market that cooled. Riskier applicants start
arriving, the model is still scoring, and two readings of the same dashboard
look identical from the outside: **the model decayed**, or **the population
changed**. One wants a retraining ticket; the other wants a business meeting.
Using one HELOC split (ExternalRiskEstimate > 63 vs ≤ 63; Gardner et al.,
2023), this example works through that story with three scores from the same
model; they answer different questions and arrive at different times.

| Signal | Requires labels? | Harmful direction | `worse` |
|--------|-------------------|-------------------|---------|
| Predicted risk | No | Higher risk | `higher` |
| Outlier score — confidence (`LogitGap`) | No | Lower certainty | `lower` |
| Prediction error (Brier) | Yes | Larger error | `higher` |

--8<-- "snippets/source-target.txt"

Which signal to lead with is a working choice; [Which signal when?](#which-signal-when) compares them. If
you are new to `samesame`, start with [Get started](../tutorials/get-started.md)
or with [Is the new drug good
enough?](../trials/check-drug-efficacy.md), the same test told as a clinical
trial with no model in sight.

## The dataset

HELOC stands for **home equity line of credit**: a revolving credit line
secured by the borrower's home. The data comes from the FICO Community
Explainable AI Challenge: anonymized credit-bureau features for each
applicant, and a target that records whether the borrower went 90 days past
due or worse at least once in the first 24 months of the account. We fetch
the 9,871 applications from the
[OpenML](https://openml.org/search?type=data&sort=runs&id=45554&status=active)
mirror (`data_id=45554`) with `fetch_openml`, so the examples run with no
manual download.

## The split

A HELOC applicant arrives with a risk estimate from a third-party bureau
service, the `ExternalRiskEstimate` feature. Higher values mean a safer
applicant. The challenge-winning model of the original FICO competition
zeroed in on a cutoff of **63** on that estimate, and the
[TableShift](https://tableshift.org) benchmark (Gardner et al., 2023) adopted
the same threshold to define its two populations. We do the same, and read it
as a deployment story:

- **Source** — 7,683 applicants with estimate > 63: the calmer book the
  lender trained on. Observed bad rate 43.5%.
- **Target** — 2,188 applicants with estimate ≤ 63: the riskier book that
  arrives after deployment. Observed bad rate 81.9%.

A model trained on the first group is deployed on the second: trained on
calm seas, sailing into a storm. Mean predicted default risk climbs from
about 44% to 73%, and the two rival readings return: model harm, or context
change? The three signals below give the evidence; the [weighting
guide](../weighting/weight-for-common-support.md) interrogates the alarm.

## Setup

The same HELOC split (source: training lower-risk, target: deployment
higher-risk) is reused for risk and confidence. The error tab uses a separate
random train/test split where the null of no error shift holds, so a large
p-value is expected.

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

## Signals

=== "Risk — no labels needed"

    Predicted risk is the model's `P(default)`; larger is worse, so
    `worse="higher"`. Declare the direction before testing; never pick it by
    p-value.

    ```python
    import numpy as np
    import samesame as ss

    --8<-- "snippets/heloc-split.py:heloc-domain"
    --8<-- "snippets/heloc-split.py:heloc-risk-model"

    # any shift? domain_prob is generic; harm on interpretable risk
    shift = ss.test_shift(
        source=domain_prob[split.values == 0],
        target=domain_prob[split.values == 1],
        rng=np.random.default_rng(12345),
    )
    harm = ss.test_harmful_shift(
        source=train_risk, target=deployment_risk,
        worse="higher", rng=np.random.default_rng(12345),
    )
    print(f"AUC {shift.statistic:.4f} p={shift.pvalue:.4f}")   # → 1.0000, 0.0002
    print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")    # → 0.2483, 0.0001
    ```

    The domain classifier separates the books almost perfectly (AUC 1.0000),
    which is expected because the split variable itself is a feature. The harm
    test confirms the move is toward higher risk, not just any change.

    | `test_shift` | `test_harmful_shift` | Meaning |
    |--------------|----------------------|---------|
    | sig. | sig. | changed **and** moved toward worse |
    | sig. | not sig. | changed, not clearly harmful |
    | not sig. | not sig. | no clear shift |
    | not sig. | sig. | rare — the tail test catches what the broad screen misses |

    Both tests provide evidence that the risk distribution changed and that the
    change is consistent with higher risk. This is a reason to investigate, not an
    automatic retraining decision. An AUC of `0.5` is chance; read the
    harmful-shift statistic against its null distribution and the model's risk
    scale (0–1). See [How the harm test
    works](../../explanation/harmful-shift-statistic.md).

=== "Outlier score — confidence — no labels needed"

    `LogitGap` is an **outlier score** for confidence: larger = more certain,
    so a drop (`worse="lower"`) is harm. It is the gap between the top logit
    and the mean of the remaining logits. The fitted risk model from the
    risk tab provides the probabilities.

    ```python
    --8<-- "snippets/heloc-split.py:heloc-domain"
    --8<-- "snippets/heloc-split.py:heloc-risk-model"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:imports"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:logit-gap"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:outlier-scores"
    import samesame as ss

    train_conf = outlier_scores_from_probabilities(rf_bad.oob_decision_function_)
    deploy_conf = outlier_scores_from_probabilities(rf_bad.predict_proba(X_deployment))

    harm = ss.test_harmful_shift(
        source=train_conf, target=deploy_conf,
        worse="lower",  # lower confidence = harm
        rng=np.random.default_rng(12345),
    )
    print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")  # → 0.0409, 1.0000
    ```

    There is no evidence of a harmful confidence drop; if anything, the
    direction of the statistic suggests the model is *more* certain on the
    deployment book. With a bad rate near 82%, predicted probabilities
    polarize toward the extremes, and certainty rides along. A model may
    become more confident while also becoming more risky: confidence is a
    complement to risk, not a substitute.

=== "Errors — needs labels"

    Once labels arrive, prediction error (Brier) is the clearest post-outcome
    check. The random split here is deliberately constructed under the null of
    no error shift, so a large p-value is expected. In a real deployment, a
    small p-value would provide evidence that prediction errors increased in
    the target.

    ```python
    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import samesame as ss

    fico = fetch_openml(data_id=45554, as_frame=True)
    X, y = fico.data, fico.target
    y_binary = (y == "Bad").astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.30, stratify=y_binary, random_state=12345,
    )
    rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10)
    rf.fit(X_train, y_train)
    train_prob = rf.oob_decision_function_[:, 1]
    test_prob = rf.predict_proba(X_test)[:, 1]

    brier_train = (y_train - train_prob) ** 2
    brier_test  = (y_test  - test_prob) ** 2
    harm = ss.test_harmful_shift(source=brier_train, target=brier_test, worse="higher", rng=np.random.default_rng(12345))
    print(f"Brier p={harm.pvalue:.4f}")  # → 0.2737
    ```

    This example provides no evidence that errors got worse. Brier score and log loss usually lead to similar conclusions, so either metric can be used for this comparison.

To reproduce these examples end to end, see the full runnable scripts in `examples/credit/_code/`.

## Which signal when?

- **Predicted risk** — use when the model output already represents the harmful outcome.
- **Outlier score — confidence** — use for an early warning when labels are not yet available.
- **Prediction error** — use for the clearest post-outcome accuracy check.

Risk and confidence moved without a single label arriving; that is what
makes them early. The error check waits for outcomes; here it runs on a
random split where nothing is wrong, and finds exactly that. Did the alarm
fire because comparable applicants got worse, or because incomparable ones
arrived? That is precisely the question [Weight for
common support](../weighting/weight-for-common-support.md) interrogates.
