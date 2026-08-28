# How to: Monitor prediction errors once labels arrive

Use this when you have ground truth for both groups and want to know: is the model worse on target?

With labels, prediction error is the cleanest signal.

## Why this signal

Prediction errors turn quality into a numeric score:

- **Brier score** — squared error on the predicted probability
- **Log-loss** — penalizes confident mistakes more heavily

For both, larger is worse, so they fit `ss.test_harmful_shift(..., worse="higher")` (or `ss.Worse.HIGHER`). See [Glossary: `worse`](../../explanation/glossary.md#worse).

## Setup

This guide uses HELOC with a stratified random split. Unlike the risk/confidence guides (which split by risk level), both groups here come from the same population — the null is true and p-values should be large. **Source** = training split; **Target** = test split. See [Glossary](../../explanation/glossary.md#source-and-target).

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
    X,
    y_binary,
    test_size=0.30,
    stratify=y_binary,
    random_state=12345,
)
```

--8<-- "snippets/honest-scores-ref.txt"

## Step 1 — Fit the model and get honest predictions

Use out-of-bag predictions for source so errors aren't optimistic (see [honest scores](../../explanation/glossary.md#honest-out-of-sample-scores)).

```python
rf = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf.fit(X_train, y_train)

train_prob = rf.oob_decision_function_[:, 1]
test_prob = rf.predict_proba(X_test)[:, 1]
```

## Step 2 — Turn predictions into error scores

```python
brier_train = (y_train - train_prob) ** 2
brier_test = (y_test - test_prob) ** 2

eps = 1e-10
train_prob_clipped = np.clip(train_prob, eps, 1 - eps)
test_prob_clipped = np.clip(test_prob, eps, 1 - eps)

logloss_train = -(
    y_train * np.log(train_prob_clipped)
    + (1 - y_train) * np.log(1 - train_prob_clipped)
)
logloss_test = -(
    y_test * np.log(test_prob_clipped)
    + (1 - y_test) * np.log(1 - test_prob_clipped)
)
```

## Step 3 — Are errors worse on target?

```python
harm_brier = ss.test_harmful_shift(
    source=brier_train,  # training errors
    target=brier_test,   # test errors — the target you are evaluating
    worse="higher",
    rng=np.random.default_rng(12345),
)

harm_logloss = ss.test_harmful_shift(
    source=logloss_train,
    target=logloss_test,
    worse="higher",
    rng=np.random.default_rng(12345),
)

print(f"Brier p-value:    {harm_brier.pvalue:.4f}")
print(f"Log-loss p-value: {harm_logloss.pvalue:.4f}")
```

On this random split, expect large p-values — no evidence the model is worse on target.

## Read the result

--8<-- "snippets/pvalue-guidance.txt"

Brier and log-loss often agree here. `ss.test_harmful_shift` is rank-based, so similarly ordered signals give similar p-values.

??? example "Full script — copy and run"
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

    rf = RandomForestClassifier(
        n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
    )
    rf.fit(X_train, y_train)

    train_prob = rf.oob_decision_function_[:, 1]
    test_prob = rf.predict_proba(X_test)[:, 1]

    brier_train = (y_train - train_prob) ** 2
    brier_test = (y_test - test_prob) ** 2

    eps = 1e-10
    train_clipped = np.clip(train_prob, eps, 1 - eps)
    test_clipped = np.clip(test_prob, eps, 1 - eps)
    logloss_train = -(y_train * np.log(train_clipped) + (1 - y_train) * np.log(1 - train_clipped))
    logloss_test = -(y_test * np.log(test_clipped) + (1 - y_test) * np.log(1 - test_clipped))

    for name, s, t in [("Brier", brier_train, brier_test), ("Log-loss", logloss_train, logloss_test)]:
        harm = ss.test_harmful_shift(source=s, target=t, worse="higher", rng=np.random.default_rng(12345))
        print(f"{name} p-value: {harm.pvalue:.4f}")
    ```

    Training probabilities use `oob_decision_function_` so errors are honest
    (out of sample). See [Glossary](../../explanation/glossary.md#honest-out-of-sample-scores).

## Next steps

- Without labels, use [Monitor predicted credit risk](monitor-credit-risk.md) or [Monitor model confidence](monitor-model-confidence.md).
- If overlap is poor, see [Focus harmful-shift testing on common support](../weighting/source-reweighting.md).

??? tip "Why clip probabilities here vs domain weights?"
    Log-loss uses `log(p)` — clipping to `[1e-10, 1-1e-10]` avoids `log(0)` without changing the ranking the test uses. This is *separate* from `domain_weights` clipping to `[1e-6, 1-1e-6]` (which keeps density ratios finite).
