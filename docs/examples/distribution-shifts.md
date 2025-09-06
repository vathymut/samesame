# Test for distribution shifts

Given two datasets `sample_P` and `sample_Q` from distributions $P$ and 
$Q$, the goal is to estimate a $p$-value for the null hypothesis of equal
distribution $P=Q$. `samesame` implements classifier two-sample tests (CTSTs)
for this use case. But why, you ask? I wax lyrical about CTSTs
[here](https://vathymut.org/posts/2022-01-22-in-gentle-praise-of-modern-tests/). 
The tl;dr is they are powerful, modular (flexible) and easy to explain, the 
elusive trifecta.

CTSTs assign your samples to different classes (you give them labels). `sample_P` 
is the positive class, and `sample_Q`, the negative one or vice versa. Then, you 
fit your favorite classifier to see if it can reliably predict the labels. If it 
can, it means the two samples are probably different. If it cannot, the two 
are more likely similar enough.

## Data

Let's see it in action. First, generate some data.

```python
from sklearn.datasets import make_classification
# Create a small dataset
X, y = make_classification(
    n_samples=100,
    n_features=4,
    n_classes=2,
    random_state=123_456,
)
```
The two samples, `sample_P` and `sample_Q`, are concatenated into a single
dataset, `X`. The binary labels, `y`, indicate sample membership, which examples
belong to which sample.

## Cross-fitted predictions

Let's run the tests using cross-fitting. Cross-fitting estimates the
performance of a model on unseen data by using out-of-sample predictions.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
# Get cross-fitted predictions
y_gb = cross_val_predict(
    estimator=HistGradientBoostingClassifier(random_state=123_456),
    X=X,
    y=y,
    cv=10,
    method='predict_proba',
)[:, 1]  # Get probabilities for the positive class
# Binarize predictions to be compatible with some metrics
from sklearn.preprocessing import binarize
y_gb_binary = binarize(y_gb.reshape(-1, 1), threshold=0.5).ravel()
```

## Classifier two-sample tests (CTSTs)

We can turn binary performance metrics into statistical tests like so:

```python
from samesame.ctst import CTST
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

metrics = [balanced_accuracy_score, matthews_corrcoef, roc_auc_score]
for metric in metrics:
    predicted = y_gb_binary if metric != roc_auc_score else y_gb
    ctst = CTST(actual=y, predicted=predicted, metric=metric)
    print(f"{metric.__name__}")
    print(f"\t statistic: {ctst.statistic:.2f}")
    print(f"\t p-value: {ctst.pvalue:.2f}")
```

```
balanced_accuracy_score
	 statistic: 0.86
	 p-value: 0.00
matthews_corrcoef
	 statistic: 0.72
	 p-value: 0.00
roc_auc_score
	 statistic: 0.93
	 p-value: 0.00
```

In all 3 cases, we reject the null hypothesis of equal distribution $P=Q$.
The classifier, `HistGradientBoostingClassifier`, was able to distinguish
between the two samples.

## Out-of-bag predictions

Instead of cross-fitting, we can also use out-of-bag predictions. Both
cross-fitted and out-of-bag predictions
[don't lose (too many) samples to estimation](https://www.cell.com/patterns/fulltext/S2666-3899(22)00237-9) whereas sample splitting typically uses half the data for 
estimation and half for inference, which incurs a loss of statistical power.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=123_456,
    oob_score=True,
    min_samples_leaf=10,
)
# Get out-of-bag predictions
y_rf = rf.fit(X, y).oob_decision_function_[:, 1]
y_rf_binary = binarize(y_rf.reshape(-1, 1), threshold=0.5).ravel()
for metric in metrics:
    predicted = y_rf_binary if metric != roc_auc_score else y_rf
    ctst = CTST(actual=y, predicted=predicted, metric=metric)
    print(f"{metric.__name__}")
    print(f"\t statistic: {ctst.statistic:.2f}")
    print(f"\t p-value: {ctst.pvalue:.2f}")
```

As before, we reject the null hypothesis of equal distribution $P=Q$.
The classifier, `RandomForestClassifier`, was able to tell apart the
two samples.

```
balanced_accuracy_score
	 statistic: 0.88
	 p-value: 0.00
matthews_corrcoef
	 statistic: 0.76
	 p-value: 0.00
roc_auc_score
	 statistic: 0.94
	 p-value: 0.00
```

## Explanations

To explain these results, we can use
[explainable methods](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
on the fitted classifiers above e.g. `rf`. These help answer questions
such as which features contribute the most to distribution shift (feature
importance).

## Conclusion

And voilà! You have successfully run classifier two-sample tests
(CTSTs). Note the flexibility (modularity) of the approach. You can use both 
cross-fitted and out-of-bag predictions, instead of sample splitting, for
inference. You can use any classifier (e.g., `HistGradientBoostingClassifier`, 
`RandomForestClassifier`, etc.), and any binary classification metric
(e.g., `AUC`, `balanced accuracy`, `Matthews correlation coefficient`, etc.).
