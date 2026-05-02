from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

import samesame
from samesame import (
    AdverseShiftDetails,
    ShiftDetails,
    test_adverse_shift as run_adverse_shift_test,
    test_shift as run_shift_test,
)


def test_root_exports() -> None:
    assert hasattr(samesame, "test_shift")
    assert hasattr(samesame, "test_adverse_shift")
    assert hasattr(samesame, "weights")
    assert not hasattr(samesame, "advanced")
    assert not hasattr(samesame, "CTST")
    assert not hasattr(samesame, "DSOS")
    assert not hasattr(samesame, "ctst")
    assert not hasattr(samesame, "nit")


def test_signatures_are_keyword_only() -> None:
    shift_sig = inspect.signature(run_shift_test)
    adverse_sig = inspect.signature(run_adverse_shift_test)
    assert shift_sig.parameters["source"].kind is inspect.Parameter.KEYWORD_ONLY
    assert adverse_sig.parameters["source"].kind is inspect.Parameter.KEYWORD_ONLY


def test_test_shift_returns_shift_details(shift_samples: dict[str, np.ndarray]) -> None:
    result = run_shift_test(**shift_samples, n_resamples=64)
    assert isinstance(result, ShiftDetails)
    assert result.statistic_name == "roc_auc"
    assert isinstance(result.statistic, float)
    assert 0.0 <= result.pvalue <= 1.0
    assert result.null_distribution.shape == (64,)


def test_test_shift_requires_keyword_arguments(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        run_shift_test(shift_samples["source"], shift_samples["target"])


def test_test_shift_rejects_unknown_statistic(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="statistic must be one of"):
        run_shift_test(**shift_samples, statistic="f1")  # type: ignore[arg-type]


def test_binary_only_statistics_require_binary_scores(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="requires binary outlier scores"):
        run_shift_test(**shift_samples, statistic="balanced_accuracy")


@pytest.mark.parametrize("statistic", ["balanced_accuracy", "matthews_corrcoef"])
def test_binary_only_statistics_accept_binary_scores(
    binary_shift_samples: dict[str, np.ndarray],
    statistic: str,
) -> None:
    result = run_shift_test(**binary_shift_samples, statistic=statistic)  # type: ignore[arg-type]
    assert isinstance(result, ShiftDetails)
    assert result.statistic_name == statistic


def test_results_are_frozen(shift_samples: dict[str, np.ndarray]) -> None:
    result = run_shift_test(**shift_samples, n_resamples=64)
    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0.0  # type: ignore[misc]


def test_test_adverse_shift_requires_direction(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        run_adverse_shift_test(**confidence_samples)


def test_test_adverse_shift_rejects_unknown_direction(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="direction must be one of"):
        run_adverse_shift_test(
            **confidence_samples,
            direction="up-is-bad",  # type: ignore[arg-type]
        )


def test_test_adverse_shift_handles_higher_is_better(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    primary = run_adverse_shift_test(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
    )
    mirrored = run_adverse_shift_test(
        source=-confidence_samples["source"],
        target=-confidence_samples["target"],
        direction="higher-is-worse",
        n_resamples=64,
    )
    assert isinstance(primary, AdverseShiftDetails)
    assert primary.direction == "higher-is-better"
    assert np.isclose(primary.statistic, mirrored.statistic)
    assert np.isclose(primary.pvalue, mirrored.pvalue)


def test_shift_null_distribution_matches_n_resamples(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = run_shift_test(**shift_samples, n_resamples=99)
    assert result.null_distribution.shape == (99,)


def test_shift_supports_explicit_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    sample_weight = np.linspace(1.0, 3.0, 600)
    base = run_shift_test(**shift_samples, n_resamples=64)
    weighted = run_shift_test(**shift_samples, n_resamples=64, weights=sample_weight)
    assert isinstance(weighted, ShiftDetails)
    assert base.statistic != weighted.statistic


def test_shift_supports_membership_prob(
    shift_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.2, 0.8, size=600)
    base = run_shift_test(**shift_samples, n_resamples=64)
    contextual = run_shift_test(
        **shift_samples, n_resamples=64, membership_prob=probs, mode="source"
    )
    assert isinstance(contextual, ShiftDetails)
    assert base.statistic != contextual.statistic


def test_shift_rejects_both_weights_and_membership_prob(
    shift_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(7)
    probs = rng.uniform(0.2, 0.8, size=600)
    sample_weights = np.linspace(1.0, 2.0, 600)
    with pytest.raises(ValueError, match="Provide either weights or membership_prob"):
        run_shift_test(
            **shift_samples,
            n_resamples=64,
            weights=sample_weights,
            membership_prob=probs,
        )


def test_shift_weights_and_membership_prob_are_distinct(
    shift_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(7)
    probs = rng.uniform(0.2, 0.8, size=600)
    sample_weights = np.linspace(1.0, 2.0, 600)
    sw_result = run_shift_test(**shift_samples, n_resamples=64, weights=sample_weights)
    mp_result = run_shift_test(
        **shift_samples, n_resamples=64, membership_prob=probs, mode="source"
    )
    assert sw_result.statistic != mp_result.statistic


def test_adverse_shift_bayesian_opt_in(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    base = run_adverse_shift_test(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
    )
    bayesian = run_adverse_shift_test(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
        bayesian=True,
        rng=np.random.default_rng(42),
    )
    assert isinstance(base, AdverseShiftDetails)
    assert base.posterior is None
    assert base.bayes_factor is None
    assert bayesian.posterior is not None
    assert bayesian.posterior.shape == (64,)
    assert bayesian.bayes_factor is not None


def test_adverse_shift_supports_membership_prob(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(99)
    probs = rng.uniform(0.2, 0.8, size=500)
    base = run_adverse_shift_test(
        **confidence_samples, direction="higher-is-better", n_resamples=64
    )
    contextual = run_adverse_shift_test(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
        membership_prob=probs,
        mode="target",
    )
    assert isinstance(contextual, AdverseShiftDetails)
    assert base.statistic != contextual.statistic
