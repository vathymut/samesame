from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

import samesame
from samesame import (
    AdverseShiftResult,
    ShiftResult,
    test_adverse_shift as run_adverse_shift_test,
    test_shift as run_shift_test,
)
from samesame.advanced import AdverseShiftDetails, ShiftDetails


def test_root_exports() -> None:
    assert hasattr(samesame, "test_shift")
    assert hasattr(samesame, "test_adverse_shift")
    assert hasattr(samesame, "advanced")
    assert not hasattr(samesame, "CTST")
    assert not hasattr(samesame, "DSOS")
    assert not hasattr(samesame, "ctst")
    assert not hasattr(samesame, "nit")


def test_primary_signatures_are_keyword_only() -> None:
    shift_signature = inspect.signature(run_shift_test)
    adverse_signature = inspect.signature(run_adverse_shift_test)
    assert (
        shift_signature.parameters["reference"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        adverse_signature.parameters["reference"].kind is inspect.Parameter.KEYWORD_ONLY
    )


def test_test_shift_defaults_to_roc_auc(shift_samples: dict[str, np.ndarray]) -> None:
    result = run_shift_test(**shift_samples)

    assert isinstance(result, ShiftResult)
    assert result.statistic_name == "roc_auc"
    assert isinstance(result.statistic, float)
    assert 0.0 <= result.pvalue <= 1.0


def test_test_shift_requires_keyword_arguments(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        run_shift_test(shift_samples["reference"], shift_samples["candidate"])


def test_test_shift_rejects_unknown_statistic(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="statistic must be one of"):
        run_shift_test(**shift_samples, statistic="f1")  # type: ignore[arg-type]


def test_binary_only_statistics_require_binary_scores(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="requires binary score vectors"):
        run_shift_test(**shift_samples, statistic="balanced_accuracy")


@pytest.mark.parametrize("statistic", ["balanced_accuracy", "matthews_corrcoef"])
def test_binary_only_statistics_accept_binary_scores(
    binary_shift_samples: dict[str, np.ndarray],
    statistic: str,
) -> None:
    result = run_shift_test(**binary_shift_samples, statistic=statistic)  # type: ignore[arg-type]
    assert isinstance(result, ShiftResult)
    assert result.statistic_name == statistic


def test_primary_results_are_frozen(shift_samples: dict[str, np.ndarray]) -> None:
    result = run_shift_test(**shift_samples)
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
    )
    mirrored = samesame.advanced.test_adverse_shift(
        reference=-confidence_samples["reference"],
        candidate=-confidence_samples["candidate"],
        direction="higher-is-worse",
        n_resamples=9999,
    )

    assert isinstance(primary, AdverseShiftResult)
    assert primary.direction == "higher-is-better"
    assert np.isclose(primary.statistic, mirrored.statistic)
    assert np.isclose(primary.pvalue, mirrored.pvalue)


def test_primary_api_rejects_expert_keywords(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        run_shift_test(**shift_samples, sample_weight=np.ones(600))  # type: ignore[call-arg]


def test_advanced_shift_returns_detail_result(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = samesame.advanced.test_shift(**shift_samples, n_resamples=64)

    assert isinstance(result, ShiftDetails)
    assert result.statistic_name == "roc_auc"
    assert result.null_distribution.shape == (64,)


def test_advanced_shift_supports_sample_weight(
    shift_samples: dict[str, np.ndarray],
) -> None:
    sample_weight = np.linspace(1.0, 3.0, 600)
    base = samesame.advanced.test_shift(**shift_samples, n_resamples=64)
    weighted = samesame.advanced.test_shift(
        **shift_samples,
        n_resamples=64,
        sample_weight=sample_weight,
    )

    assert isinstance(weighted, ShiftDetails)
    assert base.statistic != weighted.statistic


def test_advanced_adverse_shift_bayesian_opt_in(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    base = samesame.advanced.test_adverse_shift(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
    )
    bayesian = samesame.advanced.test_adverse_shift(
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
