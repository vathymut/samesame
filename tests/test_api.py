from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import samesame as ss
from samesame import shift
from samesame.shift import HarmfulShiftResult, ShiftResult
from samesame.weights import ImportanceWeights, domain_weights


def test_root_exports() -> None:
    assert ss.test_shift is shift.test_shift
    assert ss.test_harmful_shift is shift.test_harmful_shift
    assert ss.domain_weights is domain_weights
    assert ss.ImportanceWeights is ImportanceWeights
    assert ss.shift is shift
    assert ss.weights is ss.weights
    assert set(ss.__all__) == {
        "ImportanceWeights",
        "test_harmful_shift",
        "test_shift",
        "domain_weights",
        "shift",
        "weights",
    }


def test_detect_shift_signature_is_minimal() -> None:
    params = set(inspect.signature(shift.test_shift).parameters)
    assert params == {"source", "target", "n_resamples", "rng", "weights"}


def test_detect_harm_signature_is_minimal() -> None:
    params = set(inspect.signature(shift.test_harmful_shift).parameters)
    assert params == {
        "source",
        "target",
        "worse",
        "n_resamples",
        "rng",
        "weights",
    }


def test_flat_harm_detection_matches_namespace(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    flat = ss.test_harmful_shift(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
    )
    namespaced = shift.test_harmful_shift(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
    )
    assert flat.statistic == namespaced.statistic
    assert flat.pvalue == namespaced.pvalue


def test_signatures_take_positional_source_target() -> None:
    shift_sig = inspect.signature(shift.test_shift)
    harm_sig = inspect.signature(shift.test_harmful_shift)
    assert (
        shift_sig.parameters["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert harm_sig.parameters["source"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_detect_shift_returns_shift_result(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.test_shift(**shift_samples, n_resamples=64)
    assert isinstance(result, ShiftResult)
    assert isinstance(result.statistic, float)
    assert 0.0 <= result.pvalue <= 1.0
    assert result.null_distribution.shape == (64,)


def test_detect_shift_accepts_positional_source_target(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.test_shift(shift_samples["source"], shift_samples["target"])
    assert isinstance(result, ShiftResult)


def test_results_are_frozen(shift_samples: dict[str, np.ndarray]) -> None:
    result = shift.test_shift(**shift_samples, n_resamples=64)
    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0.0  # type: ignore[misc]


def test_detect_harmful_shift_requires_higher_is_worse(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        shift.test_harmful_shift(**confidence_samples)


def test_detect_harmful_shift_accepts_higher_is_worse(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.test_harmful_shift(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
    )
    assert isinstance(result, HarmfulShiftResult)
    assert isinstance(result, ShiftResult)
    assert result.worse == "lower"


def test_detect_harm_rejects_non_bool_higher_is_worse(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="worse must be"):
        shift.test_harmful_shift(
            **confidence_samples,
            worse="sideways",  # type: ignore[arg-type]
        )


def test_detect_harm_handles_higher_is_better(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    primary = shift.test_harmful_shift(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
    )
    mirrored = shift.test_harmful_shift(
        source=-confidence_samples["source"],
        target=-confidence_samples["target"],
        worse="higher",
        n_resamples=64,
    )
    assert isinstance(primary, HarmfulShiftResult)
    assert primary.worse == "lower"
    assert np.isclose(primary.statistic, mirrored.statistic)
    assert np.isclose(primary.pvalue, mirrored.pvalue)


def test_shift_null_distribution_matches_n_resamples(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.test_shift(**shift_samples, n_resamples=99)
    assert result.null_distribution.shape == (99,)


def test_shift_supports_explicit_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    source, target = shift_samples["source"], shift_samples["target"]
    sample_weight = ImportanceWeights(
        source=np.linspace(1.0, 3.0, len(source)),
        target=np.linspace(1.0, 3.0, len(target)),
    )
    base = shift.test_shift(**shift_samples, n_resamples=64)
    weighted = shift.test_shift(**shift_samples, n_resamples=64, weights=sample_weight)
    assert isinstance(weighted, ShiftResult)
    assert base.statistic != weighted.statistic


def test_shift_rejects_wrong_length_importance_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    source, target = shift_samples["source"], shift_samples["target"]
    sample_weight = ImportanceWeights(
        source=np.ones(len(source) - 1),
        target=np.ones(len(target)),
    )
    with pytest.raises(ValueError, match="weights.source has wrong length"):
        shift.test_shift(**shift_samples, weights=sample_weight)


def test_shift_rejects_invalid_importance_weight_values(
    shift_samples: dict[str, np.ndarray],
) -> None:
    source, target = shift_samples["source"], shift_samples["target"]
    with pytest.raises(ValueError, match="weights.target must contain only finite"):
        sample_weight = ImportanceWeights(
            source=np.ones(len(source)),
            target=np.full(len(target), np.inf),
        )
        shift.test_shift(**shift_samples, weights=sample_weight)


def test_shift_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="source must contain only finite"):
        shift.test_shift(
            source=np.array([0.1, np.nan, 0.3]),
            target=np.array([0.4, 0.5, 0.6]),
        )


def test_shift_supports_contextual_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(42)
    source, target = shift_samples["source"], shift_samples["target"]
    source_prob = rng.uniform(0.2, 0.5, size=len(source))
    target_prob = rng.uniform(0.5, 0.8, size=len(target))
    weights = domain_weights(source=source_prob, target=target_prob, reweight="source")
    base = shift.test_shift(**shift_samples, n_resamples=64)
    contextual = shift.test_shift(**shift_samples, n_resamples=64, weights=weights)
    assert isinstance(contextual, ShiftResult)
    assert base.statistic != contextual.statistic


def test_detect_harm_has_no_posterior_fields(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.test_harmful_shift(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
        rng=np.random.default_rng(0),
    )
    assert isinstance(result, HarmfulShiftResult)
    assert not hasattr(result, "posterior")
    assert not hasattr(result, "bayes_factor")


def test_harm_detection_supports_importance_weights(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(99)
    source, target = confidence_samples["source"], confidence_samples["target"]
    source_prob = rng.uniform(0.2, 0.5, size=len(source))
    target_prob = rng.uniform(0.5, 0.8, size=len(target))
    weights = domain_weights(source=source_prob, target=target_prob, reweight="target")
    base = shift.test_harmful_shift(**confidence_samples, worse="lower", n_resamples=64)
    contextual = shift.test_harmful_shift(
        **confidence_samples,
        worse="lower",
        n_resamples=64,
        weights=weights,
    )
    assert isinstance(contextual, HarmfulShiftResult)
    assert base.statistic != contextual.statistic


def test_shift_result_repr_omits_null_distribution(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.test_shift(**shift_samples, n_resamples=64)
    rendered = repr(result)
    assert "null_distribution" not in rendered
    assert "ShiftResult(" in rendered
    assert "statistic=" in rendered
    assert "pvalue=" in rendered


def test_harm_result_repr_includes_higher_is_worse(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = shift.test_harmful_shift(
        **confidence_samples,
        worse="higher",
        n_resamples=64,
    )
    assert "worse='higher'" in repr(result)


def test_rng_accepts_integer_seed(shift_samples: dict[str, np.ndarray]) -> None:
    # int seeds are a delightful UX — they must be accepted and reproducible
    r1 = shift.test_shift(**shift_samples, rng=42, n_resamples=64)
    r2 = shift.test_shift(**shift_samples, rng=42, n_resamples=64)
    assert r1.statistic == r2.statistic
    assert r1.pvalue == r2.pvalue
    assert np.array_equal(r1.null_distribution, r2.null_distribution)


def test_rng_rejects_invalid_type(shift_samples: dict[str, np.ndarray]) -> None:
    with pytest.raises(TypeError, match="rng must be"):
        shift.test_shift(**shift_samples, rng="bad")  # type: ignore[arg-type]


def test_results_do_not_expose_significant_method(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = shift.test_shift(**shift_samples, n_resamples=64)
    assert not hasattr(result, "significant")


def test_shift_rejects_non_positive_n_resamples(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="n_resamples must be"):
        shift.test_shift(**shift_samples, n_resamples=0)
    with pytest.raises(ValueError, match="n_resamples must be"):
        shift.test_harmful_shift(**shift_samples, worse="higher", n_resamples=-1)


def test_shift_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="source must not be empty"):
        shift.test_shift(source=np.array([]), target=np.array([0.5]))


def test_shift_rejects_non_numeric_scores() -> None:
    with pytest.raises(ValueError, match="source must be a one-dimensional numeric"):
        shift.test_shift(source=np.array(["a", "b"]), target=np.array([0.5, 0.6]))


def test_importance_weights_repr_truncates_long_arrays() -> None:
    weights = ImportanceWeights(source=np.ones(300), target=np.ones(300))
    rendered = repr(weights)
    assert "ImportanceWeights(" in rendered
    assert "..." in rendered
    assert len(rendered) < 200


def test_importance_weights_repr_shows_short_arrays_fully() -> None:
    weights = ImportanceWeights(source=np.array([1.0, 2.0]), target=np.array([1.0]))
    rendered = repr(weights)
    assert "[0.66666667 1.33333333]" in rendered
    assert "[1.]" in rendered
