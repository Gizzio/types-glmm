from typing import assert_type

import numpy as np
import pandas as pd
import pytest

import glmm
from numpy.typing import NDArray

DATA: dict[str, list[float]] = {
    "y": [1.0, 2.1, 2.9, 4.2, 5.1, 5.8],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
}


def test_fit_dict() -> None:
    m = glmm.fit(DATA, "y ~ x")
    assert_type(m, glmm.Fit)
    assert isinstance(m, glmm.Fit)


def test_fit_dataframe_and_options() -> None:
    m = glmm.fit(
        pd.DataFrame(DATA),
        "y ~ x",
        "gaussian",
        link="identity",
        weights=np.ones(6),
        offset=[0.0] * 6,
        wald_se="rx",
        satterthwaite=np.bool_(False),
        reml=False,
        anova_test="Wald",
    )
    assert_type(m, glmm.Fit)
    assert isinstance(m, glmm.Fit)


def test_rejected_by_types() -> None:
    # Each call must be a type error; an unused ignore means the stub got too loose.
    for bad in (
        lambda: glmm.fit(DATA, "y ~ x", "logistic"),  # pyrefly: ignore[bad-argument-type]
        lambda: glmm.fit(DATA, "y ~ x", link="logitt"),  # pyrefly: ignore[bad-argument-type]
        lambda: glmm.fit(DATA, "y ~ x", wald_se="exact"),  # pyrefly: ignore[bad-argument-type]
        lambda: glmm.fit(42, "y ~ x"),  # pyrefly: ignore[bad-argument-type]
        lambda: glmm.fit({"y": 5.0, "x": [0.0, 1.0]}, "y ~ x"),  # pyrefly: ignore[bad-assignment]
    ):
        try:
            bad()
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("runtime accepted an invalid call")


def test_warm_start_unknown_key_is_a_type_error() -> None:
    # Runtime only warns and drops the key; the stub rejects it outright.
    with pytest.warns(UserWarning):
        glmm.fit(DATA, "y ~ x", warm_start={"betaa": [0.0]})  # pyrefly: ignore[bad-argument-type]


GROUPED: dict[str, list[float] | list[str]] = {
    "y": [1.0, 2.1, 2.9, 4.2, 5.1, 5.8, 1.3, 2.4],
    "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5],
    "g": ["a", "a", "b", "b", "c", "c", "a", "b"],
}


def test_predict() -> None:
    m = glmm.fit(DATA, "y ~ x")
    p = m.predict()
    assert_type(p, NDArray[np.float64])
    assert isinstance(p, np.ndarray) and p.dtype == np.float64

    pr = m.predict({"x": [0.5, 1.5]}, "response", re_form="NA")
    assert_type(pr, NDArray[np.float64])

    se = m.predict(se_fit=True)
    assert_type(se["lower"], NDArray[np.float64])
    assert set(se) == {"fit", "se_fit", "lower", "upper"}

    flag: bool = bool(len(DATA))
    either = m.predict(se_fit=flag)
    assert isinstance(either, dict)


def test_predict_future() -> None:
    m = glmm.fit(GROUPED, "y ~ x + (1 | g)", "gamma")
    fut = m.predict(future=True)
    assert_type(fut["future_upper"], NDArray[np.float64])
    assert set(fut) == {"fit", "future_lower", "future_upper"}

    both = m.predict(se_fit=True, future=True)
    assert_type(both["se_fit"], NDArray[np.float64])
    assert_type(both["future_lower"], NDArray[np.float64])
    assert set(both) == {"fit", "se_fit", "lower", "upper", "future_lower", "future_upper"}


def test_predict_rejected_by_types() -> None:
    m = glmm.fit(DATA, "y ~ x")
    se = m.predict(se_fit=True)
    for bad in (
        lambda: m.predict(type="mean"),  # pyrefly: ignore[no-matching-overload]
        lambda: m.predict(re_form="~0"),  # pyrefly: ignore[no-matching-overload]
        lambda: se["future_lower"],  # pyrefly: ignore[bad-typed-dict-key]
    ):
        try:
            bad()
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("runtime accepted an invalid call")
