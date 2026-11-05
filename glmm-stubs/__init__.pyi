from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only

import numpy as np
from _typeshed import Incomplete
from numpy.typing import ArrayLike, NDArray

@type_check_only
class _ArrowTableLike(Protocol):
    # pyarrow.Table: read via `column_names` / `column(name)`
    @property
    def column_names(self) -> list[str]: ...
    def column(self, i: str, /) -> Any: ...

@type_check_only
class _DataFrameLike(Protocol):
    # pandas / polars DataFrame: read via `columns` / `data[name]`
    @property
    def columns(self) -> Iterable[str]: ...
    def __getitem__(self, key: str, /) -> Any: ...

# Every real column type (list, ndarray, pandas Series/Categorical, pyarrow and
# polars arrays) is iterable; glmm reads them duck-typed, so the items stay Any.
_Data: TypeAlias = Mapping[str, Iterable[Any]] | _DataFrameLike | _ArrowTableLike
_Family: TypeAlias = Literal[
    "gaussian",
    "binomial",
    "poisson",
    "gamma",
    "negativebinomial",
    "inversegaussian",
]
_Link: TypeAlias = Literal[
    "identity", "logit", "probit", "cloglog", "log", "inverse", "inverse_squared"
]

@type_check_only
class _WarmStart(TypedDict, total=False):
    beta: Iterable[float]
    theta: Iterable[float]
    dispersion: float | None

# Fit.predict(se_fit=..., future=...) return dicts; keys follow the two flags.
@type_check_only
class _Prediction(TypedDict):
    fit: NDArray[np.float64]

@type_check_only
class _PredictionSE(_Prediction):
    se_fit: NDArray[np.float64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

@type_check_only
class _PredictionFuture(_Prediction):
    future_lower: NDArray[np.float64]
    future_upper: NDArray[np.float64]

@type_check_only
class _PredictionSEFuture(_PredictionSE, _PredictionFuture): ...

__all__ = [
    "AgqFallbackWarning",
    "ArgumentIgnoredWarning",
    "Comparison",
    "ComparisonTable",
    "ConstantResponseWarning",
    "DesignUnsolvableWarning",
    "DiagnosticWarning",
    "ExactProfileFallbackWarning",
    "FewRowsForRandomEffectsWarning",
    "Fit",
    "FitFailedWarning",
    "GlmDivergedWarning",
    "GradientSearchFallbackWarning",
    "HessianSeFallbackWarning",
    "IllConditionedWarning",
    "NbShapeUnsettledWarning",
    "NewGroupingLevelsWarning",
    "NoCoefficientsWarning",
    "NonIntegerResponseWarning",
    "PairwiseTable",
    "PirlsExhaustedWarning",
    "PrecisionLimitedWarning",
    "ProfileTable",
    "RowsDroppedWarning",
    "SatterthwaiteCurvatureWarning",
    "SeUnavailableWarning",
    "SearchLimitWarning",
    "SingleLevelGroupingDroppedWarning",
    "SingularFitWarning",
    "TermTable",
    "TooFewRowsWarning",
    "VarianceAtLimitWarning",
    "anova",
    "compare",
    "fit",
    "model_matrix",
    "pairwise",
    "term_labels",
]

class DiagnosticWarning(UserWarning): ...
class IllConditionedWarning(DiagnosticWarning): ...
class PirlsExhaustedWarning(DiagnosticWarning): ...
class NewGroupingLevelsWarning(DiagnosticWarning): ...
class SingleLevelGroupingDroppedWarning(DiagnosticWarning): ...
class FewRowsForRandomEffectsWarning(DiagnosticWarning): ...
class HessianSeFallbackWarning(DiagnosticWarning): ...
class GradientSearchFallbackWarning(DiagnosticWarning): ...
class ExactProfileFallbackWarning(DiagnosticWarning): ...
class SeUnavailableWarning(DiagnosticWarning): ...
class PrecisionLimitedWarning(DiagnosticWarning): ...
class VarianceAtLimitWarning(DiagnosticWarning): ...
class SatterthwaiteCurvatureWarning(DiagnosticWarning): ...
class SearchLimitWarning(DiagnosticWarning): ...
class FitFailedWarning(DiagnosticWarning): ...
class GlmDivergedWarning(DiagnosticWarning): ...
class DesignUnsolvableWarning(DiagnosticWarning): ...
class SingularFitWarning(DiagnosticWarning): ...
class AgqFallbackWarning(DiagnosticWarning): ...
class ArgumentIgnoredWarning(DiagnosticWarning): ...
class RowsDroppedWarning(DiagnosticWarning): ...
class ConstantResponseWarning(DiagnosticWarning): ...
class TooFewRowsWarning(DiagnosticWarning): ...
class NoCoefficientsWarning(DiagnosticWarning): ...
class NbShapeUnsettledWarning(DiagnosticWarning): ...
class NonIntegerResponseWarning(DiagnosticWarning): ...

@type_check_only
class Diagnostics(TypedDict):
    converged: bool
    singular: bool
    aliased: NDArray[np.bool_]
    boundary: Literal["interior"]
    pinned: list
    notes: list

@dataclass
class Fit:
    beta: NDArray[np.float64]
    se: NDArray[np.float64]
    vcov: NDArray[np.float64]
    tau2: NDArray[np.float64]
    varcorr: list
    stddev_se: NDArray[np.float64]
    diagnostics: Diagnostics
    dispersion: float
    names: list
    re_groups: list
    re_term_index: list
    n_eval: int
    n_grad: int
    outer_stop: str | None
    outer_decrement: float
    deviance: float
    loglik: float
    df: int
    reml: bool
    fitted: NDArray[np.float64]
    ranef: NDArray[np.float64]
    ranef_levels: NDArray[np.int_]
    ranef_blocks: list
    formula: str
    family: str
    link: str
    nagq: int
    nobs: int
    y: NDArray[np.float64]
    weights: NDArray[np.float64] | None
    coef_df: NDArray[np.float64]
    coef_p: NDArray[np.float64]
    df_method: str
    rss: float
    r_squared: float
    adj_r_squared: float
    warnings: list = ...
    dispersion_held: float | None = None
    design_summary: dict[str, NDArray[np.float64]] | None = None
    offset: NDArray[np.float64] | None = None
    objective: str | None = None
    objective_nodes: int = 0
    term_table_type3: dict | None = None
    term_table_type2: dict | None = None
    term_table_type3_lrt: dict | None = None
    term_table_type2_lrt: dict | None = None
    term_test_auto: dict | None = None
    satterthwaite_record: dict | None = None
    _kernel_parts: dict | None = None

    @property
    def converged(self) -> Incomplete: ...
    @property
    def singular(self) -> Incomplete: ...
    @property
    def aliased(self) -> Incomplete: ...
    @property
    def aic(self) -> Incomplete: ...
    @property
    def bic(self) -> Incomplete: ...
    def stddev_corr(self, group_idx: Incomplete) -> Incomplete: ...
    def summary(self) -> Incomplete: ...
    def summary_object(self) -> Incomplete: ...
    def confint(self, level: Incomplete = 0.95) -> Incomplete: ...
    def profile_confint(
        self, level: Incomplete = 0.95, parm: Incomplete = None
    ) -> Incomplete: ...
    def contrast_test(self, L: NDArray[np.float64]) -> TermTable: ...
    @overload
    def predict(
        self,
        newdata: _Data | None = None,
        type: Literal["link", "response"] = "link",
        re_form: Literal["NA"] | None = None,
        allow_new_levels: bool = True,
        se_fit: Literal[False] = False,
        level: float = 0.95,
        future: Literal[False] = False,
    ) -> NDArray[np.float64]: ...
    @overload
    def predict(
        self,
        newdata: _Data | None = None,
        type: Literal["link", "response"] = "link",
        re_form: Literal["NA"] | None = None,
        allow_new_levels: bool = True,
        *,
        se_fit: Literal[True],
        level: float = 0.95,
        future: Literal[False] = False,
    ) -> _PredictionSE: ...
    @overload
    def predict(
        self,
        newdata: _Data | None = None,
        type: Literal["link", "response"] = "link",
        re_form: Literal["NA"] | None = None,
        allow_new_levels: bool = True,
        se_fit: Literal[False] = False,
        level: float = 0.95,
        *,
        future: Literal[True],
    ) -> _PredictionFuture: ...
    @overload
    def predict(
        self,
        newdata: _Data | None = None,
        type: Literal["link", "response"] = "link",
        re_form: Literal["NA"] | None = None,
        allow_new_levels: bool = True,
        *,
        se_fit: Literal[True],
        level: float = 0.95,
        future: Literal[True],
    ) -> _PredictionSEFuture: ...
    @overload
    def predict(
        self,
        newdata: _Data | None = None,
        type: Literal["link", "response"] = "link",
        re_form: Literal["NA"] | None = None,
        allow_new_levels: bool = True,
        se_fit: bool = False,
        level: float = 0.95,
        future: bool = False,
    ) -> NDArray[np.float64] | _Prediction: ...
    def simulate(self, nsim: Incomplete = 1, seed: Incomplete = None) -> Incomplete: ...
    def r2(self) -> Incomplete: ...
    def icc(self) -> Incomplete: ...
    def __setstate__(self, state: Incomplete) -> Incomplete: ...
    def residuals(self, type: Incomplete = "response") -> Incomplete: ...

def fit(
    data: _Data,
    formula: str,
    family: _Family = "gaussian",
    *,
    link: _Link | None = None,
    dispersion: float | Literal["estimate"] | None = None,
    init_theta: float | None = None,
    weights: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    wald_se: Literal["hessian", "rx"] = "hessian",
    nagq: int = 1,
    warm_start: _WarmStart | None = None,
    satterthwaite: bool | np.bool_ = True,
    condvar: bool | np.bool_ = True,
    reml: bool | np.bool_ | None = None,
    anova_test: Literal["auto", "Wald", "LRT"] = "auto",
) -> Fit: ...
def model_matrix(data: Incomplete, formula: Incomplete) -> Incomplete: ...
def pairwise(
    m: Incomplete,
    factor: Incomplete,
    by: Incomplete = None,
    at: Incomplete = None,
    adjust: Incomplete = "holm",
) -> Incomplete: ...
def term_labels(formula: Incomplete) -> Incomplete: ...

from glmm.anova import (
    Comparison,
    ComparisonTable,
    PairwiseTable,
    ProfileTable,
    TermTable,
    anova,
    compare,
)
