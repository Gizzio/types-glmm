from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, overload, type_check_only

import numpy as np
from glmm.summary import Summary
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

_Boundary: TypeAlias = Literal["interior", "at_boundary", "no_optimum", "upper_limit"]
_DfMethod: TypeAlias = Literal["satterthwaite", "residual", "none", "disabled", "failed"]
_Objective: TypeAlias = Literal[
    "least_squares", "maximum_likelihood", "reml", "laplace", "agq"
]
_Adjust: TypeAlias = Literal["holm", "bonferroni", "tukey", "none"]

@type_check_only
class _Note(TypedDict):
    kind: str
    columns: list[int]
    pivot: float
    evals: int
    final_eval: bool
    detail: str

@type_check_only
class Diagnostics(TypedDict):
    converged: bool
    singular: bool
    aliased: NDArray[np.bool_]
    boundary: _Boundary
    pinned: list[list[bool]]
    notes: list[_Note]

@type_check_only
class _WarningRecord(TypedDict):
    tier: Literal["severe", "caution", "note"]
    kind: str
    title: str
    message: str

@type_check_only
class _RanefBlock(TypedDict):
    group: str
    terms: list[str]
    levels: list[str]
    values: NDArray[np.float64]
    postvar: NDArray[np.float64] | None
    condsd: NDArray[np.float64] | None

@type_check_only
class _R2(TypedDict):
    marginal: float
    conditional: float

@type_check_only
class _ICC(TypedDict):
    adjusted: float
    unadjusted: float

@type_check_only
class _TermTestAuto(TypedDict):
    test: str
    reason: str

@dataclass
class Fit:
    beta: NDArray[np.float64]
    se: NDArray[np.float64]
    vcov: NDArray[np.float64]
    tau2: NDArray[np.float64]
    varcorr: list[list[float]]
    stddev_se: NDArray[np.float64]
    diagnostics: Diagnostics
    dispersion: float
    names: list[str]
    re_groups: list[tuple[str, list[str]]]
    re_term_index: list[list[int]]
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
    ranef_blocks: list[_RanefBlock]
    formula: str
    family: _Family
    link: _Link
    nagq: int
    nobs: int
    y: NDArray[np.float64]
    weights: NDArray[np.float64] | None
    coef_df: NDArray[np.float64]
    coef_p: NDArray[np.float64]
    df_method: _DfMethod
    rss: float
    r_squared: float
    adj_r_squared: float
    warnings: list[_WarningRecord] = ...
    dispersion_held: float | None = None
    design_summary: dict[str, NDArray[np.float64]] | None = None
    offset: NDArray[np.float64] | None = None
    objective: _Objective | None = None
    objective_nodes: int = 0
    # Heterogeneous per-term tables (str / list / ndarray values); see Fit docs.
    # fit() always fills these three; the runtime default None only shows on a
    # Fit pickled by an older glmm, which is outside the stub's contract.
    term_table_type3: dict[str, Any] = ...
    term_table_type2: dict[str, Any] = ...
    term_table_type3_lrt: dict[str, Any] | None = None
    term_table_type2_lrt: dict[str, Any] | None = None
    term_test_auto: _TermTestAuto = ...
    satterthwaite_record: dict[str, Any] | None = None
    _kernel_parts: dict[str, Any] | None = None

    @property
    def converged(self) -> bool: ...
    @property
    def singular(self) -> bool: ...
    @property
    def aliased(self) -> NDArray[np.bool_]: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    def stddev_corr(
        self, group_idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def summary(self) -> str: ...
    def summary_object(self) -> Summary: ...
    def confint(self, level: float = 0.95) -> NDArray[np.float64]: ...
    def profile_confint(
        self,
        level: float = 0.95,
        parm: str | list[str] | tuple[str, ...] | None = None,
    ) -> ProfileTable: ...
    def contrast_test(self, L: ArrayLike) -> TermTable: ...
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
    def simulate(
        self, nsim: int | np.integer = 1, seed: int | np.integer | None = None
    ) -> NDArray[np.float64]: ...
    def r2(self) -> _R2: ...
    def icc(self) -> _ICC: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def residuals(
        self, type: Literal["response", "pearson"] = "response"
    ) -> NDArray[np.float64]: ...

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
def model_matrix(
    data: _Data, formula: str
) -> tuple[NDArray[np.float64], list[str]]: ...
def pairwise(
    m: Fit,
    factor: str,
    by: str | None = None,
    at: ArrayLike | None = None,
    adjust: _Adjust = "holm",
) -> PairwiseTable: ...
def term_labels(formula: str) -> tuple[list[str], bool]: ...

from glmm.anova import (
    Comparison,
    ComparisonTable,
    PairwiseTable,
    ProfileTable,
    TermTable,
    anova,
    compare,
)
