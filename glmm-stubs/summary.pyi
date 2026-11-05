from typing import Literal
from _typeshed import Incomplete

import html as _html
import math
from dataclasses import dataclass, field
import numpy as np

FOOTNOTE: Literal['t tests use Satterthwaite degrees of freedom (finite-difference θ-Hessian).'] = "t tests use Satterthwaite degrees of freedom (finite-difference θ-Hessian)."
ML_NOTE: Literal['Note: p-values can be somewhat anticonservative.'] = "Note: p-values can be somewhat anticonservative."
DF_NOT_COMPUTED: Literal['df not computed (satterthwaite=False)'] = "df not computed (satterthwaite=False)"


def fmt(x: Incomplete, digits: Incomplete = 4) -> Incomplete: ...


def fmt_df(x: Incomplete) -> Incomplete: ...


def fmt_p(p: Incomplete) -> Incomplete: ...


def stars(p: Incomplete) -> Incomplete: ...


def method_line(fit: Incomplete) -> Incomplete: ...


def pearson_residuals(fit: Incomplete) -> Incomplete: ...


class Rendered:
    def html(self) -> Incomplete: ...

    def latex(self) -> Incomplete: ...

    def typst(self) -> Incomplete: ...


def build_summary(fit: Incomplete, effect_sizes: Incomplete = None, need_refit: Incomplete = False) -> Incomplete: ...


@dataclass
class Summary(Rendered):
    method: str
    formula: str
    family: str
    link: str
    reml: bool
    criterion: dict
    scaled_residuals: np.ndarray | None
    random_effects: list
    residual_variance: float | None
    nobs: int
    groups: list
    coef_names: list
    estimate: np.ndarray
    std_error: np.ndarray
    coef_df: np.ndarray | None
    statistic: np.ndarray
    test: str
    p_value: np.ndarray
    residual_df: int | None
    rss: float
    r_squared: float
    adj_r_squared: float
    r2_marginal: float | None
    r2_conditional: float | None
    icc_adjusted: float | None
    r2_need_refit: bool
    aliased: np.ndarray
    corr_fixed: np.ndarray | None
    dispersion_label: str | None
    dispersion: float
    n_eval: int
    converged: bool
    singular: bool
    n_aliased: int
    footnote: str | None
    warnings: list = ...

    def __repr__(self) -> Incomplete: ...

    def text(self) -> Incomplete: ...
