import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from _typeshed import Incomplete
from glmm.summary import Rendered, fmt, fmt_df, fmt_p, stars
from numpy.typing import NDArray

from glmm import Fit

class _Table(Rendered):
    _values: dict[str, NDArray[np.float64]]
    row_names: list[str]
    heading: str
    notes: list[str]

    @property
    def columns(self) -> list[str]: ...
    def __getitem__(self, name: str) -> NDArray[np.float64]: ...
    def __contains__(self, name: object) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def to_pandas(self) -> Incomplete: ...  # pandas.DataFrame; pandas is optional

@dataclass(repr=False, eq=False)
class TermTable(_Table):
    _values: dict[str, NDArray[np.float64]]
    row_names: list
    heading: str
    notes: list
    ss_type: str | None

    def text(self) -> Incomplete: ...

@dataclass(repr=False, eq=False)
class PairwiseTable(_Table):
    _values: dict[str, NDArray[np.float64]]
    row_names: list
    heading: str
    notes: list
    _stored: dict | None

    def text(self) -> Incomplete: ...
    def confint(self, level: Incomplete = 0.95) -> Incomplete: ...

@dataclass(repr=False, eq=False)
class ProfileTable(_Table):
    _values: dict[str, NDArray[np.float64]]
    row_names: list
    heading: str
    notes: list
    level: float

    def text(self) -> Incomplete: ...

@dataclass(repr=False, eq=False)
class ComparisonTable(_Table):
    _values: dict[str, NDArray[np.float64]]
    row_names: list
    heading: str
    notes: list
    formulas: list
    reference: str
    nsim: int
    seed: int
    _blank: set[tuple[int, str]]

    def text(self) -> Incomplete: ...

@dataclass(repr=False, eq=False)
class Comparison:
    table: ComparisonTable
    fit0: Fit
    fit1: Fit

    def __repr__(self) -> Incomplete: ...

def anova(
    *fits: Incomplete,
    type: Incomplete = None,
    test: Incomplete = ...,
    nsim: Incomplete = None,
    seed: Incomplete = None,
) -> Incomplete: ...
def compare(
    data: Incomplete,
    f0: Incomplete,
    f1: Incomplete = None,
    family: Incomplete = "gaussian",
    *,
    link: Incomplete = None,
    weights: Incomplete = None,
    offset: Incomplete = None,
    nagq: Incomplete = 1,
    nsim: Incomplete = None,
    seed: Incomplete = None,
) -> Incomplete: ...
