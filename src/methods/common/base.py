"""abstract bases for density-ratio estimators.

class hierarchy:

    DRE                                  fit(p0, p1)
     |- ELDR                             fit(p0, p1, p*)
         |- triangular concrete classes

import-time signature enforcement:
    each base declares ``_fit_params: ClassVar[tuple[str, ...]]`` listing the
    required positional parameter names of ``fit``. ``__init_subclass__``
    inspects each subclass's ``fit`` and rejects the class definition if its
    parameters do not start with the inherited (or overridden) tuple. this
    check fires once when a subclass is *defined* (at import), not when ``fit``
    is *called* -- so there is no per-step runtime cost.

    a subclass may extend its base's ``_fit_params`` (as ``ELDR`` does) and
    add purely optional / keyword-only parameters after the required prefix.

alternative enforcement mechanisms (not used here):
    1. ``typing.Protocol`` + static checker (mypy / pyright): no runtime cost
       at all, but requires the type checker to be run. orthogonal to this
       hook -- the two can coexist.
    2. per-call validation: rejected; would add overhead to every ``fit``.
"""
import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

import torch


class DRE(ABC):
    """abstract base for log-density-ratio estimators.

    Attributes:
        input_dim: dimensionality of the input space.
    """

    _fit_params: ClassVar[tuple[str, ...]] = ("samples_p0", "samples_p1")

    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim

    def __init_subclass__(cls, **kwargs) -> None:
        """validate ``cls.fit``'s positional prefix matches ``cls._fit_params``.

        skipped if the subclass does not redefine ``fit`` (it inherits one), or
        if the redefined ``fit`` is itself abstract and variadic (an
        intermediate ABC that simply restates the contract).
        """
        super().__init_subclass__(**kwargs)
        if "fit" not in cls.__dict__:
            return
        fit = cls.__dict__["fit"]
        params = tuple(
            p for p in inspect.signature(fit).parameters if p != "self"
        )
        # allow purely variadic abstracts (e.g., a future intermediate base)
        if params in ((), ("args",), ("args", "kwargs")):
            return
        expected = cls._fit_params
        if params[: len(expected)] != expected:
            raise TypeError(
                f"{cls.__name__}.fit({', '.join(params)}) does not start with "
                f"required prefix {expected}; either rename parameters or "
                f"override {cls.__name__}._fit_params"
            )

    @abstractmethod
    def fit(self, samples_p0: torch.Tensor, samples_p1: torch.Tensor, *args) -> None:
        """fit to samples_p0 [N0, input_dim] and samples_p1 [N1, input_dim].

        the ``*args`` tail is for LSP-clean refinement in subclasses (e.g.
        ``ELDR.fit`` adds ``samples_pstar``). concrete two-source estimators
        omit the ``*args``.
        """
        pass

    @abstractmethod
    def predict_ldr(self, xs: torch.Tensor) -> torch.Tensor:
        """log(p0(xs) / p1(xs)); shape [N]. device handling is implementation-defined."""
        pass


class ELDR(DRE):
    """abstract base for triangular log-density-ratio estimators that take a reference p*."""

    _fit_params: ClassVar[tuple[str, ...]] = (
        "samples_p0",
        "samples_p1",
        "samples_pstar",
    )

    @abstractmethod
    def fit(
        self,
        samples_p0: torch.Tensor,
        samples_p1: torch.Tensor,
        samples_pstar: torch.Tensor,
    ) -> None:
        """fit to samples_p0 [N0,D], samples_p1 [N1,D], samples_pstar [N*,D]."""
        pass
