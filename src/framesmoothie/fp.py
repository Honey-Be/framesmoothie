from __future__ import annotations

"""Functional-programming helpers used by the training runtime.

This module intentionally keeps the abstractions lightweight:

* ``curry`` lets us build configuration-first factories such as
  ``make_seed(master_seed)("loader")(rank)(worker)(epoch, step)``.
* ``Run`` is a small Reader/Writer/State-style monad used to describe the
  training driver as a deterministic state transition plus an append-only log.

The goal is not to emulate Haskell syntax in Python. The goal is to make the
training process explicit enough that ``(dataset, preset) -> result`` is a
reasonable mental model for the orchestration layer.
"""

from dataclasses import dataclass, field
import inspect
from functools import partial, update_wrapper
from typing import Any, Callable, Generic, Iterable, Mapping, MutableMapping, TypeVar

EnvT = TypeVar("EnvT")
StateT = TypeVar("StateT")
A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class RunLog:
    """Append-only event log for ``Run`` programs."""

    events: tuple[Mapping[str, Any], ...] = ()

    def append(self, event: Mapping[str, Any]) -> "RunLog":
        return RunLog(self.events + (dict(event),))

    def extend(self, events: Iterable[Mapping[str, Any]]) -> "RunLog":
        acc = self
        for event in events:
            acc = acc.append(event)
        return acc


class Run(Generic[EnvT, StateT, A]):
    """A tiny Reader/Writer/State monad.

    ``Run[A]`` represents a computation of the form

    ``(env, state) -> (new_state, value, log)``.

    It is intentionally minimal, but expressive enough for the training runtime
    to model deterministic state transitions and structured event emission.
    """

    def __init__(
        self,
        fn: Callable[[EnvT, StateT], tuple[StateT, A, RunLog]],
    ) -> None:
        self._fn = fn

    def execute(self, env: EnvT, state: StateT) -> tuple[StateT, A, RunLog]:
        return self._fn(env, state)

    __call__ = execute

    def map(self, fn: Callable[[A], B]) -> "Run[EnvT, StateT, B]":
        def _mapped(env: EnvT, state: StateT) -> tuple[StateT, B, RunLog]:
            state2, value, log = self.execute(env, state)
            return state2, fn(value), log

        return Run(_mapped)

    def bind(self, fn: Callable[[A], "Run[EnvT, StateT, B]"]) -> "Run[EnvT, StateT, B]":
        def _bound(env: EnvT, state: StateT) -> tuple[StateT, B, RunLog]:
            state2, value, log1 = self.execute(env, state)
            state3, value2, log2 = fn(value).execute(env, state2)
            return state3, value2, RunLog(log1.events + log2.events)

        return Run(_bound)

    def then(self, nxt: "Run[EnvT, StateT, B]") -> "Run[EnvT, StateT, B]":
        return self.bind(lambda _ignored: nxt)

    @staticmethod
    def pure(value: A) -> "Run[EnvT, StateT, A]":
        return Run(lambda _env, state: (state, value, RunLog()))


def ask() -> Run[EnvT, StateT, EnvT]:
    return Run(lambda env, state: (state, env, RunLog()))


def asks(fn: Callable[[EnvT], A]) -> Run[EnvT, StateT, A]:
    return ask().map(fn)


def get_state() -> Run[EnvT, StateT, StateT]:
    return Run(lambda _env, state: (state, state, RunLog()))


def put_state(state: StateT) -> Run[EnvT, StateT, None]:
    return Run(lambda _env, _state: (state, None, RunLog()))


def modify_state(fn: Callable[[StateT], StateT]) -> Run[EnvT, StateT, None]:
    def _modify(_env: EnvT, state: StateT) -> tuple[StateT, None, RunLog]:
        return fn(state), None, RunLog()

    return Run(_modify)


def tell(event: Mapping[str, Any]) -> Run[EnvT, StateT, None]:
    return Run(lambda _env, state: (state, None, RunLog((dict(event),))))


def sequence(programs: Iterable[Run[EnvT, StateT, Any]]) -> Run[EnvT, StateT, list[Any]]:
    def _run(env: EnvT, state: StateT) -> tuple[StateT, list[Any], RunLog]:
        acc_state = state
        out: list[Any] = []
        log = RunLog()
        for program in programs:
            acc_state, value, step_log = program.execute(env, acc_state)
            out.append(value)
            log = RunLog(log.events + step_log.events)
        return acc_state, out, log

    return Run(_run)


@dataclass
class Curried:
    """Callable wrapper returned by :func:`curry`."""

    fn: Callable[..., Any]
    signature: inspect.Signature
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        merged_args = self.args + args
        merged_kwargs = dict(self.kwargs)
        merged_kwargs.update(kwargs)
        bound = self.signature.bind_partial(*merged_args, **merged_kwargs)
        missing = [
            p
            for p in self.signature.parameters.values()
            if p.default is inspect._empty
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and p.name not in bound.arguments
        ]
        if missing:
            return Curried(self.fn, self.signature, merged_args, merged_kwargs)
        return self.fn(*merged_args, **merged_kwargs)


def curry(fn: Callable[..., A]) -> Callable[..., Any]:
    """Return a curried version of ``fn``.

    The returned callable accumulates arguments until every required parameter
    from ``fn`` has been provided, then executes ``fn``.
    """

    curried = Curried(fn=fn, signature=inspect.signature(fn))
    update_wrapper(curried, fn)
    return curried


__all__ = [
    "Curried",
    "Run",
    "RunLog",
    "ask",
    "asks",
    "curry",
    "get_state",
    "modify_state",
    "put_state",
    "sequence",
    "tell",
]
