from dataclasses import dataclass

from framesmoothie.fp import Run, curry, get_state, modify_state, tell


def test_curry_accumulates_arguments():
    @curry
    def add4(a, b, c, d=0):
        return a + b + c + d

    assert add4(1)(2)(3) == 6
    assert add4(1, 2)(3, d=4) == 10


@dataclass
class Env:
    bias: int


@dataclass
class State:
    value: int


def test_run_reader_writer_state_roundtrip():
    def program() -> Run[Env, State, int]:
        return (
            tell({"event": "start"})
            .then(modify_state(lambda st: State(st.value + 2)))
            .then(get_state().map(lambda st: st.value))
            .bind(lambda value: Run.pure(value + 5))
        )

    state, value, log = program().execute(Env(bias=10), State(value=3))
    assert state.value == 5
    assert value == 10
    assert log.events == ({"event": "start"},)
