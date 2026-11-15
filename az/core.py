"""The C side through ctypes: engine/c/engine.c + c/az.c (which includes dominion-v1's ga_agent.c)
built into az.so, and the GA genomes that answer every non-buy decision.

The rules come only from engine/ (this repo's submodule). dominion-v1 (submodule) provides the GA
genome agent, the released genomes and the benchmark bots; its own engine/ pin must be the same
commit (tests/test_core.py checks it).
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DV1 = os.path.join(ROOT, "dominion-v1")
ENGINE_C = os.path.join(ROOT, "engine", "c")
for _p in (os.path.join(ROOT, "engine"), DV1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dominion_engine.cards import ID, NAME, NUM_CARDS  # noqa: E402

SO = os.path.join(ROOT, "az.so")
SOURCES = [os.path.join(ENGINE_C, "engine.c"), os.path.join(ROOT, "c", "az.c")]
DEPS = SOURCES + [os.path.join(ENGINE_C, "engine.h"), os.path.join(DV1, "c", "ga_agent.c")]


def build() -> str:
    if not os.path.exists(SO) or max(os.path.getmtime(d) for d in DEPS) > os.path.getmtime(SO):
        tmp = f"{SO}.{os.getpid()}"
        subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC",
                               "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unused-function",
                               "-I", ENGINE_C, "-o", tmp, *SOURCES, "-lm"])
        os.replace(tmp, SO)
    return SO


lib = ctypes.CDLL(build())
NF = lib.az_num_features()
NA = lib.az_num_actions()            # card ids 0..NUM_CARDS-1, then PASS
PASS = NA - 1
assert NA == NUM_CARDS + 1

_I, _F = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
_U8 = ctypes.POINTER(ctypes.c_ubyte)
lib.az_num_params.argtypes = [ctypes.c_int]
lib.az_forward.restype = ctypes.c_float
lib.az_forward.argtypes = [_F, ctypes.c_int, _F, _F]
lib.az_play.restype = ctypes.c_int
lib.az_play.argtypes = [_I, _F, _F, ctypes.c_int, _I, _I, _I, _I, _I, _I, ctypes.c_float, ctypes.c_float,
                        _I, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                        _F, _F, _F, _U8, _I, _I, ctypes.c_int, _I]

lib.az_set_trace.argtypes = [_F, ctypes.c_int]
lib.az_set_target.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int]
TRACE_W = lib.az_trace_width()

GA, NET, SEARCH = 0, 1, 2
MAX_REC = 600


def _ints(xs):
    return (ctypes.c_int * len(xs))(*xs)


def _fptr(a: np.ndarray | None):
    return None if a is None else a.ctypes.data_as(_F)


# ── genomes (dominion-v1) ──────────────────────────────────────────────────────────────────────

_genome_cache: dict[str, ctypes.Array] = {}


def genome_ints(strategy) -> ctypes.Array:
    from ga.c_bridge import strategy_to_ints
    return strategy_to_ints(strategy)


def bot(name: str, kingdom: list[str]) -> ctypes.Array:
    """Benchmark bots of dominion-v1's policy.bench: 'Big Money', 'Smithy-BM', 'Witch-BM'."""
    from core.strategy import big_money_strategy, smithy_big_money_strategy, witch_big_money_strategy
    b = {"Big Money": big_money_strategy, "Smithy-BM": smithy_big_money_strategy,
         "Witch-BM": witch_big_money_strategy}[name]
    return genome_ints(b(kingdom))


PORTFOLIO = os.path.join(DV1, "models", "portfolio-v2")


def portfolio_members() -> list[dict]:
    """[{name, must, ints}] of dominion-v1's portfolio-v2 specialists."""
    import json
    from core.strategy import load_strategy
    with open(os.path.join(PORTFOLIO, "model.json")) as f:
        man = json.load(f)
    out = []
    for m in man["members"]:
        s = load_strategy(os.path.join(PORTFOLIO, m["file"]))
        out.append({"name": m["name"], "must": m["must"], "ints": genome_ints(s)})
    return out


# ── games ─────────────────────────────────────────────────────────────────────────────────────

class Seat:
    """One player: mode GA (genome buys), NET (net argmax) or SEARCH (IS-MCTS)."""

    def __init__(self, mode: int, genome: ctypes.Array, params: np.ndarray | None = None,
                 sims: int = 0, noise: bool = False, temp_turns: int = 0, record: bool = False):
        self.mode, self.genome, self.params = mode, genome, params
        self.sims, self.noise, self.temp_turns, self.record = sims, noise, temp_turns, record


def play(seats: list[Seat], kingdom: list[str], seed: int, search_seed: int = 1,
         nh: int = 0, cpuct: float = 1.5, lam: float = 1.0, trace: bool = False) -> dict:
    """One game; returns result + the recorded buy decisions of recording seats. trace: also
    every buy decision of both seats (player, turn, coins, choice, vp, both decks, visit shares)."""
    T = np.zeros((2000, TRACE_W), np.float32) if trace else None
    lib.az_set_trace(_fptr(T), 2000 if trace else 0)
    k = _ints([ID[c] for c in kingdom])
    X = np.zeros((MAX_REC, NF), np.float32)
    PI = np.zeros((MAX_REC, NA), np.float32)
    Q = np.zeros(MAX_REC, np.float32)
    LEGAL = np.zeros((MAX_REC, NA), np.uint8)
    SEAT = np.zeros(MAX_REC, np.int32)
    TURN = np.zeros(MAX_REC, np.int32)
    out = (ctypes.c_int * 8)()
    p = [s.params if s.params is not None else None for s in seats]
    for s in seats:
        if s.mode != GA:
            assert s.params is not None and s.params.dtype == np.float32 and nh > 0
    n = lib.az_play(_ints([s.mode for s in seats]), _fptr(p[0]), _fptr(p[1]), nh,
                    seats[0].genome, seats[1].genome, _ints([s.sims for s in seats]),
                    _ints([int(s.noise) for s in seats]), _ints([s.temp_turns for s in seats]),
                    _ints([int(s.record) for s in seats]), cpuct, lam, k, len(kingdom),
                    seed & (2**64 - 1), search_seed & (2**64 - 1),
                    _fptr(X), _fptr(PI), _fptr(Q), LEGAL.ctypes.data_as(_U8),
                    SEAT.ctypes.data_as(_I), TURN.ctypes.data_as(_I), MAX_REC, out)
    mask = out[0]
    tr = T[:lib.az_trace_count()] if trace else None
    lib.az_set_trace(None, 0)
    return {"winners": [i for i in range(2) if mask >> i & 1], "scores": [out[1], out[2]],
            "turns": [out[3], out[4]], "searched": [out[5], out[6]], "capped": bool(out[7] & 1),
            "error": bool(out[7] & 2), "trace": tr,
            "rec": {"X": X[:n], "PI": PI[:n], "Q": Q[:n], "LEGAL": LEGAL[:n], "SEAT": SEAT[:n],
                    "TURN": TURN[:n]}}


def outcome(result: dict, seat: int) -> float:
    """+1 win, 0 shared, -1 loss (Steam tiebreak is in the engine's winners)."""
    w = result["winners"]
    return 0.0 if len(w) == 2 else (1.0 if seat in w else -1.0)
