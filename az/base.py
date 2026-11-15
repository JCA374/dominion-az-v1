"""The base genome per kingdom: which dominion-v1 portfolio-v2 specialist answers the non-buy
decisions (actions, gains, trashing, attacks) and, in GA mode, the buys.

Rule: the best-ranked member whose `must` cards are all dealt (v2-generalist always qualifies).
Ranking = each member's score against the three benchmark bots on random kingdoms holding its
must cards, measured once and cached in data/members.json. Cheaper than portfolio-v2's trial at
game start (~3-7 s), and fixed, so every comparison uses the same base genome.
"""
from __future__ import annotations

import json
import os
import random

from .core import GA, ROOT, Seat, bot, play, portfolio_members

CACHE = os.path.join(ROOT, "data", "members.json")
BOTS = ["Big Money", "Smithy-BM", "Witch-BM"]
_members = None
_rank = None


def random_kingdom(rng: random.Random, must=()) -> list[str]:
    from dominion_engine.cards import KINGDOM_CARDS
    pool = [c for c in KINGDOM_CARDS if c not in must]
    return sorted(rng.sample(pool, 10 - len(must)) + list(must))


def kingdom_for_bot(rng: random.Random, opponent: str, must=()) -> list[str]:
    need = list(must) + {"Big Money": [], "Smithy-BM": ["Smithy"], "Witch-BM": ["Witch"]}[opponent]
    return random_kingdom(rng, sorted(set(need)))


def _measure(m: dict, n_kingdoms: int = 30, n_seeds: int = 6) -> float:
    rng = random.Random(f"rank:{m['name']}")
    pts = games = 0
    for opp in BOTS:
        for _ in range(n_kingdoms):
            k = kingdom_for_bot(rng, opp, m["must"])
            for _ in range(n_seeds):
                seed = rng.getrandbits(32)
                for seat in (0, 1):
                    seats = [None, None]
                    seats[seat] = Seat(GA, m["ints"])
                    seats[1 - seat] = Seat(GA, bot(opp, k))
                    r = play(seats, k, seed)
                    pts += 1.0 if r["winners"] == [seat] else 0.5 if len(r["winners"]) == 2 else 0
                    games += 1
    return pts / games


def members() -> list[dict]:
    global _members, _rank
    if _members is None:
        _members = portfolio_members()
        if os.path.exists(CACHE):
            with open(CACHE) as f:
                _rank = json.load(f)
        if _rank is None or set(_rank) != {m["name"] for m in _members}:
            _rank = {m["name"]: _measure(m) for m in _members}
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            with open(CACHE, "w") as f:
                json.dump(_rank, f, indent=1, sort_keys=True)
        _members.sort(key=lambda m: -_rank[m["name"]])
    return _members


def base_genome(kingdom: list[str]) -> dict:
    """{name, must, ints} of the member that plays this kingdom."""
    ks = set(kingdom)
    for m in members():
        if all(c in ks for c in m["must"]):
            return m
    raise AssertionError("v2-generalist has no must cards and always qualifies")
