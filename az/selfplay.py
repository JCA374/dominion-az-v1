"""Game jobs for a process pool: warm-start games (GA base genome), self-play with search, and the
fixed evaluation. Each job is one game; the C library is loaded once per worker process."""
from __future__ import annotations

import numpy as np

from .base import base_genome
from .core import GA, NET, SEARCH, Seat, bot, lib, outcome, play


def _records(r: dict) -> dict:
    rec = r["rec"]
    z = np.array([outcome(r, s) for s in rec["SEAT"]], np.float32)
    return {"X": rec["X"], "PI": rec["PI"], "LEGAL": rec["LEGAL"], "Z": z, "Q": rec["Q"],
            "TURN": rec["TURN"]}


def warm_game(job: dict) -> dict:
    """GA base genome vs itself (both seats recorded) or vs a bot (the base seat recorded)."""
    k, seed = job["kingdom"], job["seed"]
    g = base_genome(k)["ints"]
    if job.get("bot"):
        b = bot(job["bot"], k)
        seat = job["seat"]
        seats = [Seat(GA, g, record=True), Seat(GA, b)] if seat == 0 else \
                [Seat(GA, b), Seat(GA, g, record=True)]
    else:
        seats = [Seat(GA, g, record=True), Seat(GA, g, record=True)]
    return _records(play(seats, k, seed))


def selfplay_game(job: dict) -> dict:
    """Search vs search (both recorded), or search vs the GA base genome (search seat recorded)."""
    k, seed, cfg, params = job["kingdom"], job["seed"], job["cfg"], job["params"]
    lib.az_set_target(50, cfg["c_scale"], 0, cfg["target"])
    g = base_genome(k)["ints"]
    me = Seat(SEARCH, g, params, sims=cfg["sims"], noise=bool(cfg["noise"]), temp_turns=cfg["temp_turns"],
              record=True)
    if job.get("vs_bot"):
        other = Seat(GA, bot(job["vs_bot"], k))
    elif job.get("vs_ga"):
        other = Seat(GA, g)
    else:
        other = Seat(SEARCH, g, params, sims=cfg["sims"], noise=bool(cfg["noise"]),
                     temp_turns=cfg["temp_turns"], record=True)
    seats = [me, other] if job["seat"] == 0 else [other, me]
    r = play(seats, k, seed, search_seed=seed * 7919 + 1, nh=cfg["nh"], cpuct=cfg["cpuct"],
             lam=cfg["lam"])
    out = _records(r)
    out["result"] = outcome(r, job["seat"])
    out["turns"] = max(r["turns"])
    out["vs_ga"] = bool(job.get("vs_ga")) and not job.get("vs_bot")
    return out


def eval_game(job: dict) -> dict:
    """agent (search / net / ga) vs opponent (a bot name or 'GA base'), from seat job['seat']."""
    k, seed = job["kingdom"], job["seed"]
    g = base_genome(k)["ints"]
    a = job["agent"]
    if a == "ga":
        me = Seat(GA, g)
    elif a == "net":
        me = Seat(NET, g, job["params"])
    else:
        me = Seat(SEARCH, g, job["params"], sims=job["sims"])
    opp = Seat(GA, g if job["opponent"] == "GA base" else bot(job["opponent"], k))
    seats = [me, opp] if job["seat"] == 0 else [opp, me]
    r = play(seats, k, seed, search_seed=seed + 12345, nh=job.get("nh", 0), cpuct=job.get("cpuct", 1.5),
             lam=job.get("lam", 1.0))
    res = outcome(r, job["seat"])
    return {"agent": a, "opponent": job["opponent"], "points": (res + 1) / 2,
            "margin": r["scores"][job["seat"]] - r["scores"][1 - job["seat"]],
            "turns": max(r["turns"]), "searched": r["searched"][job["seat"]]}
