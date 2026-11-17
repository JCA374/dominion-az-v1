"""Out-of-sample value error: fresh games (the net's buys vs the GA base, both seats recorded), then
each net's value prediction against the final result. The training log's value_mse is measured on
the training buffer and can be memorised; this one cannot.

    python experiments/value_holdout.py runs/warm/net_0.npy runs/r2/net_357.npy [--games 600]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def game(job):
    from az.base import base_genome, random_kingdom
    from az.core import GA, NET, Seat, outcome, play
    i, path = job
    rng = random.Random(f"az-v1 value holdout {i}")
    k = random_kingdom(rng)
    g = base_genome(k)["ints"]
    seats = [Seat(NET, g, np.load(path), record=True), Seat(GA, g, record=True)]
    if i % 2:
        seats.reverse()
    r = play(seats, k, rng.getrandbits(32), nh=128)
    rec = r["rec"]
    return rec["X"], np.array([outcome(r, s) for s in rec["SEAT"]], np.float32), rec["TURN"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nets", nargs="+")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--player", default=None, help="net whose buys make the games (default: the last)")
    args = ap.parse_args()
    from az import net
    player = args.player or args.nets[-1]
    with mp.Pool(max(1, os.cpu_count() - 1)) as pool:
        out = pool.map(game, [(i, player) for i in range(args.games)])
    X = np.concatenate([o[0] for o in out])
    Z = np.concatenate([o[1] for o in out])
    T = np.concatenate([o[2] for o in out])
    res = {"games": args.games, "decisions": int(len(Z)), "player": player, "zero_mse": float((Z ** 2).mean()), "nets": {}}
    for path in args.nets:
        _, v, _ = net.forward(np.load(path), 128, X)
        e = (v - Z) ** 2
        res["nets"][path] = {"mse": round(float(e.mean()), 4), "mse_turn_lt_10": round(float(e[T < 10].mean()), 4),
                             "mse_turn_ge_10": round(float(e[T >= 10].mean()), 4)}
        print(path, res["nets"][path])
    print("predicting 0 everywhere:", round(res["zero_mse"], 4))
    json.dump(res, open("results/value_holdout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
