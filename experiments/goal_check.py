"""Goal check: a net (with search) on fresh games never used in training or the per-iteration
evaluation. 1,000 games against the GA base genome (500 kingdoms, both seats) and 200 against each
benchmark bot. Writes results/goal_<name>.json for the site's goal table.

    python experiments/goal_check.py runs/r1/net_12.npy --name iter12
    python experiments/goal_check.py ga --name ga_base --baseline     # the GA base itself
    python experiments/goal_check.py runs/r2/net_357.npy --name r2_net --sims 0   # the net alone, no search
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

P = None


def _init(path):
    global P
    P = None if path == "ga" else np.load(path)


def job(j):
    from az.base import base_genome
    from az.core import GA, NET, SEARCH, Seat, bot, lib, outcome, play
    lib.az_set_prior(j["ptemp"], j["peps"])
    lib.az_set_rollout(j["rollout_net"])
    k = j["kingdom"]
    g = base_genome(k)["ints"]
    me = Seat(GA, g) if P is None else Seat(NET, g, P) if j["sims"] == 0 else Seat(SEARCH, g, P, sims=j["sims"])
    opp = Seat(GA, g if j["opp"] == "GA base" else bot(j["opp"], k))
    seats = [me, opp] if j["seat"] == 0 else [opp, me]
    r = play(seats, k, j["seed"], search_seed=j["seed"] ^ 0x5EED, nh=j["nh"], cpuct=j["cpuct"], lam=j["lam"])
    return j["opp"], (outcome(r, j["seat"]) + 1) / 2


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--bake", action="store_true", help="a comparison: results/bake_<name>.json, not the goal table")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--nh", type=int, default=128)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--ptemp", type=float, default=1.0, help="search prior temperature")
    ap.add_argument("--peps", type=float, default=0.0, help="search prior: share of uniform")
    ap.add_argument("--rollout_net", type=int, default=0, help="1: rollouts buy with the net, not the GA")
    ap.add_argument("--kingdoms", type=int, default=500)
    ap.add_argument("--bot_kingdoms", type=int, default=100)
    args = ap.parse_args()
    from az.base import kingdom_for_bot, random_kingdom
    rng = random.Random("az-v1 goal check")         # same games for every model; never seen in training
    jobs = []
    for _ in range(args.kingdoms):
        k, s = random_kingdom(rng), rng.getrandbits(32)
        jobs += [{"kingdom": k, "seed": s, "seat": seat, "opp": "GA base"} for seat in (0, 1)]
    for b in ("Big Money", "Smithy-BM", "Witch-BM"):
        for _ in range(args.bot_kingdoms):
            k, s = kingdom_for_bot(rng, b), rng.getrandbits(32)
            jobs += [{"kingdom": k, "seed": s, "seat": seat, "opp": b} for seat in (0, 1)]
    for j in jobs:
        j.update(sims=args.sims, nh=args.nh, cpuct=args.cpuct, lam=args.lam, ptemp=args.ptemp, peps=args.peps,
                 rollout_net=args.rollout_net)
    t = time.time()
    with mp.Pool(max(1, os.cpu_count() - 1), _init, (args.net,)) as pool:
        res = pool.map(job, jobs, chunksize=4)
    by = {}
    for o, p in res:
        by.setdefault(o, []).append(p)
    m = float(np.mean(by["GA base"]))
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label or args.name, "note": args.note,
           "policy": None if args.net == "ga" else args.net, "n": len(by["GA base"]),
           "score": round(m, 4), "ci": wilson(m, len(by["GA base"])),
           "big_money": round(float(np.mean(by["Big Money"])), 4),
           "smithy_bm": round(float(np.mean(by["Smithy-BM"])), 4),
           "witch_bm": round(float(np.mean(by["Witch-BM"])), 4),
           "sims": None if args.net == "ga" else args.sims, "lam": args.lam, "cpuct": args.cpuct,
           "ptemp": args.ptemp, "peps": args.peps, "rollout_net": args.rollout_net, "secs": round(time.time() - t)}
    os.makedirs("results", exist_ok=True)
    path = f"results/{'baseline' if args.baseline else 'bake' if args.bake else 'goal'}_{args.name}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(path, json.dumps(out))


if __name__ == "__main__":
    main()
