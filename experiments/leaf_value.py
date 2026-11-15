"""Does the tree help, and what should score its leaves? Search vs the GA base genome on fixed
kingdoms, for several leaf-value mixes (lambda: 1 = net value, 0 = GA rollout to the end).

    python experiments/leaf_value.py runs/<run>/net_0.npy [--games 200] [--sims 100]
"""
from __future__ import annotations

import argparse
import json
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
    P = np.load(path)


def job(j):
    from az.base import base_genome
    from az.core import GA, NET, SEARCH, Seat, lib, outcome, play
    lib.az_set_target(50, j["cscale"], j["pick"], 2)
    g = base_genome(j["kingdom"])["ints"]
    if j["agent"] == "net":
        me = Seat(NET, g, P)
    else:
        me = Seat(SEARCH, g, P, sims=j["sims"])
    opp = Seat(GA, g)
    seats = [me, opp] if j["seat"] == 0 else [opp, me]
    r = play(seats, j["kingdom"], j["seed"], search_seed=j["seed"] + 1, nh=j["nh"], cpuct=j["cpuct"],
             lam=j["lam"])
    return j["name"], (outcome(r, j["seat"]) + 1) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--nh", type=int, default=128)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--lams", default="1,0.5,0")
    ap.add_argument("--pick", type=int, default=0, help="1: play the completed-Q target's argmax")
    ap.add_argument("--cscale", type=float, default=0.1)
    ap.add_argument("--no-net", action="store_true")
    args = ap.parse_args()
    from az.base import random_kingdom
    rng = random.Random("leaf_value")
    games = []
    for _ in range(args.games // 2):
        k, s = random_kingdom(rng), rng.getrandbits(32)
        games += [(k, s, 0), (k, s, 1)]
    agents = ([] if args.no_net else [("net", "net", 1.0)]) + [(f"search lam={l}", "search", float(l)) for l in args.lams.split(",")]
    jobs = [{"name": n, "agent": a, "lam": l, "kingdom": k, "seed": s, "seat": seat, "sims": args.sims,
             "nh": args.nh, "cpuct": args.cpuct, "pick": args.pick, "cscale": args.cscale} for n, a, l in agents for k, s, seat in games]
    t = time.time()
    with mp.Pool(max(1, os.cpu_count() - 1), _init, (args.net,)) as pool:
        res = pool.map(job, jobs, chunksize=2)
    out = {}
    for n, _, _ in agents:
        pts = [p for name, p in res if name == n]
        m = float(np.mean(pts))
        out[n] = {"score_vs_ga_base": round(m, 3), "ci95": round(1.96 * (m * (1 - m) / len(pts)) ** .5, 3),
                  "games": len(pts)}
        print(f"{n:22s} {m:.3f} ± {out[n]['ci95']:.3f}  ({len(pts)} games)")
    print(f"{time.time() - t:.0f}s")
    os.makedirs("results", exist_ok=True)
    with open(f"results/leaf_value_{os.path.basename(os.path.dirname(args.net))}_{args.sims}{'_pick' if args.pick else ''}.json", "w") as f:
        json.dump({"net": args.net, "sims": args.sims, "cpuct": args.cpuct, "results": out}, f, indent=1)


if __name__ == "__main__":
    main()
