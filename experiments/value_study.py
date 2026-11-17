"""Offline study of the value head: why it memorises and which training fixes it.

1. Self-play data like an r2 iteration (search with the given net, 50% vs itself, 30% vs the GA base,
   20% vs the bots), kept per game: runs/study/data.npz (made once, --games).
2. Split by game (80/20). From the given net, train each variant on the training games and measure
   on the held-out games: value error against the final result, policy agreement with the search.

    python experiments/value_study.py runs/r2/net_357.npy --games 3000
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


def game(job):
    from az.selfplay import selfplay_game
    job["params"] = P
    g = selfplay_game(job)
    return {k: g[k] for k in ("X", "PI", "LEGAL", "Z", "Q", "TURN")}


VARIANTS = {  # name: (value target mix vq, value weight, l2, epochs)
    "as r2 (z, w1, l2 1e-4, 10 passes)": (0.0, 1.0, 1e-4, 10),
    "2 passes": (0.0, 1.0, 1e-4, 2),
    "1 pass": (0.0, 1.0, 1e-4, 1),
    "value weight 0.25, 10 passes": (0.0, 0.25, 1e-4, 10),
    "l2 1e-3, 10 passes": (0.0, 1.0, 1e-3, 10),
    "target (z+Q)/2, 10 passes": (0.5, 1.0, 1e-4, 10),
    "target Q, 10 passes": (1.0, 1.0, 1e-4, 10),
    "target (z+Q)/2, 2 passes": (0.5, 1.0, 1e-4, 2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--games", type=int, default=3000)
    args = ap.parse_args()
    from az import net
    from az.base import BOTS, kingdom_for_bot, random_kingdom
    os.makedirs("runs/study", exist_ok=True)
    path = "runs/study/data.npz"
    if not os.path.exists(path):
        cfg = json.load(open("runs/r2/config.json"))
        rng = random.Random("az-v1 value study")
        jobs = []
        for i in range(args.games):
            u = rng.random()
            j = {"seed": rng.getrandbits(32), "seat": i % 2, "cfg": cfg, "vs_ga": u < 0.3}
            if 0.3 <= u < 0.5:
                j["vs_bot"] = BOTS[i % 3]
                j["kingdom"] = kingdom_for_bot(rng, j["vs_bot"])
            else:
                j["kingdom"] = random_kingdom(rng)
            jobs.append(j)
        t = time.time()
        with mp.Pool(max(1, os.cpu_count() - 1), _init, (args.net,)) as pool:
            games = pool.map(game, jobs, chunksize=4)
        gid = np.concatenate([np.full(len(g["Z"]), i) for i, g in enumerate(games)])
        np.savez(path, gid=gid, **{k: np.concatenate([g[k] for g in games]) for k in games[0]})
        print(f"data: {len(gid)} decisions from {len(games)} games in {time.time() - t:.0f}s", flush=True)
    d = dict(np.load(path))
    hold = d["gid"] % 5 == 0
    tr = {k: v[~hold] for k, v in d.items()}
    ho = {k: v[hold] for k, v in d.items()}
    p0 = np.load(args.net)

    def measure(p):
        _, v, _ = net.forward(p, 128, ho["X"])
        m = net.evaluate(p, 128, {k: ho[k] for k in ("X", "PI", "LEGAL", "Z")})
        return {"value_mse_z": round(float(((v - ho["Z"]) ** 2).mean()), 4),
                "value_mse_q": round(float(((v - ho["Q"]) ** 2).mean()), 4),
                "policy_acc": round(m["policy_acc"], 4)}

    out = {"net": args.net, "train_decisions": int((~hold).sum()), "holdout_decisions": int(hold.sum()),
           "zero_mse_z": round(float((ho["Z"] ** 2).mean()), 4),
           "q_as_prediction_mse_z": round(float(((ho["Q"] - ho["Z"]) ** 2).mean()), 4),
           "untrained": measure(p0), "variants": {}}
    print(json.dumps({k: out[k] for k in ("zero_mse_z", "q_as_prediction_mse_z", "untrained")}), flush=True)
    for name, (vq, vw, l2, ep) in VARIANTS.items():
        p = p0.copy()
        opt = net.Adam(len(p), 1e-3)
        data = {k: tr[k] for k in ("X", "PI", "LEGAL")}
        data["Z"] = ((1 - vq) * tr["Z"] + vq * tr["Q"]).astype(np.float32)
        t = time.time()
        m = net.train(p, 128, data, opt, epochs=ep, batch=512, seed=1, value_weight=vw, l2=l2)
        out["variants"][name] = dict(measure(p), train_value_mse=round(m["value_mse"], 4))
        print(name, out["variants"][name], f"{time.time() - t:.0f}s", flush=True)
        json.dump(out, open("results/value_study.json", "w"), indent=1)


if __name__ == "__main__":
    main()
