"""Speed of the pieces, single core: GA game, net forward, search per simulation (net leaf and mixed
leaf), a whole search game, and a training step. Writes results/speed.json for the Speed tab.

    python experiments/speed.py runs/r1/net_0.npy
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from az import base, core, net  # noqa: E402


def main():
    p = np.load(sys.argv[1])
    nh = 128
    rng = random.Random(1)
    ks = [base.random_kingdom(rng) for _ in range(20)]
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "cpu": os.cpu_count()}
    try:
        out["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    t = time.perf_counter()
    n = 0
    for i, k in enumerate(ks * 50):
        g = base.base_genome(k)["ints"]
        core.play([core.Seat(core.GA, g), core.Seat(core.GA, g)], k, i + 1)
        n += 1
    out["ga_game_ms"] = round((time.perf_counter() - t) / n * 1e3, 3)
    x = np.random.default_rng(0).random(core.NF, dtype=np.float32)
    lg = np.zeros(core.NA, np.float32)
    t = time.perf_counter()
    for _ in range(20000):
        core.lib.az_forward(core._fptr(p), nh, core._fptr(x), core._fptr(lg))
    out["net_forward_us_incl_ctypes"] = round((time.perf_counter() - t) / 20000 * 1e6, 2)
    for lam in (1.0, 0.5):
        t = time.perf_counter()
        dec = 0
        for i, k in enumerate(ks[:8]):
            g = base.base_genome(k)["ints"]
            r = core.play([core.Seat(core.SEARCH, g, p, sims=400), core.Seat(core.GA, g)], k, i + 1, nh=nh, lam=lam)
            dec += r["searched"][0]
        dt = time.perf_counter() - t
        out[f"search_lam{lam}_us_per_sim"] = round(dt / dec / 400 * 1e6, 2)
        out[f"search_lam{lam}_game_s"] = round(dt / 8, 3)
        out[f"search_lam{lam}_buys_per_game"] = round(dec / 8, 1)
    X = np.random.default_rng(0).random((512, core.NF)).astype(np.float32)
    PI = np.full((512, core.NA), 1 / core.NA, np.float32)
    L = np.ones((512, core.NA), np.uint8)
    Z = np.zeros(512, np.float32)
    q = p.copy()
    opt = net.Adam(len(q))
    t = time.perf_counter()
    for _ in range(50):
        _, gr, _ = net.loss_and_grad(q, nh, X, PI, L, Z)
        opt.step(q, gr)
    out["train_step_ms_batch512"] = round((time.perf_counter() - t) / 50 * 1e3, 2)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/speed.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
