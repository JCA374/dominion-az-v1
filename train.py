"""AlphaZero-light training: warm start from the GA, then self-play with IS-MCTS buys.

    python train.py --run r1                 # new run in runs/r1/
    python train.py --run r1 --continue      # resume from runs/r1/ckpt.npz

Per iteration: self-play games (search vs search, and search vs the GA base genome), train the net
on a replay buffer of the last iterations, then the fixed evaluation (same kingdoms and seeds
every iteration): search and the bare net against Big Money, Smithy-BM, Witch-BM and the GA base
genome (same non-buy play, GA buys). Writes runs/<run>/stats.csv (one row per iteration),
events.jsonl, net_<iter>.npy, ckpt.npz, config.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

CFG = {
    "nh": 128,
    "sims": 400,            # simulations per searched buy in self-play
    "eval_sims": 400,
    "cpuct": 1.5,
    "lam": 0.5,             # leaf value = lam * net value + (1 - lam) * GA rollout (results/leaf_value_*)
    "temp_turns": 6,        # sample buys by visit counts on turns <= this, then argmax
    "noise": 0,             # Dirichlet noise at the root (1 = on)
    "target": 2,            # policy target: 0 visit shares, 1 completed Q, 2 the chosen buy (ExIt)
    "c_scale": 0.1,         # completed-Q target scale (target 1)
    "games": 1000,          # self-play games per iteration
    "vs_ga_share": 0.3,     # share of them against the GA base genome
    "vs_bot_share": 0.0,    # share against Big Money / Smithy-BM / Witch-BM (r2 from iteration 6: 0.2)
    "buffer_iters": 5,
    "epochs": 2,
    "batch": 512,
    "lr": 1e-3,
    "warm_games": 12000,
    "warm_epochs": 8,
    "eval_kingdoms": 50,    # per opponent, x 2 seeds x 2 seats
    "workers": max(1, os.cpu_count() - 1),
    "iterations": 1000,
    "goal_every": 10,       # goal check (1,600 fresh games) every N iterations
    "prior_temp": 1.0,      # search prior = softmax(logits / prior_temp) ...
    "prior_eps": 0.0,       # ... mixed with this share of uniform (a sharp net starves PUCT of exploration)
    "vq": 0.0,              # value target = (1 - vq) * game result + vq * the search's root value
}
OPPONENTS = ["Big Money", "Smithy-BM", "Witch-BM", "GA base"]
BOTS = OPPONENTS[:3]

_wparams: dict = {}


def _load(job):
    """Workers read the net of the job's iteration from disk once."""
    path = job.pop("params_path", None)
    if path:
        if _wparams.get("path") != path:
            _wparams["path"], _wparams["p"] = path, np.load(path)
        job["params"] = _wparams["p"]
    return job


def _warm(job):
    from az.selfplay import warm_game
    return warm_game(job)


def _self(job):
    from az.selfplay import selfplay_game
    return selfplay_game(_load(job))


def _eval(job):
    from az.selfplay import eval_game
    return eval_game(_load(job))


def eval_jobs(cfg: dict) -> list[dict]:
    from az.base import kingdom_for_bot, random_kingdom
    rng = random.Random("az-v1 eval")
    jobs = []
    for opp in OPPONENTS:
        for _ in range(cfg["eval_kingdoms"]):
            k = random_kingdom(rng) if opp == "GA base" else kingdom_for_bot(rng, opp)
            for _ in range(2):
                seed = rng.getrandbits(32)
                for seat in (0, 1):
                    jobs.append({"kingdom": k, "seed": seed, "seat": seat, "opponent": opp})
    return jobs


def run_eval(pool, cfg, agent: str, params_path: str | None) -> dict:
    jobs = [dict(j, agent=agent, params_path=params_path, sims=cfg["eval_sims"], nh=cfg["nh"],
                 cpuct=cfg["cpuct"], lam=cfg["lam"], prior_temp=cfg["prior_temp"], prior_eps=cfg["prior_eps"])
            for j in eval_jobs(cfg)]
    res = list(pool.imap_unordered(_eval, jobs, chunksize=4))
    out = {}
    for opp in OPPONENTS:
        pts = [r["points"] for r in res if r["opponent"] == opp]
        out[opp] = float(np.mean(pts))
    out["bots"] = float(np.mean([out[o] for o in BOTS]))
    out["games"] = len(res)
    out["margin"] = float(np.mean([r["margin"] for r in res]))
    return out


def concat(parts: list[dict]) -> dict:
    keys = ["X", "PI", "LEGAL", "Z"]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)


class Log:
    def __init__(self, d):
        self.d = d

    def event(self, kind, **kw):
        with open(os.path.join(self.d, "events.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **kw}) + "\n")
        print(f"[{kind}]", json.dumps(kw)[:300], flush=True)

    def row(self, row: dict):
        path = os.path.join(self.d, "stats.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def fmt_eval(e: dict) -> str:
    return " ".join(f"{o[:5]}={e[o]:.3f}" for o in OPPONENTS) + f" bots={e['bots']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r1")
    ap.add_argument("--continue", dest="cont", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--warm_from", default=None, help="reuse another run's warm start (net_0, baseline, row 0)")
    ap.add_argument("--init_from", default=None, help="start from this net (e.g. runs/r2/net_357.npy): iteration 0 = "
                                                     "that net evaluated with this run's search settings")
    args = ap.parse_args()

    from az import net
    d = os.path.join("runs", args.run)
    os.makedirs(d, exist_ok=True)
    log = Log(d)
    cfg = dict(CFG)
    if args.cont and os.path.exists(os.path.join(d, "config.json")):
        cfg.update(json.load(open(os.path.join(d, "config.json"))))
    for kv in args.set:
        k, v = kv.split("=")
        cfg[k] = type(CFG[k])(v)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    nh = cfg["nh"]

    pool = mp.Pool(cfg["workers"])
    buffer: list[dict] = []
    t0 = time.time()

    if args.cont:
        ck = np.load(os.path.join(d, "ckpt.npz"))
        params = ck["params"]
        opt = net.Adam(len(params), cfg["lr"])
        opt.m, opt.v, opt.t = ck["m"], ck["v"], int(ck["t"])
        it = int(ck["iter"]) + 1
        base_eval = json.load(open(os.path.join(d, "baseline.json")))
        best = float(ck["best"])
        log.event("run_resume", iter=it)
    elif args.init_from:
        log.event("run_start", config=cfg, init_from=args.init_from)
        params = np.load(args.init_from)
        p0 = os.path.join(d, "net_0.npy")
        np.save(p0, params)
        base_eval = json.load(open(os.path.join(os.path.dirname(args.init_from), "baseline.json")))
        json.dump(base_eval, open(os.path.join(d, "baseline.json"), "w"), indent=1)
        opt = net.Adam(len(params), cfg["lr"])
        ev_s = run_eval(pool, cfg, "search", p0)
        ev_n = run_eval(pool, cfg, "net", p0)
        log.event("init_done", init_from=args.init_from, search=fmt_eval(ev_s), net=fmt_eval(ev_n))
        best = ev_s["GA base"]
        with open(os.path.join(os.path.dirname(args.init_from), "stats.csv")) as f:
            last = list(csv.DictReader(f))[-1]           # the source's training losses, so the columns match
        m = {k: float(last[k]) for k in ("policy_ce", "policy_kl", "value_mse", "policy_acc")}
        log.row(make_row(0, t0, 0, 0, m, {}, ev_s, ev_n, base_eval, {}, cfg))
        it = 1
        save_ckpt(d, params, opt, 0, best)
    elif args.warm_from:
        log.event("run_start", config=cfg, warm_from=args.warm_from)
        src = args.warm_from
        params = np.load(os.path.join(src, "net_0.npy"))
        np.save(os.path.join(d, "net_0.npy"), params)
        base_eval = json.load(open(os.path.join(src, "baseline.json")))
        json.dump(base_eval, open(os.path.join(d, "baseline.json"), "w"), indent=1)
        for line in open(os.path.join(src, "events.jsonl")):
            e = json.loads(line)
            if e["kind"] in ("baseline", "warm_epoch", "warm_done"):
                log.event(e["kind"], **{k: v for k, v in e.items() if k not in ("t", "kind")}, copied_from=src)
        lines = open(os.path.join(src, "stats.csv")).read().splitlines()[:2]
        open(os.path.join(d, "stats.csv"), "w").write("\n".join(lines) + "\n")
        opt = net.Adam(len(params), cfg["lr"])
        best = float(lines[1].split(",")[lines[0].split(",").index("search_GA_base")])
        it = 1
        save_ckpt(d, params, opt, 0, best)
    else:
        log.event("run_start", config=cfg)
        # Baseline: the GA base genome buying on its own, on the evaluation games.
        base_eval = run_eval(pool, cfg, "ga", None)
        json.dump(base_eval, open(os.path.join(d, "baseline.json"), "w"), indent=1)
        log.event("baseline", label="GA base genome (portfolio-v2 specialist per kingdom)",
                  **{k: base_eval[k] for k in OPPONENTS + ["bots"]})
        # Warm start: imitate the GA's buys, value = the GA games' results.
        rng = random.Random("az-v1 warm")
        from az.base import BOTS as B, kingdom_for_bot, random_kingdom
        jobs = []
        for i in range(cfg["warm_games"]):
            if i % 2 == 0:
                jobs.append({"kingdom": random_kingdom(rng), "seed": rng.getrandbits(32)})
            else:
                b = B[i // 2 % 3]
                jobs.append({"kingdom": kingdom_for_bot(rng, b), "seed": rng.getrandbits(32),
                             "bot": b, "seat": i // 6 % 2})
        tw = time.time()
        data = concat(list(pool.imap_unordered(_warm, jobs, chunksize=32)))
        n = len(data["X"])
        hold = np.arange(n) % 10 == 0
        tr = {k: v[~hold] for k, v in data.items()}
        va = {k: v[hold] for k, v in data.items()}
        params = net.init(nh, seed=0)
        opt = net.Adam(len(params), cfg["lr"])
        for ep in range(cfg["warm_epochs"]):
            m = net.train(params, nh, tr, opt, epochs=1, batch=cfg["batch"], seed=ep)
            mv = net.evaluate(params, nh, va)
            log.event("warm_epoch", epoch=ep, samples=n, train=m, holdout=mv)
        np.save(os.path.join(d, "net_0.npy"), params)
        p0 = os.path.join(d, "net_0.npy")
        ev_s = run_eval(pool, cfg, "search", p0)
        ev_n = run_eval(pool, cfg, "net", p0)
        log.event("warm_done", secs=round(time.time() - tw), samples=n,
                  search=fmt_eval(ev_s), net=fmt_eval(ev_n))
        best = ev_s["GA base"]
        log.row(make_row(0, t0, 0, n, mv, {}, ev_s, ev_n, base_eval, {}, cfg))
        it = 1
        save_ckpt(d, params, opt, 0, best)

    while it <= cfg["iterations"]:
        ti = time.time()
        path = os.path.join(d, f"net_{it - 1}.npy")
        rng = random.Random(f"az-v1 self {args.run} {it}")
        from az.base import kingdom_for_bot, random_kingdom
        jobs = []
        for i in range(cfg["games"]):
            u = rng.random()
            job = {"seed": rng.getrandbits(32), "seat": i % 2, "cfg": cfg, "params_path": path,
                   "vs_ga": u < cfg["vs_ga_share"]}
            if cfg["vs_ga_share"] <= u < cfg["vs_ga_share"] + cfg["vs_bot_share"]:
                job["vs_bot"] = BOTS[i % 3]                  # keeps it honest against simple money play
                job["kingdom"] = kingdom_for_bot(rng, job["vs_bot"])
            else:
                job["kingdom"] = random_kingdom(rng)
            jobs.append(job)
        games = list(pool.imap_unordered(_self, jobs, chunksize=4))
        t_play = time.time() - ti
        vs = [g["result"] for g in games if g["vs_ga"]]
        sp = {"selfplay_vs_ga": float(np.mean([(r + 1) / 2 for r in vs])) if vs else float("nan"),
              "selfplay_turns": float(np.mean([g["turns"] for g in games])),
              "root_q": float(np.mean(np.concatenate([g["Q"] for g in games]))),
              "games_per_s": len(games) / t_play}
        buffer.append(concat(games))
        buffer = buffer[-cfg["buffer_iters"]:]
        data = concat(buffer)
        tt = time.time()
        m = net.train(params, nh, data, opt, epochs=cfg["epochs"], batch=cfg["batch"], seed=it)
        t_train = time.time() - tt
        np.save(os.path.join(d, f"net_{it}.npy"), params)
        p = os.path.join(d, f"net_{it}.npy")
        ev_s = run_eval(pool, cfg, "search", p)
        ev_n = run_eval(pool, cfg, "net", p)
        sp.update({"t_play": t_play, "t_train": t_train, "t_iter": time.time() - ti})
        log.row(make_row(it, t0, len(games), len(buffer[-1]["X"]), m, sp, ev_s, ev_n, base_eval,
                         data, cfg))
        print(f"iter {it}: search {fmt_eval(ev_s)} | net {fmt_eval(ev_n)} | {sp['t_iter']:.0f}s",
              flush=True)
        if ev_s["GA base"] > best:
            best = ev_s["GA base"]
            log.event("goal_best", iter=it, check={"score": round(best, 4), "n": 2 * 2 * cfg["eval_kingdoms"]},
                      champion=p, search=fmt_eval(ev_s))
        save_ckpt(d, params, opt, it, best)
        if it % cfg["goal_every"] == 0:          # fresh games, never used elsewhere (the goal table)
            import subprocess
            import sys
            subprocess.run([sys.executable, "experiments/goal_check.py", p, "--name", f"{args.run}_iter{it}",
                            "--label", f"{args.run} iteration {it} + search",
                            "--note", f"{cfg['eval_sims']} sims, leaf {cfg['lam']} net + {1 - cfg['lam']:g} rollout",
                            "--sims", str(cfg["eval_sims"]), "--lam", str(cfg["lam"]), "--nh", str(nh),
                            "--cpuct", str(cfg["cpuct"]), "--ptemp", str(cfg["prior_temp"]),
                            "--peps", str(cfg["prior_eps"])], check=False)
        it += 1


def make_row(it, t0, games, samples, m, sp, ev_s, ev_n, base_eval, data, cfg) -> dict:
    n_opp = 4 * cfg["eval_kingdoms"]
    row = {"iter": it, "hours": round((time.time() - t0) / 3600, 4), "games": games,
           "samples": samples, "buffer": len(data["X"]) if data else samples}
    row.update({k: round(v, 4) for k, v in m.items()})
    for tag, ev in (("search", ev_s), ("net", ev_n), ("ga", base_eval)):
        for o in OPPONENTS:
            row[f"{tag}_{o.replace(' ', '_').replace('-', '_')}"] = round(ev[o], 4)
        row[f"{tag}_bots"] = round(ev["bots"], 4)
    row["search_vs_ga_ci"] = round(ci95(ev_s["GA base"], n_opp), 4)
    for k in ("selfplay_vs_ga", "selfplay_turns", "root_q", "games_per_s", "t_play", "t_train", "t_iter"):
        row[k] = round(sp[k], 4) if k in sp else ""
    return row


def save_ckpt(d, params, opt, it, best):
    np.savez(os.path.join(d, "ckpt.npz"), params=params, m=opt.m, v=opt.v, t=opt.t, iter=it, best=best)


if __name__ == "__main__":
    main()
