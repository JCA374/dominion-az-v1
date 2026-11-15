import random
import subprocess

import numpy as np
import pytest

from az import base, core, net
from az.core import GA, NET, SEARCH, Seat


def test_engine_pins_match():
    """dominion-v1's genome agent must have been built against the same rules."""
    head = lambda p: subprocess.check_output(["git", "-C", p, "rev-parse", "HEAD"], text=True).strip()
    assert head(f"{core.ROOT}/engine") == head(f"{core.DV1}/engine")


def test_forward_matches_c():
    nh = 32
    p = net.init(nh, seed=1)
    p += np.random.default_rng(2).normal(0, 0.05, p.shape).astype(np.float32)
    X = np.random.default_rng(3).random((5, core.NF), dtype=np.float32)
    logits, v, _ = net.forward(p, nh, X)
    for i in range(5):
        lg = np.zeros(core.NA, np.float32)
        cv = core.lib.az_forward(core._fptr(p), nh, core._fptr(np.ascontiguousarray(X[i])), core._fptr(lg))
        assert abs(cv - v[i]) < 1e-4
        assert np.allclose(lg, logits[i], atol=1e-4)


def test_gradient():
    nh = 8
    rng = np.random.default_rng(0)
    p = net.init(nh, seed=0).astype(np.float64)
    p += rng.normal(0, 0.1, p.shape)
    X = rng.random((6, core.NF))
    LEGAL = (rng.random((6, core.NA)) < 0.5).astype(np.uint8)
    LEGAL[:, core.PASS] = 1
    PI = rng.random((6, core.NA)) * LEGAL
    PI /= PI.sum(1, keepdims=True)
    Z = rng.choice([-1.0, 1.0], 6)
    _, g, _ = net.loss_and_grad(p, nh, X, PI, LEGAL, Z)
    for i in rng.choice(len(p), 25, replace=False):
        e = np.zeros_like(p); e[i] = 1e-5
        num = (net.loss_and_grad(p + e, nh, X, PI, LEGAL, Z)[0] - net.loss_and_grad(p - e, nh, X, PI, LEGAL, Z)[0]) / 2e-5
        assert abs(num - g[i]) < 1e-3 * max(1, abs(num))


def test_ga_mode_plays_the_genome_game():
    """GA seats through az_play give the very game dominion-v1's ga_play_one plays."""
    from core.strategy import load_strategy
    from ga.c_bridge import play_one_c
    import os
    rng = random.Random(5)
    files = sorted(os.listdir(f"{core.PORTFOLIO}/members"))
    for i in range(20):
        k = base.random_kingdom(rng)
        s0 = load_strategy(f"{core.PORTFOLIO}/members/{rng.choice(files)}")
        s1 = load_strategy(f"{core.PORTFOLIO}/members/{rng.choice(files)}")
        ref = play_one_c(s0, s1, i + 1, k)
        r = core.play([Seat(GA, core.genome_ints(s0)), Seat(GA, core.genome_ints(s1))], k, i + 1)
        assert r["scores"] == ref["scores"] and r["winners"] == ref["winners"]
        assert r["turns"][0] == ref["turns"]


@pytest.mark.parametrize("mode", [NET, SEARCH])
def test_net_and_search_play_legal_games(mode):
    nh = 16
    p = net.init(nh, seed=4)
    rng = random.Random(9)
    for i in range(6):
        k = base.random_kingdom(rng)
        g = base.base_genome(k)["ints"]
        r = core.play([Seat(mode, g, p, sims=40, record=True), Seat(GA, g)], k, i + 1, nh=nh)
        assert not r["error"] and not r["capped"]
        if mode == NET:
            continue
        rec = r["rec"]
        assert len(rec["X"]) > 10 and (rec["SEAT"] == 0).all()
        assert np.allclose(rec["PI"].sum(1), 1)
        assert (rec["PI"] * (1 - rec["LEGAL"]) == 0).all()      # visits only on legal buys
