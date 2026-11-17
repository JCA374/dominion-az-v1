"""The MLP (numpy): forward, loss and gradients, Adam. Same layout and math as c/az.c net_eval.

x[NF] -> relu(W1 x + b1) -> relu(W2 h1 + b2) -> policy logits Wp h2 + bp  (masked to legal buys)
                                             -> value tanh(Wv h2 + bv)    (win = +1, loss = -1)
Loss = cross-entropy(policy, visit distribution) + value_weight * (v - z)^2 + l2 * |W|^2.
"""
from __future__ import annotations

import numpy as np

from .core import NA, NF, lib


def shapes(nh: int) -> list[tuple[str, tuple[int, ...]]]:
    return [("W1", (nh, NF)), ("b1", (nh,)), ("W2", (nh, nh)), ("b2", (nh,)),
            ("Wp", (NA, nh)), ("bp", (NA,)), ("Wv", (nh,)), ("bv", (1,))]


def size(nh: int) -> int:
    n = sum(int(np.prod(s)) for _, s in shapes(nh))
    assert n == lib.az_num_params(nh)
    return n


def unpack(flat: np.ndarray, nh: int) -> dict[str, np.ndarray]:
    out, i = {}, 0
    for name, s in shapes(nh):
        k = int(np.prod(s))
        out[name] = flat[i:i + k].reshape(s)
        i += k
    return out


def init(nh: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = np.zeros(size(nh), np.float32)
    p = unpack(flat, nh)
    p["W1"][:] = rng.normal(0, np.sqrt(2 / NF), (nh, NF))
    p["W2"][:] = rng.normal(0, np.sqrt(2 / nh), (nh, nh))
    p["Wp"][:] = rng.normal(0, 0.01, (NA, nh))
    p["Wv"][:] = rng.normal(0, 0.01, nh)
    return flat


def forward(flat: np.ndarray, nh: int, X: np.ndarray, legal: np.ndarray | None = None):
    p = unpack(flat, nh)
    a1 = X @ p["W1"].T + p["b1"]
    h1 = np.maximum(a1, 0)
    a2 = h1 @ p["W2"].T + p["b2"]
    h2 = np.maximum(a2, 0)
    logits = h2 @ p["Wp"].T + p["bp"]
    v = np.tanh(h2 @ p["Wv"] + p["bv"][0])
    cache = (X, a1, h1, a2, h2)
    if legal is not None:
        logits = np.where(legal > 0, logits, -1e9)
    return logits, v, cache


def loss_and_grad(flat: np.ndarray, nh: int, X, PI, LEGAL, Z, value_weight=1.0, l2=1e-4):
    p = unpack(flat, nh)
    logits, v, (X, a1, h1, a2, h2) = forward(flat, nh, X, LEGAL)
    n = len(X)
    m = logits.max(1, keepdims=True)
    e = np.exp(logits - m) * (LEGAL > 0)
    prob = e / e.sum(1, keepdims=True)
    logp = np.log(np.maximum(prob, 1e-12))
    ce = -(PI * logp).sum(1).mean()
    ent = -(PI * np.log(np.maximum(PI, 1e-12))).sum(1).mean()
    mse = ((v - Z) ** 2).mean()
    W = [p["W1"], p["W2"], p["Wp"], p["Wv"]]
    loss = ce + value_weight * mse + l2 * sum((w ** 2).sum() for w in W)

    g = {k: np.zeros_like(a) for k, a in p.items()}
    dlog = (prob - PI) / n                                  # PI sums to 1 over legal
    dv = value_weight * 2 * (v - Z) / n * (1 - v ** 2)
    g["Wp"] = dlog.T @ h2
    g["bp"] = dlog.sum(0)
    g["Wv"] = dv @ h2
    g["bv"] = np.array([dv.sum()])
    dh2 = dlog @ p["Wp"] + np.outer(dv, p["Wv"])
    da2 = dh2 * (a2 > 0)
    g["W2"] = da2.T @ h1
    g["b2"] = da2.sum(0)
    dh1 = da2 @ p["W2"]
    da1 = dh1 * (a1 > 0)
    g["W1"] = da1.T @ X
    g["b1"] = da1.sum(0)
    for k in ("W1", "W2", "Wp", "Wv"):
        g[k] = g[k] + 2 * l2 * p[k]
    grad = np.concatenate([g[name].ravel() for name, _ in shapes(nh)]).astype(np.float32)
    acc = float((prob.argmax(1) == PI.argmax(1)).mean())
    return float(loss), grad, {"policy_ce": float(ce), "policy_kl": float(ce - ent),
                               "value_mse": float(mse), "policy_acc": acc}


class Adam:
    def __init__(self, n: int, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999):
        self.m = np.zeros(n, np.float32)
        self.v = np.zeros(n, np.float32)
        self.t, self.lr, self.b1, self.b2 = 0, lr, b1, b2

    def step(self, flat: np.ndarray, grad: np.ndarray) -> None:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        flat -= (self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)


def train(flat, nh, data: dict, opt: Adam, epochs: int = 2, batch: int = 512, seed: int = 0,
          value_weight: float = 1.0, l2: float = 1e-4) -> dict:
    """In place on flat. data: X, PI, LEGAL, Z. Returns the mean metrics of the last epoch."""
    rng = np.random.default_rng(seed)
    n = len(data["X"])
    last = {}
    for _ in range(epochs):
        order = rng.permutation(n)
        acc = []
        for i in range(0, n, batch):
            j = order[i:i + batch]
            _, grad, m = loss_and_grad(flat, nh, data["X"][j], data["PI"][j], data["LEGAL"][j],
                                       data["Z"][j], value_weight, l2)
            opt.step(flat, grad)
            acc.append(m)
        last = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
    return last


def evaluate(flat, nh, data: dict) -> dict:
    out = []
    for i in range(0, len(data["X"]), 4096):
        sl = slice(i, i + 4096)
        _, _, m = loss_and_grad(flat, nh, data["X"][sl], data["PI"][sl], data["LEGAL"][sl],
                                data["Z"][sl], l2=0)
        out.append((m, len(data["X"][sl])))
    tot = sum(n for _, n in out)
    return {k: sum(m[k] * n for m, n in out) / tot for k in out[0][0]}
