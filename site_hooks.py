"""Project hooks for publish.py (ml-project-site): the Play tab's game clips and the extra data
files of the custom pages.

A clip is one game, one frame per buy decision of either player: both decks as bars, the score,
what was bought with how many coins and, for a searching player, how the search split its
simulations and its win estimate. Drawing lives here only; no policy imports this file.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from az import core
from az.base import base_genome

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = "r2"
EVERY = 10                     # the Play tab's "newest" model moves every EVERY iterations (clips are committed)
W, H = 800, 450
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
N = core.NUM_CARDS
NAME = core.NAME

BG, INK, MUTED, LINE = (250, 248, 243), (30, 30, 34), (110, 110, 118), (220, 216, 208)
GROUPS = [  # deck bar colours
    ("Copper", (184, 115, 51)), ("Silver", (170, 174, 182)), ("Gold", (222, 180, 40)),
    ("Estate", (120, 170, 110)), ("Duchy", (70, 140, 80)), ("Province", (30, 100, 60)),
    ("Gardens", (150, 190, 90)), ("Curse", (120, 70, 150)),
]
ACTION = (80, 120, 200)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def encode_mp4(frames, out, fps=25):
    """frames: [(PIL image, hold frames)]. Plays once; never loops."""
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset",
                          "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                         stdin=subprocess.PIPE)
    for img, hold in frames:
        b = img.convert("RGB").tobytes()
        for _ in range(hold):
            p.stdin.write(b)
    p.stdin.close()
    p.wait()


# ── models and games ─────────────────────────────────────────────────────────────────────────

def _nets():
    d = os.path.join(ROOT, "runs", RUN)
    nets = sorted(glob.glob(os.path.join(d, "net_*.npy")), key=lambda p: int(p.split("_")[-1][:-4]))
    return [os.path.relpath(p, ROOT) for p in nets]          # repo-relative: paths end up on the site


def _best():
    path = os.path.join(ROOT, "runs", RUN, "events.jsonl")
    best = None
    if os.path.exists(path):
        for line in open(path):
            e = json.loads(line)
            if e["kind"] == "goal_best":
                best = e
    return best


def models():
    nets = _nets()
    out = [{"id": "ga-base", "label": "GA base genome", "path": "ga",
            "note": "dominion-v1's portfolio-v2 specialist for the kingdom, buying by its evolved priority list. "
                    "Every other model uses the same genome for actions, gains and trashing: only the buys differ."}]
    if nets:
        out.append({"id": "warm-search", "label": "Warm start + search", "path": "search:" + nets[0],
                    "note": "The net after imitating the GA (iteration 0), with IS-MCTS: 400 simulations per buy."})
    b = _best()
    if b and os.path.exists(os.path.join(ROOT, b["champion"])):
        it = int(b["champion"].split("_")[-1][:-4])
        out.append({"id": f"best-{it}", "label": f"Best so far (iter {it})", "path": "search:" + b["champion"],
                    "note": f"Highest score against the GA base in the per-iteration evaluation: {b['check']}. "
                            "Best of many evaluations, so flattered by luck."})
    nets = [p for p in nets if int(p.split("_")[-1][:-4]) % EVERY == 0]
    if len(nets) > 1:
        it = int(nets[-1].split("_")[-1][:-4])
        if not b or not b["champion"].endswith(f"net_{it}.npy"):
            out.append({"id": f"newest-{it}", "label": f"Newest (iter {it})", "path": "search:" + nets[-1],
                        "note": "The latest network, with search."})
        out.append({"id": f"net-only-{it}", "label": f"Newest, net only (iter {it})", "path": "net:" + nets[-1],
                    "note": "The latest network choosing buys alone, without any search: what the tree has taught it."})
    return out


GAMES = [
    ("bm", "Against Big Money", "Big Money", ["Cellar", "Harbinger", "Library", "Militia", "Moat", "Moneylender",
                                               "Remodel", "Sentry", "Vassal", "Workshop"],
     "A kingdom without strong draw; the bot buys only money and Provinces.", 101),
    ("smithy", "Against Smithy-Big Money", "Smithy-BM", ["Bureaucrat", "Cellar", "Chapel", "Festival", "Market",
                                                         "Merchant", "Moat", "Poacher", "Smithy", "Village"],
     "The classic benchmark: one or two Smithies and money.", 102),
    ("witch", "Against Witch-Big Money", "Witch-BM", ["Artisan", "Bandit", "Cellar", "Gardens", "Laboratory",
                                                      "Library", "Mine", "Throne Room", "Witch", "Workshop"],
     "Curses fly: the opponent buys Witches and money.", 103),
    ("ga-engine", "Mirror: village + draw", "GA base", ["Council Room", "Festival", "Laboratory", "Market", "Militia",
                                                       "Moneylender", "Remodel", "Sentry", "Smithy", "Village"],
     "Against the GA base genome on an engine kingdom: villages, Smithy, Labs, Markets.", 104),
    ("ga-chapel", "Mirror: Chapel", "GA base", ["Artisan", "Bandit", "Chapel", "Harbinger", "Laboratory", "Market",
                                               "Merchant", "Mine", "Moat", "Vassal"],
     "Against the GA base genome where trashing with Chapel is possible.", 105),
    ("ga-gardens", "Mirror: Gardens + Workshop", "GA base", ["Bureaucrat", "Cellar", "Gardens", "Harbinger", "Library",
                                                            "Moat", "Poacher", "Sentry", "Vassal", "Workshop"],
     "Against the GA base genome on a weak kingdom where the Gardens rush is the known plan.", 106),
    ("ga-attack", "Mirror: attacks", "GA base", ["Bandit", "Bureaucrat", "Council Room", "Festival", "Market",
                                                "Merchant", "Militia", "Moat", "Throne Room", "Witch"],
     "Against the GA base genome with Witch and Militia on the table.", 107),
    ("ga-plain", "Mirror: random kingdom", "GA base", ["Artisan", "Cellar", "Harbinger", "Laboratory", "Library",
                                                      "Mine", "Moneylender", "Remodel", "Sentry", "Village"],
     "Against the GA base genome on a middling kingdom with no attack.", 108),
]


def games():
    return [{"id": g, "title": t, "desc": d, "opponent": o, "kingdom": k, "seed": s}
            for g, t, o, k, d, s in GAMES]


def _seat(spec: str, kingdom):
    g = base_genome(kingdom)["ints"]
    if spec == "ga":
        return core.Seat(core.GA, g), "GA base"
    mode, path = spec.split(":", 1)
    p = np.load(os.path.join(ROOT, path))
    if mode == "net":
        return core.Seat(core.NET, g, p), "Net only"
    return core.Seat(core.SEARCH, g, p, sims=400), "AlphaZero-light"


# ── drawing ──────────────────────────────────────────────────────────────────────────────────

def _deck_bar(d, x, y, w, h, own):
    total = max(1, int(sum(own)))
    order = [(core.ID[n], c) for n, c in GROUPS]
    cx = x
    segs = []
    for cid, col in order:
        segs.append((own[cid], col, NAME[cid]))
    acts = [(own[c], ACTION, NAME[c]) for c in range(N) if own[c] and c not in [cid for cid, _ in order]]
    for n, col, _ in segs[:3] + acts + segs[3:]:
        if n <= 0:
            continue
        ww = w * n / total
        d.rectangle([cx, y, cx + ww - 1, y + h], fill=col)
        cx += ww
    d.rectangle([x, y, x + w, y + h], outline=LINE)


def _deck_text(own):
    parts = []
    for c in range(N):
        if own[c] > 0:
            parts.append(f"{NAME[c]} {int(own[c])}")
    return ", ".join(parts)


def _wrap(d, text, f, width):
    lines, line = [], ""
    for w in text.split(", "):
        t = (line + ", " + w) if line else w
        if d.textlength(t, font=f) > width:
            lines.append(line + ",")
            line = w
        else:
            line = t
    if line:
        lines.append(line)
    return lines


def frame(t, labels, kingdom, last_buy, final=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    me, turn, coins, choice = int(t[0]), int(t[1]), int(t[2]), int(t[3])
    vp = [int(t[4]), int(t[5])]
    q = float(t[6])
    own = [t[7:7 + N], t[7 + N:7 + 2 * N]]
    pi = t[7 + 2 * N:]
    d.text((16, 10), f"Turn {turn}", font=font(20, True), fill=INK)
    d.text((120, 15), "Kingdom: " + ", ".join(kingdom), font=font(11), fill=MUTED)
    for i in range(2):
        y0 = 48 + i * 170
        active = i == me and final is None
        d.rectangle([10, y0, W - 10, y0 + 160], outline=(60, 90, 160) if active else LINE, width=2 if active else 1)
        d.text((22, y0 + 8), labels[i] + ("  (first player)" if i == 0 else ""), font=font(16, True), fill=INK)
        d.text((W - 150, y0 + 6), f"{vp[i]} VP", font=font(22, True), fill=INK)
        _deck_bar(d, 22, y0 + 36, W - 44, 18, own[i])
        y = y0 + 60
        for line in _wrap(d, f"{int(sum(own[i]))} cards: " + _deck_text(own[i]), font(12), W - 44)[:3]:
            d.text((22, y), line, font=font(12), fill=MUTED)
            y += 16
        if last_buy[i]:
            d.text((22, y0 + 116), last_buy[i][0], font=font(14, True), fill=INK)
            if last_buy[i][1]:
                d.text((22, y0 + 136), last_buy[i][1], font=font(12), fill=(60, 90, 160))
    if final:
        d.rectangle([0, H - 82, W, H], fill=(30, 30, 34))
        d.text((20, H - 70), final, font=font(22, True), fill=(250, 248, 243))
    else:
        d.text((16, H - 40), "Bar: copper/silver/gold, blue = kingdom cards, green = victory, purple = curses.",
               font=font(11), fill=MUTED)
    return img


def render_play(policy_path, game, out_mp4):
    k = game["kingdom"]
    seat, label = _seat(policy_path, k)
    from az.core import GA, Seat
    from az.core import bot as botg
    og = base_genome(k)["ints"] if game["opponent"] == "GA base" else botg(game["opponent"], k)
    opp = Seat(GA, og)
    frames = []
    total = {"w": 0}
    results = []
    # one game with the model in seat 0, then the same seed with seats swapped
    for s in (0,):
        seats = [seat, opp] if s == 0 else [opp, seat]
        labels = [label, game["opponent"]] if s == 0 else [game["opponent"], label]
        r = core.play(seats, k, game["seed"], search_seed=game["seed"] + 1, nh=128, cpuct=1.5, lam=0.5,
                      trace=True)
        tr = r["trace"]
        last = [None, None]
        for t in tr:
            me, coins, choice = int(t[0]), int(t[2]), int(t[3])
            what = "nothing" if choice == core.PASS else NAME[choice]
            line2 = ""
            pi = t[7 + 2 * N:]
            if seats[me].mode == core.SEARCH and pi.max() < 0.999:
                top = np.argsort(-pi)[:3]
                line2 = "search: " + " · ".join(
                    f"{'nothing' if a == core.PASS else NAME[a]} {pi[a] * 100:.0f}%" for a in top if pi[a] > 0)
                line2 += f"   win estimate {(float(t[6]) + 1) / 2 * 100:.0f}%"
            last[me] = (f"${coins}: buys {what}", line2)
            frames.append((frame(t, labels, k, last), 11))
        res = core.outcome(r, s)
        word = "wins" if res > 0 else "loses" if res < 0 else "shares the win"
        sc = r["scores"]
        fin = f"{label} {word} {sc[s]}–{sc[1 - s]} in {max(r['turns'])} turns"
        frames.append((frame(tr[-1], labels, k, last, final=fin), 75))
        results.append((res, sc[s] - sc[1 - s], fin))
    encode_mp4(frames, out_mp4)
    res, margin, fin = results[0]
    return {"ok": res > 0, "result": "win" if res > 0 else "loss" if res < 0 else "shared",
            "detail": f"{fin}."}


# ── extra data for the custom pages ─────────────────────────────────────────────────────────

def extra(data_dir):
    """speed.json and experiments.json for the Speed and Lessons pages."""
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        out[os.path.basename(p)[:-5]] = json.load(open(p))
    json.dump(out, open(os.path.join(data_dir, "results.json"), "w"), separators=(",", ":"))
