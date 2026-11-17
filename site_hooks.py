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
                    "note": f"Highest score against the GA base in the per-iteration evaluation: {b['check'] if isinstance(b['check'], str) else format(b['check']['score'], '.1%')}. "
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


# ── The model tab: pictures of the newest net (helpers from the ml-project-site template) ─────

def cached(out_png, source_path):
    """True when out_png is newer than the model file it is drawn from (so the figure need not be redrawn)."""
    return os.path.exists(out_png) and os.path.exists(source_path) and os.path.getmtime(out_png) >= os.path.getmtime(source_path)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def fig_arch(layers, out_png, title='Architecture'):
    """layers: [("input", 8), ("dense tanh", 64), ("dense tanh", 64), ("action logits", 4)] -> a box diagram with sizes."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.7 * len(layers)))
    ax.axis('off')
    for i, (name, n) in enumerate(layers):
        y = len(layers) - 1 - i
        w = 1 + 4 * (n / max(m for _, m in layers)) ** 0.5
        ax.add_patch(plt.Rectangle((3 - w / 2, y - 0.3), w, 0.6, fc='#dbe4ff', ec='#3b5bdb'))
        ax.text(3, y, f'{name}  ·  {n}', ha='center', va='center', fontsize=10)
        if i:
            ax.annotate('', (3, y + 0.32), (3, y + 0.68), arrowprops=dict(arrowstyle='->', color='#6b6b70'))
    ax.set_xlim(0, 6); ax.set_ylim(-0.6, len(layers) - 0.4)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def fig_weight_hist(named_arrays, out_png, title='Weights per layer'):
    """named_arrays: [("layer 1", ndarray), ...] -> one histogram per layer (shared x range)."""
    plt = _plt()
    import numpy as np
    n = len(named_arrays)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.6), sharey=False)
    axes = np.atleast_1d(axes)
    lim = max(float(np.abs(a).max()) for _, a in named_arrays) or 1
    for ax, (name, a) in zip(axes, named_arrays):
        a = np.asarray(a).ravel()
        ax.hist(a, bins=40, range=(-lim, lim), color='#3b5bdb')
        ax.set_title(f'{name}\n{a.size} values, std {a.std():.2f}', fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def fig_bars(labels, values, out_png, title='', ylabel='', colors=None):
    """A bar chart: e.g. action probabilities in a fixed situation, or input sensitivity (how much the output moves when
    each input is nudged by one standard deviation)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(labels)), 3))
    ax.bar(range(len(labels)), values, color=colors or '#3b5bdb')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


FEAT_GROUPS = ["own", "draw pile", "opp", "supply", "in game"]
FEAT_EXTRA = ["coins", "buys", "my turns", "opp turns", "my VP", "opp VP", "VP lead", "Provinces left",
              "empty piles", "I go first"]
SITUATIONS = [  # (title, test on (turn, coins, provinces left, VP lead))
    ("Opening, $5", lambda t, c, p, l: t <= 2 and c == 5),
    ("Opening, $4", lambda t, c, p, l: t <= 2 and c == 4),
    ("Early, $6", lambda t, c, p, l: 3 <= t <= 8 and c in (6, 7)),
    ("Middle, $8", lambda t, c, p, l: 6 <= t <= 14 and c >= 8 and p >= 6),
    ("Endgame, behind, $5", lambda t, c, p, l: p <= 3 and l < 0 and c in (5, 6, 7)),
    ("Endgame, ahead, $5", lambda t, c, p, l: p <= 3 and l > 0 and c in (5, 6, 7)),
]


def feature_names():
    names = [f"{g}: {NAME[c]}" for g in FEAT_GROUPS for c in range(N)]
    return names + FEAT_EXTRA


def _viz_states(params):
    """Buy decisions of the net (NET seat, recorded) in the Play tab's games, for the figures."""
    from az.core import GA, NET, Seat, bot as botg, play
    rows = []
    for _, _, opp, k, _, seed in GAMES:
        g = base_genome(k)["ints"]
        og = g if opp == "GA base" else botg(opp, k)
        for s in (0, 1):
            seats = [Seat(NET, g, params, record=True), Seat(GA, og)]
            if s:
                seats.reverse()
            r = play(seats, k, seed + s, nh=128)
            rec = r["rec"]
            for i in range(len(rec["X"])):
                rows.append((rec["X"][i], rec["LEGAL"][i], int(rec["TURN"][i]), k))
    return rows


def model_viz(out_dir):
    """Architecture, weights, the buy distribution in six named situations and input sensitivity of
    the newest net of RUN. Redrawn only when that net changes."""
    from az import net as azn
    nets = _nets()
    if not nets:
        return []
    src = os.path.join(ROOT, nets[-1])
    it = nets[-1].split("_")[-1][:-4]
    p = np.load(src).astype(np.float32)
    nh = 128
    w = azn.unpack(p, nh)
    figs = [("arch", "Architecture", f"{RUN} net {it}: layer sizes read from the saved weights "
                                      f"({p.size:,} numbers). Two heads share the trunk."),
            ("weights", "Weights per layer", "A wide, centred histogram means the layer is used. A spike at 0 would "
                                             "mean dead units; the policy head is wider because its logits must be "
                                             "sharp (the net agrees with the search's buy ~98% of the time)."),
            ("situations", "What it wants to buy", "The net's probabilities over the legal buys in six situations from "
                                                   "the Play tab's games (net seat, no search). Look for: Gold over "
                                                   "Province early, Province at $8 later, Duchy when the game is ending."),
            ("sensitivity", "What it looks at", "How much the buy probabilities move (total variation, averaged over "
                                                "all recorded buy decisions) when one input is nudged by one standard "
                                                "deviation. The 25 most influential of the 175 inputs.")]
    paths = {k: os.path.join(out_dir, f"{k}.png") for k, _, _ in figs}
    if all(cached(paths[k], src) for k in paths):
        return [{"src": f"model/{k}.png", "title": t, "caption": c} for k, t, c in figs]
    plt = _plt()
    fig_arch([("input: what it knows at a buy", core.NF), ("dense ReLU", nh), ("dense ReLU", nh),
              ("policy: a card or nothing  |  value: tanh", core.NA)], paths["arch"], title=f"Net {it}")
    fig_weight_hist([("W1", w["W1"]), ("W2", w["W2"]), ("policy Wp", w["Wp"]), ("value Wv", w["Wv"])],
                    paths["weights"])
    rows = _viz_states(p)
    X = np.array([r[0] for r in rows], np.float32)
    L = np.array([r[1] for r in rows])
    E = core.NF - 10
    logits, v, _ = azn.forward(p, nh, X, L)
    prob = np.exp(logits - logits.max(1, keepdims=True)) * (L > 0)
    prob /= prob.sum(1, keepdims=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (title, test) in zip(axes.ravel(), SITUATIONS):
        pick = None
        for i, (x, _, turn, k) in enumerate(rows):
            coins, prov, lead = round(x[E] * 8), round(x[E + 7] * 8), x[E + 6] * 10
            if test(turn, coins, prov, lead):
                pick = i
                break
        if pick is None:
            ax.axis("off")
            ax.set_title(f"{title}: not in these games", fontsize=9)
            continue
        pr = prob[pick]
        top = [a for a in np.argsort(-pr)[:6] if pr[a] > 0.005]
        labels = ["nothing" if a == core.PASS else NAME[a] for a in top]
        ax.barh(range(len(top))[::-1], [pr[a] for a in top], color="#3b5bdb")
        ax.set_yticks(range(len(top))[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
        x = X[pick]
        ax.set_title(f"{title}\nturn {rows[pick][2]}, value {(v[pick] + 1) / 2 * 100:.0f}% win", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths["situations"], dpi=100)
    plt.close(fig)
    sd = X.std(0)
    base = prob
    eff = np.zeros(core.NF)
    for j in range(core.NF):
        if sd[j] == 0:
            continue
        X2 = X.copy()
        X2[:, j] += sd[j]
        lg, _, _ = azn.forward(p, nh, X2, L)
        pr = np.exp(lg - lg.max(1, keepdims=True)) * (L > 0)
        pr /= pr.sum(1, keepdims=True)
        eff[j] = 0.5 * np.abs(pr - base).sum(1).mean()
    names = feature_names()
    top = np.argsort(-eff)[:25]
    fig_bars([names[j] for j in top], [eff[j] for j in top], paths["sensitivity"],
             title="Change in the buy distribution per +1 std of one input", ylabel="total variation")
    return [{"src": f"model/{k}.png", "title": t, "caption": c} for k, t, c in figs]
