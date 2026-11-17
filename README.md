# dominion-az-v1: AlphaZero-light for Dominion

**🌐 Live site: https://jca374.github.io/dominion-az-v1/**

A light AlphaZero for Dominion (base set, 2nd edition, Steam rules): at every buy, an **information-set Monte Carlo
tree search** (IS-MCTS) with determinization deals the hidden cards anew for each simulation (own draw order, the
opponent's hand and deck), so the search averages over the luck of the draw. A small MLP (175 → 128 → 128 → policy
34 + value) guides the search and learns from it in self-play, warm-started by imitating the evolved GA genomes of
[dominion-v1](https://github.com/JCA374/dominion-v1). Every non-buy decision is played by the GA genome.

- Rules: [dominion-engine](https://github.com/JCA374/dominion-engine), submodule `engine/` (never copied here).
- GA genome agent, released genomes and benchmark bots: dominion-v1, submodule `dominion-v1/` (its `engine/` pin must
  equal ours; a test checks it).
- Everything that runs inside a game is C (`c/az.c` + the engine + dominion-v1's `c/ga_agent.c`, built into `az.so`);
  training is numpy.

```bash
git submodule update --init --recursive
python3 -m venv .venv && .venv/bin/pip install numpy pytest pillow
.venv/bin/python -m pytest -q                         # tests (builds az.so)
.venv/bin/python -u train.py --run r1                 # warm start + self-play loop -> runs/r1/
.venv/bin/python -u train.py --run r1 --continue      # resume
.venv/bin/python experiments/goal_check.py runs/r1/net_12.npy --name iter12   # 1,600 fresh games
.venv/bin/python experiments/leaf_value.py runs/r1/net_0.npy --sims 400       # leaf value: net vs rollout
.venv/bin/python publish.py [--push]                  # rebuild the site (docs/)
```

## How it works

1. **Search** (`c/az.c`): tree nodes are buy decisions only (mine and the opponent's), keyed by the sequence of buys
   and by who buys next. Each simulation: copy the game, `de_hide` (new random deal of everything the buyer can't
   see), walk down by PUCT over the buys legal in this deal, and at the first buy without a node expand it with the
   network. The leaf is scored `lam × net value + (1 − lam) × result of finishing the game with the GA genome`
   (lam = 0.5; the net alone was worse, see `results/bake_leaf_*`). 400 simulations per buy; the most-visited buy is
   played.
2. **Warm start**: 12,000 GA games → the net imitates the GA's buys and predicts the winner.
3. **Self-play**: 1,000 games per iteration (70% search vs search, 30% vs the GA base genome), Dirichlet noise at
   the root and visit-proportional buys in the first 6 turns; the net learns the visit shares and the final result.
4. **Evaluation** every iteration on fixed games against Big Money, Smithy-BM, Witch-BM and the GA base genome; goal
   checks on 1,600 games never used elsewhere.

Goal: 65% head to head against the GA base genome (same action play; only the buys differ) on 1,000 fresh games.

## Results (2026-11-17, training stopped)

| Agent (1,000 fresh games vs the GA base, ±3%) | Score |
|---|---|
| GA base genome | 50% |
| r2 iteration 20 + search | 65.2% (goal passed) |
| r2 iteration 100–350 + search (400 sims, leaf ½ net + ½ GA rollout) | 69–70% |
| r2 net 357 alone, no search | 66.7% |
| r3 (value head trained on the search's root value), iteration 20 | 67.4% |
| r4 (AlphaZero leaves: value head only), iteration 30 | 66.6% |

Best agent: `runs/r2/net_357.npy` (or any r2 net from iteration 100 on) with 400 simulations and lam 0.5.
What was tried after the plateau and why it stopped: the site's Lessons and Log tabs, and `results/bake_*`.
More commands: `experiments/value_holdout.py` (value error on fresh games), `experiments/value_study.py`
(value-target study), `goal_check.py --sims 0` (net alone) and `--bake` (a comparison, not a goal check).
