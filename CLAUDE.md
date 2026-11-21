# CLAUDE.md

`JCA374/dominion-az-v1` (folder `trainers/az-v1`): AlphaZero-light for Dominion. IS-MCTS over buy decisions with
determinization (`de_hide`) + a numpy-trained MLP, warm-started from dominion-v1's GA genomes. README.md has the
method and commands; the live site (docs/, ml-project-site skill) shows the training.

- Rules only from `engine/` (submodule). dominion-v1 is a submodule for its genome agent (`c/ga_agent.c`, included
  unchanged by `c/az.c`), `models/portfolio-v2` genomes and the bench bots. Its `engine/` pin must equal ours
  (`tests/test_core.py`). After an engine bump: bump both, rerun tests.
- `c/az.c` is the hot path: search, net forward, the playing agent, records and the clip trace. `az/net.py` must
  match its net layout (`test_forward_matches_c`); GA seats must replay dominion-v1's games exactly
  (`test_ga_mode_plays_the_genome_game`).
- The base genome per kingdom (`az/base.py`): best-ranked portfolio-v2 member whose must cards are dealt (ranking
  cached in `data/members.json`). Every comparison uses it for non-buy decisions on both sides.
  `base_genome_v2` (2026-11-21) is the new GA *opponent*: dominion-v1's portfolio-v3 members, round robin among the
  eligible ones on the dealt kingdom (+6.2 pts vs rl-v1's net over the old base). Agents keep base_genome for their own
  non-buy play.
- Training: `train.py` (config `CFG` at the top, `--set k=v` overrides). Runs in `runs/<run>/` (gitignored except the
  files in site.json `push_globs`). Measured results in `results/` (goal_*, baseline_*, bake_* feed the goal table).
- Site: `publish.py` (generic, from the skill), `site.json`, `site_hooks.py` (Play clips from the C trace),
  `docs/*.html`. `publish_loop.sh` rebuilds and pushes every 15 min.
