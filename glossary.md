# Glossary

One term per line, shown on the Study tab:

**Dominion** — a deck-building card game: each turn you play cards from your hand and buy one new card for your deck; most victory points when the game ends wins
**kingdom** — the 10 action cards (out of the base set's 26) dealt for one game; with the basic cards they form the supply
**supply** — the piles of cards on the table that players can buy from
**buy** — the decision this project learns: which card (or nothing) to add to the deck with this turn's coins
**Big Money** — a benchmark bot that buys only treasure and Provinces; Smithy-BM and Witch-BM add one copy or two of that action card
**GA genome** — an evolved priority list from dominion-v1 that plays every decision; here it plays all non-buy decisions on both sides
**GA base** — the best portfolio-v2 genome whose required cards are in the kingdom; the opponent of the goal measure
**portfolio-v2** — dominion-v1's set of specialist genomes, one per kind of kingdom
**AlphaZero** — DeepMind's method: a tree search guided by a neural network, and the network trained to predict what the search decided and who won
**MCTS** — Monte Carlo tree search: many simulated continuations, each walking down a tree of decisions and backing the result up
**IS-MCTS** — information-set MCTS: one tree over what the player knows, with the hidden cards dealt anew in every simulation
**determinization** — one random guess of the hidden information (draw order, opponent's hand), consistent with what the player has seen
**de_hide** — the engine function that makes a determinization: it reshuffles everything the buying player cannot see
**simulation** — one walk from the root of the search to a leaf, plus scoring the leaf; 400 per buy here
**PUCT** — the rule that picks which buy a simulation tries next: average result so far plus an exploration bonus proportional to the net's prior
**cpuct** — the weight of the exploration bonus in PUCT; higher means the search tries buys the net doesn't like more often
**prior** — the net's probability for each buy, used by PUCT to decide where to look first
**prior temperature** — divides the net's logits before the softmax; above 1 it flattens a sharp prior so the search explores more
**leaf value** — the score given to the position where a simulation stops: here lam × the net's win estimate + (1 − lam) × the result of a GA rollout
**rollout** — finishing a game from the leaf with the GA genome making every decision; one sample of who would win
**lam** — the mix of net value and rollout at the leaf (0 = rollout only, 1 = net only)
**value net / value head** — the part of the net that predicts the winner (tanh, +1 win, −1 loss)
**policy head** — the part of the net that gives a probability for each of the 34 possible buys (33 cards + nothing)
**visit shares** — how often the search tried each buy at the root; AlphaZero's usual policy target
**ExIt** — expert iteration: the net learns the single buy the search chose (one-hot), not the visit shares
**completed Q** — Gumbel AlphaZero's target: the prior reweighted by each buy's searched value
**root noise** — Dirichlet noise added to the prior at the root in self-play, so every buy gets tried sometimes
**self-play** — games where the searching agent plays (mostly) against itself to make training data
**replay buffer** — the training data kept: the last 5 iterations' buy decisions
**iteration** — one round of 1,000 self-play games, training on the buffer, and the fixed evaluation
**warm start** — iteration 0: the net trained to copy the GA's buys before any self-play
**goal check** — 1,000 games against the GA base on 500 kingdoms never used anywhere else, both seats; the honest number
**Wilson interval** — the 95% confidence interval for a win rate; about ±3 points at 1,000 games
**winner's curse** — the best of many noisy evaluations is flattered by luck; why "best check" is not the goal number
**score** — win = 1, shared win = ½, loss = 0, averaged over games
