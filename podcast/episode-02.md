# Episode 02: Two failures that shaped the method

In the last episode I described the plan. A network guides a search, the search teaches the network, round after round. This episode is about the two things that went wrong first, because they explain why the player works the way it does now.

The first failure was about how to score a position. When the search reaches a new position, it needs a number: how good is this for me? The textbook answer is to ask the network. So I tested that first, against the evolved priority list, on six hundred games each.

With the network's guess alone, a hundred imagined games per buy scored about fifty one percent. Four hundred scored fifty three. And sixteen hundred scored forty eight. More thinking made the player worse. The reason is that the network is bad at guessing the winner. After imitation it could copy the evolved list's buys ninety eight percent of the time. But its guess of who would win was barely better than a coin. Dominion between two similar players stays close until late. A search that trusts a bad judge just finds the positions where the judge is most wrong, and the more it searches, the better it finds them.

The fix was to mix. Half the score comes from the network, half from actually playing the game out with the evolved list. A single playout is noisy, but it is honest, and the noise averages out over four hundred of them. With the mix, the scores were fifty two, fifty eight and sixty percent. Now more thinking helps, as it should.

The second failure was about what the network learns. AlphaZero teaches the network the share of attention the search gave each move. If the search tried Gold two hundred times and Silver a hundred, the network learns two thirds Gold, one third Silver. I started the first real run that way. One round later, the network on its own had fallen from fifty two percent to twenty four percent against the evolved list. It had collapsed.

Here is why. In Dominion, many buys are almost equally good. When the options are that close, the search spreads its attention mostly by what the network already believed, plus some random noise added on purpose to make it explore. So the attention shares were mostly the network's own opinion with noise on top. Training on that taught the network to be vague and noisy. The search still held up, because the playouts carried it. But the network itself was learning nothing true.

I tried one more idea from a newer paper, a target that reweights the network's opinion by what the search found. With a network as confident as this one, it barely moved away from the network's opinion at all.

What worked is called expert iteration. The network learns only the one card the search finally chose. No shares, and no added noise at the top of the search. Exploration comes from two other places instead. In the first six turns, the player picks its buy at random in proportion to the search's attention. And three games in ten are played against the evolved list rather than against itself.

That second run started from the same imitation network. After ten rounds, the fresh check said sixty three and a half percent. After twenty rounds, sixty five, which was the goal. After a hundred, about sixty nine.

One more adjustment came early. In the first five rounds, the score against the evolved list rose, but the score against three simple bots, like one that buys only money and Provinces, slowly fell. The player was learning to beat one opponent at the cost of general strength. So from round seven, two games in ten are played against those simple bots. Against them it stayed near seventy eight percent from then on.

So by the end of the first day, the goal was passed. In the next episode I'll tell you what happened over the following three hundred rounds, which is mostly that nothing happened, and why.
