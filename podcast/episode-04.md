# Episode 04: Fixing the judge, and finding the ceiling

In the last episode I found what was wrong with the player's judgement. Its win estimate had memorised its own training games. I built a new one that learns from the search's careful estimate for each position, instead of from the final result of the game. Its error on fresh games fell from one point two four to point seven. Then I trained with it. This is how that went, and why I have now stopped.

The first new run kept everything else the same. Half of each judgement still came from the network, half from playing the game out with the evolved priority list. After ten rounds, the fresh check said sixty eight percent. After twenty, sixty seven. The old run had sat at sixty nine or seventy. So an honest judge, on its own, changed nothing.

I think I understand why. Half of every judgement came from the evolved list finishing the game. That half can never get better, however long the player trains. The search was learning to buy what is good if the evolved list plays the rest of the game. And the evolved list is exactly the opponent.

So the second new run did what AlphaZero does. It judged positions with the network alone, no playouts at all. That only became possible because the judge was now honest. With the old memorising one, this setting had been the weakest. With the new one it was nearly as strong as the mix. And in this setting, the judge can improve as the network learns, so the ceiling of the evolved list is gone. It is also about a third faster.

It played thirty rounds. The fresh checks said sixty seven, sixty eight, sixty seven. Against the three simple bots, its own evaluation games rose from seventy seven to eighty percent, which looked promising. But the fresh checks against those same bots didn't confirm it. Seventy five and a half percent, the same as before.

So here is where it stands after about twenty different attempts since the plateau began. More thinking per buy. Less thinking. Bolder exploration and timid exploration. Softer advice from the network. Playouts that use the network's buys. An honest judge. Training with it. The pure AlphaZero setting. Nothing beat seventy percent. The best player is still the one from the long run, with four hundred imagined games per buy and the mixed judgement.

Why seventy? My best guess is that it is close to what better buying alone can do here. Both sides play their action cards with the same evolved rules, and those rules were evolved together with the evolved buys. A smarter buyer is stuck with a partner tuned for a different style. And the decisions that remain are close calls, too close for a few hundred noisy imagined games to tell apart reliably.

What did this teach? Three things I will carry to the next project. First, a training number can improve while the model gets worse. Always keep one number measured on data the model never trained on. Second, when a search and its network agree on almost everything, self-play has stopped teaching, however long you let it run. And third, before you ask a search to explore more, make sure its judge can be trusted. Otherwise it explores its way into its judge's mistakes.

The goal of sixty five percent was passed long ago, and the player now wins seven games in ten against the opponent it started as. All of these numbers are on the site, with the lessons, a study guide, and pictures of the network itself. Thanks for listening.
