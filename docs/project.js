// Project formatters: how this project's run events read in the log.
window.eventText = function (e) {
  const p = (x) => (x == null ? '–' : (100 * x).toFixed(1) + '%');
  switch (e.kind) {
    case 'run_start': return [`Run ${e.run} started`, `Self-play with ${e.config.sims} simulations per buy, leaf value ${e.config.lam} × net + ${1 - e.config.lam} × GA rollout, ${e.config.games} games per iteration, net ${e.config.nh}×${e.config.nh}.`];
    case 'baseline': return ['Baseline: the GA base genome on the evaluation games', `vs Big Money ${p(e['Big Money'])}, Smithy-BM ${p(e['Smithy-BM'])}, Witch-BM ${p(e['Witch-BM'])}; mean over the bots ${p(e.bots)}. Against itself 50% by definition.`];
    case 'warm_epoch': return [`Warm start, epoch ${e.epoch + 1}: imitating the GA's buys`, `${e.samples.toLocaleString()} buy decisions from 12,000 GA games. Held-out: picks the GA's buy ${p(e.holdout.policy_acc)} of the time; value error (MSE) ${e.holdout.value_mse.toFixed(3)} (1.0 = knows nothing).`];
    case 'warm_done': return ['Warm start done: iteration 0', `With search: ${e.search}\nNet alone: ${e.net}`];
    case 'goal_best': return [`New best against the GA base: iteration ${e.iter}`, `${e.check}. ${e.search}\nBest of many evaluations: flattered by luck (winner's curse). The goal checks on fresh games are the real measure.`];
    case 'run_resume': return [`Run ${e.run} resumed at iteration ${e.iter}`, ''];
  }
  return null;
};
