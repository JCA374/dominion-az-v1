/*
 * az.c -- AlphaZero-light agent for the shared C engine (engine/c/engine.c).
 *
 * Buys are chosen by IS-MCTS (information-set Monte Carlo tree search) guided by a small MLP:
 *   - every simulation starts from a fresh determinization of the hidden cards (de_hide: my draw
 *     order, the opponent's hand and draw order), so the tree averages over the unknown draws;
 *   - the tree holds buy decisions only (mine and the opponent's), keyed by the sequence of buys,
 *     and by who decides next (the next buyer can differ between determinizations);
 *   - a simulation runs the copied game (de_resume_buy) until it reaches a buy decision that has no
 *     node yet: the net gives that node's priors and value, and the simulation stops (longjmp out
 *     of the engine; the copy is a plain struct, nothing to free). A game that ends first scores
 *     its real result;
 *   - every other decision (actions, gains, trashing, attacks) is answered by a GA genome
 *     (dominion-v1's c/ga_agent.c, included unchanged).
 *
 * Built with the engine into az.so by az/core.py.
 */
#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include "../dominion-v1/c/ga_agent.c"

#define AZ_NA      (DE_NUM_CARDS + 1)      /* policy outputs: card ids, then "buy nothing" */
#define AZ_PASS    DE_NUM_CARDS
#define AZ_NF      (5 * DE_NUM_CARDS + 10) /* features, see az_features */
#define AZ_MAXNODE 8192
#define AZ_MAXD    128

int az_num_features(void) { return AZ_NF; }
int az_num_actions(void) { return AZ_NA; }

/* ── network: x -> relu(W1 x + b1) -> relu(W2 h + b2) -> policy logits (Wp), value tanh(Wv) ──
 * params (float32, row-major): W1[nh][nf] b1[nh] W2[nh][nh] b2[nh] Wp[na][nh] bp[na] Wv[nh] bv */
typedef struct { const float *p; int nh; } az_net;

int az_num_params(int nh) {
    return nh * AZ_NF + nh + nh * nh + nh + AZ_NA * nh + AZ_NA + nh + 1;
}

static void dense(const float *w, const float *b, const float *x, int nin, int nout, float *y, int relu) {
    for (int o = 0; o < nout; o++) {
        const float *r = w + (size_t)o * nin;
        float s = b[o];
        for (int i = 0; i < nin; i++) s += r[i] * x[i];
        y[o] = relu && s < 0 ? 0 : s;
    }
}

/* logits[AZ_NA], returns the value in [-1, 1] (for the player the features are from) */
static float net_eval(const az_net *net, const float *x, float *logits) {
    int nh = net->nh;
    float h1[512], h2[512];
    const float *w1 = net->p, *b1 = w1 + nh * AZ_NF, *w2 = b1 + nh, *b2 = w2 + nh * nh;
    const float *wp = b2 + nh, *bp = wp + AZ_NA * nh, *wv = bp + AZ_NA, *bv = wv + nh;
    dense(w1, b1, x, AZ_NF, nh, h1, 1);
    dense(w2, b2, h1, nh, nh, h2, 1);
    dense(wp, bp, h2, nh, AZ_NA, logits, 0);
    float v = bv[0];
    for (int i = 0; i < nh; i++) v += wv[i] * h2[i];
    return tanhf(v);
}

/* What `me` knows at a buy decision: own cards (all, and the draw pile's contents), the opponent's
 * cards (tracked), the supply, coins and buys, turns, scores. */
void az_features(const de_game *g, int me, float *x) {
    const de_player *p = &g->p[me], *o = &g->p[1 - me];
    int own[DE_NUM_CARDS], deck[DE_NUM_CARDS] = {0}, opp[DE_NUM_CARDS];
    owned(p, own);
    owned(o, opp);
    for (int i = 0; i < p->deck_n; i++) deck[p->deck[i]]++;
    const int N = DE_NUM_CARDS;
    for (int c = 0; c < N; c++) {
        x[c] = own[c] / 10.0f;
        x[N + c] = deck[c] / 10.0f;
        x[2 * N + c] = opp[c] / 10.0f;
        x[3 * N + c] = g->supply[c] > 0 ? g->supply[c] / 10.0f : 0;
        x[4 * N + c] = g->supply[c] >= 0;
    }
    float *e = x + 5 * N;
    int vp = de_score(g, me), ovp = de_score(g, 1 - me);
    e[0] = g->coins / 8.0f;
    e[1] = g->buys / 2.0f;
    e[2] = p->turns / 20.0f;
    e[3] = o->turns / 20.0f;
    e[4] = vp / 20.0f;
    e[5] = ovp / 20.0f;
    e[6] = (vp - ovp) / 10.0f;
    e[7] = (g->supply[DE_PROVINCE] > 0 ? g->supply[DE_PROVINCE] : 0) / 8.0f;
    e[8] = de_empty_piles(g) / 3.0f;
    e[9] = me == 0;
}

/* priors over every card still in the supply (cost ignored: affordability differs between
 * determinizations; selection renormalises over the legal ones) */
/* Search priors = softmax(logits / g_prior_temp) mixed with g_prior_eps of uniform. A sharp net (the
 * ExIt target makes it near one-hot) leaves PUCT no reason to try other buys; a softer prior lets the
 * search disagree with the net. Defaults (1, 0) = the plain softmax. */
static float g_prior_temp = 1, g_prior_eps = 0;
void az_set_prior(float temp, float eps) { g_prior_temp = temp > 0 ? temp : 1; g_prior_eps = eps; }

static float eval_priors(const az_net *net, const de_game *g, int me, float *prior) {
    float x[AZ_NF], lg[AZ_NA], mx = -1e30f, sum = 0;
    int nok = 0;
    az_features(g, me, x);
    float v = net_eval(net, x, lg);
    for (int a = 0; a < AZ_NA; a++) {
        int ok = a == AZ_PASS || g->supply[a] > 0;
        if (ok && lg[a] > mx) mx = lg[a];
        nok += ok;
    }
    for (int a = 0; a < AZ_NA; a++) {
        int ok = a == AZ_PASS || g->supply[a] > 0;
        prior[a] = ok ? expf((lg[a] - mx) / g_prior_temp) : 0;
        sum += prior[a];
    }
    for (int a = 0; a < AZ_NA; a++) {
        int ok = a == AZ_PASS || g->supply[a] > 0;
        prior[a] = (1 - g_prior_eps) * prior[a] / sum + (ok ? g_prior_eps / nok : 0);
    }
    return v;
}

/* ── random numbers ── */
static double unif(uint64_t *r) { return ((de_rng_next(r) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static double normal(uint64_t *r) { return sqrt(-2 * log(unif(r))) * cos(6.283185307179586 * unif(r)); }
static double gamma1(double a, uint64_t *r) {   /* Marsaglia-Tsang; a < 1 via the boost */
    if (a < 1) return gamma1(a + 1, r) * pow(unif(r), 1 / a);
    double d = a - 1.0 / 3, c = 1 / sqrt(9 * d);
    for (;;) {
        double z = normal(r), v = 1 + c * z;
        if (v <= 0) continue;
        v = v * v * v;
        if (log(unif(r)) < 0.5 * z * z + d - d * v + d * log(v)) return d * v;
    }
}

/* ── tree ── */
typedef struct {
    int player;               /* who decides here */
    int child[2][AZ_NA];      /* by the next buyer */
    int n[AZ_NA];
    float w[AZ_NA];           /* value sums for `player` */
    float p[AZ_NA];
    float v0;                 /* the net's value here, for `player` */
} az_node;

static az_node pool[AZ_MAXNODE];
static int pool_n;

static int new_node(int player) {
    if (pool_n >= AZ_MAXNODE) return -1;
    az_node *nd = &pool[pool_n];
    nd->player = player;
    memset(nd->child, -1, sizeof(nd->child));
    memset(nd->n, 0, sizeof(nd->n));
    memset(nd->w, 0, sizeof(nd->w));
    return pool_n++;
}

typedef struct {
    const az_net *net;
    const int *genome;          /* non-buy decisions of both players inside the search */
    int me;
    int cur, last;              /* node and action of the edge being walked */
    int pn[AZ_MAXD], pa[AZ_MAXD], depth;
    float cpuct, leaf_v;
    float lambda;               /* leaf value = lambda * net + (1 - lambda) * GA rollout result */
    int rolling;                /* past the leaf: the GA genome buys too, to the end */
    jmp_buf jb;
} az_sim;

static int legal_mask(const int *opts, int n, int may_pass, int *ok) {
    memset(ok, 0, AZ_NA * sizeof(int));
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[AZ_PASS] = 1;
    return n + (may_pass != 0);
}

static int select_puct(const az_node *nd, const int *ok, float cpuct) {
    int total = 0;
    float psum = 0;
    for (int a = 0; a < AZ_NA; a++) if (ok[a]) { total += nd->n[a]; psum += nd->p[a]; }
    if (psum <= 0) psum = 1;
    float sq = sqrtf((float)total + 1), best = -1e30f;
    int pick = -1;
    for (int a = 0; a < AZ_NA; a++) {
        if (!ok[a]) continue;
        float q = nd->n[a] ? nd->w[a] / nd->n[a] : nd->v0 - 0.1f;
        float u = q + cpuct * (nd->p[a] / psum) * sq / (1 + nd->n[a]);
        if (u > best) { best = u; pick = a; }
    }
    return pick;
}

/* Rollout buys: 0 = the GA genome (default), 1 = the net's argmax (actions stay with the genome). */
static int g_rollout_net;
void az_set_rollout(int net) { g_rollout_net = net; }

static int net_buy(const az_net *net, const de_game *g, int me, const int *opts, int n, int may_pass) {
    float x[AZ_NF], lg[AZ_NA], best = -1e30f;
    int ok[AZ_NA], a = AZ_PASS;
    for (int i = 0; i < AZ_NA; i++) ok[i] = 0;
    for (int i = 0; i < n; i++) ok[opts[i]] = 1;
    if (may_pass) ok[AZ_PASS] = 1;
    az_features(g, me, x);
    net_eval(net, x, lg);
    for (int i = 0; i < AZ_NA; i++) if (ok[i] && lg[i] > best) { best = lg[i]; a = i; }
    return a == AZ_PASS ? -1 : a;
}

static int sim_decide(void *ctx, const de_game *g, int kind, int player, int subject,
                      const int *opts, int n, int may_pass) {
    az_sim *s = ctx;
    if (s->rolling && g_rollout_net && kind == DE_BUY && g->nplayers == 2)
        return net_buy(s->net, g, player, opts, n, may_pass);
    if (kind != DE_BUY || g->nplayers != 2 || s->rolling) {
        ga_agent ga = {s->genome, 0};
        return ga_decide(&ga, g, kind, player, subject, opts, n, may_pass);
    }
    az_node *parent = &pool[s->cur];
    int ch = parent->child[player][s->last];
    if (ch < 0) {                                   /* leaf: expand with the net, stop here */
        ch = new_node(player);
        float v = 0;
        if (ch >= 0) {
            v = eval_priors(s->net, g, player, pool[ch].p);
            pool[ch].v0 = v;
            pool[s->cur].child[player][s->last] = ch;
        } else {
            float x[AZ_NF], lg[AZ_NA];
            az_features(g, player, x);
            v = net_eval(s->net, x, lg);
        }
        s->leaf_v = player == s->me ? v : -v;
        if (s->lambda >= 1) longjmp(s->jb, 1);
        s->rolling = 1;                             /* play it out with the genome (or the net's buys) */
        if (g_rollout_net) return net_buy(s->net, g, player, opts, n, may_pass);
        ga_agent ga = {s->genome, 0};
        return ga_decide(&ga, g, kind, player, subject, opts, n, may_pass);
    }
    int ok[AZ_NA];
    legal_mask(opts, n, may_pass, ok);
    int a = select_puct(&pool[ch], ok, s->cpuct);
    if (s->depth < AZ_MAXD) { s->pn[s->depth] = ch; s->pa[s->depth] = a; s->depth++; }
    s->cur = ch;
    s->last = a;
    return a == AZ_PASS ? -1 : a;
}

/* Policy target (Gumbel AlphaZero's completed Q): softmax over the legal buys of
 * log(clean prior) + (c_visit + max visits) * c_scale * q01, q01 = the buy's mean result in [0, 1]
 * (unvisited buys: the root's value). Free of the root noise; equal values leave the prior as is. */
static float g_c_visit = 50, g_c_scale = 0.1f;
static int g_pick_target;          /* play the target's argmax instead of the most-visited buy */
static int g_target_mode = 2;      /* recorded policy target: 0 visit shares, 1 completed Q,
                                      2 one-hot of the most-visited buy (expert iteration) */
void az_set_target(float c_visit, float c_scale, int pick_target, int mode) {
    g_c_visit = c_visit; g_c_scale = c_scale; g_pick_target = pick_target; g_target_mode = mode;
}

static void completed_q_target(const az_node *root, const float *prior, const int *ok, float *target) {
    int maxn = 0;
    float psum = 0, vmix = 0, mx = -1e30f, sum = 0;
    int total = 0;
    for (int a = 0; a < AZ_NA; a++) if (ok[a]) { psum += prior[a]; total += root->n[a]; vmix += root->w[a];
                                                if (root->n[a] > maxn) maxn = root->n[a]; }
    vmix = total ? vmix / total : root->v0;            /* the root's searched value */
    for (int a = 0; a < AZ_NA; a++) {
        if (!ok[a]) { target[a] = -1e30f; continue; }
        float q = root->n[a] ? root->w[a] / root->n[a] : vmix;
        float p = psum > 0 ? prior[a] / psum : 1.0f;
        target[a] = logf(p + 1e-8f) + (g_c_visit + maxn) * g_c_scale * (q + 1) / 2;
        if (target[a] > mx) mx = target[a];
    }
    for (int a = 0; a < AZ_NA; a++) { target[a] = ok[a] ? expf(target[a] - mx) : 0; sum += target[a]; }
    for (int a = 0; a < AZ_NA; a++) target[a] /= sum;
}

/* IS-MCTS from a buy decision of `me`. visits[AZ_NA] and the policy target out; returns the root
 * value (for me). */
static float search(const az_net *net, const int *genome, const de_game *g, int me,
                    const int *opts, int n, int may_pass, int sims, float cpuct, float lambda,
                    int noise, uint64_t *rng, int *visits, float *target) {
    pool_n = 0;
    int root = new_node(me);
    pool[root].v0 = eval_priors(net, g, me, pool[root].p);
    float clean[AZ_NA];
    memcpy(clean, pool[root].p, sizeof(clean));
    int ok[AZ_NA];
    legal_mask(opts, n, may_pass, ok);
    if (noise) {                                     /* Dirichlet(0.3) over the legal actions */
        double d[AZ_NA], sum = 0;
        for (int a = 0; a < AZ_NA; a++) { d[a] = ok[a] ? gamma1(0.3, rng) : 0; sum += d[a]; }
        for (int a = 0; a < AZ_NA; a++)
            if (ok[a]) pool[root].p[a] = 0.75f * pool[root].p[a] + 0.25f * (float)(d[a] / sum);
    }
    de_agent agents[2];
    static az_sim s;                                  /* static: jmp_buf + volatile-safe across longjmp */
    s.net = net; s.genome = genome; s.me = me; s.cpuct = cpuct; s.lambda = lambda;
    agents[0] = agents[1] = (de_agent){sim_decide, &s};
    double vsum = 0;
    for (int k = 0; k < sims; k++) {
        static de_game c;
        c = *g;
        de_hide(&c, me, de_rng_next(rng));
        int a = select_puct(&pool[root], ok, cpuct);
        s.cur = root; s.last = a; s.depth = 0; s.rolling = 0; s.leaf_v = 0;
        s.pn[0] = root; s.pa[0] = a; s.depth = 1;
        float v;
        if (setjmp(s.jb) == 0) {
            de_resume_buy(&c, agents, a == AZ_PASS ? -1 : a);
            int w = de_winners(&c);
            v = w == 3 ? 0.0f : (w >> me & 1) ? 1.0f : -1.0f;
            if (s.rolling) v = lambda * s.leaf_v + (1 - lambda) * v;
        } else {
            v = s.leaf_v;
        }
        for (int d = 0; d < s.depth; d++) {
            az_node *nd = &pool[s.pn[d]];
            nd->n[s.pa[d]]++;
            nd->w[s.pa[d]] += nd->player == me ? v : -v;
        }
        vsum += v;
    }
    memcpy(visits, pool[root].n, sizeof(pool[root].n));
    if (g_target_mode == 1) {
        completed_q_target(&pool[root], clean, ok, target);
    } else {
        int total = 0, best = -1, arg = AZ_PASS;
        for (int a = 0; a < AZ_NA; a++) {
            total += visits[a];
            if (ok[a] && visits[a] > best) { best = visits[a]; arg = a; }
        }
        for (int a = 0; a < AZ_NA; a++)
            target[a] = g_target_mode == 0 ? (total ? (float)visits[a] / total : 0) : a == arg;
    }
    return sims ? (float)(vsum / sims) : pool[root].v0;
}

/* ── the playing agent ── */
enum { AZ_GA, AZ_NET, AZ_SEARCH };

typedef struct {
    int mode;
    az_net net;
    const int *genome;
    int sims, noise, temp_turns, record;
    float cpuct, lambda;
    uint64_t rng;
    int searched;
    double qsum;
} az_agent;

typedef struct {
    float *X, *PI, *Q;
    unsigned char *LEGAL;
    int *SEAT, *TURN;
    int max, n;
} az_rec;

static az_rec *g_rec;

static void record(const de_game *g, int me, const int *ok, const float *pi, float q) {
    az_rec *r = g_rec;
    if (!r || r->n >= r->max) return;
    int k = r->n++;
    az_features(g, me, r->X + (size_t)k * AZ_NF);
    for (int a = 0; a < AZ_NA; a++) {
        r->PI[(size_t)k * AZ_NA + a] = pi[a];
        r->LEGAL[(size_t)k * AZ_NA + a] = (unsigned char)ok[a];
    }
    r->Q[k] = q;
    r->SEAT[k] = me;
    r->TURN[k] = g->p[me].turns + 1;
}

/* Optional trace of every buy decision (for the site's game clips): AZ_TRACE floats each =
 * player, turn, coins, choice (AZ_PASS = none), vp0, vp1, q, own0[N], own1[N], pi[AZ_NA]. */
#define AZ_TRACE (7 + 2 * DE_NUM_CARDS + AZ_NA)
static float *g_trace;
static int g_trace_max, g_trace_n;

int az_trace_width(void) { return AZ_TRACE; }
void az_set_trace(float *buf, int max) { g_trace = buf; g_trace_max = max; g_trace_n = 0; }
int az_trace_count(void) { return g_trace_n; }

static void trace(const de_game *g, int me, int a, const float *pi, float q) {
    if (!g_trace || g_trace_n >= g_trace_max) return;
    float *t = g_trace + (size_t)g_trace_n++ * AZ_TRACE;
    int own[DE_NUM_CARDS];
    t[0] = me; t[1] = g->p[me].turns + 1; t[2] = g->coins; t[3] = a;
    t[4] = de_score(g, 0); t[5] = de_score(g, 1); t[6] = q;
    for (int i = 0; i < 2; i++) {
        owned(&g->p[i], own);
        for (int c = 0; c < DE_NUM_CARDS; c++) t[7 + i * DE_NUM_CARDS + c] = own[c];
    }
    memcpy(t + 7 + 2 * DE_NUM_CARDS, pi, AZ_NA * sizeof(float));
}

static int az_decide(void *ctx, const de_game *g, int kind, int me, int subject,
                     const int *opts, int n, int may_pass) {
    az_agent *ag = ctx;
    ga_agent ga = {ag->genome, 0};
    if (kind != DE_BUY || g->nplayers != 2)
        return ga_decide(&ga, g, kind, me, subject, opts, n, may_pass);
    int ok[AZ_NA];
    legal_mask(opts, n, may_pass, ok);
    float pi[AZ_NA] = {0}, share[AZ_NA], q = 0;   /* share: visit shares (for the clips) */
    int a;
    if (ag->mode == AZ_GA) {
        int c = ga_decide(&ga, g, kind, me, subject, opts, n, may_pass);
        a = c < 0 ? AZ_PASS : c;
        pi[a] = 1;
        if (ag->record) record(g, me, ok, pi, 0);
    } else if (ag->mode == AZ_NET || n + (may_pass != 0) == 1) {
        float x[AZ_NF], lg[AZ_NA], best = -1e30f;
        az_features(g, me, x);
        q = net_eval(&ag->net, x, lg);
        a = AZ_PASS;
        for (int i = 0; i < AZ_NA; i++) if (ok[i] && lg[i] > best) { best = lg[i]; a = i; }
        pi[a] = 1;
        if (ag->record) record(g, me, ok, pi, q);
    } else {
        int visits[AZ_NA];
        q = search(&ag->net, ag->genome, g, me, opts, n, may_pass, ag->sims, ag->cpuct,
                   ag->lambda, ag->noise, &ag->rng, visits, pi);    /* pi = the policy target */
        ag->searched++;
        ag->qsum += q;
        int total = 0;
        for (int i = 0; i < AZ_NA; i++) total += visits[i];
        for (int i = 0; i < AZ_NA; i++) share[i] = total ? (float)visits[i] / total : 0;
        if (g->p[me].turns + 1 <= ag->temp_turns && total) {       /* explore: sample by visits */
            double u = unif(&ag->rng) * total;
            a = AZ_PASS;
            for (int i = 0; i < AZ_NA; i++) {
                if (!visits[i]) continue;
                a = i;
                if ((u -= visits[i]) < 0) break;
            }
        } else if (g_pick_target) {
            a = AZ_PASS;
            float best = -1;
            for (int i = 0; i < AZ_NA; i++) if (ok[i] && pi[i] > best) { best = pi[i]; a = i; }
        } else {
            a = AZ_PASS;
            int best = -1;
            for (int i = 0; i < AZ_NA; i++) if (ok[i] && visits[i] > best) { best = visits[i]; a = i; }
        }
        if (ag->record) record(g, me, ok, pi, q);
        trace(g, me, a, share, q);
        return a == AZ_PASS ? -1 : a;
    }
    trace(g, me, a, pi, q);
    return a == AZ_PASS ? -1 : a;
}

/* One game. Per seat: mode (0 GA genome buys, 1 net argmax, 2 search), params (NULL unless mode
 * 1/2), genome (non-buy decisions, and buys in mode 0), sims, noise, temp_turns, record.
 * lambda: leaf value mix, 1 = the net's value only, 0 = a GA rollout to the end only.
 * out: [winners mask, score0, score1, turns0, turns1, searched0, searched1, capped].
 * Records (if any seat records) go to X/PI/Q/LEGAL/SEAT/TURN, up to max_rec; returns how many. */
int az_play(const int *mode, const float *params0, const float *params1, int nh,
            const int *genome0, const int *genome1, const int *sims, const int *noise,
            const int *temp_turns, const int *rec_seat, float cpuct, float lambda,
            const int *kingdom, int kn, uint64_t seed, uint64_t search_seed,
            float *X, float *PI, float *Q, unsigned char *LEGAL, int *SEAT, int *TURN, int max_rec,
            int *out) {
    az_rec rec = {X, PI, Q, LEGAL, SEAT, TURN, max_rec, 0};
    g_rec = &rec;
    az_agent ag[2];
    const float *params[2] = {params0, params1};
    const int *genome[2] = {genome0, genome1};
    uint64_t r = search_seed | 1;
    for (int i = 0; i < 2; i++) {
        ag[i] = (az_agent){mode[i], {params[i], nh}, genome[i], sims[i], noise[i], temp_turns[i],
                           rec_seat[i], cpuct, lambda, de_rng_next(&r) | 1, 0, 0};
    }
    de_agent agents[2] = {{az_decide, &ag[0]}, {az_decide, &ag[1]}};
    de_game g;
    de_init(&g, kingdom, kn, 2, seed);
    de_run(&g, agents);
    out[0] = de_winners(&g);
    out[1] = de_score(&g, 0);
    out[2] = de_score(&g, 1);
    out[3] = g.p[0].turns;
    out[4] = g.p[1].turns;
    out[5] = ag[0].searched;
    out[6] = ag[1].searched;
    out[7] = g.capped | (g.error << 1);
    g_rec = NULL;
    return rec.n;
}

/* The net's forward pass, for tests (Python and C must agree). */
float az_forward(const float *params, int nh, const float *x, float *logits) {
    az_net net = {params, nh};
    return net_eval(&net, x, logits);
}
