"""Reproduce the paper's offline experiments (Section 5) on the simulator.

    python experiments/run_all.py                    # uses YOUR code in src/
    python experiments/run_all.py --impl solutions   # uses the reference solutions

Outputs go to results/: table1.md, table2.md, table3.md, fig2_convergence.png,
fig3_daily_miss.png, checks.md, summary.json.

Paper map
  Table 1  -> table1.md   (whole-page diversity features vs CVR baseline)
  Fig. 2   -> fig2_convergence.png (Algorithm 1 converges)
  Fig. 3   -> fig3_daily_miss.png  (yesterday's duals applied today)
  Table 2  -> table2.md   (global constraints vs slotting vs DIV)
  Table 3  -> table3.md   (who receives the constrained category)
  Sec 4.3  -> checks.md   (greedy vs brute force)
  Sec 4.2  -> checks.md   (n = 1 linear case vs an exact LP)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _use_impl(name: str):
    sys.path.insert(0, str(ROOT / name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="src", choices=["src", "solutions"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", default="synthetic", choices=["synthetic", "movielens"])
    ap.add_argument("--ml-dir", default=str(ROOT / "data" / "ml-latest-small"))
    ap.add_argument("--providers", default=None, help="providers.json from scripts/fetch_tmdb_providers.py")
    ap.add_argument("--users", type=int, default=3000, help="page contexts per day |U^t|")
    ap.add_argument("--slots", type=int, default=8, help="rows per page n")
    ap.add_argument("--target", type=float, default=1.8, help="min avg rows/page for rent_buy and channel")
    ap.add_argument("--eta0", type=float, default=0.06)
    ap.add_argument("--gamma0", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    _use_impl(args.impl)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import linprog

    from wpo.world import make_synthetic_world, sample_day
    from wpo.simulate import random_pages, simulate_clicks, true_page_probs
    from wpo.features import page_feature_matrix
    from wpo.model import BayesianLinearModel, PageScorer
    from wpo.greedy import compose_pages, model_page_values, brute_force_page, regularized_value
    from wpo.primal_dual import solve_duals
    from wpo.metrics import (auc, category_counts, expected_precision_at_k, div_pair, miss_at_n,
                             per_category_miss, coverage_by_bucket, pct_lift)
    from wpo.baselines import slotting_patterns, div_matrix

    t0 = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.world == "movielens":
        from wpo.data.movielens import world_from_movielens
        world = world_from_movielens(args.ml_dir, providers_json=args.providers, seed=args.seed)
        if getattr(world, "pseudo_categories", False):
            print("[warn] no --providers given: categories are PSEUDO-RANDOM, not real availability")
    else:
        world = make_synthetic_world(args.seed)
    wc, m, n, N = world.widget_cat, world.n_categories, args.slots, args.users
    cats = world.categories
    b = np.array([args.target if c in ("rent_buy", "channel") else 0.0 for c in cats])
    train_days, eval_days = list(range(0, 7)), list(range(7, 14))
    pd_kw = dict(eta0=args.eta0, gamma0=args.gamma0, eps=args.eps, max_iter=args.max_iter)
    print(f"[setup] world={args.world} widgets={world.n_widgets} impl={args.impl} users/day={N} slots={n} targets={dict(zip(cats, b))}")

    # ------------------------------------------------------------ training logs
    Xw, Xc, Y = [], [], []
    for d in train_days:
        u = sample_day(world, d, N, seed=args.seed)
        pg = random_pages(N, world.n_widgets, n, seed=1000 * args.seed + d)
        _, y = simulate_clicks(world, u, pg, seed=2000 + 1000 * args.seed + d)
        Xw.append(page_feature_matrix(world, u, pg, True))
        Xc.append(page_feature_matrix(world, u, pg, False))
        Y.append(y.reshape(-1))
    Xw, Xc, Y = np.vstack(Xw), np.vstack(Xc), np.concatenate(Y)
    wpo_model = BayesianLinearModel().fit(Xw, Y)
    cvr_model = BayesianLinearModel().fit(Xc, Y)
    print(f"[train] {len(Y):,} logged rows, base stream rate {Y.mean():.3f}")

    def scorers(users):
        sw = PageScorer(world, wpo_model.mean, True)
        sc = PageScorer(world, cvr_model.mean, False)
        return (sw.relevance_matrix(users), sw.diversity_matrix(users),
                sc.relevance_matrix(users), sc.diversity_matrix(users))

    # ------------------------------------------------- held-out AUC (Table 1)
    u7 = sample_day(world, eval_days[0], N, seed=args.seed)
    pg7 = random_pages(N, world.n_widgets, n, seed=1000 * args.seed + 99)
    _, y7 = simulate_clicks(world, u7, pg7, seed=2999)
    auc_wpo = auc(y7.reshape(-1), wpo_model.predict(page_feature_matrix(world, u7, pg7, True)))
    auc_cvr = auc(y7.reshape(-1), cvr_model.predict(page_feature_matrix(world, u7, pg7, False)))

    # Tune DIV's single diversity weight on the last training day (true reward).
    u6 = sample_day(world, train_days[-1], N, seed=args.seed)
    _, _, Rc6, _ = scorers(u6)
    best_lam, best_val = 0.0, -np.inf
    for lam in np.linspace(0.0, 0.6, 25):
        pg = compose_pages(Rc6, div_matrix(N, m, lam), wc, n)
        v = true_page_probs(world, u6, pg).sum(1).mean()
        if v > best_val:
            best_lam, best_val = float(lam), v
    print(f"[tune] DIV lambda = {best_lam:.3f}")

    # ----------------------------------------- Fig. 2: Algorithm 1 convergence
    Rw6, Dw6, _, _ = scorers(u6)
    _, _, hist = solve_duals(Rw6, Dw6, wc, n, b, **{**pd_kw, "eps": 0.0})
    s_idx = cats.index("rent_buy")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    it = [h["iter"] for h in hist]
    ax1.plot(it, [100 * h["miss"] for h in hist], "k-", label="avg miss@n (%)")
    ax1.plot(it, [100 * h["per_cat_miss"][s_idx] for h in hist], "o--", color="tab:orange",
             label="miss, rent_buy (%)")
    ax1.axhline(100 * args.eps, color="grey", lw=0.8, ls=":")
    ax1.set_xlabel("iteration j"); ax1.set_ylabel("percent miss")
    ax2 = ax1.twinx()
    ax2.plot(it, [h["xi"][s_idx] for h in hist], "^:", color="tab:blue", label="xi, rent_buy")
    ax2.set_ylabel("dual variable")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_title("Algorithm 1 convergence (cf. paper Fig. 2)")
    fig.tight_layout(); fig.savefig(out / "fig2_convergence.png", dpi=150); plt.close(fig)
    iters_to_eps = next((h["iter"] for h in hist if h["miss"] <= args.eps), None)

    # ----------------------------------------------- evaluation over days 7..13
    policies = ["CVR", "DIV", "WPO-Free", "WPO-Global", "Slot", "DIV-Global"]
    rows = {p: [] for p in policies}
    daily = []
    table3_pages = None
    prev_users = u6
    for d in eval_days:
        u = sample_day(world, d, N, seed=args.seed)
        Rw, Dw, Rc, Dc = scorers(u)
        Dd = div_matrix(N, m, best_lam)
        Rw_p, Dw_p, Rc_p, _ = scorers(prev_users)
        xi_prev, _, _ = solve_duals(Rw_p, Dw_p, wc, n, b, **pd_kw)
        xi_div, _, _ = solve_duals(Rc_p, div_matrix(prev_users.n, m, best_lam), wc, n, b, **pd_kw)
        xi_oracle, _, _ = solve_duals(Rw, Dw, wc, n, b, **pd_kw)
        pages = {
            "CVR": compose_pages(Rc, Dc, wc, n),
            "DIV": compose_pages(Rc, Dd, wc, n),
            "WPO-Free": compose_pages(Rw, Dw, wc, n),
            "WPO-Global": compose_pages(Rw, Dw, wc, n, xi=xi_prev),
            "Slot": compose_pages(Rw, Dw, wc, n, slot_category=slotting_patterns(N, n, b, seed=d)),
            "DIV-Global": compose_pages(Rc, Dd, wc, n, xi=xi_div),
        }
        oracle_pages = compose_pages(Rw, Dw, wc, n, xi=xi_oracle)
        for p, pg in pages.items():
            pt = true_page_probs(world, u, pg)
            avg = category_counts(pg, wc, m).mean(0)
            rows[p].append(dict(
                true_reward=pt.sum(1).mean(),
                model_reward=model_page_values(Rw, Dw, wc, pg).mean(),
                p_at_4=expected_precision_at_k(pt, 4), p_at_8=expected_precision_at_k(pt, n),
                div=div_pair(pg, wc, m), miss=miss_at_n(avg, b)))
        avg_g = category_counts(pages["WPO-Global"], wc, m).mean(0)
        avg_o = category_counts(oracle_pages, wc, m).mean(0)
        daily.append(dict(day=d, weekend=u.weekend, miss=miss_at_n(avg_g, b),
                          max_miss=float(per_category_miss(avg_g, b).max()),
                          oracle_miss=miss_at_n(avg_o, b), xi=xi_prev.tolist()))
        if d == eval_days[-1]:
            table3_pages = (u, pages)
        prev_users = u

    agg = {p: {k: float(np.mean([r[k] for r in rows[p]])) for k in rows[p][0]} for p in policies}

    # ------------------------------------------------------------- Table 1
    def lift(p, base, k):
        return f"{pct_lift(agg[p][k], agg[base][k]):+.2f}%"
    t1 = ["| Metric | CVR | WPO-Free | DIV |", "|---|---|---|---|",
          f"| AUC (held-out logs) | (B) {auc_cvr:.4f} | {pct_lift(auc_wpo, auc_cvr):+.2f}% | same as CVR |",
          f"| Precision@4 (true) | (B) | {lift('WPO-Free','CVR','p_at_4')} | {lift('DIV','CVR','p_at_4')} |",
          f"| Precision@{n} (true) | (B) | {lift('WPO-Free','CVR','p_at_8')} | {lift('DIV','CVR','p_at_8')} |",
          f"| Div-pair@{n} | (B) | {lift('WPO-Free','CVR','div')} | {lift('DIV','CVR','div')} |",
          f"| miss@{n} (absolute) | {100*agg['CVR']['miss']:.1f}% | {100*agg['WPO-Free']['miss']:.1f}% | {100*agg['DIV']['miss']:.1f}% |",
          f"| True page reward | (B) | {lift('WPO-Free','CVR','true_reward')} | {lift('DIV','CVR','true_reward')} |"]
    (out / "table1.md").write_text("# Table 1 replication (unconstrained models)\n\n" + "\n".join(t1) + "\n")

    # ------------------------------------------------------------- Table 2
    cols = ["WPO-Free", "WPO-Global", "Slot", "DIV-Global"]
    def cell(p, k):
        return "(B)" if p == "WPO-Free" else lift(p, "WPO-Free", k)
    t2 = ["| Metric | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1),
          "| Precision@4 (true) | " + " | ".join(cell(p, "p_at_4") for p in cols) + " |",
          f"| Precision@{n} (true) | " + " | ".join(cell(p, "p_at_8") for p in cols) + " |",
          f"| Div-pair@{n} | " + " | ".join(cell(p, "div") for p in cols) + " |",
          f"| miss@{n} (absolute) | " + " | ".join(f"{100*agg[p]['miss']:.1f}%" for p in cols) + " |",
          "| Page reward (model) | " + " | ".join(cell(p, "model_reward") for p in cols) + " |",
          "| Page reward (TRUE) | " + " | ".join(cell(p, "true_reward") for p in cols) + " |"]
    (out / "table2.md").write_text("# Table 2 replication (global constraints)\n\n" + "\n".join(t2) + "\n")

    # ------------------------------------------------------------- Table 3
    u, pages = table3_pages
    k = cats.index("rent_buy")
    aff = u.aff[:, k]
    labels = ["0 (never)", "low", "moderate", "high", "extremely high"]
    bucket = np.zeros(u.n, dtype=int)
    active = aff > -1.2
    qs = np.quantile(aff[active], [0.25, 0.5, 0.75])
    bucket[active] = 1 + np.searchsorted(qs, aff[active])
    t3 = ["| True propensity for rent_buy | DIV-Global | WPO-Global | Slot |", "|---|---|---|---|"]
    cov = {p: coverage_by_bucket(category_counts(pages[p], wc, m)[:, k], bucket, labels, b[k])
           for p in ["DIV-Global", "WPO-Global", "Slot"]}
    for lab in labels:
        t3.append(f"| {lab} | {cov['DIV-Global'][lab]:+.1f}% | {cov['WPO-Global'][lab]:+.1f}% | {cov['Slot'][lab]:+.1f}% |")
    (out / "table3.md").write_text("# Table 3 replication (coverage vs target, last eval day)\n\n" + "\n".join(t3) + "\n")

    # ---------------------------------------------------------------- Fig. 3
    fig, ax = plt.subplots(figsize=(7, 4))
    days = [r["day"] - eval_days[0] + 1 for r in daily]
    ax.plot(days, [100 * r["miss"] for r in daily], "k-o", label="avg test miss@n (xi from day t-1)")
    ax.plot(days, [100 * r["max_miss"] for r in daily], "--s", color="tab:orange", label="worst category")
    ax.plot(days, [100 * r["oracle_miss"] for r in daily], ":", color="tab:green", label="oracle (xi tuned on day t)")
    for r, dd in zip(daily, days):
        if r["weekend"]:
            ax.axvspan(dd - 0.5, dd + 0.5, color="grey", alpha=0.12)
    ax.set_xlabel("evaluation day (shaded = weekend)"); ax.set_ylabel("percent miss")
    ax.set_title("Yesterday's duals applied today (cf. paper Fig. 3)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig3_daily_miss.png", dpi=150); plt.close(fig)

    # ------------------------------------------- checks: greedy vs brute force
    rng = np.random.default_rng(args.seed)
    Rw, Dw, _, _ = scorers(u)
    xi_last = np.array(daily[-1]["xi"])
    ratios = []
    for _ in range(30):
        i = int(rng.integers(u.n))
        cand = rng.choice(world.n_widgets, size=min(9, world.n_widgets), replace=False)
        sub_R = np.full_like(Rw[i], -1e9); sub_R[cand] = Rw[i, cand]
        g = compose_pages(sub_R[None, :], Dw[i:i + 1], wc, 4, xi=xi_last)[0]
        gv = regularized_value(Rw[i], Dw[i], wc, g, xi_last)
        _, bv = brute_force_page(Rw[i], Dw[i], wc, 4, cand, xi_last)
        ratios.append(gv / bv)

    # ------------------------------------------- checks: n = 1 linear vs LP
    nl, Wl = 400, world.n_widgets
    ul = sample_day(world, 20, nl, seed=args.seed)
    Rl = PageScorer(world, wpo_model.mean, True).relevance_matrix(ul)
    b1 = np.array([0.25 if c in ("rent_buy", "channel") else 0.0 for c in cats])
    c_obj = -Rl.reshape(-1)
    A_eq = np.kron(np.eye(nl), np.ones((1, Wl)))
    A_ub, b_ub = [], []
    for kk in range(m):
        if b1[kk] > 0:
            A_ub.append(-np.tile((wc == kk).astype(float), nl) / nl); b_ub.append(-b1[kk])
    lp = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=np.ones(nl),
                 bounds=(0, 1), method="highs")
    xi1, pg1, _ = solve_duals(Rl, np.zeros((nl, m)), wc, 1, b1, eta0=0.2, gamma0=0.0, eps=0.0, max_iter=200)
    pd_val = Rl[np.arange(nl), pg1[:, 0]].mean()
    lp_val = -lp.fun / nl
    pd_miss = miss_at_n(category_counts(pg1, wc, m).mean(0), b1)

    checks = [
        "# Algorithm checks",
        "",
        "## Greedy vs brute force (Section 4.3)",
        f"30 random users, 9 candidate widgets, n = 4, objective = Eq. (9) with xi from the last day.",
        f"- min ratio greedy/optimal: {min(ratios):.4f}",
        f"- mean ratio: {np.mean(ratios):.4f}",
        f"- theoretical floor (1 - 1/e): {1 - 1/np.e:.4f}",
        "",
        "## n = 1, linear f: primal-dual vs exact LP (Section 4.2 special case)",
        f"- LP optimum (fractional, an upper bound): {lp_val:.5f} per page",
        f"- primal-dual (integral pages): {pd_val:.5f} per page, gap {100*(1 - pd_val/lp_val):.2f}%",
        f"- primal-dual miss@1: {100*pd_miss:.2f}%",
        f"- LP duals for the constraints (marginals): {np.round(-lp.ineqlin.marginals / nl, 4).tolist()}",
        f"- primal-dual xi on constrained categories: {np.round(xi1[b1 > 0], 4).tolist()}",
        "",
        f"## Algorithm 1 on day 6: iterations to reach eps = {args.eps}: {iters_to_eps}",
    ]
    (out / "checks.md").write_text("\n".join(checks) + "\n")

    summary = dict(args=vars(args), auc=dict(cvr=auc_cvr, wpo=auc_wpo), div_lambda=best_lam,
                   aggregate=agg, daily=daily, greedy_ratio_min=float(min(ratios)),
                   lp=dict(lp=lp_val, pd=float(pd_val), pd_miss=pd_miss),
                   frac_nonneg_div_weights=float((Dw >= 0).mean()),
                   runtime_sec=time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for f in ["table1.md", "table2.md", "table3.md", "checks.md"]:
        print("\n" + (out / f).read_text())
    print(f"[done] {time.time() - t0:.1f}s, results in {out}")


if __name__ == "__main__":
    main()
