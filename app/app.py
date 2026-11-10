"""Carousel homepage demo: see what each policy shows ONE user, side by side.

    pip install streamlit
    streamlit run app/app.py                  # reference solutions
    WPO_IMPL=src streamlit run app/app.py     # your implementation

Pick a user who never rents (true rent_buy affinity very negative) and compare
"Slotting" with "Global constraints": slotting forces a rent row onto their
page; global constraints mostly don't, and pay for it on other users' pages.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMPL = os.environ.get("WPO_IMPL", "solutions")   # WPO_IMPL=src streamlit run app/app.py  -> your code
sys.path.insert(0, str(ROOT / IMPL))

from wpo.world import make_synthetic_world, sample_day  # noqa: E402
from wpo.simulate import random_pages, simulate_clicks, true_page_probs  # noqa: E402
from wpo.features import page_feature_matrix  # noqa: E402
from wpo.model import BayesianLinearModel, PageScorer  # noqa: E402
from wpo.greedy import compose_pages  # noqa: E402
from wpo.primal_dual import solve_duals  # noqa: E402
from wpo.metrics import category_counts  # noqa: E402
from wpo.baselines import slotting_patterns  # noqa: E402

COLORS = {"prime_movie": "#1F6F78", "prime_tv": "#3E8E5E", "rent_buy": "#B5651D", "channel": "#6B4FA0"}
N_USERS, N_SLOTS = 2000, 8


@st.cache_resource(show_spinner="Training the page model on a week of simulated logs")
def setup(impl: str, seed: int):
    world = make_synthetic_world(seed)
    X, Y = [], []
    for d in range(7):
        u = sample_day(world, d, N_USERS, seed=seed)
        pg = random_pages(N_USERS, world.n_widgets, N_SLOTS, seed=d)
        _, y = simulate_clicks(world, u, pg, seed=100 + d)
        X.append(page_feature_matrix(world, u, pg, True))
        Y.append(y.reshape(-1))
    model = BayesianLinearModel().fit(np.vstack(X), np.concatenate(Y))
    return world, model


@st.cache_data(show_spinner="Composing pages and solving the duals")
def compose_day(impl: str, seed: int, day: int, target: float):
    world, model = setup(impl, seed)
    scorer = PageScorer(world, model.mean, True)
    b = np.array([target if c in ("rent_buy", "channel") else 0.0 for c in world.categories])
    prev = sample_day(world, day - 1, N_USERS, seed=seed)
    xi, _, _ = solve_duals(scorer.relevance_matrix(prev), scorer.diversity_matrix(prev),
                           world.widget_cat, N_SLOTS, b, eta0=0.06, eps=0.02)
    u = sample_day(world, day, N_USERS, seed=seed)
    R, D = scorer.relevance_matrix(u), scorer.diversity_matrix(u)
    wc = world.widget_cat
    pages = {
        "Relevance + diversity, no constraints": compose_pages(R, D, wc, N_SLOTS),
        "Global constraints (xi from yesterday)": compose_pages(R, D, wc, N_SLOTS, xi=xi),
        "Slotting": compose_pages(R, D, wc, N_SLOTS, slot_category=slotting_patterns(N_USERS, N_SLOTS, b, seed=day)),
    }
    probs = {k: true_page_probs(world, u, v) for k, v in pages.items()}
    return u.aff, pages, probs, xi, b


def row_html(name: str, cat: str, prob: float, topic, genres) -> str:
    color = COLORS.get(cat, "#555")
    order = np.argsort(-topic)
    tiles = "".join(
        f'<div class="tile" style="border-top:4px solid {color}">{genres[order[j % 3]]}</div>'
        for j in range(7))
    return (f'<div class="row"><div class="rowhead"><span class="chip" style="background:{color}">'
            f'{cat.replace("_", " ")}</span> <b>{name}</b>'
            f'<span class="prob">{100 * prob:.1f}% chance to stream</span></div>'
            f'<div class="strip">{tiles}</div></div>')


st.set_page_config(page_title="Whole-page optimization demo", layout="wide")
st.markdown("""
<style>
.row{margin:0 0 14px 0}
.rowhead{font-size:0.95rem;margin-bottom:6px}
.chip{color:#fff;border-radius:3px;padding:1px 7px;font-size:0.75rem;margin-right:6px}
.prob{float:right;opacity:0.7;font-size:0.8rem}
.strip{display:flex;gap:6px;overflow-x:auto}
.tile{flex:0 0 92px;height:120px;border-radius:4px;background:rgba(127,127,127,0.12);
      display:flex;align-items:flex-end;padding:6px;font-size:0.75rem}
</style>""", unsafe_allow_html=True)

st.sidebar.caption(f"Running the `{IMPL}/` implementation")
seed = st.sidebar.number_input("World seed", 0, 99, 0)
day = st.sidebar.slider("Day (10 and 11 are weekends)", 7, 13, 7)
target = st.sidebar.slider("Min rent/buy and channel rows per page (global average)", 0.0, 3.0, 1.8, 0.1)

try:
    world, _ = setup(IMPL, int(seed))
    aff, pages, probs, xi, b = compose_day(IMPL, int(seed), int(day), float(target))
except NotImplementedError as e:
    st.error(f"Your src/ implementation is incomplete: {e}. Switch to 'solutions' or finish the exercise.")
    st.stop()

k_rent = world.categories.index("rent_buy")
st.title("What does each policy show this user?")
segment = st.sidebar.selectbox("Find a user by rent/buy appetite",
                               ["never rents", "occasional", "loves renting"])
lo, hi = {"never rents": (-9, -1.2), "occasional": (-1.2, 0.3), "loves renting": (0.3, 9)}[segment]
candidates = np.flatnonzero((aff[:, k_rent] > lo) & (aff[:, k_rent] <= hi))
user = int(st.sidebar.selectbox("User", candidates[:50]))

names = list(pages)
left, right = st.columns(2)
for col, default in [(left, 0), (right, 2)]:
    with col:
        pol = st.selectbox("Policy", names, index=default, key=f"pol{default}")
        page = pages[pol][user]
        st.caption(f"Expected streamed rows on this page: {probs[pol][user].sum():.2f}")
        st.markdown("".join(row_html(world.widget_names[c], world.categories[world.widget_cat[c]],
                                     probs[pol][user, i], world.widget_topic[c], world.genres)
                            for i, c in enumerate(page)), unsafe_allow_html=True)

st.subheader("Across all users today")
cols = st.columns(len(names))
for col, pol in zip(cols, names):
    avg = category_counts(pages[pol], world.widget_cat, world.n_categories).mean(0)
    with col:
        st.markdown(f"**{pol}**")
        st.metric("Expected streamed rows per page", f"{probs[pol].sum(1).mean():.3f}")
        for k, c in enumerate(world.categories):
            tgt = f" (target {b[k]:.1f})" if b[k] > 0 else ""
            st.write(f"{c.replace('_', ' ')}: {avg[k]:.2f} rows/page{tgt}")
st.caption("Dual variables xi (learned on yesterday's traffic): " +
           ", ".join(f"{c}={v:.3f}" for c, v in zip(world.categories, xi)))
