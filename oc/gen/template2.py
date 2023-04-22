from jinja2 import Template
import numpy as np

from template_rec import render, render_select


def main():
    import numpy as np
    import jax
    from jax import random
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from matplotlib import cm

    import streamlit as st
    from io import BytesIO

    from cog_belief import CostBelief

    def st_write_history(history, ids):
        st.write("Plan history")
        if len(history) == 0:
            st.write("None")
            return
        for i, plan in enumerate(history):
            st.write(i, str(list(ids[plan.astype(bool)])))

    num_dots = 7

    # context init
    ctx = np.array([
        0.635, -0.4,   2/3, -1/6,  # 8
        0.395, -0.7,   0.0,  3/4,  # 11
        -0.74,  0.09,  2/3, -2/3,  # 13
        -0.24, -0.63, -1/3, -1/6,  # 15
        0.15,  -0.58,  0.0,  0.24, # 40
        -0.295, 0.685, 0.0, -8/9,  # 50
        0.035, -0.79, -2/3,  0.56, # 77
    ], dtype=float).reshape(-1, 4)
    real_ids = np.array(['8', '11', '13', '15', '40', '50', '77'], dtype=int)
    # reflect across y axis
    ctx[:,1] = -ctx[:,1]

    # belief init
    belief = CostBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5, use_temporal=False)
    sc = belief.sc
    xy = belief.xy

    # settings for the original template tree
    B = 3
    inner_B = 2
    absolute_bucket = True

    N = 5

    belief_type = "cost-noTemporal"
    response_strategy = "all_yes"

    prior = belief.prior
    for n in range(N):
        st.write(f"plan {n}")
        EdHs = belief.compute_EdHs(prior)
        utt = belief.configs[EdHs.argmax()]

        # utterance
        feats = belief.get_feats(utt)
        ids = utt.nonzero()[0]
        words = render(*feats, ids, confirm=None)
        print(words)

        # viz
        uttb = utt.astype(bool)
        fig, ax = plt.subplots(figsize=(4,4))
        ax.scatter(
            xy[:,0], xy[:,1],
            marker='o',
            s=50*(1+sc[:,0]),
            c = -ctx[:,3],
            cmap="binary",
            edgecolor="black",
            linewidth=1,
        )
        ax.scatter(xy[uttb,0], xy[uttb,1], marker="x", s=100, c="r")
        for i, id in enumerate(real_ids):
            ax.annotate(id, (xy[i,0]+.025, xy[i,1]+.025))
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf, width=300)

        st.write("OLD TEMPLATE")
        st_write_history(belief.history, real_ids)
        st.write("plan:", str(list(real_ids[uttb])))
        st.write(words)

        response = None
        if response_strategy == "all_yes":
            response = 1
        elif response_strategy == "all_no":
            response = 0
        elif response_strategy == "alternate":
            response = 1 if n % 2 == 0 else 0
        else:
            raise ValueError

        print("prior", belief.marginals(prior))
        print(response)
        new_prior = belief.posterior(prior, utt, response)
        print("posterior", belief.marginals(new_prior))
        #import pdb; pdb.set_trace()

        belief.history.append(utt)
        prior = new_prior


if __name__ == "__main__":
    main()
