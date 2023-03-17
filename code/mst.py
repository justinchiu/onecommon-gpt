import numpy as np

def all_nearest(idxs, all_dots):
    import pdb; pdb.set_trace()



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

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
    #xy = np.random.rand((7,2)) * 2 - 1
    xy = ctx[:,:2]

    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(
        xy[:,0], xy[:,1],
        marker='o',
        s = 100 * (ctx[:,2] + 1),
        c = -ctx[:,3],
        cmap="binary",
        edgecolor="black",
        linewidth=1,
    )
    plan = np.array([1,0,0,1,0,0,0], dtype=bool)
    ax.scatter(xy[plan,0], xy[plan,1], marker="x", s=100, c="r")
    for i, id in enumerate(real_ids):
        ax.annotate(id, (xy[i,0]+.025, xy[i,1]+.025))
    plt.show()
    """
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, width=300)
    """

    print(all_nearest(plan, ctx))
