import matplotlib.pyplot as plt
import numpy as np


def load_data():
    PATH = "s5/"
    S = 5
    A = 3
    H = 10
    return {
        "ssr": np.loadtxt(PATH + 'ssr_s{}.txt'.format(S), unpack=True),
        "ots": np.loadtxt(PATH + 'ots_s{}.txt'.format(S), unpack=True),
        "ots_plus": np.loadtxt(PATH + 'ots_plus_s{}.txt'.format(S), unpack=True),
        "ots_n": np.loadtxt(PATH + 'ots_nonclip_s{}.txt'.format(S), unpack=True),
        "ucbvi": np.loadtxt(PATH + 'ucbvi_s{}.txt'.format(S), unpack=True),
    }


def make_figure(fig, data=None):
    PATH = "s5/"
    S = 5
    A = 3
    H = 10
    ssr = data["ssr"]
    ots = data["ots"]
    ots_plus = data["ots_plus"]
    ots_n = data["ots_n"]
    ucbvi = data["ucbvi"]


    def cum_me(r):
        return np.cumsum(r) / range(1, len(r) + 1)
        
    # Set the width of the plot to be the width of the column
    # I think this should be about right for a two-column thing

    COLUMN_WIDTH_INCHES = 3.25
    HEIGHT = 2.5
    fig.set_size_inches(COLUMN_WIDTH_INCHES, HEIGHT)

    # Set the font sizes

    SIZE_S = 9
    SIZE_M = 10
    SIZE_L = 12

    plt.rc("font", size=SIZE_M)  # default text sizes
    plt.rc("figure", titlesize=SIZE_L)  # figure title
    plt.rc("axes", titlesize=SIZE_M)  # axes title
    plt.rc("legend", fontsize=SIZE_M)  # legend
    plt.rc("axes", labelsize=SIZE_M)  # x and y labels
    plt.rc("xtick", labelsize=SIZE_S)  # tick labels
    plt.rc("ytick", labelsize=SIZE_S)  #

    # Make the axis (or use the pyplot way of doing them)

    ax = fig.add_subplot(111)

    # Increase the line width from the default

    linewidth = 2


    # Have a different line style for each algorithm

    ax.plot(cum_me(ssr), linestyle="-", linewidth=linewidth, label="SSR-Bernstein")
    ax.plot(cum_me(ots_n), linestyle="--", linewidth=linewidth, label="TS-MDP")
    ax.plot(cum_me(ots), linestyle="-.", linewidth=linewidth, label="O-TS-MDP")
    ax.plot(cum_me(ots_plus), linestyle="dotted", linewidth=linewidth, label="O-TS-MDP$^+$")
    ax.plot(cum_me(ucbvi), linestyle=(0, (5, 10)), linewidth=linewidth, label="UCB-VI")
    ax.set_xlim(xmin=1000)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("S = {}, A = {}, H = {}".format(S, A, H))
    ax.legend(framealpha=1.0, ncol=2, loc="lower right", fontsize = 8, columnspacing = 1.0, labelspacing = 0.25, frameon=False)

    # Make the figure tight with the bounding box
    fig.set_dpi(300)
    fig.tight_layout(pad=0.2)


if __name__ == "__main__":
    data = load_data()
    make_figure(plt.figure(), data)
    plt.show()
    plt.savefig("s5.pdf", format="pdf")
