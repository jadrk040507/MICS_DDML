"""
Distribution of under-five children per household in the U5 (child) sample.

Supporting figure for the Methods/Interpretation discussion of clustering.  The
U5 analysis clusters standard errors on HHID, but the cluster-size distribution
is dominated by singletons: a large majority of households contribute exactly one
under-five child, so each HHID is its own cluster and clustered vs.\ i.i.d.\
standard errors essentially coincide.  This justifies reusing i.i.d.\ SEs for
APOS (which doubleml 0.11.x does not cluster natively) on the U5 sample.

The figure plots the number of U5 rows per HHID (the cluster size) with the mean
and the singleton share marked.  Saved to the ROOT Figures/ directory.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _config import FIGURE_DIR, logger
from _runners import setup_environment, load_data
from _style import use_paper_style, despine, PRISM_BLUE


def main():
    setup_environment()
    _hh, u5_dt = load_data()
    if "HHID" not in u5_dt.columns:
        logger.error("HHID not in U5 data; cannot build cluster-size figure")
        return

    use_paper_style()
    sizes = u5_dt.groupby("HHID").size().to_numpy()
    mean = float(sizes.mean())
    share_singleton = float(np.mean(sizes == 1))
    n_clusters = len(sizes)
    n_rows = int(sizes.sum())
    top = int(min(sizes.max(), 6))

    clipped = sizes.clip(max=top)
    cats = np.arange(1, top + 1)
    counts = np.array([(clipped == c).sum() for c in cats], dtype=float)
    shares = 100 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(cats, counts, width=0.78, color=PRISM_BLUE, alpha=0.95,
                  edgecolor="white", linewidth=0.6, zorder=3)
    # percentage labels above each bar
    for c, cnt, sh in zip(cats, counts, shares):
        ax.text(c, cnt + counts.max() * 0.015, f"{sh:.0f}%",
                ha="center", va="bottom", fontsize=9, color="#222222")

    ax.set_xlabel("Under-five children per household (HHID cluster size)")
    ax.set_ylabel("Households (HHID clusters)")
    ax.set_xticks(cats)
    ax.set_xticklabels([str(i) for i in range(1, top)] + [f"{top}+"])
    ax.set_ylim(0, counts.max() * 1.12)
    # Title omitted: the LaTeX float \caption supplies it.
    despine(ax)
    fig.tight_layout()

    png = FIGURE_DIR / "children_distribution_U5.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"U5 cluster-size figure -> {png}  (mean={mean:.3f}, "
                f"singletons={share_singleton:.3f})")
    print(f"SAVED {png} mean={mean:.3f} singletons={share_singleton:.3f} clusters={n_clusters}")


if __name__ == "__main__":
    main()
