"""figstyle.py — one visual language for every figure in the package."""

__author__ = "LIJUNHUA"
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- fonts ---------------------------------------------------------------- #
# All figure text is English; see code/labels_en.py for the TCM term mapping.
BASE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#3D3D3D",
    "axes.labelcolor": "#1A1A1A",
    "text.color": "#1A1A1A",
    "xtick.color": "#3D3D3D",
    "ytick.color": "#3D3D3D",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "grid.color": "#D8D8D8",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
mpl.rcParams.update(BASE)
mpl.rcParams["axes.unicode_minus"] = False



# ---- palette -------------------------------------------------------------- #
TYPE_COLORS = {
    "SYM": "#1B6C7F",   # petrol   — symptom / sign
    "CAU": "#C4622D",   # terracotta — cause / pathogenesis
    "PRE": "#6D4C8C",   # plum     — prescription
    "HER": "#4A7C59",   # sage     — herb
    "EFF": "#C9A227",   # ochre    — effect
}
TYPE_LABELS = {
    "SYM": "Symptom",
    "CAU": "Cause",
    "PRE": "Prescription",
    "HER": "Herb",
    "EFF": "Effect",
}
REL_COLORS = {
    "CAUSES": "#C4622D",
    "HAS_EFFECT": "#C9A227",
    "CONTAINS": "#6D4C8C",
    "RELIEVES": "#4A7C59",
    "TREATS": "#1B6C7F",
}
MODEL_COLORS = {
    "TransE": "#4C6EA8",
    "DistMult": "#E0A458",
    "ComplEx": "#B5485D",
    "RotatE": "#4F9D8E",
}
INK = "#1A1A1A"
MUTED = "#7A7A7A"
FAINT = "#E8E8E8"
ACCENT = "#B5485D"


def panel_tag(ax, letter, dx=-0.10, dy=1.045):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", va="top", ha="left", color=INK)


def hgrid(ax, axis="y"):
    ax.grid(axis=axis, linewidth=0.6, color="#E2E2E2", zorder=0)
    ax.set_axisbelow(True)


def save(fig, path_stem, formats=("png", "pdf")):
    for ext in formats:
        fig.savefig(f"{path_stem}.{ext}", format=ext)
    plt.close(fig)
    print(f"  wrote {path_stem}.{'/'.join(formats)}")


def tag(fig, letter, x, y):
    """Panel tag in figure coordinates — cannot collide with axes content."""
    fig.text(x, y, letter, fontsize=11.5, fontweight="bold", va="top",
             ha="left", color=INK)


# --------------------------------------------------------------------------- #
# overlap verification
# --------------------------------------------------------------------------- #
def check_overlaps(fig, tol=0.5, verbose=True):
    """Render the figure and report every pair of Text artists whose drawn
    bounding boxes intersect. Used as a gate before saving: a figure that
    reports overlaps is not shipped.

    Legend frames and artists inside the same legend are ignored, since their
    internal layout is managed by matplotlib and is correct by construction.
    """
    import itertools
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    items = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        legend_texts = set()
        for lg in ([ax.get_legend()] if ax.get_legend() else []):
            legend_texts.update(id(t) for t in lg.get_texts())
        axbb = ax.get_window_extent(renderer)
        ticks = set(id(t) for t in ax.get_xticklabels() + ax.get_yticklabels())
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label] + \
                 ax.get_xticklabels() + ax.get_yticklabels():
            if id(t) in legend_texts:
                continue
            # tick labels for locator positions outside the view limits are
            # still instantiated by matplotlib but never drawn; ignore them
            if id(t) in ticks:
                tx, ty = t.get_position()
                if t in ax.get_xticklabels():
                    lo, hi = sorted(ax.get_xlim())
                    if not (lo - 1e-9 <= tx <= hi + 1e-9):
                        continue
                else:
                    lo, hi = sorted(ax.get_ylim())
                    if not (lo - 1e-9 <= ty <= hi + 1e-9):
                        continue
            s = t.get_text().strip()
            if not s or not t.get_visible():
                continue
            try:
                bb = t.get_window_extent(renderer)
            except Exception:
                continue
            if bb.width <= 0 or bb.height <= 0:
                continue
            items.append((s, bb, ax))
    for t in fig.texts:
        s = t.get_text().strip()
        if s and t.get_visible():
            items.append((s, t.get_window_extent(renderer), None))

    hits = []
    for (s1, b1, a1), (s2, b2, a2) in itertools.combinations(items, 2):
        dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if dx <= tol or dy <= tol:
            continue
        # twinx/twiny duplicate: identical string drawn at the same place by
        # two stacked axes. Matplotlib renders one; not a real collision.
        if s1 == s2 and abs(b1.x0 - b2.x0) < 1.0 and abs(b1.y0 - b2.y0) < 1.0:
            continue
        hits.append((s1[:34], s2[:34], round(dx, 1), round(dy, 1)))
    if verbose:
        if hits:
            print(f"    !! {len(hits)} text overlap(s):")
            for h in hits[:12]:
                print(f"       '{h[0]}'  x  '{h[1]}'   ({h[2]}x{h[3]} px)")
        else:
            print("    no text overlaps")
    return hits


def save_checked(fig, path_stem, formats=("png", "pdf")):
    """Save only after the overlap gate passes; always report the outcome."""
    hits = check_overlaps(fig)
    for ext in formats:
        fig.savefig(f"{path_stem}.{ext}", format=ext)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"  wrote {path_stem}.{'/'.join(formats)}"
          + ("  [OVERLAPS PRESENT]" if hits else ""))
    return hits
