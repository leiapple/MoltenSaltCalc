from pathlib import Path

import myplots as p
from matplotlib.patches import FancyBboxPatch

# =============================================================================
# CONFIGURATION
# =============================================================================

FONT_SIZE = 9

p.use_style(doc_fontsize=FONT_SIZE)

# Figure
FIGSIZE = (8, 6)

# Box geometry
BOX_WIDTH = 0.30
BOX_HEIGHT = 0.08

# Main column position
COLUMN_X = 0.35

# Vertical positions
Y_LOAD_MLIP = 0.88
Y_ROCKSALT = 0.72
Y_REMOVE = 0.56
Y_INITIALIZE = 0.40
Y_MD = 0.24

# Branch boxes
NPT_X = 0.08
NPT_Y = 0.05

NVT_X = 0.62
NVT_Y = 0.05

NPT_WIDTH = 0.26
NVT_WIDTH = 0.26

# Marker configuration
MARKER_OFFSET = -0.375
MARKER_SIZE = 225

# Colors
BOX_FACE = "#f7f7fb"
BOX_EDGE = "#2b2b2b"

MD_FACE = "#eef2ff"

ARROW_COLOR = "#333333"

MARKER_FACE = "#dbe7ff"
MARKER_EDGE = "#444444"

# Line widths
BOX_LINEWIDTH = 1.2
ARROW_LINEWIDTH = 1.2

# Group annotations
GROUP_X = 0.02

# =============================================================================
# HELPERS
# =============================================================================


def add_marker(ax, x, y, marker):
    ax.scatter(
        x,
        y,
        marker=marker,
        s=MARKER_SIZE,
        facecolor=MARKER_FACE,
        edgecolor=MARKER_EDGE,
        linewidth=1.0,
        zorder=10,
    )


def add_box(
    ax,
    x,
    y,
    text,
    icon=None,
    width=BOX_WIDTH,
    height=BOX_HEIGHT,
    facecolor=BOX_FACE,
    edgecolor=BOX_EDGE,
):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=BOX_LINEWIDTH,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
    )

    if icon is not None:
        add_marker(
            ax,
            x - MARKER_OFFSET if height == BOX_HEIGHT else x - MARKER_OFFSET / 1.15,
            y + height / 2,
            icon,
        )

    return {
        "left": x,
        "right": x + width,
        "bottom": y,
        "top": y + height,
        "center_x": x + width / 2,
        "center_y": y + height / 2,
    }


def add_arrow(ax, start, end):
    # Correct the y-coordinate for overlapping arrows
    if start[1] > end[1]:
        end = (end[0], end[1] + 0.01)
        start = (start[0], start[1] - 0.02)

    if start[0] != end[0]:
        end = (end[0], end[1] + 0.02)
        start = (start[0], start[1] - 0.01)

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "lw": ARROW_LINEWIDTH,
            "color": ARROW_COLOR,
        },
    )


# =============================================================================
# FIGURE
# =============================================================================

fig, ax = p.new()

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

load_box = add_box(
    ax,
    COLUMN_X,
    Y_LOAD_MLIP,
    "Load MLIP calculator\n(lazy initialization)",
    icon="D",  # diamond = MLIP
)

rocksalt_box = add_box(
    ax,
    COLUMN_X,
    Y_ROCKSALT,
    "Build rocksalt structure\n(prevents like-charge clustering)",
    icon="s",  # square = lattice
)

remove_box = add_box(
    ax,
    COLUMN_X,
    Y_REMOVE,
    "Random atom removal\n(target composition)",
    icon="X",  # vacancy/removal
)

init_box = add_box(
    ax,
    COLUMN_X,
    Y_INITIALIZE,
    "Maxwell–Boltzmann initialization\n(zero COM momentum)",
    icon="o",  # temperature
)

# =============================================================================
# MD BOX
# =============================================================================

md_box = add_box(
    ax,
    COLUMN_X,
    Y_MD,
    "MD Simulation\n(NPT → NVT)",
    icon="^",  # simulation
    facecolor=MD_FACE,
)

# =============================================================================
# PROPERTY BOXES
# =============================================================================

npt_box = add_box(
    ax,
    NPT_X,
    NPT_Y,
    "NPT equilibration\n→ density\n→ thermal expansion",
    icon="P",
    width=NPT_WIDTH,
    height=BOX_HEIGHT * 1.1,
)

nvt_box = add_box(
    ax,
    NVT_X,
    NVT_Y,
    "NVT production\n→ diffusion\n→ viscosity\n→ heat capacity",
    icon="*",
    width=NVT_WIDTH,
    height=BOX_HEIGHT * 1.1,
)

# =============================================================================
# ARROWS
# =============================================================================

add_arrow(
    ax,
    (load_box["center_x"], load_box["bottom"]),
    (rocksalt_box["center_x"], rocksalt_box["top"]),
)

add_arrow(
    ax,
    (rocksalt_box["center_x"], rocksalt_box["bottom"]),
    (remove_box["center_x"], remove_box["top"]),
)

add_arrow(
    ax,
    (remove_box["center_x"], remove_box["bottom"]),
    (init_box["center_x"], init_box["top"]),
)

add_arrow(
    ax,
    (init_box["center_x"], init_box["bottom"]),
    (md_box["center_x"], md_box["top"]),
)

branch_start = (
    md_box["center_x"],
    md_box["bottom"],
)

npt_target = (
    npt_box["center_x"],
    npt_box["top"],
)

nvt_target = (
    nvt_box["center_x"],
    nvt_box["top"],
)

add_arrow(ax, branch_start, npt_target)
add_arrow(ax, branch_start, nvt_target)

# =============================================================================
# GROUP LABELS
# =============================================================================

ax.text(
    GROUP_X,
    0.92,
    "System setup",
    weight="bold",
    fontsize=FONT_SIZE - 2,
)

ax.text(
    GROUP_X,
    0.86,
    "Construct initial molten salt\nconfiguration",
    color="#444444",
)

ax.text(
    GROUP_X,
    0.46,
    "Molecular dynamics",
    weight="bold",
    fontsize=FONT_SIZE - 2,
)

ax.text(
    GROUP_X,
    0.40,
    "Volume equilibration and\nproduction MD",
    color="#444444",
)

ax.text(
    GROUP_X,
    0.22,
    "Property analysis",
    weight="bold",
    fontsize=FONT_SIZE - 2,
)

ax.text(
    GROUP_X,
    0.18,
    "Extract observables",
    color="#444444",
)

p.save(fig, Path(__file__).parent.parent / "imgs" / "workflow_diagram.png")
