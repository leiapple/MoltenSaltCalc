from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================

N_ATOMS = 15
BOX = 1.0
SEED = 41

np.random.seed(SEED)

DT = 0.02

INIT_STATES = [
    "init",
    "add_v",
]

STATES = [
    "force",
    "acc",
    "update_v",
    "update_x",
]

NUM_FRAMES_PER_STATE = 15

# =============================================================================
# INITIAL STATE
# =============================================================================

pos = np.random.rand(N_ATOMS, 2) * BOX
vel = np.zeros_like(pos)
vel_ini = np.random.rand(N_ATOMS, 2) * 0.25
forces = np.zeros_like(pos)
acc = np.zeros_like(pos)

# =============================================================================
# TOY FORCE MODEL
# =============================================================================


def compute_forces(pos):
    center = np.array([0.1, 0.5])
    rnd_offset = np.random.rand(2) * 0.2
    return -(pos - center) + rnd_offset


# =============================================================================
# FIGURE
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, BOX)
ax.set_ylim(0, BOX)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# atoms
scat = ax.scatter(pos[:, 0], pos[:, 1], s=150, label="Atoms")

# IMPORTANT FIX:
# initialize quivers with REAL data (not empty arrays)
vel_quiver = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color="blue", scale=5, label="Velocities")

force_quiver = ax.quiver(pos[:, 0], pos[:, 1], forces[:, 0], forces[:, 1], color="red", scale=5, label="Forces")

ax.legend(loc="upper right")

# =============================================================================
# MD STEP STATE MACHINE
# =============================================================================

state_idx = 0

# =============================================================================
# UPDATE LOOP
# =============================================================================


def update(frame):
    global pos, vel, forces, acc, state_idx

    if frame < NUM_FRAMES_PER_STATE * len(INIT_STATES):
        state = INIT_STATES[state_idx]
    elif frame == NUM_FRAMES_PER_STATE * len(INIT_STATES):
        state_idx = 0  # reset to first state after initialization
        state = STATES[state_idx]
    else:
        state = STATES[state_idx]

    if state == "init":
        ax.set_title("Initial configuration", weight="bold", fontsize=14)
        vel[:] = 0
        forces[:] = 0
        acc[:] = 0

    elif state == "add_v":
        ax.set_title("Assign initial velocities", weight="bold", fontsize=14)
        vel[:] = vel_ini

    elif state == "force":
        ax.set_title("Compute forces", weight="bold", fontsize=14)
        forces[:] = compute_forces(pos)

    elif state == "acc":
        ax.set_title("Compute acceleration", weight="bold", fontsize=14)
        acc[:] = forces

    elif state == "update_v":
        ax.set_title("Update velocities", weight="bold", fontsize=14)
        vel[:] += acc * DT

    elif state == "update_x":
        ax.set_title("Update positions", weight="bold", fontsize=14)
        pos[:] += vel * DT
        pos[:] = pos % BOX

    # cycle states
    if frame % NUM_FRAMES_PER_STATE == NUM_FRAMES_PER_STATE - 1:
        state_idx = (state_idx + 1) % len(STATES)

    scat.set_offsets(pos)
    # Remove previous quivers

    vel_quiver.set_offsets(pos)
    vel_quiver.set_UVC(vel[:, 0], vel[:, 1])

    force_quiver.set_offsets(pos)
    force_quiver.set_UVC(forces[:, 0], forces[:, 1])

    return scat, vel_quiver, force_quiver


# =============================================================================
# ANIMATION
# =============================================================================

ani = FuncAnimation(fig, update, frames=tqdm(range(300)), interval=200, blit=True)

# =============================================================================
# SAVE VIDEO
# =============================================================================

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
writer = FFMpegWriter(
    fps=10,
    codec="libx264",
    bitrate=3000,
    extra_args=["-pix_fmt", "yuv420p", "-profile:v", "baseline"],
)

ani.save(
    Path(__file__).parent.parent / "imgs" / "md_workflow.mp4",
    writer=writer,
)
