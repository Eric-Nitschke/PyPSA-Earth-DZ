# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cmcrameri import cm

# Generate n equally spaced colors from romaO
n_colors = 10
colors = cm.romaO(np.linspace(0, 1, n_colors + 1))

# Create the figure
fig, ax = plt.subplots(figsize=(10, 2))

# Draw color swatches with hex codes
for i, c in enumerate(colors):
    hex_color = mcolors.to_hex(c)
    # Draw color rectangle
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c, transform=ax.transData))
    # Add hex label
    ax.text(
        i + 0.5, -0.25, hex_color, ha="center", va="top", fontsize=9, family="monospace"
    )

# Formatting
ax.set_xlim(0, n_colors + 1)
ax.set_ylim(-0.5, 1)
ax.axis("off")
plt.title(f"romaO – {n_colors} Equally Spaced Colors", fontsize=12)
plt.tight_layout()
plt.show()


# %%
