# %% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import json
from load_cost_data import load_cost_data

# %%
# Set parameters

SIZE = 8

# %% Setup paths

base_path = "/".join(os.getcwd().split("\\")[:-1])
network_path = base_path + "/networks/"
resource_path = base_path + "/resources/"
result_path = base_path + "/ext_results/"
# override_path = base_path + "/data/override_component_attrs"

# %% Import colors

color_map = json.load(open(f"{base_path}/analysis/color_map.json"))

# %% Load cost data multiple configurations

run_name = "Run_2040_final"

time_resolution = 1
clusters = 50
mode = "free"
offgrids = [False, True]
locations = ["Transmed"]
loads = [[50], [500], [5000]]

networks = pd.DataFrame()

load_labels = [f"{sum(load):,.0f} MW" for load in loads]
offgrid_labels = {
    False: "On-grid",
    True: "Off-grid",
}
categories = ["CAPEX", "OPEX", "Total", "LCOH"]
costs_cols = pd.MultiIndex.from_product(
    [categories, offgrid_labels.values(), load_labels]
)
costs = pd.DataFrame(columns=costs_cols)

for offgrid in offgrids:
    offgrid_label = offgrid_labels[offgrid]
    for load in loads:
        load_str = f"{sum(load):,.0f} MW"
        network_name = f"elec_s_{clusters}_{mode}_{offgrid}_{locations}_{load}_ec_lcopt_{time_resolution}H.nc"
        fp = f"{result_path}{run_name}/networks/{network_name}"

        n = pypsa.Network(fp)
        networks[load_str] = n

        cost_data = load_cost_data(n)
        for category in categories:
            costs[category, offgrid_label, load_str] = cost_data[category]

# %% Plot costs multiple configurations

figsize = (6.4, 4)

fig, ax = plt.subplots(figsize=figsize)
order = [
    "Onshore Wind",
    "Photovoltaic",
    "Power Line",
    "Electrolysis",
    "Desalination",
    "Hydrogen Pipeline",
    "Water Pipeline",
    "Battery Storage",
    "Hydrogen Storage",
]

costs = costs.reindex(order)

costs["LCOH"].T.plot(ax=ax, kind="bar", stacked=True, color=color_map, zorder=10)

# setup the first level xticks
ax.tick_params(axis="both", which="major", labelsize=SIZE)
ax.tick_params(axis="x", rotation=0)
ax.set_xticklabels(costs["LCOH"].columns.get_level_values(1))
ax.set_ylabel(ax.get_ylabel(), fontsize=SIZE)
ax.set_ylim(bottom=0, top=4.0)

ax.grid(axis="y", zorder=0)
ax.set_ylabel("Hydrogen cost at the export location in €/$\mathrm{kg_{H2}}$")

# leave space on the right for the legend
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])

# invert legends to be in the same order as the stacked bars
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels[::-1],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=SIZE,
)

# setup the second level xticks
sec = ax.secondary_xaxis(location=0)
sec_ticks = [1, 4]
sec_tick_labels = [
    f"\n\n{label}" for label in costs["LCOH"].columns.get_level_values(0)[sec_ticks]
]
sec.set_xticks(sec_ticks, labels=sec_tick_labels)
sec.tick_params(axis="x", length=0, labelsize=SIZE)

# add the overall LCOH as a label for each bar
for i, run_i in enumerate(costs["LCOH"]):
    ax.annotate(
        f'{costs["LCOH"][run_i].sum():.2f} €/kg',
        xy=(i, 3.70),
        ha="center",
        fontsize=SIZE,
    )

# minor plot setup
linewidth = ax.spines["bottom"].get_linewidth()
line_spacing = 1.2 # guess
tick_lenth = ax.xaxis.get_major_ticks()[0].tick1line.get_markersize()

# line between the configurations
sec2 = ax.secondary_xaxis(location=0)
sec2.set_xticks([-0.5, 2.5, 5.5], labels=[])
sec2.tick_params("x", length=SIZE * 3 * line_spacing + tick_lenth, width=linewidth)
ax.axvline(x=2.5, color="black", linestyle="-", linewidth=linewidth)

# %%

plt.show()

tracker = "V1.1"
fp = f"..\\images\\{run_name}\\costs_grouped_{tracker}.svg"

if not os.path.exists(f"..\\images\\{run_name}"):
    os.makedirs(f"..\\images\\{run_name}")

fig.savefig(fp, bbox_inches="tight")
# %%
