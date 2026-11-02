# %%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import geopandas as gpd
import pypsa
import json
from load_expansion_data import load_expansion_data
from _plot_helpers import (
    get_line_collections,
    get_generator_collection,
    get_link_collection,
    get_legends_merged,
)
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker
from collections import defaultdict

# %%
# Set parameters

SIZE = 8

# %% Setup paths

base_path = "/".join(os.getcwd().split("\\")[:-1])
network_path = base_path + "/networks/"
resource_path = base_path + "/resources/"
result_path = base_path + "/ext_results/"

# %% Import colors

color_map = json.load(open(f"{base_path}/analysis/color_map.json"))

# %% Load expansion data for both configurations

run_name = "Run_2040_final"

time_resolution = 1
clusters = 50
mode = "free"
offgrid_status = ["False", "True"]
locations = ["Transmed"]
load = [50]

networks = {}
expansions = defaultdict(dict)
title_dict = {
    "False": f"On-grid {sum(load):,.0f} MW",
    "True": f"Off-grid {sum(load):,.0f} MW",
}

for offgrid in offgrid_status:
    network_name = f"elec_s_{clusters}_{mode}_{offgrid}_{locations}_{load}_ec_lcopt_{time_resolution}H.nc"
    fp = f"{result_path}{run_name}/networks/{network_name}"
    networks[offgrid] = pypsa.Network(fp)

    point_expansion, line_expansion = load_expansion_data(
        networks[offgrid], offgrid=offgrid
    )
    expansions[offgrid]["point_expansion"] = point_expansion
    expansions[offgrid]["line_expansion"] = line_expansion

# %% Setup plot parameters

base_factor = sum(load) / 10 * 2
# water_demand = 0.1668 # (m^3/h)/MW_el
water_demand = 0.1943  # (m^3/h)/MW_el

factor_dict = {
    "Power Line": base_factor,
    "Water Pipeline": base_factor * water_demand,
    "Hydrogen Pipeline": base_factor,
    "Electrolysis": base_factor * 40,
    "Desalination": base_factor * 20 * water_demand,
    "Generator": base_factor * 40,
}

offset_dict = {
    "Power Line": 0,
    "Water Pipeline": -1,
    "Hydrogen Pipeline": 1,
}

min_expansion = 0.5  # MW
min_expansion_dict = {
    "Electrolysis": min_expansion,  # MW
    "Desalination": min_expansion * water_demand,  # (m^3/h)
    "Generator": min_expansion,  # MW
}

figsize = (6.4, 4.7)  # 6.4, 18
height_ratios = [0.6, 0.4]
plot_boundaries = [-4.7, 11.3, 28, 38.3]
component_alpha = 1
line_gap = 2


# %% Create plot

# setup plot
fig, axs = plt.subplot_mosaic(
    [["False", "True"], ["Legend", "Legend"]],
    figsize=figsize,
    layout="tight",
    height_ratios=height_ratios,
)

# configure axes
axs["Legend"].axis("off")
for offgrid in offgrid_status:
    axs[offgrid].set_xlim([plot_boundaries[0], plot_boundaries[1]])  # -10,13
    axs[offgrid].set_ylim([plot_boundaries[2], plot_boundaries[3]])  # 18,
    axs[offgrid].set_aspect("equal")
    axs[offgrid].tick_params(axis="both", which="major", labelsize=SIZE)

# plot map
map_fp = r"..\data\country_shapes.geojson"
shapes = gpd.read_file(map_fp)
for offgrid in offgrid_status:
    shapes.plot(ax=axs[offgrid], edgecolor="black", facecolor=color_map["Background"])
    axs[offgrid].set_facecolor(color_map["Ocean"])

for offgrid in offgrid_status:
    ratio = (plot_boundaries[1] - plot_boundaries[0]) / axs[offgrid].bbox.width
    # plot line expansion
    line_collections = get_line_collections(
        expansions[offgrid]["line_expansion"],
        ratio,
        factor_dict,
        offset_dict,
        line_gap,
    )

    for component in line_collections:
        lines = line_collections[component]
        lines.set_color(color_map[component])
        lines.set_zorder(2)
        axs[offgrid].add_collection(lines)

    # plot generator expansion
    gen_collection = get_generator_collection(
        expansions[offgrid]["point_expansion"],
        factor_dict["Generator"],
        min_expansion_dict["Generator"],
        color_map,
        component_alpha,
        zorder=2.5,
    )
    gen_collection.set_linewidth(0.5)
    axs[offgrid].add_collection(gen_collection)

    # plot link expansion
    link_collection = get_link_collection(
        expansions[offgrid]["point_expansion"],
        factor_dict,
        min_expansion_dict,
        color_map,
        component_alpha,
        zorder=2.5,
    )
    link_collection.set_linewidth(0.5)
    axs[offgrid].add_collection(link_collection)

    # plot existing network
    base_buses = networks[offgrid].buses.loc[networks[offgrid].buses["carrier"] == "AC"]
    axs[offgrid].plot(
        base_buses["x"],
        base_buses["y"],
        marker="o",
        markersize=2.5,
        color=color_map["Network Node"],
        linestyle="None",
        zorder=1.5,
    )
    base_lines = LineCollection(line_collections["Power Line"].get_segments())
    base_lines.set_linewidth(1)
    base_lines.set_color(color_map["Existing Power Line"])
    base_lines.set_zorder(1)
    axs[offgrid].add_collection(base_lines)

    # plot export location
    export_buses = networks[offgrid].buses.loc[
        networks[offgrid].loads.loc[networks[offgrid].loads.carrier == "H2", "bus"]
    ]
    axs[offgrid].plot(
        export_buses["x"],
        export_buses["y"],
        marker="x",
        markersize=5,
        markeredgewidth=2,
        color=color_map["Export Location"],
        zorder=6,
    )

    # add axis labels
    xlabel = "Longitude in degree"
    ylabel = "Latitude in degree"
    if offgrid == "False":
        axs[offgrid].set_ylabel(ylabel, fontsize=SIZE)
    axs[offgrid].set_xlabel(xlabel, fontsize=SIZE)

    # add title
    axs[offgrid].set_title(title_dict[offgrid], fontsize=SIZE)

# move plots closer together
axs[offgrid].yaxis.set_ticks([])
axs[offgrid].yaxis.set_ticklabels([])
fig.tight_layout()
fig.subplots_adjust(wspace=0.025)

# add legend
ratio = (plot_boundaries[1] - plot_boundaries[0]) / axs[offgrid_status[0]].bbox.width
legends = get_legends_merged(
    axs[offgrid_status[0]], sum(load), color_map, factor_dict, ratio
)

for legend_name in legends:
    axs[offgrid_status[0]].add_artist(legends[legend_name])

# %%

plt.show()
tracker = "V1.7"
file_type = "svg"

dir_name = f"..\\images\\{run_name}\\"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

fp = f"{dir_name}\\expansion_merged_{load}_{tracker}.{file_type}"
fig.savefig(fp, bbox_inches="tight")

# %%
