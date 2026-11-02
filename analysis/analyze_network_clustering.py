# %%

import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import geopandas as gpd
import json
import pypsa

# %%
# Set parameters

SIZE = 8

# %% Setup paths

base_path = "/".join(os.getcwd().split("\\")[:-1])
network_path = base_path + "/ext_networks/"
resource_path = base_path + "/ext_resources/"
result_path = base_path + "/ext_results/"

# %% Import colors

color_map = json.load(open(f"{base_path}/analysis/color_map.json"))

# %% Load networks and vornois

clusters = 50

networks = {}

titel_dict = {
    "Initial": "Initial power network",
    "Clustered": "Power network after clustering",
}

run_name = "Run_2040_V08_comp"

fp_pre = f"{network_path}{run_name}/base.nc"
networks["Initial"] = pypsa.Network(fp_pre)
voronois_pre = gpd.read_file(
    rf"{resource_path}/{run_name}/bus_regions/regions_onshore.geojson"
).set_index("name")

run_name = "Run_2040_V08_run"

fp_c = f"{network_path}{run_name}/elec_s_{clusters}.nc"
networks["Clustered"] = pypsa.Network(fp_c)
voronois_c = gpd.read_file(
    rf"{resource_path}/{run_name}/bus_regions/regions_onshore_elec_s_50.geojson"
).set_index("name")

voronois = {
    "Initial": voronois_pre,
    "Clustered": voronois_c,
    "Excluded": gpd.overlay(
        voronois_pre,
        voronois_c,
        how="difference",
    ).dissolve(),
}

map_fp = r"..\data\country_shapes.geojson"
shapes = gpd.read_file(map_fp).set_index("name")

# %% Plot setup

figsize = (6.4, 3.5)
plot_boundaries = [-10, 13, 18, 38]

# different colors than usual for better visibility
bus_color = color_map["Hydrogen Pipeline"]
line_color = color_map["Network Node"]
voronois_fill = color_map["Background"]
voronois_edge = "black"
voronoi_width = 0.25
excluder_fill = color_map["Background"]
excluder_hatch = "///"
excluder_hatch_linewidth = 0.25

# % Create plot

# setup plot
fig, axs = plt.subplot_mosaic(
    [["Initial", "Clustered"]],
    figsize=figsize,
    layout="tight",
)

# configure axes
for plot_type in titel_dict:
    axs[plot_type].set_xlim([plot_boundaries[0], plot_boundaries[1]])  # -10,13
    axs[plot_type].set_ylim([plot_boundaries[2], plot_boundaries[3]])  # 18,
    axs[plot_type].set_aspect("equal")
    axs[plot_type].tick_params(axis="both", which="major", labelsize=SIZE)

# plot map
for plot_type in titel_dict:
    shapes.loc[["DZ"]].boundary.plot(
        ax=axs[plot_type],
        edgecolor="black",
        linewidth=voronoi_width,
    )
    voronois[plot_type].plot(
        ax=axs[plot_type],
        color=voronois_fill,
        edgecolor=voronois_edge,
        linewidth=voronoi_width,
    )
    
    if plot_type == "Clustered":
        voronois["Excluded"].plot(
            ax=axs[plot_type],
            color=excluder_fill,
            edgecolor=voronois_edge,
            linewidth=voronoi_width,
            hatch=excluder_hatch,
            hatch_linewidth=excluder_hatch_linewidth,
        )

    # plot existing network
    base_buses = networks[plot_type].buses.loc[
        networks[plot_type].buses["carrier"] == "AC"
    ]
    axs[plot_type].plot(
        base_buses["x"],
        base_buses["y"],
        marker="o",
        markeredgecolor="black",
        markeredgewidth=0.25,
        # markeredgewidth=0,
        markersize=2.5,  # 4,
        color=bus_color,
        linestyle="None",
        zorder=1.5,
    )

    # plot base lines
    lines = networks[plot_type].lines.apply(
        lambda line: [
            (networks[plot_type].buses.loc[line.bus0, ["x", "y"]].values),
            (networks[plot_type].buses.loc[line.bus1, ["x", "y"]].values),
        ],
        axis=1,
    )

    line_collections = LineCollection(
        lines, linewidths=1, color=line_color, zorder=1  # 2.5,
    )

    axs[plot_type].add_collection(line_collections)

    # add axis labels
    xlabel = "Longitude in degree"
    ylabel = "Latitude in degree"
    if plot_type != "Clustered":
        axs[plot_type].set_ylabel(ylabel, fontsize=SIZE)
    axs[plot_type].set_xlabel(xlabel, fontsize=SIZE)

    # add title
    axs[plot_type].set_title(titel_dict[plot_type], fontsize=SIZE)

    # set tick params
    axs[plot_type].tick_params(axis="both", which="major", labelsize=SIZE)
    axs[plot_type].yaxis.set_major_locator(MultipleLocator(4))

# move plots closer together
axs[plot_type].yaxis.set_ticks([])
axs[plot_type].yaxis.set_ticklabels([])
fig.tight_layout()
fig.subplots_adjust(wspace=0.025)

# legend
legend_elements = [
    Line2D(
        [0],
        [0],
        color=line_color,
        lw=2,#3
        label="Power line",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color=bus_color,
        label="Substation",
        markerfacecolor=bus_color,
        markeredgecolor="black",
        markeredgewidth=0.5,
        markersize=4,
        linestyle="none",
    ),
    Patch(
        #fill=False,
        edgecolor=voronois_edge,
        facecolor=voronois_fill,
        lw=voronoi_width,
        label="Voronoi cell",
    ),
    Patch(
        edgecolor=voronois_edge,
        facecolor=excluder_fill,
        hatch="///",
        lw=voronoi_width,
        label="Excluded area",
    ),
]

# rc_context change required for the hatch in the patch
with plt.rc_context({"hatch.linewidth": excluder_hatch_linewidth}):
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        fontsize=SIZE,
    )


# %% Save network clustering plot

tracker = "V3.0"
file_type = "svg"

dir_name = f"..\\images\\{run_name}\\"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

fp = f"{dir_name}\\network_clustering_{tracker}.{file_type}"
fig.savefig(fp, bbox_inches="tight")

# %%
