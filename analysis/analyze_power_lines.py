# %%

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import pandas as pd
import geopandas as gpd
import json
import pypsa
from cmcrameri import cm

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

# %% Load loads

run_name = "Run_2040_V08_run"

time_resolution = 1
clusters = 50
mode = "free"
offgrid = "False"
locations = ["Transmed"]
load = [5000]

network_name = f"elec_s_{clusters}_{mode}_{offgrid}_{locations}_{load}_ec_lcopt_{time_resolution}H.nc"
fp = f"{result_path}{run_name}/networks/{network_name}"
n = pypsa.Network(fp)

elec_buses = n.buses.loc[n.buses.carrier == "AC"].index
gen_capacity = (
    n.generators.loc[
        (n.generators["bus"].isin(elec_buses)) & (n.generators.carrier != "load"),
        ["bus", "p_nom"],
    ]
    .groupby("bus")
    .sum()
)

# %% Load Voronois and country shape

voronois = gpd.read_file(r"regions_onshore_elec_s_50.geojson").set_index("name")
voronois["gen_capacity"] = gen_capacity

map_fp = r"..\data\country_shapes.geojson"
shapes = gpd.read_file(map_fp)

# %% Plot setup

figsize = (6.4, 4.0)  # 6.4, 18
plot_boundaries = [-4.7, 11.3, 28, 38.3]

## %% Plot peak loads

fig, ax = plt.subplots(figsize=figsize, layout="tight")

# plot map
shapes.plot(ax=ax, edgecolor="black", facecolor=color_map["Background"])
ax.set_facecolor(color_map["Ocean"])

voronois_ax = voronois.plot(
    ax=ax,
    column="gen_capacity",
    cmap=cm.davos_r,
    vmin=0,
    legend=True,
)

voronois.boundary.plot(
    ax=ax,
    color=color_map["Existing Power Line"],
    linewidth=0.5,
)

cbar = voronois_ax.get_figure().axes[-1]
cbar.yaxis.labelpad = 8
cbar.tick_params(labelsize=SIZE)
cbar.set_ylabel("Generator capacity in MW", fontsize=SIZE)

# redraw borders to fix overlap
shapes.boundary.plot(
    ax=ax,
    edgecolor="black",
    linewidth=0.5,
)

# plot power lines
ax.add_collection(
    LineCollection(
        n.lines.apply(
            lambda row: [
                (n.buses.loc[row.bus0, "x"], n.buses.loc[row.bus0, "y"]),
                (n.buses.loc[row.bus1, "x"], n.buses.loc[row.bus1, "y"]),
            ],
            axis=1,
        ),
        color="dimgray",
        linewidths=1,
    )
)

# plot affected power lines
ax.add_collection(
    LineCollection(
        n.lines.loc[(n.lines.bus0 == "DZ0 33") | (n.lines.bus1 == "DZ0 33")].apply(
            lambda row: [
                (n.buses.loc[row.bus0, "x"], n.buses.loc[row.bus0, "y"]),
                (n.buses.loc[row.bus1, "x"], n.buses.loc[row.bus1, "y"]),
            ],
            axis=1,
        ),
        color=color_map["Electrolysis"],
        linewidths=3,
    )
)

# add legend
legend_line = Line2D(
    [0],
    [0],
    color=color_map["Electrolysis"],
    linewidth=3,
    label="Expanded power lines",
)
ax.legend(
    handles=[legend_line],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.225),
    fontsize=SIZE,
)

# add axis labels
xlabel = "Longitude in degree"
ylabel = "Latitude in degree"
ax.set_ylabel(ylabel, fontsize=SIZE)
ax.set_xlabel(xlabel, fontsize=SIZE)


ax.set_xlim([plot_boundaries[0], plot_boundaries[1]])  # -10,13
ax.set_ylim([plot_boundaries[2], plot_boundaries[3]])  # 18,

ax.tick_params(axis="both", which="major", labelsize=SIZE)

# %% Save power line plot

tracker = "V1.0"
file_type = "svg"

dir_name = f"..\\images\\{run_name}\\"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

fp = f"{dir_name}\\limited_power_lines_{tracker}.{file_type}"
fig.savefig(fp, bbox_inches="tight")


#  %% Compare power line expansion through the iterations

its = {
    i: pd.read_csv(rf"../ext_data/line_extensions_V08_it{i}.csv", index_col="Line")
    for i in range(1, 6)
}
its[0] = pd.read_csv(r"../ext_data/original_lines.csv", index_col="Line").rename(
    columns={"s_nom": "s_nom_opt"}
)

# print line expansion
# Line 76: North-South, Line 77: East-West
for i in range(1, 6):
    print(f"it{i}:")

    dif = (its[i] - its[i - 1])["s_nom_opt"]
    
    print("Power Limit")
    print(its[i].loc[[76, 77], "s_nom_opt"])

    print("Change")
    print(dif.loc[[76, 77]])


# %%
