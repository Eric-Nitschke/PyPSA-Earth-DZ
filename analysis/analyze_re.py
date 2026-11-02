# %%

import os
import atlite
import pypsa
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from cmcrameri import cm
import rasterio as rio
import xarray as xr

from shapely.geometry import box
from matplotlib.patches import Patch
from rasterio.warp import reproject, Resampling

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

# %% Load network

run_name = "Run_2040_final"

time_resolution = 1
clusters = 50
mode = "free"
offgrid = False
locations = ["Transmed"]
load = [5000]

network_name = f"elec_s_{clusters}_{mode}_{offgrid}_{locations}_{load}_ec_lcopt_{time_resolution}H.nc"
fp = f"{result_path}{run_name}/networks/{network_name}"

n = pypsa.Network(fp)

# %% Get full load hours

voronois = gpd.read_file(r"regions_onshore_elec_s_50.geojson").set_index("name")
wind_flh: pd.DataFrame = n.generators_t.p_max_pu.loc[
    :, n.generators_t.p_max_pu.columns.str.contains("onwind PPA")
].sum()
solar_flh: pd.DataFrame = n.generators_t.p_max_pu.loc[
    :, n.generators_t.p_max_pu.columns.str.contains("solar PPA")
].sum()

# %%
wind_flh.index = wind_flh.index.str.removesuffix(" onwind PPA")
solar_flh.index = solar_flh.index.str.removesuffix(" solar PPA")

voronois["wind_flh"] = wind_flh
voronois["solar_flh"] = solar_flh

# %%
# setup era5

long_range = slice(-9.5, 12.5)
lat_range = slice(18.5, 37.5)

plot_boundaries_large = [-9.5, 12.5, 18.5, 38.3]

cutout = atlite.Cutout(base_path + "/cutouts/cutout-2013-era5.nc")

cutout.data = cutout.data.sel(
    x=long_range,
    y=lat_range,
)

panel = "CSi"
orientation = "latitude_optimal"
correction_factor_pv = 0.854337

turbine = "Vestas_V112_3MW"
correction_factor_wind = 1

era5 = {}
era5["solar_flh"] = (
    cutout.pv(panel=panel, orientation=orientation).rename(
        "Annual solar full-load hours (h/a)"
    )
    * correction_factor_pv
)
era5["wind_flh"] = (
    cutout.wind(turbine=turbine).rename("Annual wind full-load hours (h/a)")
    * correction_factor_wind
)

# %% select data within Algeria only

map_fp = r"..\data\country_shapes.geojson"
shapes = gpd.read_file(map_fp, index="name").set_index("name")
dz_shape = shapes.loc[["DZ"]]

# identify all cutout cells in Algeria
grid = cutout.grid
cells_in_dz = gpd.sjoin(grid, dz_shape, predicate="intersects", how="inner")

mask_df = grid[["x", "y"]].copy()
mask_df["in_dz"] = grid.index.isin(cells_in_dz.index)

dz_mask = mask_df.set_index(["y", "x"])["in_dz"].to_xarray()

era5["solar_flh"] = era5["solar_flh"].where(dz_mask)
era5["wind_flh"] = era5["wind_flh"].where(dz_mask)


# %% prepare protected areas

natura_fp = r"..\data\natura\natura.tiff"
ds = xr.open_dataset(natura_fp)

protected_dz = ds.sel(
    y=slice(plot_boundaries_large[3], plot_boundaries_large[2]),
    x=slice(plot_boundaries_large[0], plot_boundaries_large[1]),
    band=1,
)["band_data"]

# clip to just algeria
protected_dz = protected_dz.rio.clip(
    shapes.loc[["DZ"], "geometry"],
    shapes.crs,
    drop=True,
    all_touched=True,
)

# %% plot the solar and wind full load hours, combining voronois with era5

cbar_step = 50

voronois_edge = "black"
voronoi_width = 0.5
excluder_linewidth = 0.5
excluder_hatch = "xx"
excluder_hatch_linewidth = 0.5

figsize = (6.4, 4.5)
carriers = ["solar_flh", "wind_flh"]
labels = {
    "solar_flh": "photovoltaic",
    "wind_flh": "wind",
}

# setup plot
fig, axs = plt.subplot_mosaic(
    [carriers],
    figsize=figsize,
    layout="tight",
)

# configure axes
for carrier in carriers:
    axs[carrier].set_xlim(
        [plot_boundaries_large[0], plot_boundaries_large[1]]
    )  # -10,13
    axs[carrier].set_ylim([plot_boundaries_large[2], plot_boundaries_large[3]])  # 18,
    axs[carrier].set_aspect("equal")
    axs[carrier].tick_params(axis="both", which="major", labelsize=SIZE)

for carrier in carriers:
    # plot Algeria
    shapes.loc[shapes.index == "DZ"].plot(
        ax=axs[carrier],
        edgecolor="black",
        facecolor=color_map["Background"],
    )

    vmin = min(voronois[carrier].min(), era5[carrier].min())
    vmax = max(voronois[carrier].max(), era5[carrier].max())

    era5_ax = era5[carrier].plot.pcolormesh(
        ax=axs[carrier],
        cmap=cm.lipari,  # navia
        vmin=(vmin // cbar_step) * cbar_step,
        vmax=(vmax // cbar_step + 1) * cbar_step,
        add_colorbar=False,
    )

    voronois_ax = voronois.plot(
        ax=axs[carrier],
        column=carrier,
        cmap=cm.lipari,  # navia
        vmin=(vmin // cbar_step) * cbar_step,
        vmax=(vmax // cbar_step + 1) * cbar_step,
        edgecolor=voronois_edge,
        linewidth=voronoi_width,
        legend=True,
        legend_kwds={
            "label": f"Annual {labels[carrier]} full-load hours in h/a",
            "location": "bottom",
            "pad": 0.125,
        },
    )

    cbar = voronois_ax.get_figure().axes[-1]
    cbar.xaxis.labelpad = 10
    cbar.xaxis.label.set_size(SIZE)
    cbar.tick_params(labelsize=SIZE)

    # plot protected areas
    # plot hatch
    with plt.rc_context({"hatch.linewidth": excluder_hatch_linewidth}):
        protected_dz.plot.contourf(
            ax=axs[carrier],
            add_colorbar=False,
            add_labels=False,
            # colors=["black"],
            colors="none",
            hatches=[excluder_hatch],
            hatch_linewidth=excluder_hatch_linewidth,
            levels=[0.9, 1.1],  # selects values of one (protected areas)
        )
    # plot outline
    protected_dz.fillna(0).plot.contour(
        ax=axs[carrier],
        add_colorbar=False,
        add_labels=False,
        colors="black",
        linewidths=excluder_linewidth,
        levels=[0.9, 1.1],  # selects values of one (protected areas)
    )

    # plot map
    # plot ocean to fix overlap in the north
    ocean = gpd.GeoSeries(
        box(*plot_boundaries_large).difference(shapes.union_all()),
        crs=shapes.crs,
    )
    ocean.plot(
        ax=axs[carrier],
        facecolor=color_map["Ocean"],
        edgecolor="none",
    )

    # plot all other countries
    shapes.loc[shapes.index != "DZ"].plot(
        ax=axs[carrier],
        edgecolor="black",
        facecolor=color_map["Background"],
    )
    # plot the Algerian borders
    shapes.loc[shapes.index == "DZ"].boundary.plot(ax=axs[carrier], edgecolor="black")
    axs[carrier].set_facecolor(color_map["Ocean"])

    # add axis labels
    xlabel = "Longitude in degree"
    ylabel = "Latitude in degree"
    if carrier != "wind_flh":
        axs[carrier].set_ylabel(ylabel, fontsize=SIZE)
    else:
        axs[carrier].set_ylabel("")
    axs[carrier].set_xlabel(xlabel, fontsize=SIZE)

    # setup tick steps
    axs[carrier].xaxis.set_major_locator(MaxNLocator(steps=[4], integer=False))
    axs[carrier].yaxis.set_major_locator(MaxNLocator(steps=[4], integer=False))

# move plots closer together
axs[carrier].yaxis.set_ticks([])
axs[carrier].yaxis.set_ticklabels([])
fig.tight_layout()
fig.subplots_adjust(wspace=0.025)

# legend
legend_elements = [
    Patch(
        # fill=False,
        edgecolor=voronois_edge,
        facecolor="none",
        lw=excluder_linewidth,
        label="Voronoi cell",
    ),
    Patch(
        edgecolor=voronois_edge,
        facecolor="none",
        hatch=excluder_hatch,
        lw=excluder_linewidth,
        label="Protected area",
    ),
]

# rc_context change required for the hatch in the patch
with plt.rc_context({"hatch.linewidth": excluder_hatch_linewidth}):
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=SIZE,
    )

# %% save pv and wind full load hours

plt.show()
tracker = "V1.0"

dir_name = f"..\\images\\{run_name}\\"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

fp = f"{dir_name}\\flh_comparison_{tracker}.svg"
fig.savefig(fp, bbox_inches="tight")


# %% print max flh

era5_re = {}
flh_unprotected = {}

# explicitely set the rio crs for the flh
era5["solar_flh"] = era5["solar_flh"].rio.write_crs("EPSG:4326")
era5["wind_flh"] = era5["wind_flh"].rio.write_crs("EPSG:4326")

for carrier in carriers:

    # reproject flh to the higher res grid of the protected areas
    era5_re[carrier] = (
        era5[carrier]
        .drop_vars(["lat", "lon"])
        .rio.reproject_match(
            protected_dz,
            resampling=Resampling.nearest,
        )
    )

    flh_unprotected[carrier] = (
        era5_re[carrier]
        .where(protected_dz.isnull())
        .rio.clip(
            shapes.loc[["DZ"], "geometry"].buffer(-0.05), # slight buffer to remove cells that get falsly added during resampling 
            shapes.crs,
            drop=True,
            all_touched=False,
        )
    )
    
    print(f"{carrier} max: {flh_unprotected[carrier].max().item()} flh")
    print(f"{carrier} max incl. protected: {era5_re[carrier].max().item()} flh")
    print(f"{carrier} max Voronois: {voronois[carrier].max()} flh")


# %% plot era5 data larger than the voronoi values

cbar_step = 50

voronois_edge = "black"
voronoi_width = 0.5
excluder_linewidth = 0.5
excluder_hatch = "xx"
excluder_hatch_linewidth = 0.5

figsize = (6.4, 4.5)
carriers = ["solar_flh", "wind_flh"]
labels = {
    "solar_flh": "photovoltaic",
    "wind_flh": "wind",
}

# setup plot
fig, axs = plt.subplot_mosaic(
    [carriers],
    figsize=figsize,
    layout="tight",
)

# configure axes
for carrier in carriers:
    axs[carrier].set_xlim(
        [plot_boundaries_large[0], plot_boundaries_large[1]]
    )  # -10,13
    axs[carrier].set_ylim([plot_boundaries_large[2], plot_boundaries_large[3]])  # 18,
    axs[carrier].set_aspect("equal")
    axs[carrier].tick_params(axis="both", which="major", labelsize=SIZE)

for carrier in carriers:
    # plot Algeria
    shapes.loc[shapes.index == "DZ"].plot(
        ax=axs[carrier],
        edgecolor="black",
        facecolor=color_map["Background"],
    )

    vmin = voronois[carrier].max()
    vmax = era5[carrier].max()

    era5_ax = era5[carrier].where(era5[carrier] > vmin).plot.pcolormesh(
        ax=axs[carrier],
        cmap=cm.lipari,  # navia
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    # keep only for the cbar
    voronois_ax = voronois.plot(
        ax=axs[carrier],
        column=carrier,
        cmap=cm.lipari,  # navia
        vmin=vmin,
        vmax=vmax,
        edgecolor=voronois_edge,
        linewidth=voronoi_width,
        legend=True,
        legend_kwds={
            "label": f"Annual {labels[carrier]} full-load hours in h/a",
            "location": "bottom",
            "pad": 0.125,
        },
    )

    # plot over the initial voronoi plot to leave them empty
    voronois_ax = voronois.plot(
            ax=axs[carrier],
            color=color_map["Background"],
            edgecolor=voronois_edge,
            linewidth=voronoi_width,
            legend=False,
        )

    cbar = voronois_ax.get_figure().axes[-1]
    cbar.xaxis.labelpad = 10
    cbar.xaxis.label.set_size(SIZE)
    cbar.tick_params(labelsize=SIZE)

    # plot protected areas
    # plot hatch
    with plt.rc_context({"hatch.linewidth": excluder_hatch_linewidth}):
        protected_dz.plot.contourf(
            ax=axs[carrier],
            add_colorbar=False,
            add_labels=False,
            # colors=["black"],
            colors="none",
            hatches=[excluder_hatch],
            hatch_linewidth=excluder_hatch_linewidth,
            levels=[0.9, 1.1],  # selects values of one (protected areas)
        )
    # plot outline
    protected_dz.fillna(0).plot.contour(
        ax=axs[carrier],
        add_colorbar=False,
        add_labels=False,
        colors="black",
        linewidths=excluder_linewidth,
        levels=[0.9, 1.1],  # selects values of one (protected areas)
    )

    # plot map
    # plot ocean to fix overlap in the north
    ocean = gpd.GeoSeries(
        box(*plot_boundaries_large).difference(shapes.union_all()),
        crs=shapes.crs,
    )
    ocean.plot(
        ax=axs[carrier],
        facecolor=color_map["Ocean"],
        edgecolor="none",
    )

    # plot all other countries
    shapes.loc[shapes.index != "DZ"].plot(
        ax=axs[carrier],
        edgecolor="black",
        facecolor=color_map["Background"],
    )
    # plot the Algerian borders
    shapes.loc[shapes.index == "DZ"].boundary.plot(ax=axs[carrier], edgecolor="black")
    axs[carrier].set_facecolor(color_map["Ocean"])

    # add axis labels
    xlabel = "Longitude in degree"
    ylabel = "Latitude in degree"
    if carrier != "wind_flh":
        axs[carrier].set_ylabel(ylabel, fontsize=SIZE)
    else:
        axs[carrier].set_ylabel("")
    axs[carrier].set_xlabel(xlabel, fontsize=SIZE)

    # setup tick steps
    axs[carrier].xaxis.set_major_locator(MaxNLocator(steps=[4], integer=False))
    axs[carrier].yaxis.set_major_locator(MaxNLocator(steps=[4], integer=False))

# move plots closer together
axs[carrier].yaxis.set_ticks([])
axs[carrier].yaxis.set_ticklabels([])
fig.tight_layout()
fig.subplots_adjust(wspace=0.025)

# legend
legend_elements = [
    Patch(
        # fill=False,
        edgecolor=voronois_edge,
        facecolor="none",
        lw=excluder_linewidth,
        label="Voronoi cell",
    ),
    Patch(
        edgecolor=voronois_edge,
        facecolor="none",
        hatch=excluder_hatch,
        lw=excluder_linewidth,
        label="Protected area",
    ),
]

# rc_context change required for the hatch in the patch
with plt.rc_context({"hatch.linewidth": excluder_hatch_linewidth}):
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=SIZE,
    )

# %% save pv and wind full load hours where flh > voronois max

plt.show()
tracker = "V1.0"

dir_name = f"..\\images\\{run_name}\\"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

fp = f"{dir_name}\\flh_comparison_max_{tracker}.svg"
fig.savefig(fp, bbox_inches="tight")