# %%
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pypsa
import json

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
load = [50]

network_name = f"elec_s_{clusters}_{mode}_{offgrid}_{locations}_{load}_ec_lcopt_{time_resolution}H.nc"
fp = f"{result_path}{run_name}/networks/{network_name}"
n = pypsa.Network(fp)

# %% Calculate spectral amplitudes for the battery and hydrogen storages

batteries = n.stores_t.e.loc[
    :, n.stores_t.e.columns.str.contains("PPA generator battery")
]
hydrogen = n.stores_t.e.loc[:, n.stores_t.e.columns.str.contains("H2 Store")]

batteries_amps = batteries.apply(np.fft.fft, axis=0).abs() / len(batteries)
hydrogen_amps = hydrogen.apply(np.fft.fft, axis=0).abs() / len(hydrogen)

freqs = np.fft.fftfreq(len(batteries))

mask = freqs > 0
freqs = freqs[mask]
periods = 1 / freqs

batteries_amps = batteries_amps[mask].set_index(periods)
hydrogen_amps = hydrogen_amps[mask].set_index(periods)

battery_color = color_map["Battery Storage"]
hydrogen_color = color_map[
    "Hydrogen Pipeline"
]  # used instead of "Hydrogen Storage" for better contrast

# %% Plot the spectral amplitudes

figsize = (6.4, 4)

fig, axs = plt.subplots(2, 1, figsize=figsize)

axs[0].plot(
    batteries_amps.mean(axis=1),
    label="Battery storages",
    color=battery_color,
)
axs[0].plot(
    hydrogen_amps.mean(axis=1),
    label="Hydrogen storages",
    color=hydrogen_color,
)

axs[1].plot(
    batteries_amps.mean(axis=1),
    color=battery_color,
)
axs[1].plot(
    hydrogen_amps.mean(axis=1),
    color=hydrogen_color,
)

axs[0].set_xlim([2, 36])
axs[1].set_xlim([2, 8760])
axs[1].set_xscale("log")

axs[0].legend(fontsize=SIZE, framealpha=1)
axs[0].grid()
axs[1].grid()

fig.supylabel(
    "Mean normalized spectral amplitudes of the\nstorages' energy levels in MWh",
    fontsize=SIZE,
    ha="center",
    x=0.06,
)
axs[1].set_xlabel("Period in h", fontsize=SIZE)

axs[0].tick_params(axis="both", which="major", labelsize=SIZE)
axs[1].tick_params(axis="both", which="major", labelsize=SIZE)


# %% Save spectral amplitudes images

tracker = "V1.1"
file_type = "svg"

fp = f"..\\images\\{run_name}\\storage_amplitudes_{tracker}.{file_type}"

if not os.path.exists(f"..\\images\\{run_name}"):
    os.makedirs(f"..\\images\\{run_name}")

fig.savefig(fp, bbox_inches="tight")
# %%
