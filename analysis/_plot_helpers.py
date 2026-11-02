# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Wedge, Rectangle, Patch
from matplotlib.lines import Line2D
from typing import Tuple
from math import pi

SIZE = 8

# %%


def get_line_collections(
    line_expansion: pd.DataFrame,
    ratio: float,
    factor_dict: dict[str, float],
    offset_dict: dict[str, float],
    line_gap: float,
) -> dict[str, LineCollection]:
    """
    Generate a LineCollection for every line type in line_expansion DataFrame.
    """

    coords = line_expansion[["x0", "y0", "x1", "y1"]]
    expansions = line_expansion.drop(columns=coords.columns)

    lw_dict = {}

    for component in expansions:
        lw_dict[component] = _get_linewidths(
            expansions[component],
            factor_dict[component],
        )

    lines = _get_lines(
        coords,
        lw_dict,
        offset_dict,
        ratio,
        line_gap,
    )

    line_collections = {}
    for component in expansions:
        line_collections[component] = LineCollection(
            lines[component],
            linewidths=lw_dict[component],
            capstyle="round",
        )

    return line_collections


def _get_linewidths(
    expansion: pd.DataFrame,
    factor: float,
    min_lw: float = 0.1,
):

    # calculate linewidths for plotting
    linewidths = (expansion / factor).apply(lambda x: 0 if x < min_lw else x)

    return linewidths


def _get_lines(
    coords: pd.DataFrame,
    lw_dict: dict[str, pd.Series],
    offset_dict: dict[str, float],
    ratio: float,
    line_gap: float,
) -> pd.DataFrame:

    def _get_norm_vector(row: pd.Series) -> Tuple[float, float]:

        dx = row["x1"] - row["x0"]
        dy = row["y1"] - row["y0"]
        length = (dx**2 + dy**2) ** 0.5

        if length == 0:
            return (0, 0)

        return (-dy / length, dx / length)

    def _create_offset_line(
        row: pd.Series,
        lw_dict: dict[str, pd.Series],
        component: str,
        ratio: float,
        offset: float,
    ) -> list[Tuple[float, float]]:

        # get normal vector
        row["xn"], row["yn"] = _get_norm_vector(row)

        comp_dist = lw_dict[component].loc[row.name] / 2
        base_dist = lw_dict["Power Line"].loc[row.name] / 2
        x_dist = (comp_dist + base_dist + line_gap) * row["xn"] * ratio
        y_dist = (comp_dist + base_dist + line_gap) * row["yn"] * ratio

        x0 = row["x0"] + x_dist * offset
        x1 = row["x1"] + x_dist * offset
        y0 = row["y0"] + y_dist * offset
        y1 = row["y1"] + y_dist * offset

        return [(x0, y0), (x1, y1)]

    lines = pd.DataFrame()

    for component in lw_dict:
        offset = offset_dict[component]
        if offset == 0:
            lines[component] = coords.apply(
                lambda row: [(row["x0"], row["y0"]), (row["x1"], row["y1"])], axis=1
            )

        else:
            lines[component] = coords.apply(
                _create_offset_line,
                args=(lw_dict, component, ratio, offset),
                axis=1,
            )

    return lines


# %%


def get_generator_collection(
    point_expansion: pd.DataFrame,
    factor: float,
    min_expansion: float,
    color_map: dict[str, str],
    alpha: float,
    zorder: float = 2.5,
) -> PatchCollection:
    """
    Generate a PatchCollection of Wedges forming a half circle, split by every generator type.
    """

    def _get_wedges(
        row: pd.Series,
        factor: float,
        min_expansion: float,
        color_map: dict[str, str],
    ) -> list[Wedge]:

        wedges = []
        total = row["Generator"].sum()
        radius = (row["Generator"].sum() / factor / pi * 2) ** 0.5
        start_angle = 90

        for gen_type in row["Generator"].index:
            if row["Generator"][gen_type] > min_expansion:
                wedges.append(
                    Wedge(
                        (row["Bus"]["x"], row["Bus"]["y"]),
                        r=radius,
                        theta1=start_angle,
                        theta2=start_angle + 180 * row["Generator"][gen_type] / total,
                        facecolor=color_map[gen_type],
                        edgecolor=color_map["Patch Border"],
                    )
                )
                start_angle += 180 * row["Generator"][gen_type] / total

        return wedges

    wedge_series = point_expansion.apply(
        _get_wedges,
        args=(factor, min_expansion, color_map),
        axis=1,
    )
    wedges = [patch for sublist in wedge_series for patch in sublist]
    gen_collection = PatchCollection(
        wedges,
        match_original=True,
        alpha=alpha,
        zorder=zorder,
    )

    return gen_collection


# %%


def get_link_collection(
    point_expansion: pd.DataFrame,
    factor_dict: dict[str, float],
    min_expansion_dict: dict[str, float],
    color_map: dict[str, str],
    alpha: float,
    zorder: float = 2.5,
) -> PatchCollection:
    """
    Generate a PatchCollection of Rectangles next to each other for every links type.
    """

    patch_dict = {
        "Desalination": Rectangle,
        "Electrolysis": Wedge,
    }

    def _get_patches(
        row: pd.Series,
        factor_dict: float,
        patch_dict: dict[str, Patch],
        min_expansion_dict: dict[str, float],
        color_map: dict[str, str],
    ) -> list[Patch]:

        patches = []
        min_distance = 0.1
        x_distance = 0

        for link_type in row["Link"].index:
            if row["Link"][link_type] > min_expansion_dict[link_type]:
                if patch_dict[link_type] == Wedge:
                    radius = (
                        row["Link"][link_type] / factor_dict[link_type] / pi * 2
                    ) ** 0.5
                    patches.append(
                        Wedge(
                            (row["Bus"]["x"], row["Bus"]["y"]),
                            r=radius,
                            theta1=270,
                            theta2=450,
                            facecolor=color_map[link_type],
                            edgecolor=color_map["Patch Border"],
                        )
                    )
                    x_distance += radius + min_distance

                elif patch_dict[link_type] == Rectangle:
                    width = (row["Link"][link_type] / factor_dict[link_type] / 2) ** 0.5
                    patches.append(
                        Rectangle(
                            (row["Bus"]["x"] + x_distance, row["Bus"]["y"] - width),
                            width=width,
                            height=width * 2,
                            facecolor=color_map[link_type],
                            edgecolor=color_map["Patch Border"],
                        )
                    )
                    x_distance += width + min_distance
            else:
                x_distance = min_distance

        return patches

    patch_series = point_expansion.apply(
        _get_patches,
        args=(factor_dict, patch_dict, min_expansion_dict, color_map),
        axis=1,
    )
    patches = [patch for sublist in patch_series for patch in sublist]
    link_collection = PatchCollection(
        patches,
        match_original=True,
        alpha=alpha,
        zorder=zorder,
    )

    return link_collection


# %% define legend handlers


class WedgeHandler:

    def __init__(self, ratio):
        self.ratio = ratio * np.sqrt(2)

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        if orig_handle.theta1 == 270:
            center = (x0 + 0.25 * width, y0 + height / 2)
        else:
            center = (x0 + width, y0 + height / 2)

        handle = Wedge(
            center=center,  # centers handle in handlebox
            r=orig_handle.r / self.ratio,  # ensures radius fits in handlebox
            # width=orig_handle.r / self.ratio, # preserves original radius/width ratio
            theta1=orig_handle.theta1,  # copies the following parameters
            theta2=orig_handle.theta2,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            transform=handlebox.get_transform(),  # use handlebox coordinate system
        )

        handlebox.add_artist(handle)
        return handle


class RectangleHandler:

    def __init__(self, ratio):
        self.ratio = ratio * np.sqrt(2)

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        dx = orig_handle.get_width() / self.ratio
        dy = orig_handle.get_height() / self.ratio

        handle = Rectangle(
            xy=(
                x0 + width - dx,
                y0 + height / 2 - dy / 2,
            ),  # centers handle in handlebox
            width=dx,
            height=dy,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            transform=handlebox.get_transform(),  # use handlebox coordinate system
        )

        handlebox.add_artist(handle)
        return handle


# %%


def get_legends(
    ax,
    load,
    color_map,
    factor_dict,
    ratio,
):

    load_factor = load / 50

    legend_dict = {
        "Power Line": {
            "type": Line2D,
            "scale": [50, 25, 10],
            "position": [0, 0],
            "title": "Power line extension",
            "unit": "MW",
            "spacing": 0,
        },
        "Hydrogen Pipeline": {
            "type": Line2D,
            "scale": [50, 25, 10],
            "position": [1, 0],
            "title": "Hydrogen pipeline",
            "unit": "MW",
            "spacing": 0,
        },
        "Water Pipeline": {
            "type": Line2D,
            "scale": [10, 5, 2],
            "position": [2, 0],
            "title": "Water pipeline",
            "unit": "m³/h",
            "spacing": 0,
        },
        "Generator": {
            "type": Wedge,
            "start_angle": 90,
            "scale": [300, 150, 50],
            "position": [0, 1],
            "title": "Generator\n",
            "unit": "MW",
            "spacing": 1,
        },
        "Electrolysis": {
            "type": Wedge,
            "start_angle": 270,
            "scale": [100, 50, 20],
            "position": [1, 1],
            "title": "Electrolysis\n",
            "unit": "MW",
            "spacing": 1,
        },
        "Desalination": {
            "type": Rectangle,
            "scale": [10, 5, 2],
            "position": [2, 1],
            "title": "Desalination\n",
            "unit": "m³/h",
            "spacing": 1,
        },
        "Generator Type": {
            "type": LineCollection,
            "marker": ["o", "o"],
            "markersize": [7, 7],
            "names": ["Photovoltaic", "Onshore Wind"],
            "position": [0, 2],
            "title": "Generator Type",
            "spacing": 0,
        },
        "Infrastructure": {
            "type": LineCollection,
            "marker": ["o", "x"],
            "markersize": [5, 7],
            "names": ["Network Node", "Export Location"],
            "position": [1, 2],
            "title": "Infrastructure",
            "spacing": 0,
        },
        "Infrastructure_Col2": {
            "type": LineCollection,
            "marker": [None],
            "markersize": [2],
            "names": ["Existing Power Line"],
            "position": [2, 2],
            "title": " ",
            "spacing": 0,
        },
    }

    handler_map = {
        Wedge: WedgeHandler(ratio),
        Rectangle: RectangleHandler(ratio),
    }

    # legend position below image
    pos_x = [0, 0.33, 0.66]
    pos_y = [-0.075, -0.32, -0.76]
    # legend position to the right
    # pos_x = [1.05, 1.3, 1.56]
    # pos_y = [0.95, 0.7, 0.27]

    label_spacings = [0.8, 2.5]

    output_dict = {}

    for legend_name in legend_dict:
        legend = legend_dict[legend_name]
        handles = []
        labels = []

        if legend["type"] == Line2D:
            for scale in legend["scale"]:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=color_map[legend_name],
                        solid_capstyle="round",
                        linewidth=scale / factor_dict[legend_name] * load_factor,
                    )
                )
                labels.append(f"{scale*load_factor:,.0f} {legend['unit']}")

        elif legend["type"] == Wedge:
            for i, scale in enumerate(legend["scale"]):
                radius = (
                    scale / factor_dict[legend_name] * load_factor / pi * 2
                ) ** 0.5
                handles.append(
                    Wedge(
                        (0, 0),
                        r=radius,
                        theta1=legend["start_angle"],
                        theta2=legend["start_angle"] + 180,
                        facecolor=color_map[legend_name],
                        edgecolor=color_map["Patch Border"],
                        zorder=3 - i,
                    )
                )
                labels.append(f"{scale*load_factor:,.0f} {legend['unit']}")

        elif legend["type"] == Rectangle:
            for i, scale in enumerate(legend["scale"]):
                width = (scale / factor_dict[legend_name] * load_factor / 2) ** 0.5
                handles.append(
                    Rectangle(
                        (0, 0),
                        width=width,
                        height=width * 2,
                        facecolor=color_map[legend_name],
                        edgecolor=color_map["Patch Border"],
                        zorder=3 - i,
                    )
                )
                labels.append(f"{scale*load_factor:,.0f} {legend['unit']}")

        elif legend["type"] == LineCollection:
            labels = legend["names"]
            for i, label in enumerate(labels):
                if legend["marker"][i]:
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color_map[label],
                            linestyle="none",
                            marker=legend["marker"][i],
                            markersize=legend["markersize"][i],
                        )
                    )
                else:
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color_map[label],
                            linewidth=legend["markersize"][i],
                        )
                    )
        else:
            continue

        output_dict[legend_name] = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(pos_x[legend["position"][0]], pos_y[legend["position"][1]]),
            frameon=False,
            labelspacing=label_spacings[legend["spacing"]],
            handletextpad=1.5,
            title=legend["title"],
            title_fontsize=SIZE,
            fontsize=SIZE,
            handler_map=handler_map,
        )
        output_dict[legend_name]._legend_box.align = "left"

    # additional empty legend to suppress weird behaviour (last legend slightly darker)
    output_dict["Helper"] = ax.legend([], [], frameon=False)

    return output_dict


# %%


def get_legends_merged(
    ax,
    load,
    color_map,
    factor_dict,
    ratio,
):

    load_factor = load / 50

    legend_dict = {
        "Power Line": {
            "type": Line2D,
            "scale": [50, 25, 10],
            "position": [0, 0],
            "title": "Power line extension",
            "unit": "MVA",
            "spacing": 0,
        },
        "Hydrogen Pipeline": {
            "type": Line2D,
            "scale": [50, 25, 10],
            "position": [1, 0],
            "title": "Hydrogen pipeline",
            "unit": "MW",
            "spacing": 0,
        },
        "Water Pipeline": {
            "type": Line2D,
            "scale": [10, 5, 2],
            "position": [2, 0],
            "title": "Water pipeline",
            "unit": "m³/h",
            "spacing": 0,
        },
        "Generator": {
            "type": Wedge,
            "start_angle": 90,
            "scale": [300, 150, 50],
            "position": [0, 1],
            "title": "Generator",
            "unit": "MW",
            "spacing": 1,
        },
        "Electrolysis": {
            "type": Wedge,
            "start_angle": 270,
            "scale": [100, 50, 20],
            "position": [1, 1],
            "title": "Electrolysis",
            "unit": "MW",
            "spacing": 1,
        },
        "Desalination": {
            "type": Rectangle,
            "scale": [10, 5, 2],
            "position": [2, 1],
            "title": "Desalination",
            "unit": "m³/h",
            "spacing": 1,
        },
        "Generator Type": {
            "type": LineCollection,
            "marker": ["o", "o"],
            "markersize": [7, 7],
            "names": ["Photovoltaic", "Onshore Wind"],
            "position": [3, 1],
            "title": "Generator Type",
            "spacing": 0,
        },
        "Infrastructure": {
            "type": LineCollection,
            "marker": ["o", None, "x"],
            "markersize": [5, 2, 7],
            "names": ["Network Node", "Existing Power Line", "Export Location"],
            "position": [3, 0],
            "title": "Infrastructure",
            "spacing": 0,
        },
        "Infrastructure_Col2": {
            "type": LineCollection,
            "marker": [None],
            "markersize": [2],
            "names": [],
            "position": [3, 2],
            "title": " ",
            "spacing": 0,
        },
    }

    handler_map = {
        Wedge: WedgeHandler(ratio),
        Rectangle: RectangleHandler(ratio),
    }

    # legend position below image
    pos_x = [0, 0.475, 0.95, 1.425]
    pos_y = [-0.2, -0.625, -1.025]
    # legend position to the right
    # pos_x = [1.05, 1.3, 1.56]
    # pos_y = [0.95, 0.7, 0.27]

    label_spacings = [0.8, 1.2]

    output_dict = {}

    for legend_name in legend_dict:
        legend = legend_dict[legend_name]
        handles = []
        labels = []

        if legend["type"] == Line2D:
            for scale in legend["scale"]:
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=color_map[legend_name],
                        solid_capstyle="round",
                        linewidth=scale / factor_dict[legend_name] * load_factor,
                    )
                )
                labels.append(f"{scale*load_factor:,.0f} {legend['unit']}")

        elif legend["type"] == Wedge:
            for i, scale in enumerate(legend["scale"]):
                radius = (
                    scale / factor_dict[legend_name] * load_factor / pi * 2
                ) ** 0.5
                handles.append(
                    Wedge(
                        (0, 0),
                        r=radius,
                        theta1=legend["start_angle"],
                        theta2=legend["start_angle"] + 180,
                        facecolor=color_map[legend_name],
                        edgecolor=color_map["Patch Border"],
                        zorder=3 - i,
                    )
                )
                labels.append(f"{scale*load_factor:,.0f} {legend['unit']}")

        elif legend["type"] == Rectangle:
            for i, scale in enumerate(legend["scale"]):
                width = (scale / factor_dict[legend_name] * load_factor / 2) ** 0.5
                handles.append(
                    Rectangle(
                        (0, 0),
                        width=width,
                        height=width * 2,
                        facecolor=color_map[legend_name],
                        edgecolor=color_map["Patch Border"],
                        zorder=3 - i,
                    )
                )
                labels.append(f"{scale*load_factor:,.0f} {legend['unit']}")

        elif legend["type"] == LineCollection:
            labels = legend["names"]
            for i, label in enumerate(labels):
                if legend["marker"][i]:
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color_map[label],
                            linestyle="none",
                            marker=legend["marker"][i],
                            markersize=legend["markersize"][i],
                        )
                    )
                else:
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color=color_map[label],
                            linewidth=legend["markersize"][i],
                        )
                    )
        else:
            continue

        output_dict[legend_name] = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(pos_x[legend["position"][0]], pos_y[legend["position"][1]]),
            frameon=False,
            labelspacing=label_spacings[legend["spacing"]],
            handletextpad=1.5,
            title=legend["title"],
            title_fontsize=SIZE,
            fontsize=SIZE,
            handler_map=handler_map,
        )
        output_dict[legend_name]._legend_box.align = "left"

    # additional empty legend to suppress weird behaviour (last legend slightly darker)
    output_dict["Helper"] = ax.legend([], [], frameon=False)

    return output_dict


#%%
