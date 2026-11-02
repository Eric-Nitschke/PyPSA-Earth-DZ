# %%
import numpy as np
import pandas as pd
import pypsa
from typing import Tuple

# %%


def load_expansion_data(
    n: pypsa.Network,
    offgrid: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    point_expansion = get_point_expansion(n)
    line_expansion = get_line_expansion(n, offgrid)

    return point_expansion, line_expansion


def get_point_expansion(n: pypsa.Network) -> pd.DataFrame:
    """
    Get the expansion of every point component in the network n.
    """

    point_expansion = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([], names=["Type", "Component"])
    )

    clean_names = {
        "Generator": {
            "onwind PPA": "Onshore Wind",
            "solar PPA": "Photovoltaic",
        },
        "Link": {
            "Desalination": "Desalination",
            "H2 electrolysis": "Electrolysis",
        },
        "Store": {
            "H2": "Hydrogen Storage",
            "battery": "Battery Storage",
        },
    }

    attr_dict = {
        "Generator": {
            "initial": "p_nom",
            "optimized": "p_nom_opt",
        },
        "Link": {
            "initial": "p_nom",
            "optimized": "p_nom_opt",
        },
        "Store": {
            "initial": "e_nom",
            "optimized": "e_nom_opt",
        },
    }

    conv_dict = {
        "Desalination": 1 / 3.0348 * 1000  # (m^3-H2O/h)/kW*(kW/MW) -> (m^3-H2O/h)/MW
    }

    for type_name in clean_names:

        type_data = n.df(type_name)

        for component_name in clean_names[type_name]:

            comp_data = type_data.loc[type_data.carrier == component_name]

            comp_exp = (
                comp_data[attr_dict[type_name]["optimized"]]
                - comp_data[attr_dict[type_name]["initial"]]
            ).to_frame()

            if component_name in conv_dict:
                comp_exp *= conv_dict[component_name]

            comp_exp = comp_exp.rename(lambda i: " ".join(i.split()[0:2]))
            comp_exp.columns = pd.MultiIndex.from_tuples(
                [(type_name, clean_names[type_name][component_name])]
            )

            point_expansion = point_expansion.join(comp_exp, how="outer")

    point_expansion = point_expansion.fillna(0)

    buses = n.buses.loc[point_expansion.index, ["x", "y"]]
    buses.columns = pd.MultiIndex.from_product([["Bus"], buses.columns])

    point_expansion = point_expansion.join(buses, how="outer")

    return point_expansion


def get_line_expansion(
    n: pypsa.Network,
    offgrid: bool,
) -> pd.DataFrame:

    def get_line_name(
        row: pd.Series,
    ) -> str:
        """
        Return the name of a line component, comprised of the two buses it connects.
        The first bus is always west of the second bus.
        """

        bus0_i = " ".join(row["bus0"].split()[0:2])
        bus1_i = " ".join(row["bus1"].split()[0:2])

        if n.buses.loc[bus0_i, "x"] < n.buses.loc[bus1_i, "x"]:
            name = f"{bus0_i}-{bus1_i}"
        else:
            name = f"{bus1_i}-{bus0_i}"

        return name

    def get_coordinates(
        row: pd.Series,
        n: pypsa.Network,
    ) -> pd.Series:
        """
        Return the coordinates of a line component, comprised of the two buses it connects.
        """
        bus0_i, bus1_i = row.name.split("-")
        coords = pd.Series(
            {
                "x0": n.buses.loc[bus0_i, "x"],
                "y0": n.buses.loc[bus0_i, "y"],
                "x1": n.buses.loc[bus1_i, "x"],
                "y1": n.buses.loc[bus1_i, "y"],
            }
        )

        return coords

    line_expansion = pd.DataFrame()

    if offgrid == "True":
        line_str = "AC offgrid"
    else:
        line_str = "AC"

    clean_names = {
        "Line": {
            line_str: "Power Line",
        },
        "Link": {
            "H2 pipeline": "Hydrogen Pipeline",
            "water pipeline": "Water Pipeline",
        },
    }

    attr_dict = {
        "Line": {
            "initial": "s_nom",
            "optimized": "s_nom_opt",
        },
        "Link": {
            "initial": "p_nom",
            "optimized": "p_nom_opt",
        },
    }

    for type_name in clean_names:

        type_data = n.df(type_name)

        for component_name in clean_names[type_name]:

            comp_data = type_data.loc[
                (type_data.carrier == component_name)
                & ~type_data.get("reversed", pd.Series(False, index=type_data.index))
            ]

            comp_exp = (
                comp_data[attr_dict[type_name]["optimized"]]
                - comp_data[attr_dict[type_name]["initial"]]
            )

            comp_exp.index = comp_data.apply(get_line_name, axis=1)
            comp_exp.name = clean_names[type_name][component_name]

            line_expansion = line_expansion.join(comp_exp, how="outer")

    line_expansion = line_expansion.fillna(0)
    line_expansion = line_expansion.join(
        line_expansion.apply(get_coordinates, axis=1, args=(n,)), how="outer"
    )

    return line_expansion


# %%
