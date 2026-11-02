import pandas as pd
import pypsa

def load_cost_data(
    n: pypsa.Network,
) -> pd.DataFrame:

    costs = pd.DataFrame()

    opex = n.statistics.opex().droplevel(0)
    capex = n.statistics.expanded_capex().droplevel(0)

    clean_names = {
        "Onwind Ppa": "Onshore Wind",
        "Solar Ppa": "Photovoltaic",
        "AC": "Power Line",
        "Desalination": "Desalination",
        "H2 Electrolysis": "Electrolysis",
        "H2 Pipeline": "Hydrogen Pipeline",
        "Water Pipeline": "Water Pipeline",
        "Hydrogen Storage": "Hydrogen Storage",
    }

    costs["CAPEX"] = capex.loc[capex.index.isin(clean_names)].rename(index=clean_names)
    costs["OPEX"] = opex.loc[opex.index.isin(clean_names)].rename(index=clean_names)
    costs = costs.fillna(0)

    costs.loc["Battery Storage", "CAPEX"] = [
        capex["Battery Storage"] + capex["Battery Charger"]
    ]
    costs.loc["Battery Storage", "OPEX"] = [
        opex["Battery Charger"] + opex["Battery Discharger"]
    ]
    costs["Total"] = costs.sum(axis=1)

    LHV_H2 = 121  # MJ_H2/kg_H2
    load_H2 = (
        n.loads.loc[n.loads.carrier == "H2", "p_set"].sum() / LHV_H2 * 8760 * 3600
    )  # kg_H2/a

    if load_H2 == 0:
        costs["LCOH"] = costs["Total"] * 0  # €/kg_H2
    else:
        costs["LCOH"] = costs["Total"] / load_H2  # €/kg_H2

    return costs
