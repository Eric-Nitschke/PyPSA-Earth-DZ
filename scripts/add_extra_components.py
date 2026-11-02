# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Adds extra extendable components to the clustered and simplified network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        rooftop_share:
        USD2013_to_EUR2013:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        extendable_carriers:
            StorageUnit:
            Store:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec.nc``:


Description
-----------

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config.yaml`` at ``electricity: extendable_carriers:``. It processes ``networks/elec_s{simpl}_{clusters}.nc`` to build ``networks/elec_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""
import os

import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    create_logger,
    lossy_bidirectional_links,
    override_component_attrs,
    set_length_based_efficiency,
)
from add_electricity import (
    _add_missing_carriers_from_costs,
    add_nice_carrier_names,
    load_costs,
)

idx = pd.IndexSlice

logger = create_logger(__name__)


def attach_storageunits(n, costs, config):
    elec_opts = config["electricity"]
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    max_hours = elec_opts["max_hours"]

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        n.madd(
            "StorageUnit",
            buses_i,
            " " + carrier,
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[lookup_store[carrier], "efficiency"],
            efficiency_dispatch=costs.at[lookup_dispatch[carrier], "efficiency"],
            max_hours=max_hours[carrier],
            cyclic_state_of_charge=True,
        )


def attach_stores(n, costs, config):
    elec_opts = config["electricity"]
    carriers = elec_opts["extendable_carriers"]["Store"]

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ["x", "y", "country"]}

    if "battery" in carriers:
        # attach batteries to all AC buses

        _add_missing_carriers_from_costs(n, costs, ["battery"])

        b_buses_i = n.madd(
            "Bus", buses_i + " battery", carrier="battery", **bus_sub_dict
        )
        attach_batteries(n, b_buses_i, costs)

    if ("csp" in elec_opts["renewable_carriers"]) and (
        config["renewable"]["csp"]["csp_model"] == "advanced"
    ):
        # add separate buses for csp
        main_buses = n.generators.query("carrier == 'csp'").bus
        csp_buses_i = n.madd(
            "Bus",
            main_buses + " csp",
            carrier="csp",
            x=n.buses.loc[main_buses, "x"].values,
            y=n.buses.loc[main_buses, "y"].values,
            country=n.buses.loc[main_buses, "country"].values,
        )
        n.generators.loc[main_buses.index, "bus"] = csp_buses_i

        # add stores for csp
        n.madd(
            "Store",
            csp_buses_i,
            bus=csp_buses_i,
            carrier="csp",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=costs.at["csp-tower TES", "capital_cost"],
            marginal_cost=costs.at["csp-tower TES", "marginal_cost"],
        )

        # add links for csp
        n.madd(
            "Link",
            csp_buses_i,
            bus0=csp_buses_i,
            bus1=main_buses,
            carrier="csp",
            efficiency=costs.at["csp-tower", "efficiency"],
            capital_cost=costs.at["csp-tower", "capital_cost"],
            p_nom_extendable=True,
            marginal_cost=costs.at["csp-tower", "marginal_cost"],
        )

def attach_batteries(
        n: pypsa.components.Network,
        buses_i: pd.Index,
        costs: pd.core.frame.DataFrame
    ) -> None:

    '''
    Attach batteries, including chargers and dischargers, to all specified buses of network n
    based on the values specified in costs.
    '''

    # add new buses of type battery, matching the location of the original buses
    bus_sub_dict = {k: n.buses[k].loc[buses_i].values for k in ["x", "y", "country"]}
    battery_i = n.madd("Bus", buses_i + " battery", carrier="battery", **bus_sub_dict)
    
    n.add("Carrier","battery",nice_name="Battery")
    n.add("Carrier","battery charger",nice_name="Battery Charger")
    n.add("Carrier","battery discharger",nice_name="Battery Discharger")

    n.madd(
        "Store",
        battery_i,
        bus=battery_i,
        carrier="battery",
        e_cyclic=True,
        e_nom_extendable=True,
        capital_cost=costs.at["battery storage", "capital_cost"],
        marginal_cost=costs.at["battery", "marginal_cost"],
    )

    n.madd(
        "Link",
        battery_i + " charger",
        bus0=buses_i,
        bus1=battery_i,
        carrier="battery charger",
        efficiency=costs.at["battery inverter", "efficiency"],
        capital_cost=costs.at["battery inverter", "capital_cost"],
        p_nom_extendable=True,
        marginal_cost=costs.at["battery inverter", "marginal_cost"],
    )

    n.madd(
        "Link",
        battery_i + " discharger",
        bus0=battery_i,
        bus1=buses_i,
        carrier="battery discharger",
        efficiency=costs.at["battery inverter", "efficiency"],
        p_nom_extendable=True,
        marginal_cost=costs.at["battery inverter", "marginal_cost"],
    )


def attach_hydrogen_network(
        n: pypsa.components.Network,
        costs: pd.core.frame.DataFrame,
        config: dict,
        location_fp: str,
    ) -> None:

    '''
    Attach hydrogen buses, links and stores to the network n,
    based on the extendable components listed in the config.
    The costs file determines the costs of the respective components.
    '''

    extendable = config["electricity"]["extendable_carriers"]

    if not (
        any("H2" in store for store in extendable["Store"]) or
        any("H2" in link for link in extendable["Link"]) or
        ("H2" in extendable["Export"])
    ):
        logger.info(
            "No hydrogen components attached, since none were specified by the config."
        )
        return
    
    buses_i = attach_bus_network(n, carrier="H2")

    if "H2" in extendable["Store"]:
        attach_hydrogen_stores(n, buses_i, costs)
    if "H2 electrolysis" in extendable["Link"]:
        attach_hydrogen_electrolysis(n, buses_i, costs)
    if "H2 fuel cell" in extendable["Link"]:
        attach_hydrogen_fuel_cells(n, buses_i, costs)
    if "H2 pipeline" in extendable["Link"]:  
        connections = indetify_connected_buses(n)
        transmission_efficiency = config["sector"]["transmission_efficiency"]
        attach_hydrogen_pipelines(n, connections, costs, transmission_efficiency)
    if "H2" in extendable["Export"]:
        export_buses = identify_buses_from_file(n, location_fp, "H2", "H2 export")
        export_buses, loads = merge_identical_locations(export_buses, config["H2_export"], drop_without_val=True)
        attach_hydrogen_export(n, export_buses, loads)


def attach_hydrogen_stores(
        n: pypsa.components.Network,
        buses_i: pd.Index,
        costs: pd.core.frame.DataFrame,
    ) -> None:
    
    '''
    Attach hydrogen stores to all specified buses of network n based on the values specified in costs.
    '''

    n.add("Carrier","H2",nice_name="H2")

    n.madd(
        "Store",
        buses_i,
        suffix=" Store",
        bus=buses_i,
        carrier="H2",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=costs.at["hydrogen storage tank", "capital_cost"],
    )


def attach_hydrogen_electrolysis(
        n: pypsa.components.Network,
        buses_i: pd.Index,
        costs: pd.core.frame.DataFrame,
    ) -> None:
    
    '''
    Attach hydrogen electrolyzers to all specified buses of network n based on the values specified in costs.
    '''

    n.add("Carrier","H2 electrolysis",nice_name="H2 Electrolysis")

    n.madd(
        "Link",
        buses_i,
        suffix=" Electrolysis",
        bus0=buses_i.str.removesuffix(" H2"),                   # AC buses
        bus1=buses_i,                                           # H2 buses
        carrier="H2 electrolysis",
        p_nom_extendable=True,
        efficiency=costs.at["electrolysis", "efficiency"],      # electricity demand
        capital_cost=costs.at["electrolysis", "capital_cost"],
        marginal_cost=costs.at["electrolysis", "marginal_cost"],
    )


def add_electrolysis_water_demand(
        n: pypsa.components.Network,
        costs: pd.core.frame.DataFrame,
    ) -> None:

    '''
    Add a water demand to all electrolyzers in network n. The demand is specified by the costs file.
    '''

    elec_i = n.links.loc[n.links.carrier == "H2 electrolysis"].index
    n.links.loc[elec_i,"bus2"] = n.links.loc[elec_i,"bus0"] + " water"
    n.links.loc[elec_i,"efficiency2"] = -costs.at["electrolysis", "water demand"]


def add_electrolysis_PPA_requirement(
        n: pypsa.components.Network,
        counter_i: pd.Index,
    ) -> None:

    '''
    Add the requirement to all electrolyzers in network n, that all electricity used must be covered by a PPA.
    This is ensured through the PPA counter(s) counter_i.
    '''

    elec_i = n.links.loc[n.links.carrier == "H2 electrolysis"].index
    n.links.loc[elec_i,"bus3"] = counter_i
    n.links.loc[elec_i,"efficiency3"] = -1  


def attach_hydrogen_fuel_cells(
        n: pypsa.components.Network,
        buses_i: pd.Index,
        costs: pd.core.frame.DataFrame,
    ) -> None:
    
    '''
    Attach hydrogen fuel cells to all specified buses of network n based on the values specified in costs.
    '''

    n.add("Carrier","H2 fuel cell",nice_name="H2 Fuel Cell")

    n.madd(
        "Link",
        buses_i,
        suffix=" Fuel Cell",
        bus0=buses_i,                                           # H2 buses
        bus1=buses_i.str.removesuffix(" H2"),                   # AC buses
        carrier="H2 fuel cell",
        p_nom_extendable=True,
        efficiency=costs.at["fuel cell", "efficiency"],      # electricity demand
        capital_cost=costs.at["fuel cell", "capital_cost"],
        marginal_cost=costs.at["fuel cell", "marginal_cost"],
    )


def attach_hydrogen_pipelines(
        n: pypsa.Network,
        connections: pd.core.frame.DataFrame,
        costs: list,
        transmission_efficiency: dict,
    ) -> None:

    '''
    Attach hydrogen pipelines to network n based on the connections.
    The pipeline costs are specified in the costs file.
    '''

    # set pipeline names
    names = connections.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    n.add("Carrier","H2 pipeline",nice_name="H2 Pipeline")

    # add pipelines to the network
    n.madd(
        "Link",
        names,
        bus0=connections.bus0.values + " H2",
        bus1=connections.bus1.values + " H2",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=connections.length.values,
        capital_cost=costs.at["H2 pipeline", "capital_cost"] * connections.length.values,
        carrier="H2 pipeline",
    )

    # setup the pipeline as bidirectional and lossy
    lossy_bidirectional_links(n, "H2 pipeline")

    # set the pipelines efficiency and the electricity required by the pipeline for compression
    set_length_based_efficiency(n, "H2 pipeline", " H2", transmission_efficiency)


def attach_hydrogen_export(
        n: pypsa.components.Network,
        export_buses: dict[str, pd.Index],
        loads: dict[str, float],
    ) -> None:

    '''
    Attach hydrogen export loads to the network n.
    The locations are specified in export_buses, with corresponding loads in loads.
    '''

    # check if all loads are mapped to locations
    if not all(key in export_buses for key in loads.keys()):
        logger.error(
            f"Not all hydrogen export locations defined in the config are specified in custom_locations.csv"
        )
        raise IndexError

    # set export names
    names = [f"{bus} - {name} H2 export" for name, bus in export_buses.items()]

    # attach export loads
    n.madd(
        "Load",
        names=names,
        bus=export_buses.values(),
        carrier="H2",
        p_set=[loads[name] for name in export_buses.keys()]
    )

def attach_water_network(
        n: pypsa.components.Network,
        costs: pd.core.frame.DataFrame,
        config: dict,
        location_fp: str,
    ) -> None:

    '''
    Attach water buses, links and stores to the network n,
    based on the extendable components listed in the config.
    The costs file determines the costs of the respective components.
    '''

    extendable = config["electricity"]["extendable_carriers"]

    if not (
        any("water" in link for link in extendable["Link"])
    ):
        logger.info(
            "No water components attached, since none were specified by the config."
        )
        return
    
    buses_i = attach_bus_network(n, carrier="water")

    if "seawater desalination" in extendable["Link"]:
        desal_buses = identify_buses_from_file(n, location_fp, "water", "seawater desalination")
        desal_buses, _ = merge_identical_locations(desal_buses)
        attach_seawater_desalination(n, desal_buses, costs)
    if "water pipeline" in extendable["Link"]:  
        connections = indetify_connected_buses(n)
        transmission_efficiency = config["sector"]["transmission_efficiency"]
        attach_water_pipelines(n, connections, costs, transmission_efficiency)
    if "H2 electrolysis" in extendable["Link"]:
        add_electrolysis_water_demand(n, costs)


def attach_seawater_desalination(
        n: pypsa.Network,
        buses: dict[str, pd.Index],
        costs: pd.core.frame.DataFrame,
    ) -> None:
    
    '''
    Attach seawater desalination plants to all specified buses of network n based on the values specified in costs.
    '''

    # set desalination names
    names = [f"{bus} seawater desalination - {name}" for name, bus in buses.items()]

    n.add("Carrier","Desalination",nice_name="Desalination")

    # add desalination plants to the network
    n.madd(
        "Link",
        names,
        bus0=[name.removesuffix(" water") for name in buses.values()],  # AC buses
        bus1=buses.values(),                                            # water buses
        carrier="Desalination",
        p_nom_extendable=True,
        efficiency=costs.at["seawater desalination", "efficiency"],     # electricity demand
        capital_cost=costs.at["seawater desalination", "capital_cost"],
        marginal_cost=costs.at["seawater desalination", "marginal_cost"],
    )


def attach_water_pipelines(
        n: pypsa.Network,
        connections: pd.core.frame.DataFrame,
        costs: list,
        transmission_efficiency: dict,
    ) -> None:

    '''
    Attach water pipelines to network n based on the connections.
    The pipeline costs are specified in the costs file.
    '''

    # set pipeline names
    names = connections.apply(lambda c: f"water pipeline {c.bus0}-{c.bus1}", axis=1)

    n.add("Carrier","water pipeline",nice_name="Water Pipeline")

    # add pipelines to the network
    n.madd(
        "Link",
        names,
        bus0=connections.bus0.values + " water",
        bus1=connections.bus1.values + " water",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=connections.length.values,
        capital_cost=costs.at["water pipeline", "capital_cost"] * connections.length.values,
        carrier="water pipeline",
    )

    # setup the pipeline as bidirectional and lossy
    lossy_bidirectional_links(n, "water pipeline")

    # set the pipelines efficiency and the electricity required by the pipeline for compression
    set_length_based_efficiency(n, "water pipeline", " water", transmission_efficiency)


def attach_PPA_network(
        n: pypsa.components.Network,
        costs: pd.core.frame.DataFrame,
        config: dict,
    ) -> None:

    '''
    Attach a PPA counter system to the network n,
    based on the extendable PPA generators listed in the config.
    The costs file determines the costs of the respective components.
    Additionally, the setup depends on the PPA section of the config.
    '''

    extendable = config["electricity"]["extendable_carriers"]

    if not extendable["PPA"]:
        logger.info(
            "No PPA components attached, since none were specified by the config."
        )
        return
    
    PPA_i = attach_bus_network(n, carrier="PPA generator")
    counter_i = attach_PPA_counters(n, config)

    for carrier in extendable["PPA"]:
        attach_PPA_generators(n, PPA_i, carrier, costs)
    connect_PPA_generators(n, PPA_i, counter_i, config)

    if "H2 electrolysis" in extendable["Link"]:
        add_electrolysis_PPA_requirement(n, counter_i)

    #if "battery" in extendable["Store"]:
    #    add_battery_PPA_storage(n, counter_i)

    if "battery RE" in extendable["Store"]:
        attach_batteries(n, PPA_i, costs)


def attach_PPA_generators(
        n: pypsa.components.Network,
        PPA_i: pd.Index,
        carrier: str,
        costs: pd.core.frame.DataFrame,
    ) -> None:

    '''
    Attach generators of type carrier to all specified buses of network n based on the values specified in costs.
    '''

    # set generator names
    names = PPA_i.str.removesuffix(" PPA generator") + f" {carrier}"

    n.add("Carrier",f"{carrier} PPA",nice_name=f"{carrier.capitalize()} PPA")

    # add generators to the network
    n.madd(
        "Generator",
        names=names,
        suffix=" PPA",
        bus=PPA_i,
        carrier=f"{carrier} PPA",
        p_max_pu=n.generators_t.p_max_pu[names],
        p_nom_extendable=True,
        p_nom_max=n.generators.loc[names].apply(lambda x: x.p_nom_max - x.p_nom, axis=1),
        capital_cost=costs.at[carrier, "capital_cost"],
        marginal_cost=costs.at[carrier, "marginal_cost"]
    )


def attach_PPA_counters(
        n: pypsa.components.Network,
        config: dict,
    ) -> pd.Index:
    
    '''
    Attach a PPA counter either locally or globally to all buses_i in network n depending on the config.
    '''

    if config["PPA"]["local_PPA"]:
        # add a PPA counter to each AC bus of the network, if the PPA can be use only localy
        counter_i = attach_bus_network(n, carrier="PPA counter")
    else:
        # add a single PPA counter, if the PPA can be used globally
        n.add(
            "Bus",
            name="PPA counter",
            carrier="PPA counter",
            x=n.buses.x.mean(),     # purely esthetic
            y=n.buses.x.mean(),     # purely esthetic
        )
        counter_i = "PPA counter"

    return counter_i


def connect_PPA_generators(
        n: pypsa.components.Network,
        PPA_i: pd.Index,
        counter_i: pd.Index,
        config: dict,
    ) -> None:

    '''
    Connect the PPA generator PPA_i to the PPA counter(s) counter_i and to either
    the main electricity network or an offgrid network based on the config.
    '''
    
    # check whether PPA operates offgrid or on the main grid
    AC_i = PPA_i.str.removesuffix(" PPA generator")

    n.add("Carrier","PPA Link",nice_name="PPA Link")

    # connect the PPA generator with the counter(s) and electricity network
    n.madd(
        "Link",
        names=AC_i + " PPA Link",
        carrier="PPA Link",
        bus0=PPA_i,
        bus1=AC_i,
        bus2=counter_i,
        p_nom=1e6, # Choose a number as low as possible but larger than any value that can realistically occure.
    )


def add_battery_PPA_storage(
        n: pypsa.components.Network,
        counter_i: pd.Index,
    ) -> None:


    '''
    Enable all batteries in the network n to store PPA electricty, by reducing the PPA counter(s) counter_i when charging
    and increasing the counter when discharging.
    '''

    # link the battery chargers to the PPA counter(s)
    charge_i = n.links.loc[n.links.carrier == "battery charger"].index
    n.links.loc[charge_i,"bus2"] = counter_i
    n.links.loc[charge_i,"efficiency2"] = -1

    # link the battery dischargers to the PPA counter(s)
    discharge_i = n.links.loc[n.links.carrier == "battery discharger"].index
    n.links.loc[discharge_i,"bus2"] = counter_i
    n.links.loc[discharge_i,"efficiency2"] = n.links.loc[discharge_i,"efficiency"]


def attach_offgrid_network(
        n: pypsa.components.Network,
        config: dict
    ) -> None:
    
    '''
    Attach an offgrid AC network to the network n if required by the config.
    The offgrid network matches the 
    '''
    
    if not config["PPA"]["offgrid"]:
        logger.info(
            "No offgrid network added, as non was required for this run by the config."
        )
        return

    extendable = config["electricity"]["extendable_carriers"]
    carrier = "AC offgrid"
    
    attach_bus_network(n, carrier)
    attach_offgrid_powerlines(n, carrier)

    if "battery" in extendable["Store"]:
        move_batteries_offgrid(n, carrier)

    if "H2 electrolysis" in extendable["Link"]:
        move_electrolysis_offgrid(n, carrier)

    if extendable["PPA"]:
        move_PPA_offgrid(n, carrier)


def attach_offgrid_powerlines(
        n: pypsa.components.Network,
        carrier: str,
    ) -> None:

    '''
    Attach new powerlines to the network n to the AC buses of type carrier.
    The powerlines match the network structure of the original powerlines but are not "build" yet.
    '''

    # identify all original AC lines
    line_i = n.lines.loc[n.lines.carrier == "AC"].index

    # copy the original lines and change them to the new type
    new_lines = n.lines.loc[line_i].copy().set_index(line_i + f" {carrier}")
    new_lines["carrier"] = carrier
    new_lines["bus0"] += f" {carrier}"
    new_lines["bus1"] += f" {carrier}"

    # set the new lines maximum nominal power to zero
    new_lines["s_nom"] = 0
    new_lines["num_parallel"] = 0

    # add the new lines to the network
    n.lines = pd.concat([n.lines, new_lines], sort=False)


def move_electrolysis_offgrid(
        n: pypsa.components.Network,
        carrier: str,
    ) -> None:

    '''
    Move all electrolyzers in network n from the AC buses of the main network to the corresponding buses of type carrier.
    '''

    # identify all electrolyzers
    elec_i = n.links.loc[n.links.carrier == "H2 electrolysis"].index

    # move all electrolyzers offgrid
    n.links.loc[elec_i, "bus0"] += f" {carrier}"


def move_PPA_offgrid(
        n: pypsa.components.Network,
        carrier: str,
    ) -> None:

    '''
    Move all PPA in network n offgrid, by disconnecting the PPA generators from the main network and
    reconnecting them to the corresponding buses of type carrier.
    '''

    # identify all PPA links
    PPA_i = n.links.loc[n.links.carrier == "PPA Link"].index

    # move all PPA links offgrid
    n.links.loc[PPA_i, "bus1"] += f" {carrier}"


def move_batteries_offgrid(
        n: pypsa.components.Network,
        carrier: str,
    ) -> None:

    '''
    Move all batteries in network n from the AC buses of the main network to the corresponding buses of type carrier.
    '''

    # identify all battery chargers and dischargers
    charge_i = n.links.loc[n.links.carrier == "battery charger"].index
    discharge_i = n.links.loc[n.links.carrier == "battery discharger"].index

    # move all battery chargers and dischargers offgrid
    n.links.loc[charge_i, "bus0"] += f" {carrier}"
    n.links.loc[discharge_i, "bus1"] += f" {carrier}"


def attach_bus_network(
        n: pypsa.components.Network,
        carrier: str   
    ) -> pd.Index:

    '''
    Identify all original AC and DC buses of the network n.
    Add new buses of the specified carrier to n that mirror the location of the original buses.
    Return the indices of the new buses.
    '''

    # identify all original AC and DC buses
    power_i = n.buses.loc[n.buses["carrier"].isin(["AC","DC"])].index

    # add new buses of type carrier, matching the location of the original buses
    bus_sub_dict = {k: n.buses[k].loc[power_i].values for k in ["x", "y", "country"]}
    new_i = n.madd("Bus", power_i + f" {carrier}", carrier=carrier, **bus_sub_dict)

    return new_i


def indetify_connected_buses(
        n: pypsa.Network 
    ) -> pd.core.frame.DataFrame:

    '''
    Identify all bus pairs in network n, that are connected by either an AC line or a DC link.
    Return the bus pairs including the length of the connection.
    '''

    def adjust_bus_order(row: pd.core.series.Series):
        
        '''
        Switches bus0 and bus1, if bus0 is larger (str comparison) than bus1.
        '''

        if row["bus0"] > row["bus1"]:
            row["bus0"], row["bus1"] = row["bus1"], row["bus0"]
        
        return row
        
    # determine AC and DC bus pairs
    attrs = ["bus0", "bus1", "length"]
    candidates = pd.concat(
        [n.lines[attrs], n.links.query('carrier=="DC"')[attrs]]
    ).reset_index(drop=True)

    # drop dublicates, regardless of the order of bus0 and bus1
    connections = candidates.apply(adjust_bus_order,axis=1).drop_duplicates(["bus0","bus1"])

    return connections


def identify_buses_from_file(
        n: pypsa.Network,
        fp: str,
        carrier: str,
        loc_type: str
    ) -> dict[str, pd.Index]:

    '''
    Identify the buses in network n of a specified carrier, that match the locations specified on the filepath fp
    that matches the required location type loc_type.
    '''

    # load all locations of type loc_type
    df = pd.read_csv(fp,index_col="name")
    locations = df.loc[df.type == loc_type,["lat","lon"]]

    # get all buses of type carrier
    buses = n.buses.loc[n.buses.carrier == carrier,["x","y"]]

    def identify_nearest_bus(
            location,
            buses,
        ) -> pd.Index:
        
        '''
        Identify the bus out of buses nearest to the location.
        '''

        distances = pypsa.geo.haversine(buses[["x","y"]],[location["lon"],location["lat"]])
        best_connection = buses.iloc[distances.argmin()].name

        return best_connection
    
    bus_map = {}

    # identify the nearest bus for every location
    for location in locations.index:
        bus_map[location] = identify_nearest_bus(locations.loc[location], buses)

    return bus_map


def merge_identical_locations(
        bus_map: dict[str, pd.Index],
        values: dict[str, float] = {},
        drop_without_val: bool = False,
    ) -> tuple[dict[str, pd.Index], dict[str, float]]:

    '''
    Merge locations which share the same bus in bus_map.
    If a dict of values is provided, merge them accordingly.
    Locations without a value are either assigned a value of 0 or dropped.
    '''

    new_map = {}
    new_values = {}

    # merge locations that share the same bus
    for name,bus in bus_map.items():
        if drop_without_val and name not in values:
            continue
        if bus not in new_map.keys():
            new_map[bus] = name
            new_values[name] = values.pop(name,0)
        else:
            old_name = new_map.pop(bus)
            new_map[bus] = f"{old_name}-{name}"
            new_values[new_map[bus]] = new_values.pop(old_name) + values.pop(name,0)


    # return to the original bus_map order
    new_map = {name: bus for bus, name in new_map.items()}

    # keep values of unused locations
    new_values.update(values)

    return new_map, new_values


def apply_line_extension(
        n: pypsa.Network,
        line_extension_fp: str,
    ) -> None:

    '''
    Set the nominal power of all AC lines in network n to the expanded nominal power
    specified in the line_extension_fp file.
    '''

    # load line extensions
    line_extensions = pd.read_csv(line_extension_fp, index_col="Line", dtype={0: str})

    # calculate new num_parallel
    n.lines.num_parallel *= line_extensions.s_nom_opt / n.lines.s_nom

    # set lines to the expanded nominal power
    n.lines.s_nom = line_extensions.s_nom_opt


def setup_mode_config(
        mode: str,
        config: dict,
    ) -> None:

    '''
    Change the config to match the selected mode.
    '''

    extendable = config["electricity"]["extendable_carriers"]

    match mode:
        case "base":
            # considere only the base electricity network and don't add any additional components or export loads
            for carrier in extendable.values():
                carrier.clear()
        case "free":
            # allow all components
            pass
        case "desal":
            # ensure that electrolyzers are located next to desalination plants by removing water pipelines
            extendable["Link"].remove("water pipeline")
        case "export":
            # ensure that electrolyzers are located next to the export locations by removing H2 pipelines
            extendable["Link"].remove("H2 pipeline")
        case "gen":
            # ensure that electrolyzers are located next to the new generators by only allowing for local PPAs
            config["PPA"]["local_PPA"] = True
        case _:
            # handle any case that is not addressed by the other cases
            logger.error(
                f"Mode '{mode}' not recognised. Please provide a supported mode.\n"
                f"Currently supported modes: 'base', 'free', 'desal', 'export', 'gen'"
            )
            raise ValueError(f"Mode '{mode}' not supported!")
        
    logger.info(
        f"Used mode '{mode}' to setup the config."
    )


def lossy_bidirectional_links(
        n: pypsa.components.Network,
        carrier: str
    ):

    "Split bidirectional links into two unidirectional links to include transmission losses."

    carrier_i = n.links.query("carrier == @carrier").index

    if carrier_i.empty:
        return
    
    logger.info(
        f"Splitting bidirectional links with the carrier {carrier}"
    )

    n.links.loc[carrier_i, "p_min_pu"] = 0

    rev_links = (
        n.links.loc[carrier_i].copy().rename({"bus0": "bus1", "bus1": "bus0"}, axis=1)
    )
    rev_links["length_original"] = rev_links["length"]
    rev_links["capital_cost"] = 0
    rev_links["length"] = 0
    rev_links["reversed"] = True
    rev_links.index = rev_links.index.map(lambda x: x + "-reversed")

    n.links = pd.concat([n.links, rev_links], sort=False)
    n.links["reversed"] = n.links["reversed"].fillna(False).infer_objects(copy=False)
    n.links["length_original"] = n.links["length_original"].fillna(n.links.length)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_extra_components", simpl="", clusters=10)

    configure_logging(snakemake)

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    transmission_efficiency = snakemake.params.transmission_efficiency
    config = snakemake.config

    costs = load_costs(
        snakemake.input.tech_costs,
        config["costs"],
        config["electricity"],
        Nyears,
        custom_cost_path = snakemake.input.custom_costs
    )

    mode = snakemake.wildcards.mode
    export_locations = snakemake.wildcards.export_locations.removeprefix("['").removesuffix("']").split("', '")
    export_loads = snakemake.wildcards.export_loads.removeprefix("[").removesuffix("]").split(", ")

    config["PPA"] = {"local_PPA": False}
    config["PPA"]["offgrid"] = snakemake.wildcards.offgrid == "True" # convert the wildcard from str to bool
    config["H2_export"] = {loc: float(load) for loc, load in zip(export_locations, export_loads)}
    
    if snakemake.params.extended_lines:
        apply_line_extension(n, snakemake.input.line_extensions)
    
    setup_mode_config(mode, config)
    attach_storageunits(n, costs, config)
    attach_stores(n, costs, config)
    attach_hydrogen_network(n, costs, config, snakemake.input.custom_locations)
    attach_water_network(n, costs, config, snakemake.input.custom_locations)
    attach_PPA_network(n, costs, config)
    attach_offgrid_network(n, config)

    add_nice_carrier_names(n, config=snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
