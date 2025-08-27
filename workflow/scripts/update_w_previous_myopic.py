"""Add previous solutions to the next optimisation (myopic mode)

- add all brownfield from previous solution
- add_existing_myopic is still run because it allows to have
    (pre-) construction pipelines extending beyond the base year
- add previous optimal as brownfield with previous build year
- correct technical potential available for newbuilds (rm prev solution)
- retire all capacities that have reached end of life

"""

# SPDX-FileCopyrightText:2025 The PyPSA-China-PIK Authors
#
# SPDX-License-Identifier: MIT


import pypsa
import logging
import pandas as pd

from _helpers import mock_snakemake, configure_logging
from add_electricity import load_costs
from add_existing_baseyear import add_coal_retrofit
from constants import YEAR_HRS

logger = logging.getLogger(__name__)


EXCLUDE_CARRIERS = ["hydro_inflow", "", "stations"]


def _find_last_year(yr: int, years: list[int]) -> int:
    """find previous time horizon
    Args:
        yr (int): current time horizon
        years (list): all time horizons
    """
    years = sorted([int(y) for y in years])
    prev_idx = max(0, years.index(yr) - 1)
    return years[prev_idx]


def add_previous_brownfield(network: pypsa.Network, previous_network: pypsa.Network, threshold=0.0):
    """Add previous brownfield capacities from previous_network to network.
    OVERWRITES existing brownfield capacities (e.g. from add_existing_baseyear_myopic)


    Args:
        network (pypsa.Network): the network for this time horizon optimisation
        previous_network (pypsa.Network): the network optimised at the last time horizon
        threshold (float, optional): min capacity to consider for brownfield. Defaults to 0.0.
    """

    for component in ["generators", "links", "stores", "storage_units"]:
        prefix = "e" if component == "stores" else "p"

        # previously network brownfield
        query = f"{prefix}_nom_extendable==False and abs({prefix}_nom_opt) > {threshold}"
        query += " and carrier not in @EXCLUDE_CARRIERS"
        comp = getattr(previous_network, component).query(query).copy()

        # add previous solution to network (overwrites brownfield from add_existing_baseyear_myopic)
        getattr(network, component).loc[comp.index] = comp


def add_previous_optima(
    network: pypsa.Network, previous_network: pypsa.Network, previous_year: int, threshold=0.0
):
    """Add optimal previous capacities from previous_network to network.


    Args:
        network (pypsa.Network): the network for this time horizon optimisation
        previous_network (pypsa.Network): the network optimised at the last time horizon
        previous_year (int): the previous time horizon year
        threshold (float, optional): min capacity to consider for brownfield. Defaults to 0.0.
    """
    component_settings = {
        "generators": {
            "join_col": "carrier",
            "attrs_to_fix": ["p_min_pu", "p_max_pu"],
        },
        "links": {
            "join_col": "carrier",
            "attrs_to_fix": ["p_min_pu", "p_max_pu", "efficiency", "efficiency2"],
        },
        "stores": {
            "join_col": "carrier",
            "attrs_to_fix": [],
        },
    }

    for component in ["generators", "links", "stores", "storage_units"]:
        prefix = "e" if component == "stores" else "p"

        # previously solved
        query = f"{prefix}_nom_extendable==True and abs({prefix}_nom_opt) > {threshold}"
        query += " and carrier not in @EXCLUDE_CARRIERS"
        comp = getattr(previous_network, component).query(query).copy()

        comp[f"{prefix}_nom"] = comp[f"{prefix}_nom_opt"]
        comp[f"{prefix}_nom_extendable"] = False
        comp["build_year"] = previous_year

        # correct technical potentials (subtract brownfield from avail)
        new_max = comp[f"{prefix}_nom_max"] - comp[f"{prefix}_nom_opt"]
        getattr(network, component).loc[comp.index, f"{prefix}_nom_max"] = new_max

        # combine any existing capacities for that year and brownfield
        # this needs to be done after tech potential correction
        #   because tech potential is corrected for in add_existing_baseyear
        # Align indexes and sum p_nom_max for overlapping indexes
        original_index = comp.index.copy()
        comp.index = (comp.index + f"-{previous_year}").str.replace(
            f"-{previous_year}-{previous_year}", f"-{previous_year}"
        )
        brownfield_exist = getattr(network, component).query("index in @comp.index")
        comp.loc[brownfield_exist.index, f"{prefix}_nom"] += brownfield_exist[f"{prefix}_nom"]

        # add previous solution to network
        getattr(network, component).loc[comp.index + f"-{previous_year}"] = comp

        # fix missing time dependent profiles
        # now add the dynamic attributes not carried over by n.add (per unit avail etc)
        for missing_attr in component_settings[component]["attrs_to_fix"]:
            df_t = getattr(network, component + "_t")[missing_attr]
            to_copy = [idx for idx in original_index if idx in df_t.columns]
            additions = [idx + f"-{previous_year}" for idx in to_copy]
            df_t.loc[:, additions] = df_t.loc[:, to_copy]


def correct_retrofitted_potentials(network: pypsa.Network):
    """Remove retrofitted coal from the p_nom_max of coal brownfield

    Args:
        network (pypsa.Network): _description_
    """

    # TODO loop over components
    query = "carrier == 'coal' and p_nom !=0 and not index.str.contains('fuel')"
    retrofitted_q = "carrier=='coal ccs' and index.str.contains('retrofit') and not index.str.contains('fuel') and p_nom !=0"
    coal_brownfield = network.generators.query(query)
    ccs_retroffited = network.generators.query(retrofitted_q)

    updated = coal_brownfield.merge(
        ccs_retroffited[["p_nom", "bus", "build_year"]],
        on=["bus", "build_year"],
        suffixes=("", "_retro"),
    )
    updated["p_nom_max"] -= updated["p_nom_retro"]

    network.generators.loc[updated.index, "p_nom_max"] = updated["p_nom_max"]
    network.generators.loc[updated.index, "p_nom"] = updated["p_nom_max"]
    network.generators.loc[updated.index, "p_nom_min"] = 0

    # do the same for CHP
    query = "carrier == 'CHP coal' and p_nom !=0 and not index.str.contains('fuel')"
    retrofitted_q = "carrier=='CHP coal CCS' and index.str.contains('retrofit') and p_nom !=0"
    coal_brownfield = network.links.query(query)
    ccs_retroffited = network.links.query(retrofitted_q)

    updated = coal_brownfield.merge(
        ccs_retroffited[["p_nom", "bus1", "build_year"]],
        on=["bus1", "build_year"],
        suffixes=("", "_retro"),
    )
    updated["p_nom_max"] -= updated["p_nom_retro"]

    network.generators.loc[updated.index, "p_nom_max"] = updated["p_nom_max"]
    network.generators.loc[updated.index, "p_nom"] = updated["p_nom_max"]
    network.generators.loc[updated.index, "p_nom_min"] = 0


def remove_end_of_life(network: pypsa.Network, year: int):
    """Remove brownfield assets that have reached end of life

    Args:
        network (pypsa.Network): the pypsa network
        year (int): the current optimisation year
    """

    # remove generators, links and stores that are at the end of their life
    for component in ["generators", "links", "stores", "storage_units"]:
        comp = getattr(network, component)
        comp = comp[comp.build_year + comp.lifetime < year]
        network.remove(component, comp.index)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "update_brownfield_with_solved",
            co2_pathway="exp175default",
            planning_horizons="2030",
            topology="current+FCG",
            heating_demand="positive",
            configfiles="config/myopic.yml",
        )

    configure_logging(snakemake)
    config = snakemake.config
    yr = int(snakemake.wildcards.planning_horizons)
    years = sorted([int(y) for y in config["scenario"]["planning_horizons"]])
    baseyear = min(years)

    network = pypsa.Network(snakemake.input.network)
    if yr != baseyear:
        prev_year = _find_last_year(yr, years)
        prev_solution = pypsa.Network(snakemake.input.solved_network)
        min_capacity = config["existing_capacities"]["threshold_capacity"]

        remove_end_of_life(prev_solution, yr)

        add_previous_brownfield(network, prev_solution, min_capacity)

        add_previous_optima(network, prev_solution, prev_year, min_capacity)

        remove_end_of_life(network, yr)

        if config["Techs"].get("coal_ccs_retrofit", True):
            correct_retrofitted_potentials(network)

            tech_costs = snakemake.input.tech_costs
            n_years = network.snapshot_weightings.generators.sum() / YEAR_HRS
            costs = load_costs(tech_costs, config["costs"], config["electricity"], yr, n_years)

            add_coal_retrofit(network, costs, yr, config)

    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    network.export_to_netcdf(snakemake.output[0], compression=compression)

    logger.info("Update existing capacities with previous optima")
