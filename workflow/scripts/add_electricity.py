"""
Misc collection of functions supporting network prep
    still to be cleaned up
"""

import logging
from os import PathLike

import geopandas as gpd

import pandas as pd
import pypsa
from _helpers import mock_snakemake, configure_logging
from functions import calculate_annuity
from _pypsa_helpers import (
    assign_locations,
    make_periodic_snapshots,
    shift_profile_to_planning_year,
    rename_techs
)
from readers import load_costs
from constants import NICE_NAMES_DEFAULT


logger = logging.getLogger(__name__)


# TODO understand why this is in make_summary but not in the main optimisation
def update_transmission_costs(n, costs, length_factor=1.0):
    # TODO: line length factor of lines is applied to lines and links.
    # Separate the function to distinguish.

    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "capital_cost"]
    )

    if n.links.empty:
        return

    dc_b = n.links.carrier == "DC"

    # If there are no dc links, then the 'underwater_fraction' column
    # may be missing. Therefore we have to return here.
    if n.links.loc[dc_b].empty:
        return

    costs = (
        n.links.loc[dc_b, "length"]
        * length_factor
        * (
            (1.0 - n.links.loc[dc_b, "underwater_fraction"])
            * costs.at["HVDC overhead", "capital_cost"]
            + n.links.loc[dc_b, "underwater_fraction"] * costs.at["HVDC submarine", "capital_cost"]
        )
        + costs.at["HVDC inverter pair", "capital_cost"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs


def add_missing_carriers(n: pypsa.Network, carriers: list | set) -> None:
    """Function to add missing carriers to the network without raising errors.

    Args:
        n (pypsa.Network): the pypsa network object
        carriers (list | set): a list of carriers that should be included
    """
    missing_carriers = set(carriers) - set(n.carriers.index)
    if len(missing_carriers) > 0:
        n.add("Carrier", missing_carriers)


# TODO figure out whether still relevant
def sanitize_carriers(n: pypsa.Network, config: dict) -> None:
    """Sanitize the carrier information in a PyPSA Network object.

    The function ensures that all unique carrier names are present in the network's
    carriers attribute, and adds nice names and colors for each carrier according
    to the provided configuration dictionary.

    Args:
        n (pypsa.Network): PyPSA Network object representing the electrical power system.
        config (dict): A dictionary containing configuration information, specifically the
               "plotting" key with "nice_names" and "tech_colors" keys for carriers.
    """
    # update default nice names w user settings
    nice_names = NICE_NAMES_DEFAULT.update(config["plotting"].get("nice_names", {}))
    for c in n.iterate_components():
        if "carrier" in c.df:
            add_missing_carriers(n, c.df.carrier)

    # sort the nice names to match carriers and fill missing with "ugly" names
    carrier_i = n.carriers.index
    nice_names = pd.Series(nice_names).reindex(carrier_i).fillna(carrier_i.to_series())
    # replace empty nice names with nice names
    n.carriers.nice_name.where(n.carriers.nice_name != "", nice_names, inplace=True)

    # TODO make less messy, avoid using map
    tech_colors = config["plotting"]["tech_colors"]
    colors = pd.Series(tech_colors).reindex(carrier_i)
    # try to fill missing colors with tech_colors after renaming
    missing_colors_i = colors[colors.isna()].index
    colors[missing_colors_i] = missing_colors_i.map(lambda x: rename_techs(x, nice_names)).map(
        tech_colors
    )
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = n.carriers.color.where(n.carriers.color != "", colors)



if "snakemake" not in globals():
    snakemake = mock_snakemake(
        "prepare_networks",
        topology="current+FCG",
        co2_pathway="exp175default",
        # co2_pathway="SSP2-PkBudg1000-pseudo-coupled",
        planning_horizons=2025,
        heating_demand="positive",
        # configfiles="resources/tmp/pseudo-coupled.yaml",
    )

    configure_logging(snakemake)

    # if not node_cfg["split_provinces"]:
    #     assign_mode = "simple"
    #     cleaned["node"] = cleaned[ADM_LVL1]
    # elif assign_mode == "simple":
    #     splits_inv = {}  # invert to get admin2 -> node
    #     for admin1, splits in node_cfg["splits"].items():
    #         splits_inv.update({vv: admin1 + "_" + k for k, v in splits.items() for vv in v})
    #     cleaned["node"] = cleaned[ADM_LVL2].map(splits_inv).fillna(cleaned[ADM_LVL1])
    # else:
    #     if config["fetch_regions"]["simplify_tol"]["land"] > 0.05:
    #         logger.warning(
    #             "Using GPS assignment for existing capacities with land simplify_tol > 0.05. "
    #             "This may lead to inaccurate assignments (eg. Shanxi vs InnerMongolia coal power)."
    #         )
    #     cleaned["node"] = assign_node_from_gps(cleaned, nodes)