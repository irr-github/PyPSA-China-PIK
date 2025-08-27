"""Add solution to all brownfield capacity files

The approach is based on adding previously solved capacities to the brownfield
 capacities in prepare_myopic_capacities and adding them to the network.
 The downside is that "all" components have to be exported and reimported.

Another possible approach is to copy the components from the previous solution
This was the original PyPSA-China approach, with p_max_pu copied from n.generators_t etc.
However more work is then needed to respect technical potentials, especially with multiple VRE grades.
"""

# SPDX-FileCopyrightText: 2025 The PyPSA-China-PIK Authors
#
# SPDX-License-Identifier: MIT


import logging
import pypsa
import pandas as pd

from _helpers import configure_logging, mock_snakemake
from constants import YEAR_HRS
from add_electricity import load_costs
from add_existing_baseyear import (
    add_coal_retrofit,
    add_existing_vre_capacities,
    add_power_capacities_installed_before_baseyear,
    filter_capacities,
)
from readers import read_edges


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_existing_baseyear_myopic",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2025",
            configfiles="config/myopic.yml",
            heating_demand="positive",
        )

    configure_logging(snakemake, logger=logger)

    vre_techs = ["solar", "onwind", "offwind"]

    config = snakemake.config

    if config["existing_capacities"].get("collapse_years", False):
        # Not compatible due to coal retrofit (fixable) and removal of overly old/expired data
        raise ValueError(
            "collapse_years option not compatible with myopic pathway, please switch it off!"
        )

    tech_costs = snakemake.input.tech_costs
    cost_year = int(snakemake.wildcards["planning_horizons"])
    data_paths = {k: v for k, v in snakemake.input.items()}

    n = pypsa.Network(snakemake.input.network)
    n_years = n.snapshot_weightings.generators.sum() / YEAR_HRS

    existing_capacities = pd.read_csv(snakemake.input.installed_capacities, index_col=0)
    existing_capacities = filter_capacities(existing_capacities, cost_year)

    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    vre_caps = existing_capacities.query("Tech in @vre_techs | Fueltype in @vre_techs")
    # vre_caps.loc[:, "Country"] = coco.CountryConverter().convert(["China"], to="iso2")
    vres = add_existing_vre_capacities(n, costs, vre_caps, config)
    # TODO: fix bug, installed has less vre/wind cap than vres.
    installed = pd.concat(
        [existing_capacities.query("Tech not in @vre_techs & Fueltype not in @vre_techs"), vres],
        axis=0,
    )

    # add to the network
    add_power_capacities_installed_before_baseyear(n, costs, config, installed)

    if config["Techs"].get("coal_ccs_retrofit", True):
        # make the coal brownfield extendable (constrain in solve constraints)
        query = "carrier == 'coal' and p_nom !=0 and not index.str.contains('fuel')"
        coal_brownfield = n.generators.query(query)
        n.generators.loc[coal_brownfield.index, "p_nom_extendable"] = True
        coal_CHP_brownfield = n.links.query("carrier == 'CHP coal' & p_nom !=0")
        n.links.loc[coal_CHP_brownfield.index, "p_nom_extendable"] = True

    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    n.export_to_netcdf(snakemake.output[0], compression=compression)

    logger.info("Existing capacities successfully added to network")
