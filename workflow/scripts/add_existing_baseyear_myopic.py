"""Add solution to all brownfield capacity files

The approach is based on adding previously solved capacities to the brownfield
 capacities in prepare_myopic_capacities and adding them to the network.
 The downside is that "all" components have to be exported and reimported.

Another possible approach is to copy the components from the previous solution
This was the original PyPSA-China approach, with p_max_pu copied from n.generators_t etc.
However more work is then needed to respect technical potentials, especially with multiple VRE grades.
"""

# TODO ADD CCS TO EXISTING BASE YEAR TECHS

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


def add_build_year_to_new_assets(n: pypsa.Network, baseyear: int):
    """add a build year to new assets

    Args:
        n (pypsa.Network): the network
        baseyear (int): year in which optimized assets are built
    """

    # Give assets with lifetimes and no build year the build year baseyear
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        assets = c.df.index[(c.df.lifetime != np.inf) & (c.df[attr + "_nom_extendable"] is True)]

        # add -baseyear to name
        renamed = pd.Series(c.df.index, c.df.index)
        renamed[assets] += "-" + str(baseyear)
        c.df.rename(index=renamed, inplace=True)

        assets = c.df.index[
            (c.df.lifetime != np.inf)
            & (c.df[attr + "_nom_extendable"] is True)
            & (c.df.build_year == 0)
        ]
        c.df.loc[assets, "build_year"] = baseyear

        # rename time-dependent
        selection = n.component_attrs[c.name].type.str.contains("series") & n.component_attrs[
            c.name
        ].status.str.contains("Input")
        for attr in n.component_attrs[c.name].index[selection]:
            c.pnl[attr].rename(columns=renamed, inplace=True)


def update_edges(n: pypsa.Network, prev_edges: pd.DataFrame, lossy_edges: bool, carrier="AC"):
    """Update the p_nom of the HV network edges based on the previous brownfield state
    Args:
        n (pypsa.Network): the pypsa network to update with a previous state
        prev_edges (pd.DataFrame): bronwfield edges (myopic output or brownfield input)
        lossy_edges (bool): whether edges are lossy (config["line_losses"] adds 'positive' suffix)
        carrier (Optional, str): the link carrier. Defaults to AC
    """
    if prev_edges.empty:
        return

    connects_mask = n.links.query(
        "bus0.map(@n.buses.carrier) == "
        "bus1.map(@n.buses.carrier) & "
        f"carrier == '{carrier}' & "
        "not index.str.contains('reverse')"
    ).index

    edges = prev_edges.copy()
    suffix = " positive" if lossy_edges else ""
    edges.index = edges["bus0"] + "-" + edges["bus1"] + suffix

    # Check if indices align between connects and edges
    if not edges.index.isin(connects_mask).all():
        raise ValueError(
            "Indices between connects and edges do not align. Please check input data."
        )

    n.links.loc[connects_mask, ["bus0", "bus1", "p_nom"]] = edges


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_existing_baseyear_myopic",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2030",
            configfiles="config/myopic.yml",
            heating_demand="positive",
        )

    configure_logging(snakemake, logger=logger)

    vre_techs = ["solar", "onwind", "offwind"]

    config = snakemake.config
    tech_costs = snakemake.input.tech_costs
    cost_year = int(snakemake.wildcards["planning_horizons"])
    baseyear = snakemake.params["baseyear"]
    data_paths = {k: v for k, v in snakemake.input.items()}

    n = pypsa.Network(snakemake.input.network)
    n_years = n.snapshot_weightings.generators.sum() / YEAR_HRS

    existing_capacities = pd.read_csv(snakemake.input.installed_capacities, index_col=0)
    existing_capacities = filter_capacities(existing_capacities, cost_year)
    prev_edges = {}
    for edge_carrier in snakemake.params.edge_carriers:
        prev_edges[edge_carrier] = pd.read_csv(
            snakemake.input[f"edges_{edge_carrier}"], header=None, names=["bus0", "bus1", "p_nom"]
        )
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    if snakemake.params["add_baseyear_to_assets"]:
        # call before adding new assets
        add_build_year_to_new_assets(n, baseyear)

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
    for edge_carrier in prev_edges:
        update_edges(n, prev_edges[edge_carrier], config["line_losses"], edge_carrier)

    if config["Techs"].get("coal_ccs_retrofit", False):
        add_coal_retrofit(n, costs, cost_year)

    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    n.export_to_netcdf(snakemake.output[0], compression=compression)

    logger.info("Existing capacities successfully added to network")
