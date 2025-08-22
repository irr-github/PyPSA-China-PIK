""" Add solution to all brownfield capacity files"""
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
    add_paid_off_capacity,
    add_power_capacities_installed_before_baseyear,
    filter_capacities
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

def update_edges(n: pypsa.Network, solved_edges: pd.DataFrame):

    connects :pd.DataFrame = n.links.query(
        "bus0.map(@n_p.buses.carrier) == "
        "bus1.map(@n_p.buses.carrier) & "
        "carrier == 'AC' & "
        "not index.str.contains('reverse')"
    )[["bus0", "bus1", "p_nom_opt"]]

    edges = solved_edges.reset_index()
    edges.index = edges["bus0"] + "-" + edges["bus1"]

    # TODO fix

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "add_existing_baseyear_myopic",
            topology="current+FCG",
            # co2_pathway="exp175default",
            co2_pathway="SSP2-PkBudg1000-pseudo-coupled",
            planning_horizons="2040",
            configfiles="resources/tmp/pseudo_coupled.yml",
            # heating_demand="positive",
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
    if snakemake.params["add_baseyear_to_assets"]:
        # call before adding new assets
        add_build_year_to_new_assets(n, baseyear)

    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    existing_capacities = pd.read_csv(snakemake.input.installed_capacities, index_col=0)
    existing_capacities = filter_capacities(existing_capacities, cost_year)

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

    if config["Techs"].get("coal_ccs_retrofit", False):
        add_coal_retrofit(n, costs, cost_year)

    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    n.export_to_netcdf(snakemake.output[0], compression=compression)

    logger.info("Existing capacities successfully added to network")
