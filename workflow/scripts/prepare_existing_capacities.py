"""
Functions to prepare existing assets for the network

SHORT TERM FIX until PowerPlantMatching is implemented
- required as split from add_existing_baseyear for remind compat
"""

# TODO improve docstring
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from _helpers import configure_logging, mock_snakemake
from _pypsa_helpers import determine_simulation_timespan
from readers import load_costs
from constants import YEAR_HRS

logger = logging.getLogger(__name__)
idx = pd.IndexSlice
spatial = SimpleNamespace()

CARRIER_MAP = {
    "coal": "coal power plant",
    "CHP coal": "central coal CHP",
    "CHP gas": "central gas CHP",
    "OCGT": "gas OCGT",
    "CCGT": "gas CCGT",
    "solar": "solar",
    "solar thermal": "central solar thermal",
    "onwind": "onwind",
    "offwind": "offwind",
    "coal boiler": "central coal boiler",
    "ground heat pump": "central ground-sourced heat pump",
    "nuclear": "nuclear",
    "PHS": "PHS",
    "biomass": "biomass",
    "hydro": "hydro",
}


def build_capacities(powerplant_table: pd.DataFrame, cost_data: pd.DataFrame) -> pd.DataFrame:
    """Read existing capacities from csv files and format them
    Args:
        powerplant_table (pd.DataFrame):power plant data
        cost_data (pd.DataFrame): technoeconomic cost data
    Returns:
        pd.DataFrame: DataFrame with existing capacities
    """

    powerplant_table.rename(columns={"capacity": "Capacity", "Type": "Fueltype"}, inplace=True)

    df = powerplant_table.merge(cost_data, how="left", left_on=["Tech"], right_index=True)
    df2 = powerplant_table.merge(cost_data, how="left", left_on=["Fueltype"], right_index=True)
    missed = df.capital_cost.isna()
    df.loc[missed, :] = df2.loc[missed, :]
    still_missed = df.loc[df.capital_cost.isna()]

    if not still_missed.empty:
        missed_techs = still_missed.Fueltype.unique() + still_missed.Tech.unique()
        raise ValueError(
            f"Cost Data could not be found for requested existing techs or types {missed_techs}"
        )

    carrier = {k: v for k, v in CARRIER_MAP.items() if k in techs}
    df["Tech"] = df["Fueltype"].map(carrier)
    df["DateIn"] = df["grouping_year"]
    df["lifetime"] = df["lifetime"].astype(int)
    df["DateOut"] = df["DateIn"].astype(int) + df["lifetime"]
    df["ID"] = df["cluster"] + "-" + df["Fueltype"] + "-" + df["grouping_year"].astype(str)

    return df.set_index("ID", drop=True)


def convert_CHP_to_poweronly(capacities: pd.DataFrame) -> pd.DataFrame:
    """Convert CHP capacities to power-only capacities by removing the heat part

    Args:
        capacities (pd.DataFrame): DataFrame with existing capacities
    Returns:
        pd.DataFrame: DataFrame with converted capacities
    """
    # Convert CHP to power-only by removing the heat part
    chp_mask = capacities.Tech.str.contains("CHP")
    capacities.loc[chp_mask, "Fueltype"] = (
        capacities.loc[chp_mask, "Fueltype"]
        .str.replace("coal CHP", "coal")
        .replace("CHP coal", "coal")
        .str.replace("CHP gas", "gas CCGT")
        .replace("gas CHP", "gas CCGT")
    )
    # update the Tech field based on the converted Fueltype
    capacities.loc[chp_mask, "Tech"] = (
        capacities.loc[chp_mask, "Fueltype"]
        .str.replace(" CHP", "")
        .str.replace("CHP ", " ")
        .str.replace("gas ", "")
        .str.replace("coal power plant", "coal")
    )
    return capacities


def load_powerplants(path: os.PathLike, plan_year: int) -> pd.DataFrame:
    """Load powerplants, filter by existing and retired, format as table

    Args:
        path (os.PathLike): path to the cleaned powerplant csv
        plan_year (int): the model year

    Returns:
        pd.DataFrame: DataFrame with cleaned powerplant data
    """

    ppl_data = pd.read_csv(path, index_col=0).reset_index()

    ppl_data["Retired year"].fillna(1e5, inplace=True)
    ppl_data = ppl_data.query("`Start year`<= @plan_year and `Retired year` > @plan_year")

    ppl_table = (
        (
            ppl_data.pivot_table(
                columns="grouping_year",
                index=["cluster", "Type"],
                values="Capacity (MW)",
                aggfunc="sum",
            )
            .fillna(0)
            .astype(int)
        )
        .stack()
        .reset_index()
    )
    ppl_table["Tech"] = ppl_table.Type.map(CARRIER_MAP)

    return ppl_table.rename(columns={0: "Capacity"})


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "prepare_baseyear_capacities",
            topology="current+FCG",
            co2_pathway="SSP2-PkBudg1000-pseudo-coupled",
            planning_horizons="2025",
            heating_demand="positive",
            # configfiles="resources/tmp/pseudo_coupled.yml",
        )

    configure_logging(snakemake, logger=logger)

    config = snakemake.config
    params = snakemake.params
    year = int(snakemake.wildcards.planning_horizons)
    # reference pypsa cost (lifetime) year is simulation Baseyar
    baseyear = min([int(y) for y in config["scenario"]["planning_horizons"]])
    tech_costs = snakemake.input.tech_costs
    data_paths = {k: v for k, v in snakemake.input.items()}

    n_years = determine_simulation_timespan(snakemake.config, baseyear)
    costs = load_costs(tech_costs, config["costs"], config["electricity"], baseyear, n_years)

    ppl_table = load_powerplants(snakemake.input.cleaned_ppls, year)

    techs = snakemake.params["techs"]
    # TODO check whether it shouldn't use the carrier map
    ppl_table = ppl_table.query("Tech in @techs or Type in @techs")

    existing_capacities = build_capacities(ppl_table, costs)

    # TODO add renewables
    if params.CHP_to_elec:
        existing_capacities = convert_CHP_to_poweronly(existing_capacities)

    if existing_capacities.empty or existing_capacities.lifetime.isna().any():
        logger.warning(
            f"The following assets have no lifetime assigned and are for ever lived: \n{existing_capacities[existing_capacities.lifetime.isna()]}"
        )

    existing_capacities.to_csv(snakemake.output.installed_capacities)

    logger.info(f"Installed capacities saved to {snakemake.output.installed_capacities}")
