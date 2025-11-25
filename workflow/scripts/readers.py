"""file reading support functions"""

import os
import pandas as pd
import geopandas as gpd
import logging

from constants import PROV_NAMES, PROV_RENAME_MAP
from functions import calculate_annuity

logger = logging.getLogger(__name__)


def load_costs(
    tech_costs: os.PathLike,
    cost_config: dict,
    elec_config: dict,
    cost_year: int,
    n_years: int,
    econ_lifetime=40,
) -> pd.DataFrame:
    """Load techno-economic data for technologies. Calculate the anualised capex costs
      and OM costs for the technologies based on the input data

    Args:
        tech_costs (PathLike): the csv containing the costs
        cost_config (dict): the snakemake china cost config
        elec_config (dict): the snakemake china electricity config
        cost_year (int): the year for which the costs are retrived
        n_years (int): the # of years represented by the snapshots/investment period
        econ_lifetime (int, optional): the max lifetime over which to discount. Defaults to 40.

    Returns:
        pd.DataFrame: costs dataframe in [CURRENCY] per MW_ ... or per MWh_ ...
    """
    idx = pd.IndexSlice

    # set all asset costs and other parameters
    costs = pd.read_csv(tech_costs, index_col=list(range(3))).sort_index()
    costs.fillna(" ", inplace=True)
    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= cost_config["USD2013_to_EUR2013"]
    costs.loc[costs.unit.str.contains("USD"), "value"] *= cost_config["USD2013_to_EUR2013"]

    cost_year = float(cost_year)
    costs = (
        costs.loc[idx[:, cost_year, :], "value"]
        .unstack(level=2)
        .groupby("technology")
        .sum(min_count=1)
    )

    # TODO set default lifetime as option
    if "discount rate" not in costs.columns:
        costs.loc[:, "discount rate"] = cost_config["discountrate"]
    costs = costs.fillna(
        {
            "CO2 intensity": 0,
            "FOM": 0,
            "VOM": 0,
            "discount rate": cost_config["discountrate"],
            "efficiency": 1,
            "hist_efficiency": 1,  # represents brownfield efficiency state, only useful for links
            "fuel": 0,
            "investment": 0,
            "lifetime": 25,
        }
    )

    discount_period = costs["lifetime"].apply(lambda x: min(x, econ_lifetime))
    costs["capital_cost"] = (
        (calculate_annuity(discount_period, costs["discount rate"]) + costs["FOM"] / 100.0)
        * costs["investment"]
        * n_years
    )

    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT-CCS", "fuel"] = costs.at["gas", "fuel"]

    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    costs.at["OCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]
    costs.at["CCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]

    def costs_for_storage(store, link1, link2=None, max_hours=1.0):
        capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
        if link2 is not None:
            capital_cost += link2["capital_cost"]
        return pd.Series(dict(capital_cost=capital_cost, marginal_cost=0.0, co2_emissions=0.0))

    max_hours = elec_config["max_hours"]
    costs.loc["battery"] = costs_for_storage(
        costs.loc["battery storage"], costs.loc["battery inverter"], max_hours=max_hours["battery"]
    )

    for attr in ("marginal_cost", "capital_cost"):
        overwrites = cost_config.get(attr)
        overwrites = cost_config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites

    return costs


def read_generic_province_data(
    data_p: os.PathLike,
    index_col: int | str = 0,
    index_name: str = "province",
) -> pd.DataFrame:
    """Read generic province data from csv

    Args:
        data_p (os.PathLike): the data path.
        index_col (int, optional): the index column. Defaults to 0.
        index_name (str, optional): the output index name. Defaults to "province".

    Returns:
        pd.DataFrame: the formatted data
    """
    data = pd.read_csv(data_p, index_col=index_col).rename_axis(index_name)

    # common fixes to province names
    data.index = data.index.map(lambda x: PROV_RENAME_MAP.get(x, x))

    missing = set(PROV_NAMES) - set(data.index)
    if missing:
        raise ValueError(f"The following provinces are missing from {data_p}: {missing}")
    return data.loc[PROV_NAMES]


def read_yearly_load_projections(
    yearly_projections_p: os.PathLike = "resources/data/load/Province_Load_2020_2060.csv",
    conversion=1,
) -> pd.DataFrame:
    """Prepare projections for model use

    Args:
        yearly_projections_p (os.PathLike, optional): the data path.
                Defaults to "resources/data/load/Province_Load_2020_2060.csv".
        conversion (int, optional): the conversion factor to MWh. Defaults to 1.

    Returns:
        pd.DataFrame: the formatted data, in MWh
    """
    yearly_proj = pd.read_csv(yearly_projections_p)
    yearly_proj.rename(columns={"Unnamed: 0": "province", "region": "province"}, inplace=True)
    if "province" not in yearly_proj.columns:
        raise ValueError(
            "The province (or region or unamed) column is missing in the yearly projections data"
            ". Index cannot be built"
        )
    yearly_proj.set_index("province", inplace=True)
    yearly_proj.rename(columns={c: int(c) for c in yearly_proj.columns}, inplace=True)

    return yearly_proj * conversion


def merge_w_admin_l2(
    data: pd.DataFrame, admin_l2: gpd.GeoDataFrame, data_col: str
) -> gpd.GeoDataFrame:
    """Merge data with admin level 2 shapes.
    1. Merge on Chinese names (NL_NAME_2).
    2. Merge on English names (NAME_2) for missing values.

    Args:
        data: DataFrame with admin level 2 names (native language: NL_NAME_2 and eng NAME_2, data: data_col)
        admin_l2: GeoDataFrame with admin level 2 shapes.
        data_col: Relevant column name in data

    Returns:
        GeoDataFrame with merged data.
    """

    duplicates_NL = data.duplicated(["NL_NAME_2", "NAME_1"], keep=False)
    duplicates_EN = data.duplicated(["NAME_2", "NAME_1"], keep=False)
    duplicates = data.loc[duplicates_NL | duplicates_EN]
    if not duplicates.empty:
        raise ValueError(
            f"Data contains duplicate (NL_NAME_2, NAME_1)  or (NAME_2, NAME_1) pairs:\n{duplicates}"
        )

    merged = admin_l2.merge(
        data,
        left_on=["NL_NAME_2", "NAME_1"],
        right_on=["NL_NAME_2", "NAME_1"],
        how="left",
        suffixes=("", "_y"),
    )

    missing = merged.NAME_2_y.isna()
    if merged.loc[missing].empty:
        return merged

    fixed = merged.loc[missing, ["NAME_2", "NAME_1"]].merge(
        data,
        left_on=["NAME_2", "NAME_1"],
        right_on=["NAME_2", "NAME_1"],
        how="left",
        suffixes=("", "_y"),
    )
    fixed.index = merged.loc[missing].index
    merged.loc[missing, [data_col, "NAME_2_y"]] = fixed[[data_col, "NAME_2"]]
    still_missing = merged.NAME_2_y.isna()
    if still_missing.sum() > 0:
        logger.warning(
            f"Could not find {data_col} data for " f"{still_missing.sum()} admin level 2 regions."
        )

    # now merge any split geometries
    return merged
