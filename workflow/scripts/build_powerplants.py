# SPDX-FileCopyrightText: : 2025 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT
"""
Build the existing capacities for each node from GEM (global energy monitor) tracker data.
This script is intended for use as part of the Snakemake workflow.

The GEM data has to be downloaded manually and placed in the source directory of the snakemake rule.
download page: https://globalenergymonitor.org/projects/global-integrated-power-tracker/download-data/

"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from _helpers import configure_logging, mock_snakemake
from shapely.geometry import Point

logger = logging.getLogger(__name__)

ADM_COLS = {
    0: "Country",
    1: "Subnational unit (state, province)",
    2: "Major area (prefecture, district)",
    3: "Local area (taluk, county)",
}
ADM_LVL1, ADM_LVL2 = ADM_COLS[1], ADM_COLS[2]


def load_gem_excel(
    path: os.PathLike, sheetname="Units", country_col="Country/area", country_names=["China"]
) -> pd.DataFrame:
    """
    Load a Global Energy monitor excel file as a dataframe.

    Args:
        path (os.PathLike): Path to the Excel file.
        sheetname (str): Name of the sheet to load. Default is "Units".
        country_col (str): Column name for country names. Default is "Country/area".
        country_names (list): List of country names to filter by. Default is ["China"].
    """

    df = pd.read_excel(path, sheet_name=sheetname, engine="openpyxl")
    # replace problem characters in column names
    df.columns = df.columns.str.replace("/", "_")
    country_col = country_col.replace("/", "_")

    if country_col not in df.columns:
        logger.warning(f"Column {country_col} not found in {path}. Returning unfiltered DataFrame.")
        return df

    return df.query(f"{country_col} in @country_names")


def clean_gem_data(gem_data: pd.DataFrame, gem_cfg: dict) -> pd.DataFrame:
    """
    Clean the GEM data by
     - mapping GEM types onto pypsa types
     - filtering for relevant project statuses
     - cleaning invalid entries (e.g "not found"->nan)

    Args:
        gem_data (pd.DataFrame): GEM dataset.
        gem_cfg (dict): Configuration dictionary, 'global_energy_monitor.yaml'
    Returns:
        pd.DataFrame: Cleaned GEM data.
    """

    _valid_project_states = gem_cfg["status"]
    GEM = gem_data.query("Status in @_valid_project_states")
    GEM.rename(columns={"Plant _ Project name": "Plant name"}, inplace=True)
    GEM.loc[:, "Retired year"] = GEM["Retired year"].replace("not found", np.nan)
    GEM.loc[:, "Start year"] = GEM["Start year"].replace("not found", np.nan)
    keep = [col for col in gem_cfg["relevant_columns"] if col in GEM.columns]
    skipped = set(gem_cfg["relevant_columns"]) - set(keep)
    if skipped:
        logger.warning(
            f"The following relevant columns were not found in the GEM data and will be skipped: {skipped}"
        )
    GEM = GEM[keep]

    # Remove whitespace from admin columns
    # Remove all whitespace (including tabs, newlines) from admin columns
    admin_cols = [col for col in ADM_COLS.values() if col in GEM.columns]
    GEM[admin_cols] = GEM[admin_cols].apply(lambda x: x.str.replace(r"\s+", "", regex=True))

    # split oil and gas, rename bioenergy
    gas_mask = GEM.query("Type == 'oil/gas' & Fuel.str.contains('gas', case=False, na=False)").index
    GEM.loc[gas_mask, "Type"] = "gas"
    GEM.Type = GEM.Type.str.replace("bioenergy", "biomass")

    # split CHP (potential issue: split before type split. After would be better)
    if gem_cfg["CHP"].get("split", False):
        GEM.loc[:, "CHP_bool"] = (
            GEM.loc[:, "CHP"]
            .map({"not found": False, "yes": True, "no": False, np.nan: False})
            .fillna(False)
        )
        chp_mask = GEM[GEM["CHP_bool"] == True].index

        aliases = gem_cfg["CHP"].get("aliases", [])
        for alias in aliases:
            chp_mask = chp_mask.append(
                GEM[GEM["Plant name"].str.contains(alias, case=False, na=False)].index
            )
        chp_mask = chp_mask.unique()
        GEM.loc[chp_mask, "Type"] = "CHP " + GEM.loc[chp_mask, "Type"]

    GEM["tech"] = ""
    for tech, mapping in gem_cfg["tech_map"].items():
        if not isinstance(mapping, dict):
            raise ValueError(
                f"Mapping for {tech} is a {type(mapping)} - expected dict. Check your config."
            )

        tech_mask = GEM.query(f"Type == '{tech}'").index
        if tech_mask.empty:
            continue
        GEM.loc[tech_mask, "Type"] = GEM.loc[tech_mask, "Technology"].map(mapping)

        # apply defaults if requested
        if "default" not in mapping:
            continue
        fill_val = mapping["default"]
        if fill_val is not None:
            GEM.loc[tech_mask, "Type"] = GEM.loc[tech_mask, "Type"].fillna(value=fill_val)
        else:
            GEM.loc[tech_mask, "Type"] = GEM.loc[tech_mask, "Type"].dropna()

    return GEM.dropna(subset=["Type"])


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


def assign_year_bins(df: pd.DataFrame, year_bins: list, base_year=2020) -> pd.DataFrame:
    """
    Group the DataFrame by year bins.

    Args:
        df (pd.DataFrame): DataFrame with a 'Start year' column.
        year_bins (list): List of year bins to group by.
        base_year (int): cut-off for histirocal period. Default is 2020.

    Returns:
        pd.DataFrame: DataFrame with a new 'grouping_year' column.
    """
    min_start_year = min(year_bins) - 2.5
    base_year = 2020
    df = df[df["Start year"] > min_start_year]
    df = df[df["Retired year"].isna() | (df["Retired year"] > base_year)].reset_index(drop=True)
    df["grouping_year"] = np.take(year_bins, np.digitize(df["Start year"], year_bins, right=True))

    # check the grouping years are appropriate
    newer_assets = (df["Start year"] > max(year_bins)).sum()
    if newer_assets:
        raise ValueError(
            f"There are {newer_assets} assets with build year "
            f"after last power grouping year {max(year_bins)}. "
            "These assets will be dropped and not considered."
            "Redefine the grouping years to keep them or"
            " remove pre-construction/construction/... state from options."
        )

    return df


def assign_node_from_gps(
    ppl: pd.DataFrame, nodes: gpd.GeoDataFrame, offshore_nodes: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Assign plant node based on GPS coordinates with robust boundary handling.

    spatial join within first, then nearest neighbor fallback
    (catches powerplants that fall on cluster boundaries due to precision issues.)

    Args:
        ppl (pd.DataFrame): Powerplants data with Latitude/Longitude columns
        nodes (gpd.GeoDataFrame): Node shape geometries with "cluster" and "geometry" columns
        offshore_nodes (gpd.GeoDataFrame): Offshore node shapes for offshore plants
    Returns:
        pd.DataFrame: DataFrame with assigned nodes and diagnostic columns
    """

    # Create GeoDataFrame from lat/lon, project to metric CRS
    gdf = gpd.GeoDataFrame(
        ppl,
        geometry=ppl.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis=1),
        crs="EPSG:4326",
    )
    # manual fix to make clearly not shanghai
    gdf.query(
        "`Plant name`=='Jiangsu Rudong (China Guangdong Nuclear) Offshore wind farm'"
    ).Latitude = 32.8
    gdf_proj = gdf.to_crs("EPSG:3036")
    nodes_proj = nodes.to_crs("EPSG:3036")

    # First pass: spatial join within
    assigned = gdf.sjoin(nodes, predicate="within", how="left")
    # Identify unassigned (likely on boundaries)
    unassigned_mask = assigned["cluster"].isna()
    if unassigned_mask.any():
        logger.info(f"Assigning {unassigned_mask.sum()} boundary powerplants via nearest neighbor")

        # Get original indices of unassigned powerplants
        unassigned_indices = assigned[unassigned_mask].index
        unassigned_ppls = gdf_proj.loc[unassigned_indices]

        # fallback: nearest
        nearest_assigned = unassigned_ppls.sjoin_nearest(
            nodes_proj.to_crs("EPSG:3036"),
            how="left",
            max_distance=10000,  # 10km sanity check
            distance_col="boundary_distance_m",
        )
        assigned.loc[unassigned_mask, "cluster"] = nearest_assigned["cluster"]

    # Try offshore shapes (wind turbines and)
    still_unassigned = assigned["cluster"].isna()
    assigned_offshore = assigned.loc[still_unassigned].sjoin_nearest(
        offshore_nodes.reset_index().to_crs("EPSG:3036"),
        how="left",
        distance_col="boundary_distance_m",
        max_distance=400000,
    )  # 300km sanity check
    # ugly hack in case at (inexact) border, flip coin
    idx = assigned_offshore.index.drop_duplicates()
    assigned.loc[still_unassigned, "cluster"] = assigned_offshore.loc[idx, "cluster_right"].values

    # Report issues
    still_unassigned = assigned["cluster"].isna().sum()
    if still_unassigned > 0:
        logger.error(f"{still_unassigned} powerplants still unassigned after fallback!")
        failed_plants = assigned[assigned["cluster"].isna()]["Plant name"].tolist()
        logger.error(f"Failed plants: {failed_plants[:5]}...")

    return assigned.drop(columns=["geometry", "index_right"], errors="ignore")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_powerplants",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2020",
            cluster_id="IM2XJ4",
            # configfiles="resources/tmp/remind_coupled.yaml",
        )

    configure_logging(snakemake, logger=logger)

    # config = snakemake.params.config
    params = snakemake.params
    cfg_GEM = params["gem"]
    grouped_years = params["grouped_years"]
    output_paths = dict(snakemake.output.items())

    gem_data = load_gem_excel(snakemake.input.GEM_plant_tracker, sheetname="Power facilities")

    cleaned = clean_gem_data(gem_data, cfg_GEM)
    cleaned = assign_year_bins(cleaned, grouped_years, base_year=cfg_GEM["base_year"])

    processed, requested = cleaned.Type.unique(), set(snakemake.params.technologies)
    missing = requested - set(processed)
    extra = set(processed) - requested
    if missing:
        raise ValueError(
            f"Some techs requested existing_baseyear missing from GEM\n\t:{missing}\nAvailable Global Energy Monitor techs after processing:\n\t{processed}."
        )
    if extra:
        logger.warning(f"Techs from GEM {extra} not covered by existing_baseyear techs.")

    # assign buses
    nodes = gpd.read_file(snakemake.input.nodes)
    offshore_nodes = gpd.read_file(snakemake.input.offshore_nodes)
    ppls = assign_node_from_gps(cleaned, nodes, offshore_nodes)
    ppls.loc[:, "Type"] = ppls.Type.str.replace(" CHP", "").str.replace("CHP ", "")

    ppls.to_csv(
        output_paths["cleaned_ppls"],
        index=False,
    )
    logger.info(f"Cleaned GEM capacities saved to {output_paths['cleaned_ppls']}")
