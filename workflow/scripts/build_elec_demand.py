"""
Build the electricity demand
"""

import logging
import os

import numpy as np
import pandas as pd

import geopandas as gpd
from itertools import product
import scipy.sparse as sp
from _helpers import configure_logging, mock_snakemake
from _pypsa_helpers import make_periodic_snapshots
from shapely.prepared import prep
import xarray as xr

from constants import (
    PROV_NAMES,
    REF_YEAR,
    REGIONAL_GEO_TIMEZONES,
    TIMEZONE,
)

logger = logging.getLogger(__name__)

TWH2MWH = 1e6


def read_historical_data(
    hourly_load_p: os.PathLike,
    prov_codes_p: os.PathLike,
) -> pd.DataFrame:
    """Read the hourly electricity demand data and prepare it for use in the model

    Args:
        hourly_load_p (os.PathLike, optional): raw elec data from zenodo, see readme in data.
        prov_codes_p (os.PathLike, optional): province mapping for data.

    Returns:
        pd.DataFrame: the hourly demand data with the right province names, in MWh/hr
    """
    hourly = pd.read_csv(hourly_load_p)
    hourly_MWh = hourly.drop(columns=["Time Series"])
    prov_codes = pd.read_csv(prov_codes_p)
    prov_codes.set_index("Code", inplace=True)
    hourly_MWh.columns = hourly_MWh.columns.map(prov_codes["Full name"])
    return hourly_MWh


def shapes_to_shapes(orig: gpd.GeoSeries, dest: gpd.GeoSeries) -> sp.lil_matrix:
    """
    Adopted from vresutils.transfer.Shapes2Shapes()
    """
    orig_prepped = list(map(prep, orig))
    transfer = sp.lil_matrix((len(dest), len(orig)), dtype=float)

    for i, j in product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest.iloc[i]):
            area = orig.iloc[j].intersection(dest.iloc[i]).area
            transfer[i, j] = area / dest.iloc[i].area

    return transfer


def _normed(s: pd.Series) -> pd.Series:
    return s / s.sum()


def upsample_load(
    prov_load: pd.DataFrame,
    l2_gdp_pop: gpd.GeoDataFrame,
    distribution_key: dict[str, float],
) -> pd.DataFrame:
    """Upsample provincial load to admin level 2 based on gdp and population distribution

    Args:
        prov_load (pd.DataFrame): the provincial load data
        l2_gdp_pop (gpd.GeoDataFrame): the level 2 gdp and population data
        distribution_key (dict[str, float]): the distribution key weights for gdp and population
    Returns:
        pd.DataFrame: the upsampled load data
    """

    gdp_weight = distribution_key.get("gdp", 0.7)
    pop_weight = distribution_key.get("pop", 0.3)
    if not gdp_weight + pop_weight == 1.0:
        raise ValueError("The gdp and pop weights must sum to 1.0")
    data_arrays = []

    for province, group in l2_gdp_pop.geometry.groupby(l2_gdp_pop.NAME_1):
        load_prov = prov_load[province]

        if len(group) == 1:
            factors = pd.Series(1.0, index=group.index)

        else:
            prov_data = l2_gdp_pop.query("NAME_1 == @province")
            transfer = shapes_to_shapes(group, prov_data.geometry).T.tocsr()
            gdp_n = pd.Series(
                transfer.dot(prov_data["gdp_l2"].fillna(1.0).values), index=group.index
            )
            pop_n = pd.Series(
                transfer.dot(prov_data["population"].fillna(1.0).values), index=group.index
            )

            factors = _normed(gdp_weight * _normed(gdp_n) + pop_weight * _normed(pop_n))

        data_arrays.append(
            xr.DataArray(
                factors.values * load_prov.values[:, np.newaxis],
                dims=["time", "Bus"],
                coords={"time": load_prov.index.values, "Bus": factors.index.values},
            )
        )

    return xr.concat(data_arrays, dim="Bus")


if __name__ == "__main__":

    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_electricity_demand",
            heating_demand="positive",
            topology="Current+Neigbor",
            co2_pathway="exp175default",
            configfiles="tests/failed_test_config_c12b88ce935c2e4988e581332e0281b5697d8b4167d48ae58939513f7f409b4b.yaml"
        )

    configure_logging(snakemake, logger=logger)

    logger.info("Building base network electricity reference demand data...")

    prov_hrly_MWh_load = read_historical_data(
        snakemake.input.hrly_regional_elec_load,
        snakemake.input.province_codes,
    )
    gdp = gpd.read_file(snakemake.input.gdp)
    population = pd.read_csv(snakemake.input.population, index_col=0)

    if not len(population) == len(gdp):
        raise ValueError("Population and GDP data admin2 length mismatch")
    gdp_with_pop = gdp.merge(population, on=["NAME_1", "NAME_2"], how="left")

    load_l2 = upsample_load(
        l2_gdp_pop=gdp_with_pop,
        prov_load=prov_hrly_MWh_load,
        distribution_key=snakemake.params.distribution_key,
    )

    load_l2.name = "electricity demand (MW)"
    comp = dict(zlib=True, complevel=9, least_significant_digit=5)
    logger.info(
        f"Saving base network electricity reference demand data to {snakemake.output.base_network_load}..."
    )
    load_l2.to_netcdf(snakemake.output.base_network_load, encoding={load_l2.name: comp})

    logger.info("Successfully built base network electricity reference demand data.")
