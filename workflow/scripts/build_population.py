"""
Rules for building the population data by region
"""

import logging
import os

import pandas as pd
from _helpers import configure_logging, mock_snakemake
from constants import POP_YEAR, PROV_NAMES, YEARBOOK_DATA2POP

logger = logging.getLogger(__name__)


def load_pop_csv(csv_path: os.PathLike, year = POP_YEAR) -> pd.DataFrame:
    """Load the national bureau of statistics of China population
    (Yearbook - Population, table 2.5 pop at year end by Region)
    
    Supports both formats:
    - Yearbook format (2.5 pop at year end by Region)
    - Historical data format with comment lines

    Args:
        csv_path (os.Pathlike): Path to the CSV file

    Returns:
        pd.DataFrame: The population for constants.POP_YEAR by province
        
    Raises:
        ValueError: If the province names do not match expected names
    """
    # Read CSV, skipping comment lines that start with #
    df = pd.read_csv(csv_path, index_col=0, header=0, comment='#')
    df = df.apply(pd.to_numeric)
    if not year in df.columns:
        raise ValueError(f"Requested year {year} not in {csv_path}. Avail: {df.columns}")
    
    df = df[year][df.index.isin(PROV_NAMES)]
    if not sorted(df.index.to_list()) == sorted(PROV_NAMES):
        raise ValueError(
            f"Province names do not match {sorted(df.index.to_list())} != {sorted(PROV_NAMES)}"
        )
    return df


def build_population(data_path: os.PathLike,
        data_unit_conversion = YEARBOOK_DATA2POP,
        data_year = POP_YEAR):
    """Build the population data by region

    Args:
        data_path (os.PathLike): the path to the pop csv table (province, year).
        data_unit_conversion (int, optional): unit conversion to head count.
            Defaults to YEARBOOK_DATA2POP
        data_year (int, optional): the year to use in the table
    """

    population = data_unit_conversion * load_pop_csv(csv_path=data_path, data_year)
    population.name = "population"
    population.to_hdf(snakemake.output.population, key=population.name)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_population")

    configure_logging(snakemake, logger=logger)
    conversion_factor = snakemake.params.pop_conversion
    year = snakemake.params.population_year
    build_population(snakemake.input.population, conversion_factor, year)
    logger.info("Population successfully built")
