"""file reading support functions"""

import os
import pandas as pd

from constants import PROV_NAMES, PROV_RENAME_MAP

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