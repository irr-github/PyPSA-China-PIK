"""File reading support functions"""

import os

import pandas as pd
import geopandas as gpd
import rioxarray
from constants import CRS, OFFSHORE_WIND_NODES, PROV_NAMES, PROV_RENAME_MAP
from xarray import DataArray


def read_raster(
    path: os.PathLike,
    clip_shape: gpd.GeoSeries = None,
    var_name="var",
    chunks=60,
    plot=False,
) -> DataArray:
    """Read raster data and optionally clip it to a given shape.

    Args:
        path (os.PathLike): The path to the raster file.
        clip_shape (gpd.GeoSeries, optional): The shape to clip the raster data. Defaults to None.
        var_name (str, optional): The variable name to assign to the raster data. Defaults to "var".
        chunks (int, optional): The chunk size for the raster data. Defaults to 60.
        plot (bool, optional): Whether to plot the raster data. Defaults to False.

    Returns:
        DataArray: The raster data as an xarray DataArray.
    """
    ds = rioxarray.open_rasterio(path, chunks=chunks, default_name="pop_density")
    ds = ds.rename(var_name)

    if clip_shape is not None:
        ds = ds.rio.clip(clip_shape.geometry)

    if plot:
        ds.plot()

    return ds


def read_pop_density(
    path: os.PathLike,
    clip_shape: gpd.GeoSeries = None,
    crs=CRS,
    chunks=25,
    var_name="pop_density",
) -> gpd.GeoDataFrame:
    """Read raster data, clip it to a clip_shape and convert it to a GeoDataFrame

    Args:
        path (os.PathLike): the target path for the raster data (tif)
        clip_shape (gpd.GeoSeries, optional): the shape to clip the data. Defaults to None.
        crs (int, optional): the coordinate system. Defaults to 4326.
        var_name (str, optional): the variable name. Defaults to "var".
        chunks (int, optional): the chunk size for the raster data. Defaults to 25.

    Returns:
        gpd.GeoDataFrame: the raster data for the aoi
    """

    ds = read_raster(path, clip_shape, var_name, plot=False)
    ds = ds.where(ds > 0)

    df = ds.to_dataframe(var_name)
    df.reset_index(inplace=True)

    # Convert the DataFrame to a GeoDataFrame
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=crs)


def _fix_gadm41(df_admin_l2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix known issues with GADM v4.1 China admin level 2 data.

    Args:
        df_admin_l2 (gpd.GeoDataFrame): GADM v4.1 China admin level 2 data.

    Returns:
        gpd.GeoDataFrame: Fixed GADM v4.1 China admin level 2 data.
    """
    RENAME_MAP = {
        "Ma'anshan": "Maanshan",
        "Ürümqi": "Urumqi",
        "Neijiang]]": "Neijiang",
        "Hainan": "Danzhou",
    }
    df_admin_l2["NAME_2"] = df_admin_l2.NAME_2.map(lambda x: RENAME_MAP.get(x, x))

    # Chaohu has been redistributed over 3 districts in 2011. Attach it to Hefei (not correct)
    # TODO check whether GADMv5 has fixed this issue

    df_admin_l2.loc[df_admin_l2.NAME_2 == "Hefei", "geometry"] = df_admin_l2.loc[
        df_admin_l2.NAME_2.isin(["Hefei", "Chaohu"])
    ].union_all()
    df_admin_l2 = df_admin_l2.query("NAME_2 != 'Chaohu'")

    df_admin_l2.loc[df_admin_l2.NAME_2 == "Jinan", "geometry"] = df_admin_l2.loc[
        df_admin_l2.NAME_2.isin(["Jinan", "Laiwu"])
    ].union_all()
    df_admin_l2 = df_admin_l2.query("NAME_2 != 'Laiwu'")

    return df_admin_l2


def read_admin2_shapes(path: str, fix=True, exclude=["Macau", "HongKong"]) -> gpd.GeoDataFrame:
    """Read and preprocess administrative level 2 shapes (GADM41).
    Merge shapes

    Args:
        path (str): Path to the GeoJSON/shape file.
        fix (bool, optional): Whether to fix inconsistencies in region names. Defaults to True.
        exclude (list, optional): Regions to exclude from the data. Defaults to ["Macau", "HongKong"].

    Returns:
        gpd.GeoDataFrame: Preprocessed administrative level 2 shapes.
    """

    admin_l2 = gpd.read_file(path).query("NAME_1 not in @exclude")
    admin_l2["NAME_1"] = admin_l2.NAME_1.map(lambda x: PROV_RENAME_MAP.get(x, x))

    # merge geometries
    unmerged_NL = admin_l2.duplicated(subset=["NL_NAME_2", "NAME_1"], keep=False)
    unmerged_EN = admin_l2.duplicated(subset=["NAME_2", "NAME_1"], keep=False)
    admin_l2.loc[unmerged_EN, "geometry"] = (
        admin_l2.loc[unmerged_EN]
        .groupby(["NAME_2", "NAME_1"])
        .geometry.transform(lambda x: x.geometry.union_all("unary"))
    )
    admin_l2.loc[unmerged_NL, "geometry"] = (
        admin_l2.loc[unmerged_NL]
        .groupby(["NL_NAME_2", "NAME_1"])
        .geometry.transform(lambda x: x.geometry.union_all("unary"))
    )
    # do this at end to not mess index
    admin_l2.drop_duplicates(subset=["NL_NAME_2", "NAME_1"], inplace=True)
    admin_l2.drop_duplicates(subset=["NAME_2", "NAME_1"], inplace=True)
    admin_l2.reset_index(drop=True, inplace=True)
    if fix:
        admin_l2 = _fix_gadm41(admin_l2)

    return admin_l2
