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


def read_province_shapes(shape_file: os.PathLike) -> gpd.GeoDataFrame:
    """Read the province shape files

    Args:
        shape_file (os.PathLike): the path to the .shp file & co

    Returns:
        gpd.GeoDataFrame: the province shapes as a GeoDataFrame
    """

    prov_shapes = gpd.GeoDataFrame.from_file(shape_file)
    prov_shapes = prov_shapes.to_crs(CRS)
    prov_shapes.set_index("province", inplace=True)
    # TODO: does this make sense? reindex after?
    if not (prov_shapes.sort_index().index == sorted(PROV_NAMES)).all():
        missing = f"Missing provinces: {set(PROV_NAMES) - set(prov_shapes.index)}"
        raise ValueError(f"Province names do not match expected names: missing {missing}")

    return prov_shapes


def read_offshore_province_shapes(
    shape_file: os.PathLike, index_name="province"
) -> gpd.GeoDataFrame:
    """Read the offshore province shape files (based on the eez)

    Args:
        shape_file (os.PathLike): the path to the .shp file & co
        index_name (str, optional): the name of the index column. Defaults to "province".

    Returns:
        gpd.GeoDataFrame: the offshore province shapes as a GeoDataFrame
    """

    offshore_regional = gpd.read_file(shape_file).set_index(index_name)
    offshore_regional = offshore_regional.reindex(OFFSHORE_WIND_NODES).rename_axis("bus")
    if offshore_regional.geometry.isnull().any():
        empty_geoms = offshore_regional[offshore_regional.geometry.isnull()].index.to_list()
        raise ValueError(
            f"There are empty geometries in offshore_regional {empty_geoms}, offshore wind will fail"
        )

    return offshore_regional


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
    """Read and preprocess administrative level 2 shape s (GADM41).

    Args:
        path (str): Path to the GeoJSON/shape file.
        fix (bool, optional): Whether to fix inconsistencies in region names. Defaults to True.
        exclude (list, optional): Regions to exclude from the data. Defaults to ["Macau", "HongKong"].

    Returns:
        gpd.GeoDataFrame: Preprocessed administrative level 2 shapes.
    """

    PROV_RENAME_MAP = {"Inner Mongolia": "InnerMongolia", "Ningxia Hui": "Ningxia", "Xizang": "Tibet"}
    admin_l2 = gpd.read_file(path).query("NAME_1 not in @exclude")
    admin_l2["NAME_1"] = admin_l2.NAME_1.map(lambda x: PROV_RENAME_MAP.get(x, x))

    # merge duplicated region geometries
    merged_geos = admin_l2.groupby(["NAME_1", "NAME_2", "NL_NAME_2", "NL_NAME_1"]).apply(
        lambda x: pd.Series(x.geometry.union_all()) if len(x) > 1 else x.geometry
    )
    merged_geos = merged_geos.to_frame().reset_index().query("NAME_1 not in @exclude")
    admin_l2 = gpd.GeoDataFrame(
        merged_geos[["NAME_1", "NAME_2", "NL_NAME_2", "NL_NAME_1"]],
        geometry=merged_geos[0],
        crs=admin_l2.crs,
    )

    if fix:
        return _fix_gadm41(admin_l2)
    else:
        return admin_l2
