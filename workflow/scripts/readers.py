import rioxarray
import rasterio
import geopandas as gpd
import os.path

from xarray import DataArray, Dataset
from constants import CRS


def read_raster(
    path: os.PathLike,
    clip_shape: gpd.GeoSeries = None,
    var_name="var",
    chunks=60,
    plot=False,
):
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
    """read raster data, clip it to a clip_shape and convert it to a GeoDataFrame

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