import io
import logging
from os import PathLike

import geopandas as gpd
import pandas as pd
import numpy as np
from _helpers import configure_logging, mock_snakemake
from readers import merge_w_admin_l2
from readers_geospatial import read_admin2_shapes


# TODO remove hardcoded offshore wind nodes
from constants import (
    CRS,
    EEZ_PREFIX,
    OFFSHORE_WIND_NODES,
)
from pandas import DataFrame

NATURAL_EARTH_RESOLUTION = "10m"
GDAM_LV1 = "NAME_1"
GDAM_LV2 = "NAME_2"

logger = logging.getLogger(__name__)


def build_gdps_from_raster():
    """ NOT USED AS THE RASTER DATA IS QUESTIONABLE FOR CHINA RIGHT NOW"""
    raise NotImplementedError("GDP from raster not implemented yet")



def cut_smaller_from_larger(
    row: gpd.GeoSeries, gdf: gpd.GeoDataFrame, overlaps: DataFrame
) -> gpd.GeoSeries:
    """Automatically assign overlapping area to the smaller region

    Example:
        areas_gdf.apply(cut_smaller_from_larger, args=(areas_gdf, overlaps), axis=1)

    Args:
        row (gpd.GeoSeries): the row from pandas apply
        gdf (gpd.GeoDataFrame): the geodataframe on which the operation is performed
        overlaps (DataFrame): the boolean overlap table

    Raises:
        ValueError: in case areas are exactly equal

    Returns:
        gpd.GeoSeries: the row with overlaps removed or not
    """
    ovrlap_idx = np.where(overlaps.loc[row.name].values == True)[0].tolist()
    for idx in ovrlap_idx:
        geom = gdf.iloc[idx].geometry
        if row.geometry.area > geom.area:
            row["geometry"] = row["geometry"].difference(geom)
        elif row.geometry.area == geom.area:
            raise ValueError(f"Equal area overlap between {row.name} and {idx} - unhandled")
    return row


def has_overlap(gdf: gpd.GeoDataFrame) -> DataFrame:
    """Check for spatial overlaps across rows

    Args:
        gdf (gpd.GeoDataFrame): the geodataframe to check

    Returns:
        DataFrame: Index x Index boolean dataframe
    """
    return gdf.apply(
        lambda row: gdf[gdf.index != row.name].geometry.apply(
            lambda geom: row.geometry.intersects(geom)
        ),
        axis=1,
    )


def remove_overlaps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove inter row overlaps from a GeoDataFrame, cutting out the smaller region from the larger one

    Args:
        gdf (gpd.GeoDataFrame): the geodataframe to be treated

    Returns:
        gpd.GeoDataFrame: the treated geodataframe
    """
    overlaps = has_overlap(gdf)
    return gdf.apply(cut_smaller_from_larger, args=(gdf, overlaps), axis=1)


def split_eez_by_region(
    eez: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    prov_shapes: gpd.GeoDataFrame,
    indx_key="region",
    prov_key="province",
    simplify_tol=0.5,
) -> gpd.GeoDataFrame:
    """Break up the eez by admin1 regions based on voronoi polygons of the centroids

    Args:
        eez (gpd.GeoDataFrame): _description_
        regions (gpd.GeoDataFrame): _description_
        indx_key (str, optional): name of the provinces col in regions. Defaults to "region".
        simplify_tol (float, optional): tolerance for simplifying the voronoi polygons. Defaults to 0.5.

    Returns:
        gpd.GeoDataFrame: _description_
    """
    # generate voronoi cells (more than one per province & can overlap)
    voronois = regions.simplify(tolerance=simplify_tol).voronoi_polygons()
    voronois_simple = gpd.GeoDataFrame(
        geometry=voronois,
        crs=regions.crs,
    )
    # assign region (order of cells is not guaranteed to be same as regions)
    reg_voronois = (
        voronois_simple.sjoin(regions, predicate="intersects")
        .groupby(indx_key)
        .apply(lambda x: x.union_all("unary"))
    )
    reg_voronois = gpd.GeoDataFrame(
        geometry=reg_voronois.values,
        crs=regions.crs,
        data={indx_key: reg_voronois.index},
    )

    # assign province
    repr_points = gpd.GeoDataFrame(index = reg_voronois.index, geometry = reg_voronois.representative_point().values)
    reg_voronois[prov_key] = repr_points.sjoin_nearest(prov_shapes)[prov_key]
    if reg_voronois[prov_key].isna().sum():
        logger.warning(
            f"{reg_voronois[prov_key].isna().sum()} voronoi regions could not be assigned a province"
        )

    # remove overlaps
    gdf_ = remove_overlaps(reg_voronois.set_index(indx_key)).reset_index()

    eez_prov = (
        gdf_.query(f"{L1_KEY} in @OFFSHORE_WIND_NODES")
        .overlay(eez, how="intersection")#[[indx_key, "geometry"]]
        .groupby([indx_key, prov_key]).geometry.apply(lambda x: x.union_all("unary"))
    )

    return gpd.GeoDataFrame(eez_prov.reset_index().rename(columns={0:"geometry"}))


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake("build_shapes")
    configure_logging(snakemake, logger=logger)

    L2_KEY = "NAME_2"
    L1_KEY = "NAME_1"

    simplify_tol = snakemake.params["simplify_tol"].get("eez", 0.1)
    eez_country = gpd.read_file(snakemake.input.offshore_shapes)
    exclude = snakemake.params["node_config"].get("exclude_provinces", [])
    regions = read_admin2_shapes(snakemake.input.admin_l2_shapes, fix = True, exclude=exclude)
    province_shapes = gpd.read_file(snakemake.input.province_shapes).rename(
        columns={"province": L1_KEY}
    )
    gdps = pd.read_csv(snakemake.input.gdp, skiprows=1)

    gdp_regions = merge_w_admin_l2(gdps, regions, data_col="gdp_l2")

    regions_offshore = split_eez_by_region(
        eez_country,
        regions,
        province_shapes,
        indx_key=L2_KEY,
        prov_key=L1_KEY,
        simplify_tol=simplify_tol,
    )

    regions_offshore[[L2_KEY, L1_KEY, "geometry"]].to_file(
        snakemake.output.offshore_shapes, driver="GeoJSON"
    )

    gdp_regions.to_file(
        snakemake.output.regions_w_gdp, driver="GeoJSON"
    )

    logger.info("Successfully built shapes")
