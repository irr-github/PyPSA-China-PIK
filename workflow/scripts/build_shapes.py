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
    DISTANCE_CRS,
)
from pandas import DataFrame

NATURAL_EARTH_RESOLUTION = "10m"
GDAM_LV1 = "NAME_1"
GDAM_LV2 = "NAME_2"

logger = logging.getLogger(__name__)


def build_gdps_from_raster():
    """NOT USED AS THE RASTER DATA IS QUESTIONABLE FOR CHINA RIGHT NOW"""
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
        if row.geometry.area >= geom.area:
            row["geometry"] = row["geometry"].difference(geom)
    return row


def find_seaside_regions(
    regions: gpd.GeoDataFrame, eez: gpd.GeoDataFrame, max_dist=2, precision_deg=0.5, tolerance=10
) -> gpd.GeoDataFrame:
    """find seaside regions (within max_dist of the eez)

    Args:
        regions (gpd.GeoDataFrame): regions shapes, eg at l2 resolution
        eez (gpd.GeoDataFrame): economic exclusive maritmie zone
        max_dist (float): max dist from seaside
        precision_deg (float): precision in degrees simplification
    Returns:
        regions filtered for seaside"""

    # Single buffered geometry
    regions_ = regions.to_crs(DISTANCE_CRS)
    # remove rivers by decreasing precision before buffering
    eez_ = eez.set_precision(precision_deg).to_crs(DISTANCE_CRS).simplify(tolerance)
    buffer = eez_.union_all("unary").buffer(max_dist * 1000)

    # Spatial-index filter
    sidx = regions_.sindex
    possible = list(sidx.intersection(buffer.bounds))
    regions_small = regions_.iloc[possible]

    # Exact-intersect only on the small subset
    result = regions_small[regions_small.intersects(buffer)]

    return regions.loc[result.index]


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
    seaside_regions: gpd.GeoDataFrame,
    indx_key="NAME_2",
    prov_key="NAME_1",
    simplify_tol=0.5,
) -> gpd.GeoDataFrame:
    """Break up the eez by admin1 seaside_regions based on voronoi polygons of the centroids

    NOTE: this requires pre-processed seaside regions

    Args:
        eez (gpd.GeoDataFrame): country eez shape
        seaside_regions (gpd.GeoDataFrame): l2 (or arbitrary) shapes
        prov_shapes (gpd.GeoDataFrame): province shapes
        indx_key (str, optional): name of relevant col in seaside_regions. Defaults to "region".
        prov_key (str, optional): name of the province col in prov_shapes. Defaults to "province".
        simplify_tol (float, optional): tolerance for simplifying the voronoi polygons. Defaults to 0.5.

    Returns:
        gpd.GeoDataFrame: _description_
    """
    prov_shapes = seaside_regions.dissolve(prov_key).reset_index()[[prov_key, "geometry"]]
    voronois = seaside_regions.simplify(tolerance=simplify_tol).voronoi_polygons()
    voronois_simple = gpd.GeoDataFrame(
        geometry=voronois,
        crs=seaside_regions.crs,
    )
    # assign region (order of cells is not guaranteed to be same as seaside_regions)
    reg_voronois = (
        (voronois_simple.sjoin(seaside_regions, predicate="intersects").dissolve(indx_key))
        .reset_index()
        .drop(columns=[prov_key, "index_right"])
    )

    # assign province
    intersections = reg_voronois.sjoin(prov_shapes, predicate="intersects")
    intersections["key_pair"] = intersections.apply(lambda x: (x[prov_key], x[indx_key]), axis=1)
    valid = seaside_regions.apply(lambda x: (x[prov_key], x[indx_key]), axis=1)  # noqa
    reg_voronois_assigned = intersections.query("key_pair in @valid")
    if not len(reg_voronois_assigned) == len(seaside_regions):
        raise ValueError(
            "Error during offshore province assignment - some (l2, l1) keys likely duplicated"
        )

    eez_split = reg_voronois_assigned.overlay(
        eez, how="intersection"
    ).dissolve(  # [[indx_key, "geometry"]]
        by="key_pair"
    )
    eez_split = gpd.GeoDataFrame(eez_split.reset_index().rename(columns={0: "geometry"}))

    cleaned = remove_overlaps(remove_overlaps(remove_overlaps(remove_overlaps(eez_split))))
    return cleaned


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_shapes",
            configfiles="tests/failed_test_config_c12b88ce935c2e4988e581332e0281b5697d8b4167d48ae58939513f7f409b4b.yaml",
        )
    configure_logging(snakemake, logger=logger)

    L2_KEY = "NAME_2"
    L1_KEY = "NAME_1"

    simplify_tol = snakemake.params["simplify_tol"].get("eez", 0.5)
    eez_country = gpd.read_file(snakemake.input.offshore_shapes)
    exclude = snakemake.params["node_config"].get("exclude_provinces", [])
    regions = read_admin2_shapes(snakemake.input.admin_l2_shapes, fix=True, exclude=exclude)
    province_shapes = gpd.read_file(snakemake.input.province_shapes).rename(
        columns={"province": L1_KEY}
    )
    gdps = pd.read_csv(snakemake.input.gdp, skiprows=1)

    gdp_regions = merge_w_admin_l2(gdps, regions, data_col="gdp_l2")
    missing = gdp_regions[gdp_regions.gdp_l2.isna()]
    if not missing.empty:
        logger.warning(
            f"Some admin level 2 regions are missing GDP data: {missing[[L2_KEY, L1_KEY]]}"
        )
    gdp_regions.fillna({"gdp_l2": 0}, inplace=True)

    logger.info("Onshore Shapes and GDP processed")

    logger.info("Building offshore regions ...")

    # this requires a high simplification tolerance to remove regions far from complicated coastline
    seaside_regions = find_seaside_regions(
        regions.query(f"{L1_KEY} in @OFFSHORE_WIND_NODES"), eez_country, 2, tolerance=10
    )
    logger.info("found seaside regions")
    regions_offshore = split_eez_by_region(
        eez_country,
        seaside_regions,
        indx_key=L2_KEY,
        prov_key=L1_KEY,
        simplify_tol=simplify_tol,
    )

    regions_offshore[[L2_KEY, L1_KEY, "geometry"]].to_file(
        snakemake.output.offshore_shapes, driver="GeoJSON"
    )

    gdp_regions.to_file(snakemake.output.regions_w_gdp, driver="GeoJSON")

    logger.info(
        "Successfully built shapes \n\t- Offshore shapes saved to {snakemake.output.offshore_shapes} \n\t- GDP regions saved to {snakemake.output.regions_w_gdp}"
    )
