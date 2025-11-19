# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>,
# Adapted for PyPSA-China-PIK
#
# SPDX-License-Identifier: MIT
"""
Build population layouts for all clustered model regions as total as well as
split by urban and rural population.
"""
"""
Build mapping between population raster (higher resolution than cutout) and provinces.
Determine rural population cells based on provincial urban fractions.
Aggregate back to cutout resolution
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  # enable .rio accessor
from atlite import Cutout
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

from _helpers import configure_logging, mock_snakemake
from build_population_layouts import add_shape_id


logger = logging.getLogger(__name__)


def population_at_cluster_cutout_intersection(
    cutout: Cutout,
    population_raster: xr.DataArray,
    cluster_names: list | pd.Series,
    band_name: str = "population",
) -> pd.DataFrame:
    """
    Aggregate population at cluster x cutout grid resolution.
    
    Hybrid approach for memory efficiency and accuracy:
    1. Assign high-res population pixels to clusters (rasterize once)
    2. For each cluster, regrid masked population to cutout resolution
    3. Use conservative resampling to preserve population totals
    
    Args:
        cutout (Cutout): atlite Cutout with grid cells at 0.25° resolution
        population_raster (xr.DataArray): High-resolution population raster with band dimension
        cluster_shapes (gpd.GeoDataFrame): Cluster region polygons (e.g., provinces, nodes)
        band_name (str): Name of the band containing population data. Defaults to "population".
    
    Returns:
        pd.DataFrame: Shape (n_clusters, n_cutout_cells) with population values.
            Rows indexed by cluster IDs, columns by cutout cell indices.
    """

    n_clusters = len(cluster_names)

    logger.info(
        f"Processing population: {n_clusters} clusters "
        f"x {cutout.data['x'].size * cutout.data['y'].size} cutout cells"
    )

    # Select the population band
    if "band" not in population_raster.dims:
        raise ValueError("population_raster must have a 'band' dimension")

    pop_data = population_raster.sel(band=band_name)

    # Log initial population sum
    initial_pop_sum = pop_data.sum()
    
    # Get cutout grid bounds and resolution
    cutout_coords = cutout.data.coords
    cutout_x = cutout_coords['x'].values
    cutout_y = cutout_coords['y'].values

    # Derive cell sizes from coordinates
    cutout_dx = float(np.median(np.diff(cutout_x))) if len(cutout_x) > 1 else 0.25
    cutout_dy_abs = float(np.median(np.abs(np.diff(cutout_y)))) if len(cutout_y) > 1 else 0.25

    # Build transform from the true top-left corner based on centers
    x_left_c = float(np.min(cutout_x))
    y_top_c = float(np.max(cutout_y))
    cutout_transform = from_origin(
        x_left_c - cutout_dx / 2.0,  # west edge
        y_top_c + cutout_dy_abs / 2.0,  # north edge
        cutout_dx,
        cutout_dy_abs,
    )
    
    # Population grid transform
    # Ensure population is north-up for rasterio (row 0 is the northernmost row)
    # Then compute transform robustly from centers
    pop_data = pop_data.sortby('y', ascending=False)
    pop_x = pop_data.coords['x'].values
    pop_y = pop_data.coords['y'].values
    pop_dx = float(np.median(np.diff(pop_x))) if len(pop_x) > 1 else 0.01
    pop_dy_abs = float(np.median(np.abs(np.diff(pop_y)))) if len(pop_y) > 1 else 0.01

    x_left = float(np.min(pop_x))
    y_top = float(np.max(pop_y))
    pop_transform = from_origin(
        x_left - pop_dx / 2.0,
        y_top + pop_dy_abs / 2.0,
        pop_dx,
        pop_dy_abs,
    )
    
    # Get CRS
    pop_crs = pop_data.rio.crs if hasattr(pop_data, 'rio') else 'EPSG:4326'
    cutout_crs = 'EPSG:4326'  # atlite uses WGS84
    
    logger.debug(f"Pop grid: shape={pop_data.shape}, CRS={pop_crs}")
    logger.debug(f"Cutout grid: shape=({len(cutout_y)}, {len(cutout_x)}), CRS={cutout_crs}")
    logger.debug(
        f"Pop bounds approx: x=[{pop_x[0]:.2f}, {pop_x[-1]:.2f}], "
        f"y=[{pop_y[0]:.2f}, {pop_y[-1]:.2f}]"
    )
    logger.debug(
        f"Cutout bounds approx: x=[{cutout_x[0]:.2f}, {cutout_x[-1]:.2f}], "
        f"y=[{cutout_y[0]:.2f}, {cutout_y[-1]:.2f}]"
    )
    logger.debug(f"Pop transform: {pop_transform}")
    logger.debug(f"Cutout transform: {cutout_transform}")

    # per-cluster reprojection to minimise population error at high res
    # For each cluster, regrid masked population to cutout resolution
    logger.info(f"  Regridding population for {n_clusters} clusters...")
    result = np.zeros((n_clusters, len(cutout_y), len(cutout_x)))

    for cluster_idx, cluster_name in enumerate(cluster_names):
        # Mask population to this cluster only
        cluster_pop = population_raster.sel(band=band_name).where(
            population_raster.sel(band='cluster') == cluster_idx, 0)  # Fill with 0 instead of NaN
        
        # Prepare masked cluster data (north-up, float32, NaNs→0)
        cluster_pop_array = np.nan_to_num(
            cluster_pop.sortby('y', ascending=False).values,
            nan=0.0,
        ).astype(np.float32)
        
        # Get stats for this cluster
        cluster_pop_sum = cluster_pop_array.sum()
        logger.debug(f"    Cluster {cluster_idx} ({cluster_name}): pop={cluster_pop_sum:.2e}")
        
        # Skip if no population
        if cluster_pop_sum == 0:
            logger.debug(f"      Cluster {cluster_name} has 0 population, skipping")
            continue
        
        # Regrid this cluster's population to cutout resolution using conservative resampling
        cluster_pop_coarse = np.zeros((len(cutout_y), len(cutout_x)), dtype=np.float32)
    
        reproject(
            source=cluster_pop_array,
            destination=cluster_pop_coarse,
            src_transform=pop_transform,
            src_crs=pop_crs,
            dst_transform=cutout_transform,
            dst_crs=cutout_crs,
            resampling=Resampling.sum,  # Conservative: preserves totals
        )
        
        result[cluster_idx, :, :] = cluster_pop_coarse
        
        # Check for issues
        cluster_pop_after = cluster_pop_coarse.sum()
        logger.debug(
            f"      Regridded: before={cluster_pop_sum:.2e}, after={cluster_pop_after:.2e}"
        )
    
    # Stage 4: Flatten cutout dimensions and convert to DataFrame
    logger.info("  Converting to DataFrame...")
    result_flat = result.reshape(n_clusters, -1)
    
    df = pd.DataFrame(
        result_flat,
        index=cluster_names,
        columns=range(result_flat.shape[1])
    )
    
    total_pop = df.values.sum()
    conservation_error = abs(total_pop - initial_pop_sum) / initial_pop_sum * 100
    
    logger.info(
        f"  Result: shape={df.shape}, total={total_pop:.2e}, "
        f"error={conservation_error:.2f}%"
    )

    return df


if __name__ == "__main__":
    # Make static analyzers happy and support both direct and snakemake execution
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_clustered_population",
        )

    configure_logging(snakemake, logger=logger)

    # Load input data
    cutout = Cutout(snakemake.input.cutout)
    clusters = gpd.read_file(snakemake.input.regions)
    population_raster = xr.open_dataarray(snakemake.input.pop_layouts)

    # add cluster IDs to population raster
    population_raster = add_shape_id(population_raster, clusters, "cluster")

    # BEBUG population sum for each band
    for band in population_raster.coords['band'].values:
        band_data = population_raster.sel(band=band)
        band_sum = band_data.sum()
        band_has_nans = np.isnan(band_data.values).any()
        logger.debug(
            f"  Band '{band}': sum={band_sum:.2e}, has_NaNs={band_has_nans}"
        )
    # Process total population
    pop_total = population_at_cluster_cutout_intersection(
        cutout=cutout,
        population_raster=population_raster,
        cluster_names=list(clusters["cluster"]),
        band_name="population",
    )

    # Process urban population (if available)
    pop_urban = None
    if "is_rural" in population_raster.coords["band"].values:
        # Create urban population band: population * (1 - is_rural)
        pop_data = population_raster.sel(band="population")
        is_rural = population_raster.sel(band="is_rural")
        urban_pop = pop_data * (1 - is_rural)
        
        # Add as new band to the same raster to avoid duplicating x/y coords in memory
        urban_pop_da = urban_pop.expand_dims("band").assign_coords(band=["urban_population"])
        population_raster = xr.concat([population_raster, urban_pop_da], dim="band")
        
        pop_urban = population_at_cluster_cutout_intersection(
            cutout=cutout,
            population_raster=population_raster,
            cluster_names=list(clusters["cluster"]),
            band_name="urban_population",
        )
    
    # Save outputs
    with pd.HDFStore(snakemake.output.clustered_pop_layout, mode="w", complevel=4) as store:
        store["total"] = pop_total
        if pop_urban is not None:
            store["urban"] = pop_urban
    
    logger.info(
        f"Saved clustered population layouts with total population: "
        f"{pop_total.values.sum():.2e}"
    )
