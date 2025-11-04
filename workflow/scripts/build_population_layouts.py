# SPDX-FileCopyrightText: the PyPSA-China-PIK authors, 2025
# based on PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build mapping between population raster (higher resolution than cutout) and provinces.
Determine rural population cells based on provincial urban fractions.
Aggregate back to cutout resolution
"""
import logging
from venv import logger

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import scipy.sparse as sp

from _helpers import configure_logging, mock_snakemake
from readers import read_generic_province_data
from readers_geospatial import read_admin2_shapes

from rasterio import features
import rioxarray
from affine import Affine

logger = logging.getLogger(__name__)


def compute_shape_indicators_fast(data: xr.DataArray, shapes_gdf: gpd.GeoDataFrame, data_crs: str | None = None) -> xr.DataArray:
    """Convert polygons to raster masks. This is a sparse matrix mapping shape to pixel.
    The indicator is binary with no area fraction.

    Unlike atlite do not rely on shapely geometry overlap but rasterio.features.rasterize.
    This is faster & enables working at higher resolution than the era5 grid. Resulting assignments
    are accurate.

    Args:
        data (xr.DataArray): The input data array.
        shapes_gdf (gpd.GeoDataFrame): The GeoDataFrame containing shape geometries (e.g. provinces).
        data_crs (str | None, optional): The CRS of the data array if not stored in data. Defaults to None.
    Returns:
       xr.DataArray: indicator matrix (n_shapes x pixels)
    """

    # Handle CRS - check if ds has crs attribute, otherwise use provided ds_crs
    if hasattr(data, 'crs') and data.crs is not None:
        data_crs = data.crs
    elif data_crs is not None:
        data_crs = data_crs
    else:
        # Default to WGS84 if no CRS provided
        data_crs = 'EPSG:4326'
        logger.warning(f"Warning: No CRS found for dataset, assuming {data_crs}")
    
    # Get grid parameters
    height, width = data.shape
    x_coords = data.coords["x"].values
    y_coords = data.coords["y"].values
    
    # Map coordinates into pixel space
    x_res = (x_coords.max() - x_coords.min()) / (len(x_coords) - 1)
    y_res = (y_coords.max() - y_coords.min()) / (len(y_coords) - 1)
    transform = Affine.translation(x_coords.min() - x_res/2, y_coords.max() + y_res/2) * Affine.scale(x_res, -y_res)

    # Ensure shapes_gdf is in the same CRS as the data
    if str(shapes_gdf.crs) != str(data_crs):
        logger.info(f"Converting province shapes from {shapes_gdf.crs} to {data_crs}")
        shapes_gdf_proj = shapes_gdf.to_crs(data_crs)
    else:
        shapes_gdf_proj = shapes_gdf

    n_shapes = len(shapes_gdf_proj)
    indicator_arrays = []

    logger.info(f"Rasterizing {n_shapes} provinces or shapes...")

    for shp_idx, (_, shp_geom) in enumerate(shapes_gdf_proj.geometry.items()):
        # Rasterize this province
        province_mask = features.rasterize(
            [(shp_geom, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Flatten and store
        indicator_arrays.append(province_mask.ravel())

        if shp_idx % 10 == 0:
            logger.info(f"Rasterized {shp_idx+1}/{n_shapes} provinces | shapes | nodes")

    logger.info("Finished rasterizing all provinces | shapes | nodes.")
    
    # Stack all province masks
    indicator_matrix = np.column_stack(indicator_arrays)

    # Convert to sparse for memory efficiency
    indicator_matrix = sp.csr_matrix(indicator_matrix)

    logger.info(f"Created rasterized indicator matrix with shape {indicator_matrix.shape}")

    return indicator_matrix


def assign_cells_to_shape_fast(
    data: xr.DataArray,
    indicator_matrix: sp.csr_matrix,
    band_name: str = "prov_index",
) -> xr.DataArray:
    """Assign each cell to a shape (province) using sparse matrix operations.

    Optimized to work directly with sparse matrices without converting to
    dense. 10-100x faster than assign_cells_to_shape for typical datasets.

    Args:
        data (xr.DataArray): The input data array (should be squeezed to 2D).
        indicator_matrix (sp.csr_matrix): Sparse indicator matrix
            (pixels x shapes).
        band_name (str): Name for the assignment band.
            Defaults to "prov_index".

    Returns:
        xr.DataArray: Data array with new 'shape_assignment' band added.
    """

    if data.ndim == 3 and "band" in data.dims:
        data_ = data.sel(band=data["band"][0])
    elif data.ndim == 2:
        data_ = data
    else:
        raise ValueError("Data array must be 2D or 3D with band dimension.")
    
    n_pixels = indicator_matrix.shape[0]
    n_shapes = indicator_matrix.shape[1]

    # Initialize assignment array with -1 (no province)
    cell_assignment = np.full(n_pixels, -1, dtype=np.int16)

    # Convert to CSC format for efficient column access
    I_csc = indicator_matrix.tocsc()

    # For each shape (province), find which pixels belong to it
    for shape_idx in range(n_shapes):
        # Get pixels that belong to this shape (non-zero entries in column)
        pixel_indices = I_csc.getcol(shape_idx).nonzero()[0]
        cell_assignment[pixel_indices] = shape_idx

    # Reshape to match data shape
    pixel_shp_id = cell_assignment.reshape(data_.shape)
    
    # Create assignment band with same coordinates
    assignment_band = xr.DataArray(
        pixel_shp_id[np.newaxis, :, :],  # Add band dimension
        coords={
            "band": [band_name],
            "y": data.coords["y"],
            "x": data.coords["x"],
        },
        dims=["band", "y", "x"],
    )

    data_with_assignment = xr.concat(
        [data, assignment_band], dim="band"
    )

    return data_with_assignment


def add_shape_id(data: xr.DataArray, shape: gpd.GeoDataFrame, new_band_name: str):
    """Add a new band to the data array with shape IDs.
    
    Args:
        data (xr.DataArray): The input data array.
        shape (gpd.GeoDataFrame): The GeoDataFrame containing shape geometries (e.g. provinces).
        new_band_name (str): Name of the new band to add.
    Returns:
        xr.DataArray: Data array with new shape ID band added."""
    
    I_mat = compute_shape_indicators_fast(data.sel(band=data["band"][0]), shape)
    logger.info("Assigning shape IDs using indicator matrix..")
    data_with_assignment = assign_cells_to_shape_fast(data, I_mat, band_name=new_band_name)
    return data_with_assignment


def calculate_cell_areas(data: xr.DataArray) -> np.ndarray:
    """Calculate area of each grid cell in km^2 for lat/lon grid.
    
    Args:
        data (xr.DataArray): Data with x (longitude) and y (latitude) coordinates.
    
    Returns:
        np.ndarray: Flattened array of cell areas in km^2.
    """
    # Get coordinates
    x_coords = data.coords["x"].values
    y_coords = data.coords["y"].values
    
    # Calculate resolution
    x_res = (x_coords.max() - x_coords.min()) / (len(x_coords) - 1)
    y_res = (y_coords.max() - y_coords.min()) / (len(y_coords) - 1)
    
    # Create meshgrid for lat/lon
    lon_grid, lat_grid = np.meshgrid(x_coords, y_coords)
    
    # Calculate area using spherical Earth approximation
    # Area = (lat_spacing * R) * (lon_spacing * R * cos(lat))
    R_earth = 6371.0  # km
    lat_rad = np.deg2rad(lat_grid)
    cell_area_km2 = (np.deg2rad(y_res) * R_earth) * (np.deg2rad(x_res) * R_earth * np.cos(lat_rad))
    
    return cell_area_km2.ravel()


def scale_prov_totals_and_find_rural(assigned_data: xr.DataArray, prov_data: pd.DataFrame | gpd.GeoDataFrame) -> tuple[xr.DataArray, pd.DataFrame]:
    """Rescale population values to match expected provincial totals and identify rural cells.

    1. For each province, calculate the scaling factor to match the expected province population.
    2. For each province find the density cut-off to separate rural based province urban population.
    3. Add band indicating rural population cells.
    4. Report statistics per province for monitoring.

    Args:
        assigned_data (xr.DataArray): pop data with shape_id assignment: bands=["population", "prov_index"].
        prov_data (pd.DataFrame | gpd.GeoDataFrame): Expected population per province (e.g. indexed 0-30).

    Returns:
        tuple[xr.DataArray, pd.DataFrame]: 
            - DataArray with population band rescaled and rural band added
            - DataFrame with cut-off statistics per province
    """
    # Get the bands
    pop_data = assigned_data.sel(band="population")
    prov_idx = assigned_data.sel(band="prov_index")
    
    # Create flat arrays for fast computation
    pop_flat = pop_data.values.ravel()
    prov_flat = prov_idx.values.ravel()
    
    # Calculate cell areas once
    cell_area_flat = calculate_cell_areas(assigned_data)
    
    # Initialize rural bool and tracking arrays
    is_rural = np.zeros_like(pop_flat, dtype=bool)
    cut_off_stats = []
    scaling_factors = np.zeros(len(prov_data))
    
    # Process each province
    for i, prov in enumerate(prov_data.index):
        mask = prov_flat == i
        raster_total = pop_flat[mask].sum()
        
        # Calculate scaling factor
        if raster_total > 0:
            scaling_factors[i] = prov_data.iloc[i].population / raster_total
        else:
            scaling_factors[i] = 0.0
        
        # Get province-specific values and sort by population
        prov_pop_vals = pop_flat[mask]
        prov_areas = cell_area_flat[mask]
        sort_indices = np.argsort(prov_pop_vals)
        sorted_pop = prov_pop_vals[sort_indices]
        sorted_areas = prov_areas[sort_indices]

        # Find rural cut-off
        pop_rural = prov_data.iloc[i].population * (1 - prov_data.iloc[i].urban_fraction)
        cut_off_idx = (sorted_pop.cumsum() < pop_rural).sum()

        # Mark rural cells
        is_rural[mask] = (prov_pop_vals <= sorted_pop[cut_off_idx-1])
        
        # Get cut-off cell statistics
        if cut_off_idx < len(sorted_pop):
            cut_off_pop = sorted_pop[cut_off_idx]
            cut_off_area = sorted_areas[cut_off_idx]
            cut_off_density = cut_off_pop / cut_off_area
        else:
            cut_off_pop = 0.0
            cut_off_area = 0.0
            cut_off_density = 0.0
        
        cut_off_stats.append({
            'province': prov,
            'cut_off_idx': cut_off_idx,
            'cut_off_population': cut_off_pop,
            'cut_off_area_km2': cut_off_area,
            'cut_off_density_per_km2': cut_off_density,
            'n_cells_province': mask.sum(),
            'n_cells_rural': is_rural[mask].sum(),
            'rural_fraction_cells': is_rural[mask].sum() / mask.sum() if mask.sum() > 0 else 0.0
        })

    # Convert to DataFrame
    cut_off_df = pd.DataFrame(cut_off_stats)

    # Rescale population using scaling factors
    scaling_map = scaling_factors[prov_flat.astype(int)]
    pop_rescaled_flat = pop_flat * scaling_map
    pop_rescaled = pop_rescaled_flat.reshape(pop_data.shape)

    # Create new DataArray with rescaled population
    pop_rescaled_da = xr.DataArray(
        pop_rescaled,
        coords=pop_data.coords,
        dims=pop_data.dims
    )
    # Create rural layer
    rural = xr.DataArray(
        is_rural.reshape(pop_data.shape),
        coords=pop_data.coords,
        dims=pop_data.dims
    ).expand_dims("band").assign_coords(band=["is_rural"])

    # Replace population band and add rural band
    result = assigned_data.copy()
    result.loc[dict(band="population")] = pop_rescaled_da

    return xr.concat([result, rural], dim='band'), cut_off_df


def group_and_sum_by_mask(data: xr.DataArray, value_band: str, group_band: str) -> pd.Series:
    """Efficiently group values by mask and sum using numpy bincount.
    
    This is much faster than xarray groupby for large datasets.
    
    Args:
        data (xr.DataArray): Data with multiple bands.
        value_band (str): Band to sum (e.g., "population").
        group_band (str): Band to group by (e.g., "prov_index").
    
    Returns:
        pd.Series: Summed values per group, indexed by group ID.
    """
    # Extract bands and flatten
    values = data.sel(band=value_band).values.ravel()
    groups = data.sel(band=group_band).values.ravel().astype(int)
    
    # Filter out unassigned cells (group_id == -1)
    valid_mask = groups >= 0
    values_valid = values[valid_mask]
    groups_valid = groups[valid_mask]
    
    # Use bincount for fast aggregation
    # bincount is O(n) and much faster than groupby for this use case
    n_groups = groups_valid.max() + 1
    sums = np.bincount(groups_valid, weights=values_valid, minlength=n_groups)
    
    return pd.Series(sums, index=range(n_groups), name=value_band)


if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "build_population_layouts",

        )

    configure_logging(snakemake, logger=logger)

    pop_year = snakemake.params.pop_year


    # provincial shapes and population
    province_shapes = gpd.read_file(snakemake.input.province_shape).set_index("province")
    admin2_shapes = read_admin2_shapes(snakemake.input.admin_l2_shape)
    population_provs = read_generic_province_data(
        snakemake.input.province_populations, index_col=0, index_name="province")
    population_provs = population_provs[str(pop_year)]*snakemake.params.pop_conversion
    if not population_provs.sum() > 1e9:
        logger.warning("Low population count, check headcount conversion")

    urban_frac = read_generic_province_data(snakemake.input.urban_percent, index_col=0, index_name="province")
    urban_frac = urban_frac[str(pop_year)]/100

     # merge population and geometry
    prov_data = gpd.GeoDataFrame(
        {
            "geometry": province_shapes.geometry,
            "population": population_provs,
            "urban_fraction": urban_frac,
        },
        crs=province_shapes.crs.to_string(),
    )

    # raster data population (pop per pixel)
    high_res_pop = rioxarray.open_rasterio(snakemake.input.population_gridded).assign_coords(band=["population"])
    # Set missing value to 0
    high_res_pop = high_res_pop.clip(0)
    high_res_pop.attrs["_FillValue"] = 0
    # coarsen as requested
    coarsening_factor = snakemake.params.get("coarsen_pop_by", 3)
    high_res_pop = high_res_pop.coarsen(x=coarsening_factor, y=coarsening_factor, boundary='trim').sum()

    # assign shapes
    processed_raster = add_shape_id(high_res_pop, province_shapes, new_band_name="prov_index")
    processed_raster = add_shape_id(processed_raster, admin2_shapes, new_band_name="l2_id")
    processed_raster, cut_off_stats = scale_prov_totals_and_find_rural(processed_raster, prov_data)

    logger.info("Aggregating population by admin2 regions...")
    population_by_admin = group_and_sum_by_mask(
        processed_raster, 
        value_band="population", 
        group_band="l2_id"
    )
    logger.info("Saving admin2 population data.")
    population_by_admin = pd.concat([admin2_shapes[["NAME_2", "NAME_1"]], population_by_admin], axis=1)
    print(population_by_admin.head())
    population_by_admin.to_csv(snakemake.output.admin2_population)

    # Save data
    logger.info("Saving population layout data.")
    processed_raster.to_netcdf(snakemake.output.pop_layouts)

    logger.debug("Rural Cut-off statistics per province:")
    logger.debug(cut_off_stats)

    logger.info("Successfully built population layouts.")