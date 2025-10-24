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

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import scipy.sparse as sp
import xesmf as xe

from _helpers import configure_logging, mock_snakemake
from constants import PROV_NAMES
from readers import read_generic_province_data


from rasterio import features
import rioxarray
from affine import Affine


def compute_shape_indicators_fast(data: xr.DataArray, shapes_gdf: gpd.GeoDataFrame, data_crs: str | None = None) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert polygons to raster masks. This is a sparse matrix mapping shape to pixel.
    The indicator is binary with no area fraction.

    Unlike atlite do not rely on shapely geometry overlap but rasterio.features.rasterize.
    This is faster and enables working at higher resolution than the era5 grid. Resulting assignments
    are accurate.

    Args:
        data (xr.DataArray): The input data array.
        shapes_gdf (gpd.GeoDataFrame): The GeoDataFrame containing shape geometries (e.g. provinces).
        data_crs (str | None, optional): The CRS of the data array if not stored in data. Defaults to None.
    Returns:
        tuple(xr.DataArray, xr.DataArray): indicator matrix (n_shapes x pixels), grid coordinates (pixels x 2)
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

    # Stack all province masks
    indicator_matrix = np.column_stack(indicator_arrays)

    # Convert to sparse for memory efficiency
    indicator_matrix = sp.csr_matrix(indicator_matrix)

    # Create grid coordinates
    xs, ys = np.meshgrid(x_coords, y_coords)
    grid_coords = np.column_stack([xs.ravel(), ys.ravel()])

    logger.info(f"Created rasterized indicator matrix with shape {indicator_matrix.shape}")

    return indicator_matrix, grid_coords


def assign_cells_to_shape(data: xr.DataArray, indicator_matrix: sp.csr_matrix, band_name: str="prov_index") -> xr.DataArray:
    """
    Assign each cell to a shape (province) and add as a new band to the data.
    
    Args:
        data (xr.DataArray): The input data array (should be squeezed to 2D).
        indicator_matrix (sp.csr_matrix): Sparse indicator matrix (pixels x shapes).
    
    Returns:
        xr.DataArray: Data array with new 'shape_assignment' band added.
    """
    I_dense = indicator_matrix.T.toarray()  # Convert to dense for easier indexing
    cell_assignment = np.argmax(I_dense, axis=0)  # Get province index for each cell
    cell_assignment = np.where(I_dense.sum(axis=0) > 0, cell_assignment, -1)  # -1 for no province
    
    # Reshape to match data shape
    pixel_shp_id = cell_assignment.reshape(data.shape)
    
    # Add as new band to data with named coordinate
    # If data already has a band dimension, stack along it; otherwise create one
    if 'band' not in data.dims:
        data_expanded = data.expand_dims({'band': ['population']})
    else:
        data_expanded = data
    

    # Create assignment band with same coordinates
    assignment_band = xr.DataArray(
        pixel_shp_id[np.newaxis, :, :],  # Add band dimension
        coords={
            'band': [band_name],
            'y': data.coords['y'],
            'x': data.coords['x']
        },
        dims=['band', 'y', 'x']
    )
    data_with_assignment = xr.concat([data_expanded, assignment_band], dim='band')
    
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


def aggregate_to_cutout_grid(high_res_data: xr.DataArray, cutout: atlite.Cutout) -> xr.DataArray:
    """Aggregate high-resolution population data to cutout grid resolution using area-weighted sum.
    
    This performs CONSERVATIVE aggregation (preserves totals) unlike interp_like which interpolates.
    
    Methods ranked by accuracy:
    1. xESMF conservative regridding - BEST: true area-weighted, handles any grid misalignment
    2. xarray.coarsen() - GOOD: exact if grids aligned with integer ratio
    3. rioxarray.reproject_match(sum) - OK: approximation for misaligned grids
    
    Args:
        high_res_data (xr.DataArray): High-res population with bands [population, prov_index, is_rural]
        cutout (atlite.Cutout): Target cutout with coarse grid
        
    Returns:
        xr.DataArray: Population layouts at cutout resolution with bands:
            - 'total': Total population
            - 'rural': Rural population
            - 'rural_fraction': Fraction of cell that is rural (0-1)
            - 'prov_index': Dominant province index (integer)
    """
    
    # Create target grid template matching cutout coordinates
    target_template = xr.DataArray(
        np.zeros((len(cutout.coords['y']), len(cutout.coords['x']))),
        coords={'y': cutout.coords['y'], 'x': cutout.coords['x']},
        dims=['y', 'x']
    )
    
    # Extract bands
    population = high_res_data.sel(band='population')
    is_rural = high_res_data.sel(band='is_rural')
    prov_index = high_res_data.sel(band='prov_index')
    
    # Calculate rural population
    pop_rural_highres = population.where(is_rural == 1, 0)
    
    # For provinervative regridding (BEST - true area weighting)
    # Build regridder once (computationally expensive, but reusable)
    logger.info("Building xESMF conservative regridder...")
    
    regridder = xe.Regridder(
        population, target_template,
        method='conservative',
        periodic=False,
        ignore_degenerate=True  # Handle polar regions
    )
    
    # Apply to all quantities (fast after regridder is built)
    pop_total_coarse = regridder(population)
    pop_rural_coarse = regridder(pop_rural_highres)
    rural_fraction = pop_rural_coarse / pop_total_coarse.where(pop_total_coarse > 0, np.nan)
    rural_fraction = rural_fraction.fillna(0).clip(0, 1)  # Ensure 0-1 range

    # Validate conservation
    total_in = float(population.sum())
    total_out = float(pop_total_coarse.sum())
    conservation_error = abs(total_out - total_in) / total_in * 100
    logger.info(f"Conservation error: {conservation_error:.3f}%")
    if conservation_error > 1.0:
        logger.warning(f"Conservation error > 1%: {conservation_error:.2f}%")

    regridder.clean_weight_file()  # Clean up temporary files

    # Combine into multi-band DataArray
    coarse_layouts = xr.concat(
        [pop_total_coarse, pop_rural_coarse, rural_fraction, prov_index_coarse],
        dim='band'
    ).assign_coords(band=['total', 'rural', 'rural_fraction', 'prov_index'])

    return coarse_layouts


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




if __name__ == "__main__":
    if "snakemake" not in globals():

        snakemake = mock_snakemake(
            "build_population_layouts",
        )

    configure_logging(snakemake)

    pop_year = snakemake.params.pop_year
    cutout = atlite.Cutout(snakemake.input.cutout)
    grid_cells = cutout.grid.geometry

    # provincial shapes and population
    province_shapes = gpd.read_file(snakemake.input.province_shape).set_index("province")
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
    # TODO squeeze or not?

    # indicator matrix province -> high res grid cells
    I, coords_fast = compute_shape_indicators_fast(high_res_pop.squeeze(), province_shapes)

    shape_assigned_data = assign_cells_to_shape(high_res_pop, I)

    processed_pop_raster, cut_off_stats = scale_prov_totals_and_find_rural(shape_assigned_data, prov_data)


    # now do the same for nodes
    # TODO: add node assignment and processing

    # Aggregate high-res population to cutout grid resolution
    # Uses xESMF conservative regridding to preserve population totals
    cutout_pop_layouts = aggregate_to_cutout_grid(processed_pop_raster, cutout)
    
    # Save outputs
    logger.info(f"Cutout population layouts shape: {cutout_pop_layouts.shape}")
    logger.info(f"Bands: {list(cutout_pop_layouts.coords['band'].values)}")
    
    # Save individual bands
    for band_name in cutout_pop_layouts.coords['band'].values:
        layout_data = cutout_pop_layouts.sel(band=band_name)
        logger.info(f"Band '{band_name}' - Total population: {float(layout_data.sum()):,.0f}")
        # layout_data.to_netcdf(snakemake.output[f'pop_layout_{band_name}'])
    
