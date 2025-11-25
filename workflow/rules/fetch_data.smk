# snakemake rules for data fetch operations
from zipfile import ZipFile
import shutil
import os


rule fetch_region_shapes:
    """Fetch administrative shape polygons"""
    output:
        country_shape="resources/data/regions/country.geojson",
        province_shapes="resources/data/regions/provinces_onshore.geojson",
        offshore_shapes="resources/data/regions/country_offshore.geojson",
        admin2_shapes="resources/data/regions/admin2_shapes.geojson",
    log:
        LOGS_COMMON + "/fetch_regions_shapes.log",
    script:
        "../scripts/fetch_shapes.py"


name = "-".join([str(v) for v in config["world_population_raster"].values()])
release_yr, rev, yr, v = (
    config["world_population_raster"]["release_yr"],
    config["world_population_raster"]["revision"],
    config["world_population_raster"]["year"],
    config["world_population_raster"]["version"],
)
rule fetch_gridded_population:
    params:
        config = config["world_population_raster"],
        base_url = f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R{release_yr}{rev}/{yr}/CHN/v{v}/1km_ua/constrained/"
    output:
        pop_raster= f"resources/data/population/china_world_pop_{name}.tif",
    log:
        LOGS_COMMON + "/fetch_gridded_population.log",
    script:
        "../scripts/fetch_gridded_pop.py"


# This replaces the old approach of using separate percentage cover fraction rasters
# The discrete classification map contains integer codes for different land cover types
# See: https://zenodo.org/records/3939050
if config["enable"].get("retrieve_raster", True):

    rule retrieve_copernicus_land_cover:
        input:
            storage.http(
                "https://zenodo.org/records/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
            ),
        output:
            "resources/data/landuse_availability/Copernicus_LC100_discrete_classification_2019.tif",
        run:
            os.makedirs(os.path.dirname(output[0]), exist_ok=True)
            shutil.move(input[0], output[0])

    rule retrieve_bathymetry_raster:
        input:
            gebco=storage.http(
                "https://zenodo.org/record/17697456/files/GEBCO_tiff.zip"
            ),
        output:
            gebco="resources/data/landuse_availability/GEBCO_tiff/gebco_2025_CN.tif",
        params:
            zip_file="resources/data/landuse_availability/GEBCO_tiff.zip",
        run:
            os.rename(input.gebco, params.zip_file)
            with ZipFile(params.zip_file, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(params.zip_file))
            os.remove(params.zip_file)


if config["enable"].get("retrieve_cutout", False) and config["enable"].get(
    "build_cutout", False
):
    raise ValueError(
        "Settings error: you must choose between retrieving a pre-built cutout or building one from scratch"
    )
elif config["enable"].get("retrieve_cutout", False):

    rule retrieve_cutout:
        input:
            zenodo_cutout=storage.http(
                "https://zenodo.org/record/16792792/files/China-2020c.nc"
            ),
        output:
            cutout="resources/cutouts/China-2020c.nc",
        run:
            os.makedirs(os.path.dirname(output.cutout), exist_ok=True)
            shutil.move(input.zenodo_cutout, output.cutout)


rule retrieve_powerplants:
    input:
        powerplants=storage.http(
            "https://zenodo.org/records/16810831/files/Global-integrated-Plant-Tracker-July-2025_china.xlsx"
        ),
    output:
        powerplants="resources/data/existing_infrastructure/gem_data_raw/Global-integrated-Plant-Tracker-July-2025_china.xlsx",
    run:
        os.makedirs(os.path.dirname(output.powerplants), exist_ok=True)
        shutil.move(input.powerplants, output.powerplants)


rule retrieve_raster_gdp:
    """ See https://gee-community-catalog.org/projects/gridded_gdp_hdi/
    ML-based study https://www.nature.com/articles/s41597-025-04487-x .
    Covers many years but not trained on l2 China at all.

    Not actually needed right now but kept for reference. Also possible to get data via google earth engine.
    """
    input:
        kummu_et_al_raster_hr = storage.http("https://zenodo.org/records/16741980/files/rast_adm2_gdp_perCapita_1990_2022.tif"),
        kummu_et_al_gdp_adm2 = storage.http("https://zenodo.org/records/16741980/files/rast_gdpTot_1990_2022_30arcmin.tif")
    output:
        kummu_et_al_raster = "resources/data/population/adm2_gdp_percapita_1990_2022.tif",
        kummu_et_al_gdp_adm2 = "resources/data/population/rast_gdpTot_1990_2022_30arcmin.tif"
    run:
        # os.makedirs(os.path.dirname(output.kummu_et_al_geo, exist_ok=True)
        shutil.move(input.kummu_et_al_raster, output.kummu_et_al_raster)
        # shutil.move(input.kummu_et_al_raster_hr, output.kummu_et_al_raster_hr)
        shutil.move(input.kummu_et_al_gdp_adm2, output.kummu_et_al_gdp_adm2)

