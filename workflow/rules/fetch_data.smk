# snakemake rules for data fetch operations
from zipfile import ZipFile
import shutil
import os

# TODO rework this, save shapes with all admin levels
# build nodes with another script and save that to DERIVED_DATA
# nodes could be read by snakefile and passed as a param to the relevant rules
rule fetch_region_shapes:
    """Fetch administrative shape polygons"""
    output:
        country_shape="resources/data/regions/country.geojson",
        province_shapes="resources/data/regions/provinces_onshore.geojson",
        offshore_shapes="resources/data/regions/provinces_offshore.geojson",
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

print(f"Raster Pop target is: resources/data/population/china_world_pop_{name}.tif")
print(f"from https://data.worldpop.org/GIS/Population/Global_2015_2030/R{release_yr}{rev}/{yr}/CHN/v{v}/1km_ua/constrained/")

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

# TODO build actual fetch rules with the sentinel/copernicus APIs.
# TODO See if there are datasets succeeding the S2 LC100 cover to get newer data
if config["enable"].get("retrieve_raster", True):

    rule retrieve_Grass_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Grass-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/landuse_availability/Grass.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Bare_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Bare-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/landuse_availability/Bare.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Shrubland_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Shrub-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/landuse_availability/Shrubland.tif"
        run: shutil.move(input[0], output[0])
    rule retrieve_Build_up_raster:
        input: storage.http("https://zenodo.org/record/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_BuiltUp-CoverFraction-layer_EPSG-4326.tif")
        output: "resources/data/landuse_availability/Build_up.tif"
        run: shutil.move(input[0], output[0])

    rule retrieve_bathymetry_raster:
        input:
            gebco=storage.http(
                "https://zenodo.org/record/16792792/files/GEBCO_tiff.zip"
            ),
        output:
            gebco="resources/data/landuse_availability/GEBCO_tiff/gebco_2024_CN.tif",
        params:
            zip_file="resources/data/landuse_availability/GEBCO_tiff.zip",
        run:
            os.rename(input.gebco, params.zip_file)
            with ZipFile(params.zip_file, "r") as zip_ref:
                zip_ref.extractall(os.path.dirname(params.zip_file))
            os.remove(params.zip_file)

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

# rule retrieve_gdp:
#     """ See https://gee-community-catalog.org/projects/gridded_gdp_hdi/
#     ML-based study https://www.nature.com/articles/s41597-025-04487-x .
#     Use this until easier to get official county level data for China.
#     Also possible to get data via google earth engine.
#     """
#     input:
#         kummu_et_al_geo = storage.http("https://zenodo.org/records/16741980/files/polyg_adm2_gdp_perCapita_1990_2022.gpkg"),
#         kummu_et_al_raster = storage.http("https://zenodo.org/records/16741980/files/rast_adm2_gdp_perCapita_1990_2022_30arcmin.tif"),
#         kummu_et_al_raster_hr = storage.http("https://zenodo.org/records/16741980/files/rast_adm2_gdp_perCapita_1990_2022.tif"),
#         kummu_et_al_gdp_adm2 = storage.http("https://zenodo.org/records/16741980/files/rast_gdpTot_1990_2022_30arcmin.tif")
#     output:
#         kummu_et_al_geo = "resources/data/population/adm2_gdp_percapita_1990_2022.gpkg",
#         kummu_et_al_raster = "resources/data/population/adm2_gdp_percapita_1990_2022.tif",
#         kummu_et_al_raster_hr = "resources/data/population/adm2_gdp_percapita_1990_2022_hr.tif",
#         kummu_et_al_gdp_adm2 = "resources/data/population/rast_gdpTot_1990_2022_30arcmin.tif"
#     run:
#         os.makedirs(os.path.dirname(output.kummu_et_al_geo), exist_ok=True)
#         shutil.move(input.kummu_et_al_geo, output.kummu_et_al_geo)
#         shutil.move(input.kummu_et_al_raster, output.kummu_et_al_raster)
#         shutil.move(input.kummu_et_al_raster_hr, output.kummu_et_al_raster_hr)
#         shutil.move(input.kummu_et_al_gdp_adm2, output.kummu_et_al_gdp_adm2)


if config["enable"].get("retrieve_cutout", False) and config["enable"].get(
    "build_cutout", False
):
    raise ValueError(
        "Settings error: you must choose between retrieving a pre-built cutout or building one from scratch"
    )
elif config["enable"].get("retrieve_cutout", False):

    rule retrieve_cutout:
        input:
            zenodo_cutout = storage.http("https://zenodo.org/record/16792792/files/China-2020c.nc"),
        output:
            cutout = "resources/cutouts/China-2020c.nc",
        run:
            os.makedirs(os.path.dirname(output.cutout), exist_ok=True)
            shutil.move(input.zenodo_cutout, output.cutout)

