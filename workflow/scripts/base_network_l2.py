"""
The base network for China at admin L2 resolution.

This is a temporary step towards full network resolution using the OSM basemap.

The lines are simplified to use the transport model (dispatchable links).
"""

# TODO lines extendable
# TODO line costs (from GEM data?) -> flag use_project_costs, currency converter
# TODO key HV lines pypsa-earth
# TODO decide on whether year/snapshots belongs here / check vs pypsa-how they do it
# TODO split UHV AC lines into multiple segments (cost per km per watt then done outside)

import geopandas as gpd
import pypsa
import pandas as pd
from _helpers import configure_logging, mock_snakemake

# from readers import read_generic_province_data
from readers_geospatial import read_admin2_shapes
from _pypsa_helpers import make_periodic_snapshots, simplify_lines
from constants import PROV_RENAME_MAP, CHINA_INFLATION, COST_YEAR, YUAN_TO_EUR
from functions import apply_inflation_correction


def _buses_in_shape(buses, shape, crs="EPSG:4326"):
    bus_points = gpd.GeoDataFrame(
        gpd.points_from_xy(buses.x, buses.y),
        index=buses.index,
        crs=crs,
    )
    return gpd.sjoin_nearest(bus_points, shape)


def _clean_gem_lines(gem_lines: pd.DataFrame, exclude_provinces: list | pd.Index) -> pd.DataFrame:
    """Clean GEM UHV lines data to match model scope and years

    NOTE: ALL LINES CURRENTLY IMPLEMENTED AS LOSSY LINKS

    Args:
        gem_lines (pd.DataFrame): the raw GEM UHV lines data
        exclude_provinces (list|pd.Index): provinces to exclude
    Returns:
        pd.DataFrame: cleaned GEM UHV lines data
    """
    # rename (common fixes)
    gem_lines["Start_Region"] = gem_lines["Start_Region"].replace(PROV_RENAME_MAP)
    gem_lines["End_Region"] = gem_lines["End_Region"].replace(PROV_RENAME_MAP)

    gem_lines = gem_lines.query(
        "Start_Region not in @exclude_provinces and End_Region not in @exclude_provinces"
    )
    build_time_yrs = 3
    gem_lines["build_year"] = (
        gem_lines["Commissioned Year"].fillna(2025).astype(int) + build_time_yrs
    )

    gem_lines.rename(
        columns={
            "Technology": "type",
            "Capacity (MW)": "p_nom",
            "Transmission Distance（km）": "length",
        },
        inplace=True,
    )

    gem_lines["investment_per_km"] = (
        gem_lines["Investment (Billion RMB)"] / gem_lines["length"]
    )
    # TODO calculate capital cost using NPV and GEM investment cost & length. Convert to EUR/MW
    gem_lines = gem_lines[
        [
            "length",
            "p_nom",
            "Status",
            "Start_Admin2",
            "End_Admin2",
            "investment_per_km",
            "type",
            "build_year",
            "Start_Region",
            "End_Region",
            "UHV Power Transmission Line",
        ]
    ]
    gem_lines["p_nom"] = gem_lines.apply(
        lambda row: 0 if row["Status"] == "proposed" else row["p_nom"], axis=1
    )
    gem_lines["p_nom_min"] = gem_lines["p_nom"]
    
    # Convert investment costs to reference year (COST_YEAR) prices
    # Prepare inflation data DataFrame with yr_val column
    inflation = pd.DataFrame.from_dict(
        CHINA_INFLATION, orient='index', columns=["inflation"]
    )
    inflation["yr_val"] = inflation["inflation"] / 100 + 1
    
    # Apply inflation correction to each line based on its build_year
    gem_lines["investment"] = gem_lines["build_year"].apply(
        lambda year: apply_inflation_correction(year, inflation, COST_YEAR)
    ) * gem_lines["investment_per_km"] * 1e9 * YUAN_TO_EUR  /gem_lines.p_nom

    return gem_lines


def _validate_line_regions(lines: pd.DataFrame, admin_l2_shapes: gpd.GeoDataFrame):
    """check start and end points are valid

    Args:
        lines (pd.DataFrame): the line data
        admin_l2_shapes (gpd.GeoDataFrame): the administrative level 2 shapes

    Raises:
        ValueError: if any line has an invalid start or end region
    """
    mask_start = lines.Start_Admin2.isin(admin_l2_shapes.NAME_2.unique())
    mask_end = lines.End_Admin2.isin(admin_l2_shapes.NAME_2.unique())
    invalid = lines[~mask_start & ~mask_end]

    if not invalid.empty:
        raise ValueError(
            f"The following lines have invalid start or end regions: {invalid.index.tolist()}"
        )

def add_hv_for_sparse_regions(network: pypsa.Network, hv_data: pd.DataFrame):
    """Add HV lines for large regions with sparse admin level 2 coverage

    Args:
        network (pypsa.Network): the PyPSA network
        hv_data (pd.DataFrame): the HV line data
    """
    pass


def _build_admin2_buses(
    admin_l2_shapes: gpd.GeoDataFrame, center="centroid", country="CN"
) -> pd.DataFrame:
    """Build admin level 2 buses from admin_l2_shapes geodataframe centers

    Args:
        admin_l2_shapes (gpd.GeoDataFrame): admin level 2 shapes & names (inc lv1)
        center (str): method to determine bus location, either 'centroid' or 'representative_point'
    Returns:
        pd.DataFrame: DataFrame containing bus information
    """
    # HID index
    shapes = admin_l2_shapes[["NAME_1", "NAME_2", "NL_NAME_2", "geometry"]].copy()
    shapes.index = shapes.NAME_1 + "-" + shapes.groupby("NAME_1").cumcount().astype(str)
    if center == "centroid":
        buses = pd.DataFrame(
            index=shapes.index,
            data={
                "x": shapes.geometry.centroid.x,
                "y": shapes.geometry.centroid.y,
                "province": shapes.NAME_1,
                "prefecture": shapes.NAME_2,
                "prefecture_nl": shapes.NL_NAME_2,
                "country": country,
            },
        )
        return buses

    elif center == "representative_point":
        buses = pd.DataFrame(
            index=shapes.index,
            data={
                "x": shapes.geometry.representative_point().x,
                "y": shapes.geometry.representative_point().y,
                "province": shapes.NAME_1,
                "prefecture": shapes.NAME_2,
                "prefecture_nl": shapes.NL_NAME_2,
            },
        )
        return buses
    else:
        raise ValueError(f"Function {center} not recognized for bus location")


def _set_uhv_lines(network: pypsa.Network, lines: pd.DataFrame, buses: pd.DataFrame):
    """Set lines in the network based on the lines DataFrame and admin level 2 shapes

    Args:
        network (pypsa.Network): the PyPSA network
        lines (pd.DataFrame): the line data
        buses (pd.DataFrame): the network bus dataframe
    """

    # assign buses
    lines["bus0"] = lines.merge(
            network.buses.reset_index(),
            how="left",
            left_on=["Start_Region", "Start_Admin2"],
            right_on=["province", "prefecture"],
            suffixes=("", "_buses"),
        )["Bus"]

    lines["bus1"] = lines.merge(
            network.buses.reset_index(),
            how="left",
            left_on=["End_Region", "End_Admin2"],
            right_on=["province", "prefecture"],
            suffixes=("", "_buses"),
        )["Bus"]

    not_matched = lines.query("`bus0`.isna() or `bus1`.isna()")[
        ["UHV Power Transmission Line", "Start_Region", "Start_Admin2", "End_Region", "End_Admin2"]
    ]
    if not not_matched.empty:
        raise ValueError(f"The following UHV lines have unmatched buses: {not_matched}")

    lines_grouped = simplify_lines(lines)
    # FAKE CARRIER -> change if you have full network resolution
    lines_ = lines_grouped[[c for c in lines_grouped.columns if c in network.links.columns]]
    network.add("Link", lines_.index, **lines_, carrier="AC")

    # future ?
    # lines_["carrier"] = lines_.type.apply(lambda x: "DC" if x.str.contains("AC") else "AC")
    # hvdc_lines = lines_.query("carrier == 'DC'")
    # hvac_lines = lines_.query("carrier == 'AC'")


def _set_shapes(
    n: pypsa.Network,
    province_shapes: gpd.GeoDataFrame,
    offshore_shapes: gpd.GeoDataFrame,
):
    """Add province and offshore shapes to network.shapes component.

    Args:
        n (pypsa.Network): The PyPSA network.
        province_shapes (gpd.GeoDataFrame): Province geometries with 'idx' column.
        offshore_shapes (gpd.GeoDataFrame): Offshore geometries.
    """
    province_shapes["type"] = "province"
    offshore_shapes_renamed = offshore_shapes.rename(columns={"name": "idx"})
    offshore_shapes_renamed["type"] = "offshore"
    all_shapes = pd.concat([province_shapes, offshore_shapes_renamed], ignore_index=True)
    n.add(
        "Shape",
        all_shapes.index,
        geometry=all_shapes.geometry,
        idx=all_shapes.index,
        type=all_shapes["type"],
    )


def _set_snapshots(network: pypsa.Network, snapshot_cfg: dict, yr: int):
    """Set the snapshots for the network based on the snapshot configuration.
    Snapshots are for the specified year and in naive local time

    NOTE: drops leap days
    Args:
        network (pypsa.Network): The PyPSA network to set snapshots for.
        snapshot_cfg (dict): Configuration dictionary for snapshots.
        yr (int): The year for which to generate snapshots.
    """
    snapshots = make_periodic_snapshots(
        year=yr,
        freq=1,  # apply time resolution later
        start_day_hour=snapshot_cfg["start"],
        end_day_hour=snapshot_cfg["end"],
        bounds=snapshot_cfg["bounds"],
        tz=None,
        end_year=(None if not snapshot_cfg["end_year_plus1"] else yr + 1),
    )
    network.set_snapshots(snapshots)


def build_base_network(
    admin_l2_shapes: gpd.GeoDataFrame,
    prov_shapes: gpd.GeoDataFrame,
    offshore_shapes: gpd.GeoDataFrame,
    lines: pd.DataFrame,
    snapshot_config: dict,
    year: int,
) -> pypsa.Network:
    """Build the base network at admin level 2 resolution.

    Args:
        admin_l2_shapes (gpd.GeoDataFrame): the admin level 2 shapes
        prov_shapes (gpd.GeoDataFrame): the province shapes
        offshore_shapes (gpd.GeoDataFrame): the offshore shapes
        lines (pd.DataFrame): the UHV line data
        snapshot_config (dict): the snapshot configuration
        year (int): the model year
    Returns:
        pypsa.Network: the base network at admin l2 resolution
    """

    _validate_line_regions(lines, admin_l2_shapes)

    network = pypsa.Network(name="Base Network")

    buses = _build_admin2_buses(admin_l2_shapes)
    network.add("Bus", buses.index, **buses)

    _set_shapes(network, prov_shapes, offshore_shapes)
    _set_snapshots(network, snapshot_config, year)
    _set_uhv_lines(network, lines, buses)

    return network


if __name__ == "__main__":
    snakemake = mock_snakemake(
        "base_network",
        topology="current+FCG",
        co2_pathway="exp175default",
        heating_demand="positive",
        planning_horizons="2030",
    )

    configure_logging(snakemake)
    # mp.set_start_method("spawn", force=True)

    yr = int(snakemake.params.ref_year)
    node_config = snakemake.params.get("node_config", {})
    snapshot_config = snakemake.params.get("snapshots")
    exclude_provinces = node_config.get("exclude_provinces", [])
    lines_p = snakemake.params["lines_path"]

    province_shapes = gpd.read_file(snakemake.input.province_shapes).query(
        "~province.isin(@exclude_provinces)"
    )
    admin_l2_shapes = read_admin2_shapes(snakemake.input.admin_l2_shapes).query(
        "~NAME_1.isin(@exclude_provinces)"
    )
    offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes).query(
        "~NAME_1.isin(@exclude_provinces)"
    )

    # TODO use investment costs from gem lines
    lines = pd.read_csv(lines_p, skiprows=1)
    lines = _clean_gem_lines(lines, exclude_provinces)
    lines["p_max_pu"] = snakemake.params["line_margin"]
    # temp - APPLY NPV later in prepare network
    # TODO just export? 
    lines.rename(columns={"investment": "capital_cost"}, inplace=True)

    network = build_base_network(
        admin_l2_shapes, province_shapes, offshore_shapes, lines, snapshot_config, yr
    )

    # check order kept
    l1_ok = (network.buses.reset_index().province == admin_l2_shapes.reset_index().NAME_1).all()
    l2_ok = (network.buses.reset_index().prefecture == admin_l2_shapes.reset_index().NAME_2).all()
    if not (l1_ok and l2_ok):
        raise ValueError("Admin level 2 bus order does not match shapes order")

    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)

    network.export_to_netcdf(snakemake.output.base_network, compression=compression)
