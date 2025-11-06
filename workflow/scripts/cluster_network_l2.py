"""
Cluster admin L2 network to province-level or custom regional aggregations.

Adapted from PyPSA-EUR clustering for PyPSA-China's admin hierarchy.
Clusters the high-resolution L2 network (341 prefectures) to province
or custom regional groupings defined in config.
"""

import logging
import warnings

import geopandas as gpd
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake
from packaging.version import Version, parse
from pypsa.clustering.spatial import get_clustering_from_busmap
from readers_geospatial import read_admin2_shapes
from constants import CRS

DISTANCE_CRS = "EPSG:3035"
PD_GE_2_2 = parse(pd.__version__) >= Version("2.2")

warnings.filterwarnings(action="ignore", category=UserWarning)
logger = logging.getLogger(__name__)

GEO_CRS = "EPSG:4326"


def build_busmap_from_config(
    buses: pd.DataFrame,
    admin_l2_shapes: gpd.GeoDataFrame,
    node_config: dict,
) -> pd.Series:
    """Build busmap for clustering admin L2 buses to provinces or custom regions.

    By default, each province becomes a single node. Custom splits in the config
    allow grouping L2 regions within a province into named clusters.

    Buses are named with hierarchical IDs:
    - Single-node provinces: "ProvinceName"
    - Split provinces: "ProvinceName_ClusterName"

    Example config:
        splits:
          InnerMongolia:
            West: ["Hulunbuir", "Xing'an", "Tongliao"]
            East: ["Alxa", "Baotou", "Hohhot"]

    This creates nodes: InnerMongolia_West, InnerMongolia_East
    Other provinces remain as single nodes: Sichuan, Guangdong, etc.

    Args:
        buses (pd.DataFrame): Network buses with 'province' and 'prefecture' columns
        admin_l2_shapes (gpd.GeoDataFrame): Admin L2 shapes with NAME_1, NAME_2
        node_config (dict): Config dict with 'splits' for custom aggregations

    Returns:
        pd.Series: Busmap mapping L2 bus index to cluster name (hierarchical ID)
    """
    splits = node_config.get("splits", {})
    
    # Initialize busmap with province names (default: one cluster per province)
    busmap = buses["province"].copy()
    
    # Apply custom splits for provinces with multiple clusters
    for province, clusters in splits.items():
        province_buses = buses[buses["province"] == province]
        
        if province_buses.empty:
            logger.warning(f"Province '{province}' in splits config not found in network")
            continue
        
        # Map each L2 prefecture to its cluster
        for cluster_name, prefectures in clusters.items():
            # Match buses by prefecture name
            mask = province_buses["prefecture"].isin(prefectures)
            matched_buses = province_buses[mask]
            
            if len(matched_buses) == 0:
                logger.warning(
                    f"No buses found for cluster '{cluster_name}' in '{province}'. "
                    f"Prefectures: {prefectures}"
                )
                continue
            
            # Create hierarchical cluster ID: Province_ClusterName
            cluster_id = f"{province}_{cluster_name}"
            busmap.loc[matched_buses.index] = cluster_id
            
            logger.info(
                f"Assigned {len(matched_buses)} buses to cluster '{cluster_id}'"
            )
        
        # Check for unmapped buses in split provinces
        province_mask = buses["province"] == province
        still_unmapped = buses[province_mask & (busmap == province)]
        if not still_unmapped.empty:
            logger.warning(
                f"Province '{province}' has {len(still_unmapped)} unmapped L2 buses. "
                f"Unmapped prefectures: {still_unmapped['prefecture'].tolist()}"
            )
    
    return busmap.rename("busmap")


def aggregate_regions(
    busmap: pd.Series,
    regions: gpd.GeoDataFrame,
    id_col: str = "name",
) -> gpd.GeoDataFrame:
    """Aggregate regional shapes based on busmap clustering.

    Args:
        busmap (pd.Series): Mapping from L2 bus ID to cluster name
        regions (gpd.GeoDataFrame): L2 regional shapes
        id_col (str): Column name for region identifier

    Returns:
        gpd.GeoDataFrame: Dissolved regions by cluster
    """
    regions = regions.copy()
    
    # Ensure index matches busmap
    if id_col in regions.columns and id_col != regions.index.name:
        regions = regions.set_index(id_col)
    
    # Reindex to match busmap
    regions = regions.reindex(busmap.index)
    regions["cluster"] = busmap.values
    
    # Dissolve geometries by cluster
    clustered = regions.dissolve(by="cluster", as_index=False)
    clustered = clustered.rename(columns={"cluster": id_col})
    
    return clustered


def update_bus_coordinates(
    n: pypsa.Network,
    busmap: pd.Series,
    shapes: gpd.GeoDataFrame,
    # geo_crs: str = GEO_CRS,
    # distance_crs: str = DISTANCE_CRS,
    # tol: float = BUS_TOL,
) -> None:
    """Update bus coordinates to clustered regions points.

    Uses geometric centroids of the aggregated regions.

    Args:
        n (pypsa.Network): Original L2 network
        busmap (pd.Series): Cluster assignments
        shapes (gpd.GeoDataFrame): L2 shapes with geometry
    """
    # merge (l2) shapes by custom cluster
    shapes["cluster"] = busmap.values
    # this is equivalent to disslolve?
    clusters = admin_l2_shapes.groupby("cluster").geometry.apply(lambda x: x.union_all("unary"))
    # TODO consider switching to Pole of Inaccessibility as pypsa-eur does
    points = clusters.representative_point()
    points.name="point"

    busmap_df = shapes.merge(points, left_on="cluster", right_index=True, how="left")
    busmap_df["x"] = busmap_df.point.x
    busmap_df["y"] = busmap_df.point.y

    n.buses["x"] = busmap_df["x"].values
    n.buses["y"] = busmap_df["y"].values

    # Drop admin-level columns
    n.buses.drop(columns=["prefecture","prefecture_cn"], inplace=True)

def clustering_for_busmap(
    n: pypsa.Network,
    busmap: pd.Series,
    aggregation_strategies: dict | None = None,
) -> pypsa.clustering.spatial.Clustering:
    """Create PyPSA clustering object and aggregate network.

    Args:
        n (pypsa.Network): Original L2 network
        busmap (pd.Series): Cluster assignments
        aggregation_strategies (dict | None): Custom aggregation strategies

    Returns:
        pypsa.clustering.spatial.Clustering: Clustering object with aggregated network
    """
    if aggregation_strategies is None:
        aggregation_strategies = {}

    line_strategies = aggregation_strategies.get("lines", {})
    
    bus_strategies = aggregation_strategies.get("buses", {})
    # Preserve any substation flags
    bus_strategies.setdefault("substation_lv", lambda x: bool(x.sum()))
    bus_strategies.setdefault("substation_off", lambda x: bool(x.sum()))

    clustering = get_clustering_from_busmap(
        n,
        busmap,
        bus_strategies=bus_strategies,
        line_strategies=line_strategies,
        custom_line_groupers=["build_year"],
    )

    return clustering


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "cluster_network",
            topology="current+FCG",
            co2_pathway="exp175default",
            heating_demand="positive",
            planning_horizons="2030",
        )

    configure_logging(snakemake)

    # Load base L2 network
    n = pypsa.Network(snakemake.input.base_network)
    buses_prev = len(n.buses)
    lines_prev = len(n.lines)
    links_prev = len(n.links)

    logger.info(f"Loaded base network: {buses_prev} buses, {lines_prev} lines, {links_prev} links")

    # Load admin L2 shapes for spatial aggregation
    node_config = snakemake.params.get("node_config", {})
    exclude_provinces = node_config.get("exclude_provinces", [])
    
    admin_l2_shapes = read_admin2_shapes(snakemake.input.admin_l2_shapes).query(
        "~NAME_1.isin(@exclude_provinces)"
    )

    # Check if clustering is needed
    if not node_config.get("split_provinces", False):
        busmap = n.buses["province"]
    else:
        # Build custom busmap from config
        busmap = build_busmap_from_config(
            n.buses,
            admin_l2_shapes,
            node_config,
        )
        
    # Update bus coordinates to cluster centroids
    update_bus_coordinates(n, busmap, admin_l2_shapes)
        
    # Perform clustering
    aggregation_strategies = snakemake.params.get("aggregation_strategies", {})
    clustering = clustering_for_busmap(
        n,
        busmap,
        aggregation_strategies=aggregation_strategies,
    )

    # Extract clustered network
    nc = clustering.n

    # Save outputs
    clustering.busmap.to_csv(snakemake.output.busmap)

    # Aggregate regional shapes
    if "regions_onshore" in snakemake.input:
        regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
        clustered_onshore = aggregate_regions(
            clustering.busmap,
            regions_onshore,
            id_col="name",
        )
        clustered_onshore.to_file(snakemake.output.regions_onshore)

    if "regions_offshore" in snakemake.input:
        regions_offshore = gpd.read_file(snakemake.input.regions_offshore)
        clustered_offshore = aggregate_regions(
            clustering.busmap,
            regions_offshore,
            id_col="name",
        )
        clustered_offshore.to_file(snakemake.output.regions_offshore)

    # Export clustered network
    nc.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    compression = snakemake.config.get("io", {}).get("nc_compression", None)
    nc.export_to_netcdf(snakemake.output.clusted_network, compression=compression)

    logger.info(
        f"Clustered network:\n"
        f"  Buses: {buses_prev} → {len(nc.buses)}\n"
        f"  Lines: {lines_prev} → {len(nc.lines)}\n"
        f"  Links: {links_prev} → {len(nc.links)}\n"
        f"  Unique clusters: {clustering.busmap.nunique()}"
    )
