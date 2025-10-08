"""file reading support functions"""

import os

import pandas as pd


def read_edges(edge_path: os.PathLike, nodes: pd.Index | list) -> pd.DataFrame:  
    """read edges csv data and validate vs nodes
    Args:
        edge_path (os.PathLike): path to the edges csv
        nodes (pd.Index | list): The nodes to validate against.
    Returns:
        pd.DataFrame: The edges dataframe.
    Raises:
        ValueError: if edge path not specified (None)
        ValueError: edge file vertices are not contained in nodes )skip withnone
    """
    if edge_path is None:
        raise ValueError(f"No grid found for topology path {edge_path}")
    else:
        edges = pd.read_csv(
            edge_path, sep=",", header=None, names=["bus0", "bus1", "p_nom"]
        ).fillna(0)

    # skip qa
    if nodes is None:
        return edges

    edges_uniq = set(edges["bus0"]).union(set(edges["bus1"]))
    if not edges_uniq.issubset(nodes):
        raise ValueError(f"Edges contain unknown nodes: {edges_uniq - set(nodes)}")

    return edges


def read_yearly_load_projections(
    yearly_projections_p: os.PathLike = "resources/data/load/Province_Load_2020_2060.csv",
    conversion=1,
) -> pd.DataFrame:
    """Prepare projections for model use

    Args:
        yearly_projections_p (os.PathLike, optional): the data path.
                Defaults to "resources/data/load/Province_Load_2020_2060.csv".
        conversion (int, optional): the conversion factor to MWh. Defaults to 1.

    Returns:
        pd.DataFrame: the formatted data, in MWh
    """
    yearly_proj = pd.read_csv(yearly_projections_p)
    yearly_proj.rename(columns={"Unnamed: 0": "province", "region": "province"}, inplace=True)
    if "province" not in yearly_proj.columns:
        raise ValueError(
            "The province (or region or unamed) column is missing in the yearly projections data"
            ". Index cannot be built"
        )
    yearly_proj.set_index("province", inplace=True)
    yearly_proj.rename(columns={c: int(c) for c in yearly_proj.columns}, inplace=True)

    return yearly_proj * conversion
