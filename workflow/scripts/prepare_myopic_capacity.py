"""Add solution to brownfield capacity file"""

import logging
import pypsa
import pandas as pd

from prepare_existing_capacities import CARRIER_MAP
from _helpers import configure_logging, mock_snakemake


logger = logging.getLogger(__name__)

EXTRA_CARRIERS = {}
CARRIER_MAP.update(EXTRA_CARRIERS)


def _previous_horizon(yr) -> int:
    """find previous year"""
    years = [int(y) for y in config["scenario"]["planning_horizons"]]
    prev_idx = max(0, years.index(yr) - 1)
    return years[prev_idx]


# TODO remove coal retrofit from coal
def calculate_solved_capacities(
    previous_network: pypsa.Network, year: int, threshold: float = 1e-6
) -> pd.DataFrame:
    """
    Calculate the capacities installed in a previous solution.

    Args:
        previous_network (pypsa.Network): The solved PyPSA network from the previous horizon.
        year (int): The planning year.
        threshold (float, optional): Minimum capacity to include. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with solved capacities in existing_infrastructure format.
    """

    components = ["Generator", "Link", "Store"]
    allowed_carriers = list(CARRIER_MAP.values()) + list(CARRIER_MAP.keys())
    solved_capacities = pd.DataFrame()
    for c in components:
        # print(c)
        prefix = "e" if c == "Store" else "p"
        comp = getattr(previous_network, c.lower() + "s").query(f"{prefix}_nom_extendable==True")
        comp = comp[comp.carrier.isin(allowed_carriers)]
        installed = comp.groupby(["carrier", "location"])[f"{prefix}_nom_opt"].sum()
        lifetime = comp.groupby(["carrier", "location"])["lifetime"].mean()
        comp_data = pd.concat([lifetime, installed], axis=1).reset_index()
        comp_data["Tech"] = comp_data["carrier"].map(CARRIER_MAP)
        comp_data["Fuel"] = comp_data["carrier"]
        comp_data.index = comp_data.location + "-" + comp_data.Fuel + f"-{year}"

        # reformat as existing infrastructure table
        df = pd.DataFrame(
            {
                "DateIn": [int(year)] * len(comp_data),
                "DateOut": year + comp_data.lifetime.astype(int),
                "Tech": comp_data.carrier.map(CARRIER_MAP),
                "Fueltype": comp_data.carrier,
                "Capacity": comp_data[f"{prefix}_nom_opt"],
                "bus": comp_data.location,
            },
            index=comp_data.index,
        )

        if threshold:
            df = df[df.Capacity > threshold]
        solved_capacities = pd.concat([solved_capacities, df])

    return solved_capacities


def get_solved_edges(
    previous_network: pypsa.Network, carriers=["AC", "H2"]
) -> dict[str, pd.DataFrame]:
    """Get the previously solved network edges (transmission lines/pipelines)
    Args:
        previous_network (pypsa.Network): The solved PyPSA network from the previous horizon.
        carriers (Optional, list): carriers to include
    """

    n_p = previous_network
    edges = {}
    for carrier in carriers:
        connects = n_p.links.query(
            "bus0.map(@n_p.buses.carrier) == "
            "bus1.map(@n_p.buses.carrier) & "
            f"carrier == '{carrier}' & "
            "not index.str.contains('reverse')"
        )[["bus0", "bus1", "p_nom_opt", "p_nom"]]

        connects["p_nom_opt"] = connects.apply(lambda row: max(row.p_nom_opt, row.p_nom), axis=1)
        edges[carrier] = connects[["bus0", "bus1", "p_nom_opt"]]
    return edges


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "update_brownfield_with_solved",
            co2_pathway="exp175default",
            planning_horizons="2025",
            topology="current+FCG",
            heating_demand="positive",
            configfiles="config/myopic.yml",
        )

    configure_logging(snakemake)
    config = snakemake.config

    plan_yr = int(snakemake.wildcards.planning_horizons)
    prev_plan_yr = _previous_horizon(plan_yr)

    previous_network = pypsa.Network(snakemake.input.solved_network)
    brownfield_capacities = pd.read_csv(snakemake.input.installed_capacities)

    installed_last_step = calculate_solved_capacities(
        previous_network,
        prev_plan_yr,
        threshold=config["existing_capacities"].get("threshold_capacity", None),
    )

    installed_all = pd.concat([brownfield_capacities, installed_last_step])
    # TODO Group by name
    # installed_all.groupby([""])

    edges = get_solved_edges(previous_network, carriers=snakemake.params.edge_carriers)

    installed_all.to_csv(snakemake.output.myopic_capacities, index=False)

    for carrier, df in edges.items():
        df.to_csv(snakemake.output[f"edges_{carrier}"], index=False, header=False)
