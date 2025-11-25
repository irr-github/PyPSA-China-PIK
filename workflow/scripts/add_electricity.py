""" """

import logging
import xarray as xr
import pandas as pd
import pypsa
from os import PathLike

from _helpers import mock_snakemake, configure_logging
from functions import calculate_annuity
from _pypsa_helpers import (
    assign_locations,
    make_periodic_snapshots,
    shift_profile_to_planning_year,
    rename_techs,
)
from readers import load_costs
from constants import NICE_NAMES_DEFAULT


logger = logging.getLogger(__name__)


# TODO understand why this is in make_summary but not in the main optimisation
def update_transmission_costs(n, costs, length_factor=1.0):
    """Update transmission line and link capital costs based on length and cost data.

    Args:
        n: PyPSA Network object
        costs: Cost data DataFrame containing capital costs for transmission technologies
        length_factor (float, optional): Factor to scale line lengths. Defaults to 1.0.

    Returns:
        None: Modifies network transmission costs in place
    """
    # TODO: line length factor of lines is applied to lines and links.
    # Separate the function to distinguish.

    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "capital_cost"]
    )

    if n.links.empty:
        return

    dc_b = n.links.carrier == "DC"

    # If there are no dc links, then the 'underwater_fraction' column
    # may be missing. Therefore we have to return here.
    if n.links.loc[dc_b].empty:
        return

    costs = (
        n.links.loc[dc_b, "length"]
        * length_factor
        * (
            (1.0 - n.links.loc[dc_b, "underwater_fraction"])
            * costs.at["HVDC overhead", "capital_cost"]
            + n.links.loc[dc_b, "underwater_fraction"] * costs.at["HVDC submarine", "capital_cost"]
        )
        + costs.at["HVDC inverter pair", "capital_cost"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs


def add_reverse_links(n: pypsa.Network) -> None:
    pass


def add_missing_carriers(n: pypsa.Network, carriers: list | set) -> None:
    """Function to add missing carriers to the network without raising errors.

    Args:
        n (pypsa.Network): the pypsa network object
        carriers (list | set): a list of carriers that should be included
    """
    missing_carriers = set(carriers) - set(n.carriers.index)
    if len(missing_carriers) > 0:
        n.add("Carrier", missing_carriers)


# TODO figure out whether still relevant
def sanitize_carriers(n: pypsa.Network, plot_config: dict) -> None:
    """Sanitize the carrier information in a PyPSA Network object.

    The function ensures that all unique carrier names are present in the network's
    carriers attribute, and adds nice names and colors for each carrier according
    to the provided configuration dictionary.

    Args:
        n (pypsa.Network): PyPSA Network object representing the electrical power system.
        plot_config (dict): A dictionary containing configuration information for plotting,
               specifically the "nice_names" and "tech_colors" keys for carriers.
    """
    # update default nice names w user settings
    nice_names = NICE_NAMES_DEFAULT.update(plot_config.get("nice_names", {}))
    for c in n.iterate_components():
        if "carrier" in c.df:
            add_missing_carriers(n, c.df.carrier)

    # sort the nice names to match carriers and fill missing with "ugly" names
    carrier_i = n.carriers.index
    nice_names = pd.Series(nice_names).reindex(carrier_i).fillna(carrier_i.to_series())
    # replace empty nice names with nice names
    n.carriers.nice_name.where(n.carriers.nice_name != "", nice_names, inplace=True)

    # TODO make less messy, avoid using map
    tech_colors = plot_config["tech_colors"]
    colors = pd.Series(tech_colors).reindex(carrier_i)
    # try to fill missing colors with tech_colors after renaming
    missing_colors_i = colors[colors.isna()].index
    colors[missing_colors_i] = missing_colors_i.map(lambda x: rename_techs(x, nice_names)).map(
        tech_colors
    )
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = n.carriers.color.where(n.carriers.color != "", colors)


# ### PLACEHOLDERS FOR REFACTOR TO PYPSA-EUR STYLE ###
# TODO move heat sector to a separate rule
def attach_load(
    n: pypsa.Network,
    load_path: PathLike,
    busmap_path: PathLike,
    scaling: float = 1.0,
) -> None:
    """Attach load data to the network.

    Args:
        n (pypsa.Network): The PyPSA network to attach the load data to.
        load_path (PathLike): Path to the load data file.
        busmap_path (PathLike): Path to the busmap file.
        scaling (float): Scaling factor for the load data. Defaults to 1.0.
    """
    load = xr.open_dataarray(load_path).to_dataframe().squeeze(axis=1).unstack(level="time")

    # apply clustering busmap
    busmap = pd.read_csv(busmap_path, dtype=str, index_col=0).squeeze()
    load = load.groupby(busmap).sum().T

    logger.info(f"Load data scaled by factor {scaling}.")
    load *= scaling

    # carrier="electricity"
    n.add("Load", load.columns, bus=load.columns, p_set=load.values)


def add_generators(n, ppl_data, carrier_list):

    pass


def add_links(n, ppl_data, carrier_list):

    pass


def distribute_existing_vre_by_grade(
    cap_by_year: pd.Series, grade_capacities: pd.Series
) -> pd.DataFrame:
    """Distribute VRE capacities by grade, ensuring potentials are respected and prioritizing better grades first

    Allocates variable renewable energy (VRE) capacity additions across different
    resource quality grades. The algorithm preferentially uses higher-quality
    grades before moving to lower-quality ones, implementing a "fill-up" strategy.

    Args:
        cap_by_year (pd.Series): Annual VRE capacity additions indexed by year. Values represent
            the total capacity to be added in each year (MW or GW).
        grade_capacities (pd.Series): Available capacity potential by resource grade for a bus,
            indexed by grade identifier. Higher-quality grades should have lower indices for
            proper prioritization.

    Returns:
        DataFrame with distributed capacities where:
            - Rows are indexed by years (from cap_by_year)
            - Columns are indexed by grades (from grade_capacities)
            - Values represent allocated capacity for each year-grade combination

    Example:
        >>> cap_additions = pd.Series([100, 200], index=[2020, 2030])
        >>> grade_potentials = pd.Series([50, 75, 100], index=['grade_1', 'grade_2', 'grade_3'])
        >>> result = distribute_vre_by_grade(cap_additions, grade_potentials)
        >>> print(result.sum(axis=1))  # Should match original yearly totals

    Note:
        The function assumes grade_capacities are ordered with best grades first.
        If total demand exceeds available capacity, the algorithm allocates as much
        as possible following the grade priority.
    """

    availability = cap_by_year.sort_index(ascending=False)
    to_distribute = grade_capacities.fillna(0).sort_index()
    n_years = len(to_distribute)
    n_sources = len(availability)

    # To store allocation per year per source (shape: sources x years)
    allocation = np.zeros((n_sources, n_years), dtype=int)
    remaining = availability.values

    for j in range(n_years):
        needed = to_distribute.values[j]
        cumsum = np.cumsum(remaining)
        used_up = cumsum < needed
        cutoff = np.argmax(cumsum >= needed)

        allocation[used_up, j] = remaining[used_up]

        if needed > (cumsum[cutoff - 1] if cutoff > 0 else 0):
            allocation[cutoff, j] = needed - (cumsum[cutoff - 1] if cutoff > 0 else 0)

        # Subtract what was used from availability
        remaining -= allocation[:, j]

    return pd.DataFrame(data=allocation, columns=grade_capacities.index, index=availability.index)


def add_existing_vre_capacities(
    n: pypsa.Network,
    costs: pd.DataFrame,
    vre_caps: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Add existing VRE capacities to the network and distribute them by vre grade potential.
    Adapted from pypsa-eur but the VRE capacities are province resolved.

    NOTE that using this function requires adding the land-use constraint in solve_network so
      that the existing capacities are subtracted from the available potential

    Args:
        n (pypsa.Network): the network
        costs (pd.DataFrame): costs of the technologies
        vre_caps (pd.DataFrame): existing brownfield VRE capacities in MW
        config (dict): snakemake configuration dictionary
    Returns:
        pd.DataFrame: DataFrame with existing VRE capacities distributed by CF grade

    """

    tech_map = {"solar": "PV", "onwind": "Onshore", "offwind-ac": "Offshore", "offwind": "Offshore"}
    tech_map = {k: tech_map[k] for k in tech_map if k in config["Techs"]["vre_techs"]}

    # historical data by tech, location and build year
    grouped_vre = vre_caps.groupby(["Tech", "bus", "DateIn"]).Capacity.sum()
    vre_df = grouped_vre.unstack().reset_index()
    df_agg = pd.DataFrame()

    # iterate over vre carriers
    for carrier in tech_map:
        df = vre_df[vre_df.Tech == carrier].drop(columns=["Tech"])
        df.set_index("bus", inplace=True)
        df.columns = df.columns.astype(int)

        # fetch existing vre generators (n grade bins per node)
        gen_i = n.generators.query("carrier == @carrier").index
        carrier_gens = n.generators.loc[gen_i]
        res_capacities = []
        # for each bus, distribute the vre capacities by grade potential - best first
        for bus, group in carrier_gens.groupby("bus"):
            if bus not in df.index:
                continue
            res_capacities.append(distribute_vre_by_grade(group.p_nom_max, df.loc[bus]))

        if res_capacities:
            res_capacities = pd.concat(res_capacities, axis=0)

            for year in df.columns:
                for gen in res_capacities.index:
                    bus_bin = re.sub(f" {carrier}.*", "", gen)
                    bus, bin_id = bus_bin.rsplit(" ", maxsplit=1)
                    name = f"{bus_bin} {carrier}-{int(year)}"
                    capacity = res_capacities.loc[gen, year]
                    if capacity > 0.0:
                        cost_key = carrier.split("-", maxsplit=1)[0]
                        df_agg.at[name, "Fueltype"] = carrier
                        df_agg.at[name, "Capacity"] = capacity
                        df_agg.at[name, "DateIn"] = int(year)
                        df_agg.at[name, "grouping_year"] = int(year)
                        df_agg.at[name, "lifetime"] = costs.at[cost_key, "lifetime"]
                        df_agg.at[name, "DateOut"] = year + costs.at[cost_key, "lifetime"] - 1
                        df_agg.at[name, "bus"] = bus
                        df_agg.at[name, "resource_class"] = bin_id

    if df_agg.empty:
        return df_agg

    df_agg.loc[:, "Tech"] = df_agg.Fueltype
    return df_agg


# MOVE to a separate rule?
def add_paid_off_capacities():
    pass


def attach_existing_capacities_to_network():
    pass


if "snakemake" not in globals():
    snakemake = mock_snakemake(
        "add_electricity",
        topology="current+FCG",
        co2_pathway="exp175default",
        # co2_pathway="SSP2-PkBudg1000-pseudo-coupled",
        planning_horizons=2025,
        heating_demand="positive",
        cluster_id="IM2XJ4",
        # configfiles="resources/tmp/pseudo-coupled.yaml",
    )

    configure_logging(snakemake)

    # if not node_cfg["split_provinces"]:
    #     assign_mode = "simple"
    #     cleaned["node"] = cleaned[ADM_LVL1]
    # elif assign_mode == "simple":
    #     splits_inv = {}  # invert to get admin2 -> node
    #     for admin1, splits in node_cfg["splits"].items():
    #         splits_inv.update({vv: admin1 + "_" + k for k, v in splits.items() for vv in v})
    #     cleaned["node"] = cleaned[ADM_LVL2].map(splits_inv).fillna(cleaned[ADM_LVL1])
    # else:
    #     if config["fetch_regions"]["simplify_tol"]["land"] > 0.05:
    #         logger.warning(
    #             "Using GPS assignment for existing capacities with land simplify_tol > 0.05. "
    #             "This may lead to inaccurate assignments (eg. Shanxi vs InnerMongolia coal power)."
    #         )
    #     cleaned["node"] = assign_node_from_gps(cleaned, nodes)
