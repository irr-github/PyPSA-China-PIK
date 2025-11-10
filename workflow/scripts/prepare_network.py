"""Function suite and script to define the network to be solved. Network components are added here.
Additional constraints require the linopy model and are added in the solve_network script.

These functions are currently only for the overnight mode. Myopic pathway mode contains near
        duplicates which need to merged in the future. Idem for solve_network.py
"""

# SPDX-FileCopyrightText: : 2022 The PyPSA-China Authors
#
# SPDX-License-Identifier: MIT

# for non-pathway network
# TODO WHY DO WE USE VRESUTILS ANNUITY IN ONE PLACE AND OUR OWN CALC ELSEWHERE?

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import ConfigManager, configure_logging, mock_snakemake
from _pypsa_helpers import (
    assign_locations,
    shift_profile_to_planning_year,
    calc_link_length,
)
from readers import load_costs
from add_electricity import sanitize_carriers, attach_load
from build_biomass_potential import estimate_co2_intensity_xing
from build_load_profiles import project_elec_demand, read_yearly_load_projections
from constants import (
    CRS,
    FOM_LINES,
    INFLOW_DATA_YR,
    NUCLEAR_EXTENDABLE,
    PROV_NAMES,
)
from functions import haversine
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# TODO add a heat bus that can absorb heat for free in non-coupled mode
#   (e.g. Hydrogen electrolysis, sabatier)
# TODO add heat disipator?


def add_biomass_chp(
    network: pypsa.Network,
    costs: pd.DataFrame,
    biomass_potential: pd.DataFrame,
    add_beccs: bool = True,
):
    """Add biomass combined heat and power (CHP) systems to the network.

    Integrates biomass CHP technology as new-build capacity (not retrofits or
    co-firing with coal). Optionally includes biomass with carbon capture and
    storage (BECCS) for negative emissions.

    Note:
        The carbon capture component is not currently constrained to biomass capacity

    Args:
        network (pypsa.Network): The PyPSA network object to modify.
        costs (pd.DataFrame): Techno-economic data (for all techs including biomass CHP)
        biomass_potential (pd.DataFrame): biomass feedstock potential
            by node, indexed by location with values in energy units.
        add_beccs (bool, optional): Whether to include biomass with carbon capture
            and storage technology. Defaults to True.
    """

    suffix = " biomass"
    nodes = network.buses.query("carrier == 'AC'")
    biomass_potential.index = biomass_potential.index.map(
        lambda x: x + suffix if not x.endswith(suffix) else x
    )

    network.add(
        "Bus",
        nodes.index,
        suffix=suffix,
        x=nodes.x,
        y=nodes.y,
        carrier="biomass",
        province=nodes.province,
    )
    logger.info("Adding biomass buses")
    logger.info(f"{nodes.index + suffix}")
    logger.info("potentials")
    # aggricultural residue biomass
    # NOTE THIS CURRENTLY DOESN'T INCLUDE TRANSPORT between nodes
    # NOTE additional emissions from treatment/remedials are missing
    network.add(
        "Store",
        nodes.index + suffix,
        bus=nodes.index + suffix,
        e_nom_extendable=False,
        e_nom=biomass_potential,
        e_initial=biomass_potential,
        carrier="biomass",
    )
    biomass_co2_intsty = estimate_co2_intensity_xing()
    network.add(
        "Link",
        nodes.index + " central biomass CHP",
        bus0=nodes.index + " biomass",
        bus1=nodes.index,
        bus2=nodes.index + " central heat",
        bus3=nodes.index + " CO2",
        p_nom_extendable=True,
        carrier="biomass",
        efficiency=costs.at["biomass CHP", "efficiency"],
        efficiency2=costs.at["biomass CHP", "efficiency-heat"],
        efficiency3=biomass_co2_intsty,
        capital_cost=costs.at["biomass CHP", "efficiency"]
        * costs.at["biomass CHP", "capital_cost"],
        marginal_cost=costs.at["biomass CHP", "efficiency"]
        * costs.at["biomass CHP", "marginal_cost"]
        + costs.at["solid biomass", "fuel"],
        lifetime=costs.at["biomass CHP", "lifetime"],
    )
    if add_beccs:
        network.add(
            "Link",
            nodes.index + " central biomass CHP capture",
            bus0=nodes.index + " CO2",
            bus1=nodes.index + " CO2 capture",
            bus2=nodes.index,
            p_nom_extendable=True,
            carrier="CO2 capture",
            efficiency=costs.at["biomass CHP capture", "capture_rate"],
            efficiency2=-1
            * costs.at["biomass CHP capture", "capture_rate"]
            * costs.at["biomass CHP capture", "electricity-input"],
            capital_cost=costs.at["biomass CHP capture", "capture_rate"]
            * costs.at["biomass CHP capture", "capital_cost"],
            lifetime=costs.at["biomass CHP capture", "lifetime"],
        )

    network.add(
        "Link",
        nodes.index + " decentral biomass boiler",
        bus0=nodes.index + " biomass",
        bus1=nodes.index + " decentral heat",
        p_nom_extendable=True,
        carrier="biomass",
        efficiency=costs.at["biomass boiler", "efficiency"],
        capital_cost=costs.at["biomass boiler", "efficiency"]
        * costs.at["biomass boiler", "capital_cost"],
        marginal_cost=costs.at["biomass boiler", "efficiency"]
        * costs.at["biomass boiler", "marginal_cost"]
        + costs.at["biomass boiler", "pelletizing cost"]
        + costs.at["solid biomass", "fuel"],
        lifetime=costs.at["biomass boiler", "lifetime"],
    )


def add_carriers(network: pypsa.Network, config: dict, costs: pd.DataFrame):
    """Add various carriers to the network based on configuration settings.

    Creates carrier objects for different energy types including electricity (AC),
    heat, renewable energy sources, storage technologies, and fuel carriers with
    their associated CO2 emissions factors.

    Args:
        network (pypsa.Network): The PyPSA network object to modify.
        config (dict): Config dictionary containing technology specifications
            and settings for carriers to be added.
        costs (pd.DataFrame): Cost database containing CO2 emissions factors for
            fuel carriers, indexed by carrier name.
    """

    network.add("Carrier", "AC")
    if config.get("heat_coupling", False):
        network.add("Carrier", "heat")
    for carrier in config["Techs"]["vre_techs"]:
        network.add("Carrier", carrier)
        if carrier == "hydroelectricity":
            network.add("Carrier", "hydro_inflow")
    for carrier in config["Techs"]["store_techs"]:
        network.add("Carrier", carrier)
        if carrier == "battery":
            network.add("Carrier", "battery discharger")

    # add fuel carriers, emissions in # in t_CO2/MWht
    if config["add_gas"]:
        network.add("Carrier", "gas", co2_emissions=costs.at["gas", "co2_emissions"])
    if config["add_coal"]:
        network.add("Carrier", "coal", co2_emissions=costs.at["coal", "co2_emissions"])
    if "CCGT-CCS" in config["Techs"]["conv_techs"]:
        network.add("Carrier", "gas ccs", co2_emissions=costs.at["gas ccs", "co2_emissions"])
    if "coal-CCS" in config["Techs"]["conv_techs"]:
        network.add("Carrier", "coal ccs", co2_emissions=costs.at["coal ccs", "co2_emissions"])


def add_co2_capture_support(network: pypsa.Network):
    """Add CO2 capture carriers and storage to the network.

    Creates the necessary CO2-related carriers, buses, and storage components
    to support carbon capture and storage (CCS) technologies in the network.

    Args:
        network (pypsa.Network): The PyPSA network object to modify.
    """
    nodes = network.buses.query("carrier == 'AC'")

    network.add("Carrier", "CO2", co2_emissions=0)
    network.add(
        "Bus",
        nodes.index,
        suffix=" CO2",
        x=nodes.x,
        y=nodes.y,
        carrier="CO2",
        province=nodes.province,
    )

    network.add("Store", nodes.index + " CO2", bus=nodes.index + " CO2", carrier="CO2")
    # normally taking away from carrier generates CO2, but here we are
    # adding CO2 stored, so the emissions will point the other way ?
    network.add("Carrier", "CO2 capture", co2_emissions=1)
    network.add(
        "Bus",
        nodes.index,
        suffix=" CO2 capture",
        x=nodes.x,
        y=nodes.y,
        carrier="CO2 capture",
        province=nodes.province,
    )

    network.add(
        "Store",
        nodes.index + " CO2 capture",
        bus=nodes.index + " CO2 capture",
        e_nom_extendable=True,
        # TODO change to capture to improve reporting (add carrier too?)
        carrier="CO2",
    )


def add_conventional_generators(
    network: pypsa.Network,
    config: dict,
    costs: pd.DataFrame,
):
    """Add conventional generation techs to the network.

    Integrates fossil fuel generators/links including gas (OCGT, CCGT), coal power plants,
    and carbon capture and storage (CCS) variants based on configuration settings.

    Args:
        network (pypsa.Network): The PyPSA network object to modify.
        nodes (pd.Index): Index of network nodes where generators will be added.
        config (dict): Config with tech choices etc
        costs (pd.DataFrame): Cost database containing techno-economic parameters
    """
    nodes = network.buses.query("carrier == 'AC'")

    if config["add_gas"]:
        # add converter from fuel source
        network.add(
            "Bus",
            nodes.index,
            suffix=" gas",
            x=nodes.x,
            y=nodes.y,
            carrier="gas",
            location=nodes.index,
            province=nodes.province,
        )

        network.add(
            "Generator",
            nodes.index,
            suffix=" gas fuel",
            bus=nodes.index + " gas",
            carrier="gas",
            p_nom_extendable=True,
            p_nom=1e7,
            marginal_cost=costs.at["gas", "fuel"],
        )

        # gas prices identical per region, pipelines ignored
        network.add(
            "Store",
            nodes.index + " gas Store",
            bus=nodes.index + " gas",
            e_nom_extendable=True,
            carrier="gas",
            e_nom=1e7,
            e_cyclic=True,
        )

    # add gas will then be true
    gas_techs = ["OCGT", "CCGT"]
    for tech in gas_techs:
        if tech in config["Techs"]["conv_techs"]:
            network.add(
                "Link",
                nodes.index,
                suffix=f" {tech}",
                bus0=nodes.index + " gas",
                bus1=nodes.index,
                marginal_cost=costs.at[tech, "efficiency"]
                * costs.at[tech, "VOM"],  # NB: VOM is per MWel
                capital_cost=costs.at[tech, "efficiency"]
                * costs.at[tech, "capital_cost"],  # NB: capital cost is per MWel
                p_nom_extendable=True,
                efficiency=costs.at[tech, "efficiency"],
                lifetime=costs.at[tech, "lifetime"],
                carrier=f"gas {tech}",
            )

    if "CCGT-CCS" in config["Techs"]["conv_techs"]:
        network.add(
            "Generator",
            nodes.index,
            suffix=" CCGT-CCS",
            bus=nodes.index,
            carrier="gas ccs",
            p_nom_extendable=True,
            efficiency=costs.at["CCGT-CCS", "efficiency"],
            marginal_cost=costs.at["CCGT-CCS", "marginal_cost"],
            capital_cost=costs.at["CCGT-CCS", "efficiency"]
            * costs.at["CCGT-CCS", "capital_cost"],  # NB: capital cost is per MWel
            lifetime=costs.at["CCGT-CCS", "lifetime"],
            p_max_pu=0.9,  # planned and forced outages
        )

    if config["add_coal"]:
        ramps = config.get(
            "fossil_ramps", {"coal": {"ramp_limit_up": np.nan, "ramp_limit_down": np.nan}}
        )
        ramps = ramps.get("coal", {"ramp_limit_up": np.nan, "ramp_limit_down": np.nan})
        ramps = {k: v * config["snapshots"]["frequency"] for k, v in ramps.items()}
        # this is the non sector-coupled approach
        # for industry may have an issue in that coal feeds to chem sector
        # no coal in Beijing - political decision

        # This is ugly but works because Beijing is also l2
        # TODO separate
        network.add(
            "Generator",
            nodes.index[nodes.index != "Beijing"],
            suffix=" coal power",
            bus=nodes.index[nodes.index != "Beijing"],
            carrier="coal",
            p_nom_extendable=True,
            efficiency=costs.at["coal", "efficiency"],
            marginal_cost=costs.at["coal", "marginal_cost"],
            capital_cost=costs.at["coal", "capital_cost"],  # NB: capital cost is per MWel
            lifetime=costs.at["coal", "lifetime"],
            ramp_limit_up=ramps["ramp_limit_up"],
            ramp_limit_down=ramps["ramp_limit_down"],
            p_max_pu=0.9,  # planned and forced outages
        )

        # TODO separate - not conventional
        if "coal-CCS" in config["Techs"]["conv_techs"]:
            network.add(
                "Generator",
                nodes.index,
                suffix=" coal-CCS",
                bus=nodes.index,
                carrier="coal ccs",
                p_nom_extendable=True,
                efficiency=costs.at["coal ccs", "efficiency"],
                marginal_cost=costs.at["coal ccs", "marginal_cost"],
                capital_cost=costs.at["coal ccs", "capital_cost"],  # NB: capital cost is per MWel
                lifetime=costs.at["coal ccs", "lifetime"],
                p_max_pu=0.9,  # planned and forced outages
            )

        # TODO fi
        if "nuclear" in config["Techs"]["vre_techs"]:
            nuclear_nodes = network.buses.query(
                "carrier=='AC' and province in @NUCLEAR_EXTENDABLE"
            ).index
            network.add(
                "Generator",
                nuclear_nodes,
                suffix=" nuclear",
                p_nom_extendable=True,
                p_max_pu=config["nuclear_reactors"]["p_max_pu"],
                p_min_pu=config["nuclear_reactors"]["p_min_pu"],
                bus=nuclear_nodes,
                carrier="nuclear",
                efficiency=costs.at["nuclear", "efficiency"],
                capital_cost=costs.at["nuclear", "capital_cost"],  # NB: capital cost is per MWel
                marginal_cost=costs.at["nuclear", "marginal_cost"],
                lifetime=costs.at["nuclear", "lifetime"],
            )


def add_H2(network: pypsa.Network, config: dict, costs: pd.DataFrame, planning_year: int):
    """Add hydrogen infrastructure including production, storage, and transport.

    Adds hydrogen electrolysis, fuel cells, turbines, storage systems, and
    optionally hydrogen pipelines for transport and methanation for power-to-gas

    Args:
        network (pypsa.Network): PyPSA network object to which H2 comps are added
        config (dict): Configuration with H2 settings (inc geo storage)
        costs (pd.DataFrame): Cost database containing techno-economic parameters
            for hydrogen technologies and infrastructure.
        planning_year (int): Planning year for cost selection and temporal alignment.
    """
    nodes = network.buses.query("carrier=='AC'")

    # TODO, does it make sense?
    if config.get("heat_coupling", False):
        network.add(
            "Link",
            name=nodes.index + " H2 Electrolysis",
            bus0=nodes.index,
            bus1=nodes.index + " H2",
            bus2=nodes.index + " central heat",
            p_nom_extendable=True,
            carrier="H2 Electrolysis",
            efficiency=costs.at["electrolysis", "efficiency"],
            efficiency2=costs.at["electrolysis", "efficiency-heat"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            lifetime=costs.at["electrolysis", "lifetime"],
        )
    else:
        network.add(
            "Link",
            name=nodes.index + " H2 Electrolysis",
            bus0=nodes.index,
            bus1=nodes.index + " H2",
            p_nom_extendable=True,
            carrier="H2 Electrolysis",
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            lifetime=costs.at["electrolysis", "lifetime"],
        )

    if "fuel cell" in config["Techs"]["vre_techs"]:
        network.add(
            "Link",
            name=nodes.index + " H2 Fuel Cell",
            bus0=nodes.index + " H2",
            bus1=nodes.index,
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            capital_cost=costs.at["fuel cell", "efficiency"]
            * costs.at["fuel cell", "capital_cost"],
            lifetime=costs.at["fuel cell", "lifetime"],
            carrier="H2 fuel cell",
        )
    if "H2 turbine" in config["Techs"]["vre_techs"]:
        network.add(
            "Link",
            name=nodes.index + " H2 turbine",
            bus0=nodes.index + " H2",
            bus1=nodes.index,
            p_nom_extendable=True,
            efficiency=costs.at["H2 turbine", "efficiency"],
            capital_cost=costs.at["H2 turbine", "efficiency"]
            * costs.at["H2 turbine", "capital_cost"],
            lifetime=costs.at["H2 turbine", "lifetime"],
            carrier="H2 turbine",
        )

    H2_under_nodes_ = pd.Index(config["H2"]["geo_storage_nodes"])
    H2_type1_nodes_ = nodes.index.difference(H2_under_nodes_)
    H2_under_nodes = H2_under_nodes_.intersection(nodes.index)
    H2_type1_nodes = H2_type1_nodes_.intersection(nodes.index)
    if not (
        H2_under_nodes_.shape == H2_under_nodes.shape
        and H2_type1_nodes_.shape == H2_type1_nodes.shape
    ):
        logger.warning("Some H2 storage nodes are not in the network buses")

    network.add(
        "Store",
        H2_under_nodes + " H2 Store",
        bus=H2_under_nodes + " H2",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=costs.at["hydrogen storage underground", "capital_cost"],
        lifetime=costs.at["hydrogen storage underground", "lifetime"],
    )

    # TODO harmonize with remind (add if in techs)
    network.add(
        "Store",
        H2_type1_nodes + " H2 Store",
        bus=H2_type1_nodes + " H2",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=costs.at["hydrogen storage tank type 1 including compressor", "capital_cost"],
        lifetime=costs.at["hydrogen storage tank type 1 including compressor", "lifetime"],
    )
    if config["add_methanation"]:
        cost_year = planning_year
        network.add(
            "Link",
            nodes.index + " Sabatier",
            bus0=nodes.index + " H2",
            bus1=nodes.index + " gas",
            carrier="Sabatier",
            p_nom_extendable=True,
            efficiency=costs.at["methanation", "efficiency"],
            capital_cost=costs.at["methanation", "efficiency"]
            * costs.at["methanation", "capital_cost"]
            + costs.at["direct air capture", "capital_cost"]
            * costs.at["gas", "co2_emissions"]
            * costs.at["methanation", "efficiency"],
            # TODO fix me
            lifetime=costs.at["methanation", "lifetime"],
            marginal_cost=(400 - 5 * (int(cost_year) - 2020))
            * costs.at["gas", "co2_emissions"]
            * costs.at["methanation", "efficiency"],
        )

    if config["Techs"]["hydrogen_lines"]:
        edge_path = config["edge_paths"].get(config["scenario"]["topology"], None)
        if edge_path is None:
            raise ValueError(f"No grid found for topology {config['scenario']['topology']}")
        else:
            edges_ = pd.read_csv(
                edge_path, sep=",", header=None, names=["bus0", "bus1", "p_nom"]
            ).fillna(0)
            edges = edges_[edges_["bus0"].isin(nodes) & edges_["bus1"].isin(nodes)]
            if edges_.shape[0] != edges.shape[0]:
                logger.warning("Some edges are not in the network buses")

        # fix this to use map with x.y
        lengths = config["lines"]["line_length_factor"] * np.array(
            [
                haversine(
                    [network.buses.at[bus0, "x"], network.buses.at[bus0, "y"]],
                    [network.buses.at[bus1, "x"], network.buses.at[bus1, "y"]],
                )
                for bus0, bus1 in edges[["bus0", "bus1"]].values
            ]
        )

        # TODO harmonize with remind (add if in techs)
        cc = costs.at["H2 (g) pipeline", "capital_cost"] * lengths

        # === h2 pipeline with losses ====
        # NB this only works if there is an equalising constraint, which is hidden in solve_ntwk
        network.add(
            "Link",
            edges["bus0"] + "-" + edges["bus1"] + " H2 pipeline",
            suffix=" positive",
            bus0=edges["bus0"].values + " H2",
            bus1=edges["bus1"].values + " H2",
            bus2=edges["bus0"].values,
            carrier="H2 pipeline",
            p_nom_extendable=True,
            p_nom=0,
            p_nom_min=0,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["H2 pipeline"]["efficiency_static"]
            * config["transmission_efficiency"]["H2 pipeline"]["efficiency_per_1000km"]
            ** (lengths / 1000),
            efficiency2=-config["transmission_efficiency"]["H2 pipeline"]["compression_per_1000km"]
            * lengths
            / 1e3,
            length=lengths,
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
            capital_cost=cc,
        )

        network.add(
            "Link",
            edges["bus0"] + "-" + edges["bus1"] + " H2 pipeline",
            suffix=" reversed",
            carrier="H2 pipeline",
            bus0=edges["bus1"].values + " H2",
            bus1=edges["bus0"].values + " H2",
            bus2=edges["bus1"].values,
            p_nom_extendable=True,
            p_nom=0,
            p_nom_min=0,
            p_min_pu=0,
            efficiency=config["transmission_efficiency"]["H2 pipeline"]["efficiency_static"]
            * config["transmission_efficiency"]["H2 pipeline"]["efficiency_per_1000km"]
            ** (lengths / 1000),
            efficiency2=-config["transmission_efficiency"]["H2 pipeline"]["compression_per_1000km"]
            * lengths
            / 1e3,
            length=lengths,
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
            capital_cost=0,
        )


# TODO harmonize with remind
def add_voltage_links(network: pypsa.Network, config: dict):
    """Add high-voltage transmission links (HVDC/AC) to the network.

    Creates transmission infrastructure between network nodes based on topology config.
    Supports both lossy and lossless transmission models with
    configurable efficiency parameters and security margins.

    Args:
        network (pypsa.Network): The PyPSA network object to modify.
        config (dict): Config containing transmission topology,
            line parameters, efficiency settings, and security margins.

    Raises:
        ValueError: If the specified topology file is not found.
    """

    hv_links = network.links.query("carrier in ['DC', 'AC']")

    # fill any missing lengths
    lengths = (
        hv_links.apply(lambda row: calc_link_length(row, network), axis=1)
        * config["lines"]["line_length_factor"]
    )
    missing_lengths = hv_links[hv_links.length == 0 | hv_links.length.isna()]
    hv_links.loc[missing_lengths.index, "length"] = lengths.loc[missing_lengths.index]

    security_config = config.get("security", {"line_security_margin": 70})
    line_margin = security_config.get("line_security_margin", 70) / 100

    represented_hours = network.snapshot_weightings.sum()[0]
    n_years = represented_hours / 8760.0
    line_cost = (
        hv_links.length * costs.at["HVDC overhead", "capital_cost"] * FOM_LINES * n_years
    ) 
    missing_costs = hv_links[hv_links.capital_cost == 0 | hv_links.capital_cost.isna()]
    hv_links.loc[missing_costs.index, "capital_cost"] = line_cost.loc[missing_costs.index]
    hv_links["capital_cost"] += costs.at["HVDC inverter pair", "capital_cost"]  # /MW

    # AVOID degeneracies by grouping (questionnable... collapse all as DC)
    hv_links_orig = hv_links.copy()
    sum_columns = ["type", "p_nom"]
    avg_columns = ["length"]
    other_columns = hv_links.columns.difference(sum_columns + avg_columns + ["bus0", "bus1"])

    hv_links = hv_links.groupby(["bus0", "bus1"])[sum_columns].sum()
    hv_links[avg_columns] = hv_links_orig.groupby(["bus0", "bus1"])[avg_columns].mean()
    hv_links[other_columns] = hv_links_orig.groupby(["bus0", "bus1"])[other_columns].first()
    hv_links = hv_links.reset_index()
    hv_links.index = hv_links["bus0"] + "-" + hv_links["bus1"] + " HV"

    hv_links["p_max_pu"] = line_margin

    # remove original HV links
    network.links = network.links.query("index not in @hv_links_orig.index")
    
    # ==== lossy transport model (split into 2) ====
    # NB this only works if there is an equalising constraint, which is hidden in solve_ntwk
    if config["line_losses"]:
        eta_stat = config["transmission_efficiency"]["DC"]["efficiency_static"]
        hv_links["efficiency"] = eta_stat * config["transmission_efficiency"]["DC"][
            "efficiency_per_1000km"
        ] ** (hv_links.length / 1000)
        hv_links["p_min_pu"] = 0
        network.add("Link", hv_links.index, suffix ="positive", **hv_links)
        # 0 len for reversed in case line limits are specified in km. 
        # Limited in constraints to fwdcap
        hv_links["length"] = 0
        hv_links["capital_cost"] = 0
        network.add("Link", hv_links.index, suffix ="reversed", **hv_links)
    else:
        hv_links["p_min_pu"] = -hv_links["p_max_pu"]
        network.add("Link", hv_links.index, **hv_links)
        


def add_wind_and_solar(
    network: pypsa.Network,
    techs: list[str],
    paths: dict,
    year: int,
    costs: pd.DataFrame,
):
    """Add wind and solar generators with resource grade differentiation.

    Add ariable renewable energy (VRE) generators for each tech and quality grade
    Loads capacity factors and potential from profiles and adds generators.

    Args:
        network (pypsa.Network): PyPSA network to which generators will be added.
        techs (list): renewable energy technologies to add. Supported
            technologies are ["solar", "onwind", "offwind"].
        paths (dict): Dictionary containing file paths to renewable energy profiles
            with keys like 'profile_solar', 'profile_onwind', etc.
        year (int): Planning year for which profiles should be aligned.
        costs (pd.DataFrame): Technoeconomic data incl. renewable technologies

    Raises:
        ValueError: If unsupported technologies are specified or if paths not specified
    """

    unsupported = set(techs).difference({"solar", "onwind", "offwind"})
    if unsupported:
        raise ValueError(f"Carrier(s) {unsupported} not wind or solar pv")
    prof_paths = {f"profile_{tech}": paths[f"profile_{tech}"] for tech in techs}
    if len(prof_paths) != len(techs):
        raise ValueError(f"Paths do not correspond to techs  ({prof_paths} vs {techs})")

    for tech in techs:
        # load the renewable profiles
        logger.info(f"Attaching {tech} to network")
        with xr.open_dataset(prof_paths[f"profile_{tech}"]) as ds:
            if ds.indexes["bus"].empty:
                continue
            if "year" in ds.indexes:
                ds = ds.sel(year=ds.year.min(), drop=True)

            timestamps = pd.DatetimeIndex(ds.time)

            def shift_weather_to_planning_yr(t):
                """Shift weather data to planning year."""
                return t.replace(year=int(year))

            timestamps = timestamps.map(shift_weather_to_planning_yr)
            ds = ds.assign_coords(time=timestamps)

            mask = ds.time.isin(network.snapshots)
            ds = ds.sel(time=mask)

            if not len(ds.time) == len(network.snapshots):
                err = f"{len(ds.time)} and {len(network.snapshots)}"
                raise ValueError("Mismatch in profile and network timestamps " + err)
            ds = ds.stack(bus_bin=["bus", "bin"])

        # remove low potential bins
        cutoff = config.get("renewable_potential_cutoff", 0)
        ds = ds.where(ds["p_nom_max"] > cutoff, drop=True)

        # bins represent renewable generation grades
        def flatten(t):
            """Flatten tuple to string with ' grade' separator."""
            return " grade".join(map(str, t))

        buses = ds.indexes["bus_bin"].get_level_values("bus")
        bus_bins = ds.indexes["bus_bin"].map(flatten)

        p_nom_max = ds["p_nom_max"].to_pandas()
        p_nom_max.index = p_nom_max.index.map(flatten)

        p_max_pu = ds["profile"].to_pandas()
        p_max_pu.columns = p_max_pu.columns.map(flatten)

        # add renewables
        network.add(
            "Generator",
            bus_bins,
            suffix=f" {tech}",
            bus=buses,
            carrier=tech,
            p_nom_extendable=True,
            p_nom_max=p_nom_max,
            capital_cost=costs.at[tech, "capital_cost"],
            marginal_cost=costs.at[tech, "marginal_cost"],
            p_max_pu=p_max_pu,
            lifetime=costs.at[tech, "lifetime"],
        )


def add_heat_coupling(
    network: pypsa.Network,
    config: dict,
    costs: pd.DataFrame,
    planning_year: int,
    paths: dict,
):
    """Add heat sector coupling technologies and infrastructure to the network.

    Integrates heat pumps, thermal storage, heating demand, and other heat-related
    technologies for both centralized and decentralized heating systems.

    Args:
        network (pypsa.Network): PyPSA network object to modify.
        config (dict): Config containing heat coupling and demand settings
        costs (pd.DataFrame): Cost database containing techno-economic parameters
            for heat technologies and heating systems.
        planning_year (int): Planning year for profile alignment and cost calculations.
        paths (dict): Dictionary containing file paths to heat demand profiles,
            COP (coefficient of performance) data, and other heat-related data files.
    """

    nodes = network.buses.query("carrier=='AC'")

    central_fraction = pd.read_hdf(paths["central_fraction"])
    with pd.HDFStore(paths["heat_demand_profile"], mode="r") as store:
        heat_demand = store["heat_demand_profiles"]
        # TODO fix this if not working
        heat_demand.index = heat_demand.index.tz_localize(None)
        heat_demand = heat_demand.loc[network.snapshots]
        # NOTE electric boilers not yet subtracted from load
        hot_water_demand = store.get("hot_water_demand")
        hot_water_demand = hot_water_demand.loc[network.snapshots]

    network.add(
        "Bus",
        nodes.index,
        suffix=" decentral heat",
        x=nodes.x,
        y=nodes.y,
        carrier="heat",
        location=nodes.index,
        province=nodes.province,
    )

    network.add(
        "Bus",
        nodes.index,
        suffix=" central heat",
        x=nodes.x,
        y=nodes.y,
        carrier="heat",
        location=nodes.index,
        province=nodes.province,
    )

    network.add(
        "Load",
        nodes.index,
        suffix=" decentral heat",
        bus=nodes.index + " decentral heat",
        p_set=heat_demand[nodes].multiply(1 - central_fraction[nodes]) + hot_water_demand[nodes],
    )

    network.add(
        "Load",
        nodes.index,
        carrier="heat",
        suffix=" central heat",
        bus=nodes.index + " central heat",
        p_set=heat_demand[nodes].multiply(central_fraction[nodes]),
    )

    if "heat pump" in config["Techs"]["vre_techs"]:
        logger.info(f"loading cop profiles from {paths['cop_name']}")
        with pd.HDFStore(paths["cop_name"], mode="r") as store:
            ashp_cop = store["ashp_cop_profiles"]
            ashp_cop.index = ashp_cop.index.tz_localize(None)
            ashp_cop = shift_profile_to_planning_year(ashp_cop, planning_year)
            gshp_cop = store["gshp_cop_profiles"]
            gshp_cop.index = gshp_cop.index.tz_localize(None)
            gshp_cop = shift_profile_to_planning_year(gshp_cop, planning_year)

        for cat in [" decentral ", " central "]:
            network.add(
                "Link",
                nodes.index,
                suffix=cat + "heat pump",
                bus0=nodes.index,
                bus1=nodes.index + cat + "heat",
                carrier="heat pump",
                efficiency=(
                    ashp_cop.loc[network.snapshots, nodes.index]
                    if config["time_dep_hp_cop"]
                    else costs.at[cat.lstrip() + "air-sourced heat pump", "efficiency"]
                ),
                capital_cost=costs.at[cat.lstrip() + "air-sourced heat pump", "efficiency"]
                * costs.at[cat.lstrip() + "air-sourced heat pump", "capital_cost"],
                marginal_cost=costs.at[cat.lstrip() + "air-sourced heat pump", "efficiency"]
                * costs.at[cat.lstrip() + "air-sourced heat pump", "marginal_cost"],
                p_nom_extendable=True,
                lifetime=costs.at[cat.lstrip() + "air-sourced heat pump", "lifetime"],
            )

        network.add(
            "Link",
            nodes.index,
            suffix=" ground heat pump",
            bus0=nodes.index,
            bus1=nodes.index + " decentral heat",
            carrier="heat pump",
            efficiency=(
                gshp_cop.loc[network.snapshots, nodes.index]
                if config["time_dep_hp_cop"]
                else costs.at["decentral ground-sourced heat pump", "efficiency"]
            ),
            marginal_cost=costs.at[cat.lstrip() + "ground-sourced heat pump", "efficiency"]
            * costs.at[cat.lstrip() + "ground-sourced heat pump", "marginal_cost"],
            capital_cost=costs.at[cat.lstrip() + "ground-sourced heat pump", "efficiency"]
            * costs.at["decentral ground-sourced heat pump", "capital_cost"],
            p_nom_extendable=True,
            lifetime=costs.at["decentral ground-sourced heat pump", "lifetime"],
        )

    if "water tanks" in config["Techs"]["store_techs"]:
        for cat in [" decentral ", " central "]:
            network.add(
                "Bus",
                nodes.index,
                suffix=cat + "water tanks",
                x=nodes.x,
                y=nodes.y,
                carrier="water tanks",
                location=nodes.index,
                province=nodes.province,
            )

            network.add(
                "Link",
                nodes.index + cat + "water tanks charger",
                bus0=nodes.index + cat + "heat",
                bus1=nodes.index + cat + "water tanks",
                carrier="water tanks",
                efficiency=costs.at["water tank charger", "efficiency"],
                p_nom_extendable=True,
            )

            network.add(
                "Link",
                nodes.index + cat + "water tanks discharger",
                bus0=nodes.index + cat + "water tanks",
                bus1=nodes.index + cat + "heat",
                carrier="water tanks",
                efficiency=costs.at["water tank discharger", "efficiency"],
                p_nom_extendable=True,
            )
            # [HP] 180 day time constant for centralised, 3 day for decentralised
            tes_tau = config["water_tanks"]["tes_tau"][cat.strip()]
            network.add(
                "Store",
                +cat + "water tank",
                bus=+cat + "water tanks",
                carrier="water tanks",
                e_cyclic=True,
                e_nom_extendable=True,
                standing_loss=1 - np.exp(-1 / (24.0 * tes_tau)),
                capital_cost=costs.at[cat.lstrip() + "water tank storage", "capital_cost"],
                lifetime=costs.at[cat.lstrip() + "water tank storage", "lifetime"],
            )

    if "resistive heater" in config["Techs"]["vre_techs"]:
        for cat in [" decentral ", " central "]:
            network.add(
                "Link",
                nodes.index + cat + "resistive heater",
                bus0=nodes.index,
                bus1=nodes.index + cat + "heat",
                carrier="resistive heater",
                efficiency=costs.at[cat.lstrip() + "resistive heater", "efficiency"],
                capital_cost=costs.at[cat.lstrip() + "resistive heater", "efficiency"]
                * costs.at[cat.lstrip() + "resistive heater", "capital_cost"],
                marginal_cost=costs.at[cat.lstrip() + "resistive heater", "efficiency"]
                * costs.at[cat.lstrip() + "resistive heater", "marginal_cost"],
                p_nom_extendable=True,
                lifetime=costs.at[cat.lstrip() + "resistive heater", "lifetime"],
            )

    if (
        "H2 CHP" in config["Techs"]["vre_techs"]
        and config["add_H2"]
        and config.get("heat_coupling", False)
    ):
        htpr = config["chp_parameters"]["OCGT"]["heat_to_power"]

        network.add(
            "Link",
            nodes.index,
            suffix=" CHP H2 generator",
            bus0=nodes.index + " H2",
            bus1=nodes.index,
            bus2=nodes.index + " central heat",
            p_nom_extendable=True,
            marginal_cost=costs.at["central hydrogen CHP", "efficiency"]
            * costs.at["central hydrogen CHP", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central hydrogen CHP", "efficiency"]
            * costs.at["central hydrogen CHP", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=costs.at["central hydrogen CHP", "efficiency"],
            heat_to_power=config["chp_parameters"]["gas"]["heat_to_power"],
            lifetime=costs.at["central hydrogen CHP", "lifetime"],
            carrier="CHP H2",
            location=nodes.index,
        )
        network.add(
            "Link",
            nodes.index,
            suffix=" CHP H2 boiler",
            bus0=nodes.index + " H2",
            bus1=nodes.index + " central heat",
            carrier="CHP H2",
            marginal_cost=costs.at["central hydrogen CHP", "efficiency"]
            * costs.at["central hydrogen CHP", "VOM"],  # NB: VOM is per MWel
            p_nom_extendable=True,
            # total eff will be fixed by CHP constraints
            efficiency=config["chp_parameters"]["gas"]["total_eff"],
            lifetime=costs.at["central hydrogen CHP", "lifetime"],
            heat_to_power=config["chp_parameters"]["gas"]["heat_to_power"],
            location=nodes.index,
        )

    if "CHP gas" in config["Techs"]["conv_techs"]:
        # Extraction mode CCGT CHP -> heat to power ratio is maximum
        network.add(
            "Link",
            nodes.index,
            suffix=" CHP gas generator",
            bus0=nodes.index + " gas",
            bus1=nodes.index,
            bus2=nodes.index + " central heat",
            p_nom_extendable=True,
            marginal_cost=costs.at["central gas CHP CC", "efficiency"]
            * costs.at["central gas CHP CC", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central gas CHP CC", "efficiency"]
            * costs.at["central gas CHP CC", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=costs.at["central gas CHP CC", "efficiency"],
            heat_to_power=config["chp_parameters"]["gas"]["heat_to_power"],
            lifetime=costs.at["central gas CHP CC", "lifetime"],
            carrier="CHP gas",
            location=nodes.index,
        )

        network.add(
            "Link",
            nodes.index,
            suffix=" CHP gas boiler",
            bus0=nodes.index + " gas",
            bus1=nodes.index + " central heat",
            carrier="CHP gas",
            marginal_cost=costs.at["central gas CHP CC", "efficiency"]
            * costs.at["central gas CHP CC", "VOM"],  # NB: VOM is per MWel
            p_nom_extendable=True,
            # total eff will be fixed by CHP constraints
            efficiency=config["chp_parameters"]["gas"]["total_eff"],
            lifetime=costs.at["central gas CHP CC", "lifetime"],
            heat_to_power=config["chp_parameters"]["gas"]["heat_to_power"],
            location=nodes.index,
        )

    if "CHP OCGT gas" in config["Techs"]["conv_techs"]:
        # OCGT CHPs have fixed heat to power ratio
        # F*eta_tot = Qout + Pout = (1+htpr)*Pout = (1+1/htpr)*Qout
        # eta_el = Pout/F = eta_tot/(1+htpr) etc
        eta_tot = config["chp_parameters"]["OCGT"]["total_eff"]
        htpr = config["chp_parameters"]["OCGT"]["heat_to_power"]

        network.add(
            "Link",
            nodes.index,
            suffix=" CHP OCGT gas",
            bus0=nodes.index + " gas",
            bus1=nodes.index,
            bus2=nodes.index + " central heat",
            p_nom_extendable=True,
            marginal_cost=costs.at["central gas CHP", "efficiency"]
            * costs.at["central gas CHP", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central gas CHP", "efficiency"]
            * costs.at["central gas CHP", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=eta_tot / (1 + htpr),
            efficiency2=eta_tot / (1 + 1 / htpr),
            lifetime=costs.at["central gas CHP", "lifetime"],
            carrier="CHP OCGT gas",
        )

    if "CHP coal" in config["Techs"]["conv_techs"]:
        logger.info("Adding CHP coal to network")
        # Extraction mode super critical CHP -> heat to power ratio is maximum
        # in ideal model there would be plant level "commitment" of extraction mode
        # to reflect that real plants have a min heat output in extraction mode
        # however, over many plants can avoid this (remembering no heat in condensing mode)

        network.add(
            "Bus",
            nodes.index,
            suffix=" coal fuel",
            x=nodes.x,
            y=nodes.y,
            carrier="coal",
            location=nodes.index,
            province=nodes.province,
        )

        network.add(
            "Generator",
            nodes.index + " coal fuel",
            bus=nodes.index + " coal fuel",
            carrier="coal",
            p_nom_extendable=False,
            p_nom=1e8,
            marginal_cost=costs.at["coal", "fuel"],
        )

        # NOTE CHP + generator | boiler is a key word for the constraint
        network.add(
            "Link",
            name=nodes[nodes != "Beijing"],
            suffix=" CHP coal generator",
            bus0=nodes.index[nodes.index != "Beijing"] + " coal fuel",
            bus1=nodes.index[nodes.index != "Beijing"],
            p_nom_extendable=True,
            marginal_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "VOM"],  # NB: VOM is per MWel
            capital_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at["central coal CHP", "capital_cost"],  # NB: capital cost is per MWel
            efficiency=costs.at["central coal CHP", "efficiency"],
            heat_to_power=config["chp_parameters"]["coal"]["heat_to_power"],
            lifetime=costs.at["central coal CHP", "lifetime"],
            carrier="CHP coal",
            location=nodes.index[nodes.index != "Beijing"],
        )

        network.add(
            "Link",
            nodes.index[nodes.index != "Beijing"],
            suffix=" central CHP coal boiler",
            bus0=nodes.index[nodes.index != "Beijing"] + " coal fuel",
            bus1=nodes.index[nodes.index != "Beijing"] + " central heat",
            carrier="CHP coal",
            p_nom_extendable=True,
            marginal_cost=costs.at["central coal CHP", "efficiency"]
            * costs.at[
                "central coal CHP", "VOM"
            ],  # NB: VOM is per MWel            # total eff will be fixed by CHP constraints
            efficiency=config["chp_parameters"]["coal"]["total_eff"],
            lifetime=costs.at["central coal CHP", "lifetime"],
            heat_to_power=config["chp_parameters"]["coal"]["heat_to_power"],
            location=nodes.index[nodes.index != "Beijing"],
        )

    if "coal boiler" in config["Techs"]["conv_techs"]:
        use_hist_eff = config["use_historical_efficiency"].get("coal_boiler", False)
        for cat in ["decentral", "central"]:
            eff = (
                costs.at[f"{cat} coal boiler", "hist_efficiency"]
                if use_hist_eff
                else costs.at[f"{cat} coal boiler", "efficiency"]
            )
            network.add(
                "Link",
                nodes.index[nodes.index != "Beijing"] + f" {cat} coal boiler",
                p_nom_extendable=True,
                bus0=nodes.index[nodes.index != "Beijing"] + " coal fuel",
                bus1=nodes.index[nodes.index != "Beijing"] + f" {cat} heat",
                efficiency=eff,
                marginal_cost=eff * costs.at[f"{cat} coal boiler", "VOM"],
                capital_cost=eff * costs.at[f"{cat} coal boiler", "capital_cost"],
                lifetime=costs.at[f"{cat} coal boiler", "lifetime"],
                carrier=f"coal boiler {cat}",
            )

    if "gas boiler" in config["Techs"]["conv_techs"]:
        for cat in ["decentral", "central"]:
            network.add(
                "Link",
                nodes.index + cat + "gas boiler",
                p_nom_extendable=True,
                bus0=nodes.index + " gas",
                bus1=nodes.index + f" {cat} heat",
                efficiency=costs.at[f"{cat} gas boiler", "efficiency"],
                marginal_cost=costs.at[f"{cat} gas boiler", "VOM"],
                capital_cost=costs.at[f"{cat} gas boiler", "efficiency"]
                * costs.at[f"{cat} gas boiler", "capital_cost"],
                lifetime=costs.at[f"{cat} gas boiler", "lifetime"],
                carrier=f"gas boiler {cat}",
            )

    if "solar thermal" in config["Techs"]["vre_techs"]:
        # this is the amount of heat collected in W per m^2, accounting
        # for efficiency
        with pd.HDFStore(paths["solar_thermal_name"], mode="r") as store:
            # 1e3 converts from W/m^2 to MW/(1000m^2) = kW/m^2
            solar_thermal = config["solar_cf_correction"] * store["solar_thermal_profiles"] / 1e3

        solar_thermal.index = solar_thermal.index.tz_localize(None)
        solar_thermal = shift_profile_to_planning_year(solar_thermal, planning_year)
        solar_thermal = solar_thermal.loc[network.snapshots]

        for cat in [" decentral ", " central "]:
            network.add(
                "Generator",
                nodes.index,
                suffix=cat + "solar thermal",
                bus=nodes.index + cat + "heat",
                carrier="solar thermal",
                p_nom_extendable=True,
                capital_cost=costs.at[cat.lstrip() + "solar thermal", "capital_cost"],
                p_max_pu=solar_thermal[nodes.index].clip(1.0e-4),
                lifetime=costs.at[cat.lstrip() + "solar thermal", "lifetime"],
            )


def add_hydro(
    network: pypsa.Network,
    config: dict,
    region_shapes: gpd.GeoDataFrame,
    costs: pd.DataFrame,
    planning_horizons: int,
):
    """Add the hydropower plants (dams) to the network.
    Due to the spillage/basin calculations these have real locations not just nodes.
    WARNING: the node is assigned based on the damn province name (turbine link)
            NOT future proof

    Args:
        network (pypsa.Network): the network object
        config (dict): the yaml config
        nodes (pd.Index): the buses
        region_shapes (gpd.GeoDataFrame): the region shapes GDF
        costs (pd.DataFrame): the costs dataframe
        planning_horizons (int): the year
    """

    nodes = network.buses.index

    logger.info("\tAdding dam cascades")

    # load dams
    df = pd.read_csv(config["hydro_dams"]["dams_path"], index_col=0)

    points = df.apply(lambda row: Point(row.Lon, row.Lat), axis=1)
    dams = gpd.GeoDataFrame(df, geometry=points, crs=CRS)
    all_dams = dams.index.values
    dams = dams.sjoin(region_shapes, predicate="within")

    if dams.cluster.isna().sum() > 0:
        logger.warning(
            f"{dams.loc[dams.cluster.isna()].index} dams not assigned to a region - dropping"
        )
        dams = dams.loc[~dams.cluster.isna()]

    hourly_rng = pd.date_range(
        config["hydro_dams"]["inflow_date_start"],
        config["hydro_dams"]["inflow_date_end"],
        freq="1h",  # THIS IS THE INFLOW RES
        inclusive="left",
    )
    # TODO implement inflow calc, understand resolution (seems daily!)
    inflow = pd.read_pickle(config["hydro_dams"]["inflow_path"])
    # select inflow year
    hourly_rng = hourly_rng[hourly_rng.year == INFLOW_DATA_YR]
    inflow = inflow.loc[inflow.index.year == INFLOW_DATA_YR]
    inflow = inflow.reindex(hourly_rng, fill_value=0)
    inflow.columns = all_dams  # TODO dangerous
    # select only the inflwo to dams in the network
    inflow = inflow.loc[:, inflow.columns.map(dams.cluster).isin(nodes)]
    inflow = shift_profile_to_planning_year(inflow, planning_horizons)
    inflow = inflow.loc[network.snapshots]
    # m^3/KWh -> m^3/MWh
    water_consumption_factor = dams.loc[:, "Water_consumption_factor_avg"] * 1e3

    #######
    # ### Add hydro stations as buses
    network.add(
        "Bus",
        dams.index,
        suffix=" station",
        carrier="stations",
        x=dams["geometry"].to_crs("+proj=cea").centroid.to_crs(region_shapes.crs).x,
        y=dams["geometry"].to_crs("+proj=cea").centroid.to_crs(region_shapes.crs).y,
        location=dams["cluster"],
        province=dams["Province"],
    )

    dam_buses = network.buses[network.buses.carrier == "stations"]

    # ===== add hydro reservoirs as stores ======
    initial_capacity = pd.read_pickle(config["hydro_dams"]["reservoir_initial_capacity_path"])
    effective_capacity = pd.read_pickle(config["hydro_dams"]["reservoir_effective_capacity_path"])
    initial_capacity.index = all_dams
    effective_capacity.index = all_dams
    initial_capacity = initial_capacity / water_consumption_factor
    effective_capacity = effective_capacity / water_consumption_factor

    # select relevant dams in nodes
    effective_capacity = effective_capacity.loc[
        effective_capacity.index.map(dams.cluster).isin(nodes)
    ]
    initial_capacity = initial_capacity.loc[initial_capacity.index.map(dams.cluster).isin(nodes)]

    network.add(
        "Store",
        dams.index,
        suffix=" reservoir",
        bus=dam_buses.index,
        e_nom=effective_capacity,
        e_initial=initial_capacity,
        e_cyclic=True,
        # TODO fix all config["costs"]
        marginal_cost=config["hydro"]["marginal_cost"]["reservoir"],
    )

    # add hydro turbines to link stations to provinces
    network.add(
        "Link",
        dams.index,
        suffix=" turbines",
        bus0=dam_buses.index,
        bus1=dams["Province"],
        carrier="hydroelectricity",
        p_nom=10 * dams["installed_capacity_10MW"],
        capital_cost=(
            costs.at["hydro", "capital_cost"] if config["hydro"]["hydro_capital_cost"] else 0
        ),
        efficiency=1,
        location=dams["Province"],
        p_nom_extendable=False,
    )

    # ===  add rivers to link station to station
    dam_edges = pd.read_csv(config["hydro_dams"]["damn_flows_path"], delimiter=",")
    in_nodes = dam_edges.bus0.map(dams.cluster).isin(nodes) & dam_edges.end_bus.map(
        dams.cluster
    ).isin(nodes)
    dam_edges = dam_edges[in_nodes]

    # === normal flow ====
    for row in dam_edges.iterrows():
        bus0 = row[1].bus0 + " turbines"
        bus2 = row[1].end_bus + " station"
        network.links.at[bus0, "bus2"] = bus2
        network.links.at[bus0, "efficiency2"] = 1.0

    # === spillage ====
    # TODO WHY EXTENDABLE - weather year?
    for row in dam_edges.iterrows():
        bus0 = row[1].bus0 + " station"
        bus1 = row[1].end_bus + " station"
        network.add(
            "Link",
            f"{bus0}-{bus1}" + " spillage",
            bus0=bus0,
            bus1=bus1,
            p_nom_extendable=True,
        )

    dam_ends = [
        dam
        for dam in np.unique(dams.index.values)
        if dam not in dam_edges["bus0"]
        or dam not in dam_edges["end_bus"]
        or (dam in dam_edges["end_bus"].values & dam not in dam_edges["bus0"])
    ]
    # need some kind of sink to absorb spillage (e,g ocean).
    # here hack by flowing to existing bus with 0 efficiency (lose)
    # TODO make more transparent -> generator with neg sign and 0 c0st
    for bus0 in dam_ends:
        network.add(
            "Link",
            bus0 + " spillage",
            bus0=bus0 + " station",
            bus1=bus0 + " station",
            p_nom_extendable=True,
            efficiency=0.0,
        )

    # add inflow as generators
    # only feed into hydro stations which are the first of a cascade
    inflow_stations = [
        dam for dam in np.unique(dams.index.values) if dam not in dam_edges["end_bus"].values
    ]

    for inflow_station in inflow_stations:
        # p_nom = 1 and p_max_pu & p_min_pu = p_pu, compulsory inflow
        p_nom = (inflow / water_consumption_factor)[inflow_station].max()
        p_pu = (inflow / water_consumption_factor)[inflow_station] / p_nom
        p_pu.index = network.snapshots
        network.add(
            "Generator",
            inflow_station + " inflow",
            bus=inflow_station + " station",
            carrier="hydro_inflow",
            p_max_pu=p_pu.clip(1.0e-6),
            # p_min_pu=p_pu.clip(1.0e-6),
            p_nom=p_nom,
        )

        # p_nom*p_pu = XXX m^3 then use turbines efficiency to convert to power

    return

    # ======= add other existing hydro power (no basin cascades) ===
    # 1. Subtract cascades from GEM

    # 2. Work out basins and inflow

    # 3. Add remaining hydro

    hydro_p_nom = pd.read_hdf(config["hydro_dams"]["p_nom_path"]).loc[nodes]
    hydro_p_max_pu = (
        pd.read_hdf(
            config["hydro_dams"]["p_max_pu_path"],
            key=config["hydro_dams"]["p_max_pu_key"],
        ).tz_localize(None)
    )[nodes]

    hydro_p_max_pu = shift_profile_to_planning_year(hydro_p_max_pu, planning_horizons)
    # sort buses (columns) otherwise stuff will break
    hydro_p_max_pu.sort_index(axis=1, inplace=True)

    hydro_p_max_pu = hydro_p_max_pu.loc[network.snapshots]
    hydro_p_max_pu.index = network.snapshots

    logger.info("\tAdding extra hydro capacity (regionally aggregated)")

    network.add(
        "Generator",
        nodes,
        suffix=" hydroelectricity",
        bus=nodes,
        carrier="hydroelectricity",
        p_nom=hydro_p_nom,
        p_nom_min=hydro_p_nom,
        p_nom_extendable=False,
        p_max_pu=hydro_p_max_pu,
        capital_cost=(
            costs.at["hydro", "capital_cost"] if config["hydro"]["hydro_capital_cost"] else 0
        ),
    )


# TODO fix timezones/centralsie, think Shanghai won't work on its own
def generate_periodic_profiles(
    dt_index=None,
    col_tzs=pd.Series(index=PROV_NAMES, data=len(PROV_NAMES) * ["Shanghai"]),
    weekly_profile=range(24 * 7),
):
    """Give a 24*7 long list of weekly hourly profiles, generate this
    for each country for the period dt_index, taking account of time
    zones and Summer Time.
    """

    weekly_profile = pd.Series(weekly_profile, range(24 * 7))
    # TODO fix, no longer take into accoutn summer time
    # ALSO ADD A TODO in base_network
    week_df = pd.DataFrame(index=dt_index, columns=col_tzs.index)
    for ct in col_tzs.index:
        week_df[ct] = [24 * dt.weekday() + dt.hour for dt in dt_index.tz_localize(None)]
        week_df[ct] = week_df[ct].map(weekly_profile)
    return week_df


def prepare_network(
    config: dict,
    costs: pd.DataFrame,
    planning_year: int,
    biomass_potential: pd.DataFrame = None,
    paths: dict = None,
) -> pypsa.Network:
    """Prepares/makes the network object for overnight mode according to config &
    at 1 node per region/province

    Args:
        config (dict): the snakemake config
        costs (pd.DataFrame): the costs dataframe (anualised capex and marginal costs)
        planning_year (int): the planning year
        biomass_potential (Optional, pd.DataFrame): biomass potential dataframe. Defaults to None.
        paths (Optional, dict): the paths to the data files. Defaults to None.

    Returns:
        pypsa.Network: the pypsa network object
    """

    # determine whether gas/coal to be added depending on specified conv techs
    config["add_gas"] = (
        True
        if [tech for tech in config["Techs"]["conv_techs"] if ("gas" in tech or "CGT" in tech)]
        else False
    )
    config["add_coal"] = (
        True if [tech for tech in config["Techs"]["conv_techs"] if "coal" in tech] else False
    )

    planning_horizons = planning_year
    nodes = network.buses.query("carrier =='AC'")

    # add carriers
    add_carriers(network, config, costs)

    ws_carriers = [c for c in config["Techs"]["vre_techs"] if c.find("wind") >= 0 or c == "solar"]
    add_wind_and_solar(network, ws_carriers, paths, planning_horizons, costs)

    add_conventional_generators(network, config, costs)

    # TODO add coal CC? no retrofit option

    if "PHS" in config["Techs"]["store_techs"]:
        # pure pumped hydro storage, fixed, 6h energy by default, no inflow
        cc = costs.at["PHS", "capital_cost"]

        network.add(
            "StorageUnit",
            nodes.index,
            suffix=" PHS",
            bus=nodes.index,
            carrier="PHS",
            p_nom_extendable=True,
            max_hours=config["hydro"]["PHS_max_hours"],
            efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
            efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
            cyclic_state_of_charge=True,
            capital_cost=cc,
            marginal_cost=0.0,
        )

    if config["add_hydro"]:
        logger.info("Adding hydro to network")
        region_shapes = gpd.read_file(paths["regions"])
        add_hydro(network, config, region_shapes, costs, planning_horizons)

    if config["add_H2"]:
        logger.info("Adding H2 buses to network")
        # do beore heat coupling to avoid warning
        network.add(
            "Bus",
            nodes.index,
            suffix=" H2",
            x=nodes.x,
            y=nodes.y,
            carrier="H2",
            location=nodes.index,
            province=nodes.province,
        )

    if config.get("heat_coupling", False):
        logger.info("Adding heat and CHP to the network")
        add_heat_coupling(network, config, costs, planning_horizons, paths)

        if config["add_biomass"]:
            logger.info("Adding biomass to network")
            add_co2_capture_support(network)
            add_biomass_chp(
                network,
                costs,
                biomass_potential[nodes.province.unique()],
                add_beccs="beccs" in config["Techs"]["vre_techs"],
            )

    if config["add_H2"]:
        logger.info("Adding H2 to network")
        add_H2(network, config, costs, yr)

    if "battery" in config["Techs"]["store_techs"]:
        network.add(
            "Bus",
            nodes.index,
            suffix=" battery",
            x=nodes.x,
            y=nodes.y,
            carrier="battery",
            location=nodes.index,
            province=nodes.province,
        )

        # TODO Why no standing loss?: test with
        min_charge = config["electricity"].get("min_charge", {"battery": 0})
        min_charge = min_charge.get("battery", 0)
        network.add(
            "Store",
            nodes.index + " battery",
            bus=nodes.index + " battery",
            e_cyclic=True,
            e_nom_extendable=True,
            e_min_pu=min_charge,
            capital_cost=costs.at["battery storage", "capital_cost"],
            lifetime=costs.at["battery storage", "lifetime"],
        )

        # TODO understand/remove sources, data should not be in code
        # Sources:
        # [HP]: Henning, Palzer http://www.sciencedirect.com/science/article/pii/S1364032113006710
        # [B]: Budischak et al. http://www.sciencedirect.com/science/article/pii/S0378775312014759

        network.add(
            "Link",
            nodes.index + " battery charger",
            bus0=nodes.index,
            bus1=nodes.index + " battery",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["battery inverter", "efficiency"]
            * costs.at["battery inverter", "capital_cost"],
            p_nom_extendable=True,
            carrier="battery",
            lifetime=costs.at["battery inverter", "lifetime"],
        )

        network.add(
            "Link",
            nodes.index + " battery discharger",
            bus0=nodes.index + " battery",
            bus1=nodes.index,
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            marginal_cost=0.0,
            p_nom_extendable=True,
            carrier="battery discharger",
        )

    # ============= add lines =========
    # The lines are implemented according to the transport model (no KVL) and without losses.
    # see Neumann et al 10.1016/j.apenergy.2022.118859
    # TODO make not lossless optional (? - increases computing cost)

    if not config["no_lines"]:
        add_voltage_links(network, config)

    assign_locations(network)
    return network


def apply_time_resolution(network: pypsa.Network, snapshot: dict):
    """Apply time resolution to network snapshots and weightings
    WARNING: THIS IS DUMB 1 from N SAMPLING

    Args:
        network (pypsa.Network): the network object
        config (dict): the snakemake config
    """
    freq = config["snapshots"]["frequency"]
    if freq != 1.0:
        logger.info(f"Applying time resolution of {freq}h to network")
        # resample snapshots
        new_snapshots = network.snapshots[:: int(freq)]
        network.snapshots = new_snapshots
        # resample snapshot weightings
        network.snapshot_weightings[:] = freq


def attach_load(
    n: pypsa.Network,
    load_path: os.PathLike,
    busmap_path: os.PathLike,
    scaling: float = 1.0,
) -> None:
    """Attach load data to the network.

    Args:
        n (pypsa.Network): The PyPSA network to attach the load data to.
        load_path (PathLike): Path to the load data file.
        busmap_path (PathLike): Path to the busmap file.
        scaling (float): Scaling factor for the load data. Defaults to 1.0.
    """
    load = (
        xr.open_dataarray(load_path).to_dataframe().squeeze(axis=1).unstack(level="time")
    ).rename_axis("Bus")

    # apply clustering busmap
    busmap = pd.read_csv(busmap_path, dtype=str, index_col=0).squeeze()
    load = load.groupby(busmap).sum().T

    logger.info(f"Load data scaled by factor {scaling}.")
    load *= scaling

    n.add("Load", load.columns, bus=load.columns, p_set=load.values)  # carrier="electricity"


def scale_load_by_projection(n: pypsa.Network, prov_projection: pd.Series) -> pd.Series:
    """Scale the load in the network according to yearly projections.

    Args:
        n (pypsa.Network): The PyPSA network.
        prov_projection (pd.Series): provincial estimates for planning year.
        yr (int): The planning year.

    Returns:
        pd.Series: The projected demand.
    """
    loads_yr = n.loads_t.p_set.sum()
    provinces = n.loads_t.p_set.columns.map(n.buses.province)
    df = pd.DataFrame(index=loads_yr.index, data={"load": loads_yr.values, "province": provinces})
    df["fraction"] = df.groupby("province").transform(lambda x: x / x.sum())
    df["projected_total"] = df.province.map(prov_projection) * df.fraction

    # apply rescale
    n.loads_t.p_set = n.loads_t.p_set * df["projected_total"] / loads_yr


if __name__ == "__main__":
    # Detect running outside of snakemake and mock snakemake for testing
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "prepare_networks",
            topology="current+FCG",
            co2_pathway="exp175default",
            # co2_pathway="SSP2-PkBudg1000-pseudo-coupled",
            planning_horizons=2025,
            heating_demand="positive",
            # configfiles="resources/tmp/pseudo-coupled.yaml",
        )

    configure_logging(snakemake)

    config = snakemake.params.config
    snapshots_config = snakemake.params.snapshots
    yr = int(snakemake.wildcards["planning_horizons"])
    pathway = snakemake.wildcards.co2_pathway
    co2_opts = ConfigManager(config).fetch_co2_restriction(pathway, yr)

    logger.info(f"Preparing network for {yr} and scenario:")
    logger.info(config["scenario"])
    logger.info(config["co2_scenarios"])

    # load network and change from data year to plan year
    network = pypsa.Network(snakemake.input.clustered_network)
    network.snapshots = shift_profile_to_planning_year(
        pd.DataFrame(index=network.snapshots), yr
    ).index
    network.snapshot_weightings[:] = config["snapshots"]["frequency"]
    network.name = f"China {pathway} {yr}  {config['scenario']}"

    # add load
    prov_demand_projection = read_yearly_load_projections(
        snakemake.input.elec_projections, snakemake.params.elec_proj_conversion
    )[yr]
    attach_load(network, snakemake.input.base_elec_load, snakemake.input.busmap, scaling=1)
    scale_load_by_projection(network, prov_demand_projection)

    # load costs
    n_years = config["snapshots"]["frequency"] * len(network.snapshots) / 8760.0
    tech_costs = snakemake.input["tech_costs"]
    input_paths = {k: v for k, v in snakemake.input.items()}
    cost_year = yr
    costs = load_costs(tech_costs, config["costs"], config["electricity"], cost_year, n_years)

    # biomass
    if config["add_biomass"]:
        biomass_potential = pd.read_hdf(input_paths["biomass_potential"])
    else:
        biomass_potential = None

    network = prepare_network(snakemake.config, costs, yr, biomass_potential, paths=input_paths)
    sanitize_carriers(network, snakemake.config)

    apply_time_resolution(network, snapshots_config)

    outp = snakemake.output.network_name
    compression = snakemake.config.get("io", None)
    if compression:
        compression = compression.get("nc_compression", None)
    network.export_to_netcdf(outp, compression=compression)

    logger.info(f"Network for {yr} prepared and saved to {outp}")

    costs_outp = os.path.dirname(outp) + f"/costs_{yr}.csv"
    costs.to_csv(costs_outp)
