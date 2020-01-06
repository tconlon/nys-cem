import numpy as np
from gurobipy import *
from utils import annualization_rate, load_timeseries
import pandas as pd

def create_model(args, lowc_target, nuclear_boolean, h2_boolean, heating_cap_mw, ev_cap_mw):

    # Set up model parameters
    m = Model("capacity_optimization_renewable_targets")
    T = args.num_years*8760
    trange = range(T)

    # Load in time-series data
    baseline_demand, heating_load_mw, ev_load_mw, onshore_wind_pot, \
    offshore_wind_pot,solar_pot, fixed_hydro_mw, flex_hydro_daily_mwh = load_timeseries(args, heating_cap_mw, ev_cap_mw)

    # Annualize capacity costs for model
    onshore_cap_cost  = args.num_years * annualization_rate(args.i_rate, args.annualize_years_cap) * \
                        float(args.onshore_cost_mw)
    offshore_cap_cost = args.num_years * annualization_rate(args.i_rate, args.annualize_years_cap) * \
                        float(args.offshore_cost_mw)
    solar_cap_cost    = args.num_years * annualization_rate(args.i_rate, args.annualize_years_cap) * \
                        float(args.solar_cost_mw)
    battery_cost_mwh  = args.num_years * annualization_rate(args.i_rate, args.annualize_years_storage) * \
                        float(args.battery_cost_mwh)
    battery_cost_mw   = args.num_years * annualization_rate(args.i_rate, args.annualize_years_storage) * \
                        float(args.battery_cost_mw)
    h2_cost_mwh       = args.num_years * annualization_rate(args.i_rate, args.annualize_years_storage) * \
                        float(args.h2_cost_mwh)
    h2_cost_mw        = args.num_years * annualization_rate(args.i_rate, args.annualize_years_storage) * \
                        float(args.h2_cost_mw)
    gt_cost_mw        = [args.num_years * annualization_rate(args.i_rate, args.annualize_years_cap) *
                         args.reserve_req * float(x) for x in args.gt_cost_mw]

    # Load transmission cost and current capacity paramters
    tx_matrix_limits = pd.read_excel(os.path.join(args.data_dir, 'transmission_matrix_limits.xlsx'),
                                     header=0, index_col=0)
    tx_matrix_costs = pd.read_excel(os.path.join(args.data_dir, 'transmission_matrix_costs.xlsx'),
                                    header=0, index_col=0)

    ## Define and populate transmission dictionaries
    # Dictionary for transmission parameters
    tx_dict = {}

    # Dictionaries to store the transmission time series
    tx_ts_dict = {}
    tx_ts_abs_dict = {}

    # Base string for tx dictionary keys
    base_tx_cap_string = 'net_export_limits_{}_{}'

    # Create our transmission cost and limits dictionary
    for i in range(len(tx_matrix_limits)):
        for j in range(len(tx_matrix_limits.columns)):
            if tx_matrix_limits.iloc[i, j] > 0:
                tx_dict[base_tx_cap_string.format(i + 1, j + 1)] = (tx_matrix_limits.iloc[i, j],
                                                                    args.num_years *
                                                                    annualization_rate(args.i_rate,
                                                                                       args.annualize_years_cap) *
                                                                    tx_matrix_costs.iloc[i, j])

    # Initialize gas variables
    gt_current_cap = [x / args.reserve_req for x in args.gt_current_cap_mw]
    gt_diff_cost = args.gt_startup_cost_mw * 0.5  # start-up cost, multiply by 0.5 to change to start-up/shut-down

    # Initialize nuclear generation constraint base on nuclear boolean
    nuc_gen_mw = [int(nuclear_boolean) * args.nuc_gen_mw[i] for i in range(4)]

    ## Print statements to check model configuration is correct:

    print('Low-Carbon Supply Target')
    print(lowc_target)
    print('Heating Capacity [MW]')
    print(heating_cap_mw)
    print('EV Capacity [MW]')
    print(ev_cap_mw)
    print('Nuclear Generation [MW]')
    print(nuc_gen_mw)
    print('H2 Storage Availability')
    print(h2_boolean)


    for i in range(0, args.num_regions):

        # Find all interconnections for the current region
        tx_lines_set = sorted([j for j in tx_dict.keys() if str(i + 1) in j])
        # Find the complementary indices for the interconnections
        tx_export_regions_set = np.unique([j.split('_')[-1] for j in tx_lines_set])
        # Only take the indices larger than i
        tx_export_regions_set = [j for j in tx_export_regions_set if int(j) > i + 1]

        # Create new tx capacity variables, two for each interface, an 'export' and an 'import'
        for export_region in tx_export_regions_set:
            new_export_cap = m.addVar(obj=tx_dict[base_tx_cap_string.format(i + 1, export_region)][1],
                                      name='new_export_limits_{}_{}'.format(i + 1, export_region))
            new_import_cap = m.addVar(obj=tx_dict[base_tx_cap_string.format(export_region, i + 1)][1],
                                      name='new_export_limits_{}_{}'.format(export_region, i + 1))

        m.update()


        # Initialize capacity variables
        onshore_cap     = m.addVar(ub=args.onshore_wind_limit_mw[i], obj=onshore_cap_cost,
                                   name = 'onshore_cap_region_{}'.format(i + 1))
        offshore_cap    = m.addVar(obj=offshore_cap_cost, name = 'offshore_cap_region_{}'.format(i + 1))
        solar_cap       = m.addVar(ub=float(args.solar_limit_mw[i]), obj=solar_cap_cost,
                                   name='solar_cap_region_{}'.format(i + 1))
        gt_cap          = m.addVar(obj=gt_cost_mw[i], name='new_gt_cap_region_{}'.format(i + 1))
        battery_cap_mwh = m.addVar(obj=battery_cost_mwh, name = 'batt_energy_cap_region_{}'.format(i + 1))
        battery_cap_mw  = m.addVar(obj=battery_cost_mw, name = 'batt_power_cap_region_{}'.format(i + 1))
        h2_cap_mwh      = m.addVar(obj=h2_cost_mwh, name = 'h2_energy_cap_region_{}'.format(i + 1))
        h2_cap_mw       = m.addVar(obj=h2_cost_mw, name = 'h2_power_cap_region_{}'.format(i + 1))

        # Initialize time-series variables
        flex_hydro_mw     = m.addVars(trange, ub=args.flex_hydro_cap_mw[i],
                                      name = 'flex_hydro_region_{}'.format(i + 1))
        batt_charge       = m.addVars(trange, name= 'batt_charge_region_{}'.format(i + 1))
        batt_discharge    = m.addVars(trange, name= 'batt_discharge_region_{}'.format(i + 1))
        h2_charge         = m.addVars(trange, name= 'h2_charge_region_{}'.format(i + 1))
        h2_discharge      = m.addVars(trange, name= 'h2_discharge_region_{}'.format(i + 1))


        # Create transmission time series and total export/import capacity variables
        for export_region in tx_export_regions_set:
            # Time series variable, must set lowerbound to -Infinity since we are allowing 'negative' flow
            tx_export_vars = m.addVars(trange, name= 'net_exports_ts_{}_to_{}'.format(i+1, export_region),
                                       lb = -GRB.INFINITY)

            # Export cap is = new cap + existing cap (from dictionary)
            tx_export_cap  = m.getVarByName("new_export_limits_{}_{}".format(i + 1, export_region)) + \
                                tx_dict[base_tx_cap_string.format(i + 1, export_region)][0]

            # Import cap is = new cap + existing cap (from dictionary)
            tx_import_cap  = m.getVarByName("new_export_limits_{}_{}".format(export_region, i + 1))  + \
                                tx_dict[base_tx_cap_string.format(export_region, i + 1)][0]

            m.update()

            # Constrain individual Tx flow variables to the export import capacity
            for j in trange:
                m.addConstr(tx_export_vars[j] - tx_export_cap <= 0)
                m.addConstr(tx_export_vars[j] + tx_import_cap >= 0)


            # Store these tx flow variables in the time series dictionary for energy balance equation
            tx_ts_dict['net_exports_ts_{}_to_{}'.format(i+1, export_region)] = tx_export_vars

        m.update()



        # Initialize hq_imports
        hq_imports = m.addVars(trange, ub=args.hq_limit_mw[i], obj=args.hq_cost_mwh[i],
                               name="hq_import_region_{}".format(i + 1))

        # Initialize battery level and EV charging variables
        batt_level   = m.addVars(trange, name = 'batt_level_region_{}'.format(i + 1))
        h2_level     = m.addVars(trange, name = 'h2_level_region_{}'.format(i + 1))
        ev_charging  = m.addVars(trange, ub=ev_load_mw[i] / float(args.ev_charging_p2e_ratio),
                                 name = 'ev_charging_region_{}'.format(i + 1))

        # Initialize netload variables
        netload_diff = m.addVars(trange, lb=-GRB.INFINITY, name = "netload_diff_region_{}".format(i + 1))
        netload_abs  = m.addVars(trange, obj=gt_diff_cost, name =  "netload_abs_region_{}".format(i + 1))
        netload      = m.addVars(trange, obj=args.netload_cost_mwh[i], name="netload_region_{}".format(i + 1))

        # Set up initial battery cap constraints
        batt_level[0] = battery_cap_mwh
        h2_level[0]   = h2_cap_mwh

        # Initialize H2 constraints based on model run specifics
        if not h2_boolean:
            m.addConstr(h2_cap_mwh == 0)
            m.addConstr(h2_cap_mw == 0)


        # Find all export/import time series for energy balance -- these variables will find the same time series
        # but in different regions
        tx_export_keys     = [k for k in tx_ts_dict.keys() if 'ts_{}'.format(i + 1) in k]
        tx_import_keys     = [k for k in tx_ts_dict.keys() if 'to_{}'.format(i + 1) in k]

        m.update()



        # Add time-series Constraints
        for j in trange:
            # Sum all the transmission export timeseries for region i at time step j
            if len(tx_export_keys) > 0:
                total_exports = quicksum(tx_ts_dict[tx_export_keys[k]][j] for k in range(len(tx_export_keys)))
            else:
                total_exports = 0

            # Sum all the transmission import timeseries for region i at time step j
            if len(tx_import_keys) > 0:
                total_imports     = quicksum(tx_ts_dict[tx_import_keys[k]][j] for k in range(len(tx_import_keys)))
            else:
                total_imports = 0


            if j == 0:
                # Load constraint: No battery/H2 operation in time t=0
                # First transmission loss constraint
                m.addConstr((offshore_cap * offshore_wind_pot[j, i]) + (onshore_cap * onshore_wind_pot[j, i]) +
                            (solar_cap * solar_pot[j, i]) + flex_hydro_mw[j] -
                            ev_charging[j] - total_exports +  (1+args.trans_loss) *total_imports +
                            hq_imports[j] + netload[j] >=
                            baseline_demand[j, i] + heating_load_mw[j, i] - fixed_hydro_mw[j, i] - nuc_gen_mw[i])

                # Second transmission loss constraint
                m.addConstr((offshore_cap * offshore_wind_pot[j, i]) + (onshore_cap * onshore_wind_pot[j, i]) +
                            (solar_cap * solar_pot[j, i]) + flex_hydro_mw[j] -
                            ev_charging[j] - total_exports + (1 - args.trans_loss) * total_imports +
                            hq_imports[j] + netload[j] >=
                            baseline_demand[j, i] + heating_load_mw[j, i] - fixed_hydro_mw[j, i] - nuc_gen_mw[i])


            else:
                # Load constraint: Add battery constraints for all other times series
                # First transmission loss constraint
                m.addConstr((offshore_cap * offshore_wind_pot[j, i]) + (onshore_cap * onshore_wind_pot[j, i]) +
                            (solar_cap * solar_pot[j, i]) + flex_hydro_mw[j] -
                            batt_charge[j] + batt_discharge[j] - h2_charge[j] + h2_discharge[j] -
                            ev_charging[j] - total_exports +  (1+args.trans_loss) *total_imports +
                            hq_imports[j] + netload[j] >=
                            baseline_demand[j, i] + heating_load_mw[j, i] - fixed_hydro_mw[j, i] - nuc_gen_mw[i])

                # Second transmission loss constraint
                m.addConstr((offshore_cap * offshore_wind_pot[j, i]) + (onshore_cap * onshore_wind_pot[j, i]) +
                            (solar_cap * solar_pot[j, i]) + flex_hydro_mw[j] -
                            batt_charge[j] + batt_discharge[j] - h2_charge[j] + h2_discharge[j] -
                            ev_charging[j] - total_exports + (1 - args.trans_loss) * total_imports +
                            hq_imports[j] + netload[j] >=
                            baseline_demand[j, i] + heating_load_mw[j, i] - fixed_hydro_mw[j, i] - nuc_gen_mw[i])

                # Battery/H2 energy conservation constraints
                m.addConstr(batt_discharge[j] / args.battery_eff - args.battery_eff * batt_charge[j] ==
                            ((1 - args.self_discharge) * batt_level[j - 1] - batt_level[j]))
                m.addConstr(h2_discharge[j] / args.h2_eff - args.h2_eff * h2_charge[j] ==
                            ((1 - args.self_discharge) * h2_level[j - 1] - h2_level[j]))

                # Battery operation constraints
                m.addConstr(batt_charge[j] - battery_cap_mw <= 0)
                m.addConstr(batt_discharge[j] - battery_cap_mw <= 0)
                m.addConstr(batt_level[j] - battery_cap_mwh <= 0)
                m.addConstr(battery_cap_mwh - 4 * battery_cap_mw <= 0)
                m.addConstr(battery_cap_mw - 8 * battery_cap_mwh <= 0)

                # H2 operation constraints
                m.addConstr(h2_charge[j] - h2_cap_mw <= 0)
                m.addConstr(h2_discharge[j] - h2_cap_mw <= 0)
                m.addConstr(h2_level[j] - h2_cap_mwh <= 0)

                # Net load ramping constraints
                m.addConstr(netload_diff[j] - (netload[j] - netload[j - 1]) == 0)
                m.addConstr(netload_abs[j] >= netload_diff[j])
                m.addConstr(netload_abs[j] >= -netload_diff[j])

            ## Net load constraints
            m.addConstr(netload[j] - (gt_cap + gt_current_cap[i]) <= 0)


            # Add constraints for new HQ imports into NYC -- This is to ensure constant flow of power
            if i == 2:
                m.addConstr(hq_imports[j] - args.hqch_capacity_factor * args.hq_limit_mw[i] == 0)

        m.update()

        # Initialize flexible hydro dispatch + EV charging constraints
        for j in range(0, int(T / 24) - 1):
            jrange_hydro_daily = range(j * 24, (j + 1) * 24)
            jrange_ev = range(j * 24, j * 24 + args.ev_charging_hours)

            m.addConstr(quicksum(flex_hydro_mw[k + 1] for k in jrange_hydro_daily) == flex_hydro_daily_mwh[j, i])

            if args.ev_charging_method == 'flexible':
                m.addConstr(quicksum(ev_charging[args.ev_hours_start + k] for k in jrange_ev) == ev_load_mw[i] * 24)
            elif args.ev_charging_method == 'fixed':
                for k in jrange_ev:
                    m.addConstr(ev_charging[args.ev_hours_start + k]  == ev_load_mw[i] * 24/args.ev_charging_hours)
            else:
                print('Invalid EV charging method')

        m.update()


    ## Initialize constraints for multi-region variables

    # Data structures for setting net load equal to a percent of the load
    model_data_netload_region_1 = {}
    model_data_netload_region_2 = {}
    model_data_netload_region_3 = {}
    model_data_netload_region_4 = {}

    # Data structure for setting offshore wind capacity equal to NREL limit
    model_data_offshore_cap = {}

    ## Data structures for retrieving Hydro Quebec import time series
    hq_import_region_1 = {}
    hq_import_region_3 = {}

    for j in trange:

        model_data_netload_region_1[j]    = m.getVarByName("netload_region_1[{}]".format(j))
        model_data_netload_region_2[j]    = m.getVarByName("netload_region_2[{}]".format(j))
        model_data_netload_region_3[j]    = m.getVarByName("netload_region_3[{}]".format(j))
        model_data_netload_region_4[j]    = m.getVarByName("netload_region_4[{}]".format(j))

        hq_import_region_1[j]             = m.getVarByName("hq_import_region_1[{}]".format(j))
        hq_import_region_3[j]             = m.getVarByName("hq_import_region_3[{}]".format(j))

    m.update()

    # Offshore generation constraints
    for i in range(args.num_regions):
        model_data_offshore_cap[i] = m.getVarByName("offshore_cap_region_{}".format(i+1))

    m.addConstr((model_data_offshore_cap[2] + model_data_offshore_cap[3]) <= args.offshore_wind_limit_mw)

    # Low-carbon electricity constraint
    full_netload_sum_mwh = quicksum(model_data_netload_region_1[j] + model_data_netload_region_2[j] +
                                    model_data_netload_region_3[j] + model_data_netload_region_4[j] for j in trange)
    full_demand_sum_mwh  = np.sum(baseline_demand[0:T]) + np.sum(heating_load_mw[0:T]) + np.sum(ev_load_mw) * T
    full_imports_sum_mwh = quicksum(hq_import_region_1[j] + hq_import_region_3[j] for j in trange)
    full_nuclear_sum_mwh = np.sum(nuc_gen_mw) * T

    frac_netload = 1 - lowc_target

    m.update()

    if args.rgt_boolean:
        m.addConstr(full_netload_sum_mwh + full_nuclear_sum_mwh -
                    (full_demand_sum_mwh - full_imports_sum_mwh) * (np.max((frac_netload, args.supp_limit))) <= 0)
    else:
        m.addConstr(full_netload_sum_mwh -
                (np.max((frac_netload, args.supp_limit))) * (full_demand_sum_mwh - full_imports_sum_mwh)  <= 0)



    m.update()

    return m
