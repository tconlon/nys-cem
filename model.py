import numpy as np
from gurobipy import *
from utils import annualization_rate, load_timeseries


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

    # Need the 0.5 multiplier in here because of the way the transmission constraints are set up:
    # There are variables for WE export and an EW import lines (and vice versa), need to ascribe
    # half of the cost to each
    trans_cost = [args.num_years * annualization_rate(args.i_rate, args.annualize_years_cap) * 0.5 * float(x)
                  for x in np.array([0, args.trans_cost_mw, args.trans_cost_mw, args.trans_cost_mw, 0])]  # $/MW

    # Initialize gas variables
    gt_current_cap = [x / args.reserve_req for x in args.gt_current_cap_mw]
    gt_diff_cost = args.gt_startup_cost_mw * 0.5  # start-up cost, multiply by 0.5 to change to start-up/shut-down

    # Initialize nuclear generation constraint base on nuclear boolean
    nuc_gen_mw = [int(nuclear_boolean) * args.nuc_gen_mw[i] for i in range(4)]

    ## Print statements to check model configuration is correct:
    print('Demand')
    print(np.sum(np.mean(baseline_demand[0:T], axis=0)))
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

        # Initialize transmission capacity variables
        exports_EW_cap = m.addVar(obj=trans_cost[i], lb=args.translimits_EW[i],
                                  name="exports_EW_region" + str(i + 1) + "_cap")
        imports_EW_cap = m.addVar(obj=trans_cost[i + 1], lb=args.translimits_EW[i + 1],
                                  name="imports_EW_region" + str(i + 1) + "_cap")
        exports_WE_cap = m.addVar(obj=trans_cost[i + 1], lb=args.translimits_WE[i + 1],
                                  name="exports_WE_region" + str(i + 1) + "_cap")
        imports_WE_cap = m.addVar(obj=trans_cost[i], lb=args.translimits_WE[i],
                                  name="imports_WE_region" + str(i + 1) + "_cap")

        m.update()

        # Constrain transmission edge cases
        if args.translimits_EW[i] == 0:
            m.addConstr(exports_EW_cap == 0)

        if args.translimits_EW[i + 1] == 0:
            m.addConstr(imports_EW_cap == 0)

        if args.translimits_WE[i + 1] == 0:
            m.addConstr(exports_WE_cap == 0)

        if args.translimits_WE[i] == 0:
            m.addConstr(imports_WE_cap == 0)

        # Initialize capacity variables
        onshore_cap     = m.addVar(ub=args.onshore_wind_limit_mw[i], obj=onshore_cap_cost)
        offshore_cap    = m.addVar(obj=offshore_cap_cost, name="offshore_cap_region" + str(i + 1))
        solar_cap       = m.addVar(ub=float(args.solar_limit_mw[i]), obj=solar_cap_cost)
        gt_cap          = m.addVar(obj=gt_cost_mw[i], lb=gt_current_cap[i])
        battery_cap_mwh = m.addVar(obj=battery_cost_mwh)
        battery_cap_mw  = m.addVar(obj=battery_cost_mw)
        h2_cap_mwh      = m.addVar(obj=h2_cost_mwh)
        h2_cap_mw       = m.addVar(obj=h2_cost_mw)

        # Initialize time-series variables
        offshore_windutil = m.addVars(trange)
        onshore_windutil  = m.addVars(trange)
        solar_util        = m.addVars(trange)
        flex_hydro_mw     = m.addVars(trange, ub=args.flex_hydro_cap_mw[i])
        batt_charge       = m.addVars(trange)
        batt_discharge    = m.addVars(trange)
        h2_charge         = m.addVars(trange)
        h2_discharge      = m.addVars(trange)

        # Initialize transmission time-series variables
        exports_EW = m.addVars(trange, name="exports_EW_region" + str(i + 1))
        imports_EW = m.addVars(trange, name="imports_EW_region" + str(i + 1))
        exports_WE = m.addVars(trange, name="exports_WE_region" + str(i + 1))
        imports_WE = m.addVars(trange, name="imports_WE_region" + str(i + 1))

        # Initialize hq_imports
        hq_imports = m.addVars(trange, ub=args.hq_limit_mw[i], obj=args.hq_cost_mwh[i],
                               name="hq_import_region" + str(i + 1))

        # Initialize battery level and EV charging variables
        batt_level   = m.addVars(trange)
        h2_level     = m.addVars(trange)
        ev_charging  = m.addVars(trange, ub=ev_load_mw[i] * 6)

        # Initialize netload variables
        netload_diff = m.addVars(trange, lb=-GRB.INFINITY)
        netload_abs  = m.addVars(trange, obj=gt_diff_cost)
        netload      = m.addVars(trange, obj=args.netload_cost_mwh[i], name="netload_region" + str(i + 1))

        m.update()

        # Set up initial battery cap constraints
        batt_level[0] = 0  # battery_cap_mwh/2
        h2_level[0] = h2_cap_mwh / 2

        m.update()

        # Initialize H2 constraints based on model run specifics
        if not h2_boolean:
            m.addConstr(h2_cap_mwh == 0)
            m.addConstr(h2_cap_mw == 0)

        m.update()

        # Add time-series Constraints
        for j in trange:

            if j == 0:
                # Load constraint: No battery/H2 operation in time t=0
                m.addConstr(offshore_windutil[j] + onshore_windutil[j] + solar_util[j] + flex_hydro_mw[j] - \
                            ev_charging[j] - (exports_EW[j] + exports_WE[j]) + \
                            args.trans_eff * (imports_EW[j] + imports_WE[j]) + hq_imports[j] + netload[j] == \
                            baseline_demand[j, i] + heating_load_mw[j, i] - fixed_hydro_mw[j, i] - nuc_gen_mw[i])
            else:
                # Load constraint: Add battery constraints for all other times series
                m.addConstr(offshore_windutil[j] + onshore_windutil[j] + solar_util[j] + flex_hydro_mw[j] - \
                            batt_charge[j] + batt_discharge[j] - h2_charge[j] + h2_discharge[j] - \
                            ev_charging[j] - (exports_EW[j] + exports_WE[j]) + \
                            args.trans_eff * (imports_EW[j] + imports_WE[j]) + hq_imports[j] + netload[j] == \
                            baseline_demand[j, i] + heating_load_mw[j, i] - fixed_hydro_mw[j, i] - nuc_gen_mw[i])

                # Battery/H2 energy conservation constraints
                m.addConstr(batt_discharge[j] / args.battery_eff - args.battery_eff * batt_charge[j] == \
                            ((1 - args.self_discharge) * batt_level[j - 1] - batt_level[j]))
                m.addConstr(h2_discharge[j] / args.h2_eff - args.h2_eff * h2_charge[j] == \
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

            # Renewable generation utilization constraints
            m.addConstr(offshore_windutil[j] - (offshore_cap * offshore_wind_pot[j, i]) <= 0)
            m.addConstr(onshore_windutil[j] - (onshore_cap * onshore_wind_pot[j, i]) <= 0)
            m.addConstr(solar_util[j] - (solar_cap * solar_pot[j, i]) <= 0)

            ## Net load constraints
            m.addConstr(netload[j] - gt_cap <= 0)

            # Add Import/Export Cap Constraints
            m.addConstr(exports_EW[j] - exports_EW_cap <= 0)
            m.addConstr(imports_EW[j] - imports_EW_cap <= 0)
            m.addConstr(exports_WE[j] - exports_WE_cap <= 0)
            m.addConstr(imports_WE[j] - imports_WE_cap <= 0)

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
    # Data structures for equating transmission variables
    model_data_exports_EW_region1 = {}
    model_data_imports_EW_region1 = {}
    model_data_exports_WE_region1 = {}
    model_data_imports_WE_region1 = {}
    model_data_exports_EW_region2 = {}
    model_data_imports_EW_region2 = {}
    model_data_exports_WE_region2 = {}
    model_data_imports_WE_region2 = {}
    model_data_exports_EW_region3 = {}
    model_data_imports_EW_region3 = {}
    model_data_exports_WE_region3 = {}
    model_data_imports_WE_region3 = {}
    model_data_exports_EW_region4 = {}
    model_data_imports_EW_region4 = {}
    model_data_exports_WE_region4 = {}
    model_data_imports_WE_region4 = {}

    # Data structures for setting net load equal to a percent of the load
    model_data_netload_region1 = {}
    model_data_netload_region2 = {}
    model_data_netload_region3 = {}
    model_data_netload_region4 = {}

    # Data structure for setting offshore wind capacity equal to NREL limit
    model_data_offshore_cap = {}

    ## Data structures for retrieving Hydro Quebec import time series
    hq_import_region1 = {}
    hq_import_region3 = {}

    for i in trange:
        model_data_exports_EW_region1[i] = m.getVarByName("exports_EW_region1[" + str(i) + "]")
        model_data_imports_EW_region1[i] = m.getVarByName("imports_EW_region1[" + str(i) + "]")
        model_data_exports_WE_region1[i] = m.getVarByName("exports_WE_region1[" + str(i) + "]")
        model_data_imports_WE_region1[i] = m.getVarByName("imports_WE_region1[" + str(i) + "]")

        model_data_exports_EW_region2[i] = m.getVarByName("exports_EW_region2[" + str(i) + "]")
        model_data_imports_EW_region2[i] = m.getVarByName("imports_EW_region2[" + str(i) + "]")
        model_data_exports_WE_region2[i] = m.getVarByName("exports_WE_region2[" + str(i) + "]")
        model_data_imports_WE_region2[i] = m.getVarByName("imports_WE_region2[" + str(i) + "]")

        model_data_exports_EW_region3[i] = m.getVarByName("exports_EW_region3[" + str(i) + "]")
        model_data_imports_EW_region3[i] = m.getVarByName("imports_EW_region3[" + str(i) + "]")
        model_data_exports_WE_region3[i] = m.getVarByName("exports_WE_region3[" + str(i) + "]")
        model_data_imports_WE_region3[i] = m.getVarByName("imports_WE_region3[" + str(i) + "]")

        model_data_exports_EW_region4[i] = m.getVarByName("exports_EW_region4[" + str(i) + "]")
        model_data_imports_EW_region4[i] = m.getVarByName("imports_EW_region4[" + str(i) + "]")
        model_data_exports_WE_region4[i] = m.getVarByName("exports_WE_region4[" + str(i) + "]")
        model_data_imports_WE_region4[i] = m.getVarByName("imports_WE_region4[" + str(i) + "]")

        model_data_netload_region1[i]    = m.getVarByName("netload_region1[" + str(i) + "]")
        model_data_netload_region2[i]    = m.getVarByName("netload_region2[" + str(i) + "]")
        model_data_netload_region3[i]    = m.getVarByName("netload_region3[" + str(i) + "]")
        model_data_netload_region4[i]    = m.getVarByName("netload_region4[" + str(i) + "]")

        hq_import_region1[i]             = m.getVarByName("hq_import_region1[" + str(i) + "]")
        hq_import_region3[i]             = m.getVarByName("hq_import_region3[" + str(i) + "]")

    m.update()

    # Set transmission equal across lines
    for i in trange:
        ## Zone 1-2
        m.addConstr(model_data_exports_WE_region1[i] - model_data_imports_WE_region2[i] == 0)
        m.addConstr(model_data_imports_EW_region1[i] - model_data_exports_EW_region2[i] == 0)

        ## Zone 2-3
        m.addConstr(model_data_exports_WE_region2[i] - model_data_imports_WE_region3[i] == 0)
        m.addConstr(model_data_imports_EW_region2[i] - model_data_exports_EW_region3[i] == 0)

        # Zone 3-4
        m.addConstr(model_data_exports_WE_region3[i] - model_data_imports_WE_region4[i] == 0)
        m.addConstr(model_data_imports_EW_region3[i] - model_data_exports_EW_region4[i] == 0)

    # Offshore generation constraints
    for i in range(args.num_regions):
        model_data_offshore_cap[i] = m.getVarByName("offshore_cap_region" + str(i + 1))

    m.addConstr((model_data_offshore_cap[0] + model_data_offshore_cap[1] +
                 model_data_offshore_cap[2] + model_data_offshore_cap[3]) <= 37572)

    # Low-carbon electricity constraint
    full_netload_sum_mwh = quicksum(model_data_netload_region1[i] + model_data_netload_region2[i] +
                                    model_data_netload_region3[i] + model_data_netload_region4[i] for i in trange)
    full_demand_sum_mwh  = (np.sum(baseline_demand[0:T]) + np.sum(heating_load_mw[0:T]) + np.sum(ev_load_mw) * T)
    full_imports_sum_mwh = quicksum(hq_import_region1[i] + hq_import_region3[i] for i in trange)
    full_nuclear_sum_mwh = np.sum(nuc_gen_mw) * T

    frac_netload = 1 - lowc_target

    m.addConstr(full_netload_sum_mwh  - ((full_demand_sum_mwh - full_imports_sum_mwh) * frac_netload -
                full_nuclear_sum_mwh) <= 0)



    m.update()

    return m
