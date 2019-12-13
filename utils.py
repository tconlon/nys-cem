import os
import argparse
import yaml
import numpy as np
import pandas as pd


def annualization_rate(i, years):
    return (i*(1+i)**years)/((1+i)**years-1)

def get_args():
    # Store all parameters for easy retrieval
    parser = argparse.ArgumentParser(
        description = 'nys-cem')
    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help = 'Loads model parameters')
    args = parser.parse_args()
    config = yaml.load(open(args.params_filename), Loader=yaml.FullLoader)
    for k,v in config.items():
        args.__dict__[k] = v

    return args

def load_timeseries(args, heating_cap_mw, ev_cap_mw):
    T = 8760 * args.num_years

    # Load all potential generation and actual hydro generation time-series
    onshore_pot_hourly        = np.load(os.path.join(args.data_dir, 'onshore_pot_hourly.npy'))
    offshore_pot_hourly       = np.load(os.path.join(args.data_dir, 'offshore_pot_hourly.npy'))
    solar_pot_hourly          = np.load(os.path.join(args.data_dir, 'solar_pot_hourly.npy'))
    flex_hydro_daily_mwh      = np.load(os.path.join(args.data_dir, 'flex_hydro_daily_mwh.npy'))
    fixed_hydro_hourly_mw     = np.load(os.path.join(args.data_dir, 'fixed_hydro_hourly_mw.npy'))

    # Load baseline and full heating demand time series
    baseline_demand_hourly_mw = np.load(os.path.join(args.data_dir, 'baseline_demand_hourly_mw.npy'))
    nys_heating_mw            = np.load(os.path.join(args.data_dir, 'nys_heating_mw.npy'))

    # Scale heating and EV load time series based on the amount of capacity present
    heating_load_hourly_mw  = np.multiply(nys_heating_mw,
                                          heating_cap_mw/np.sum(np.mean(nys_heating_mw[0:T], axis=0)))
    ev_avg_load_hourly_mw = np.multiply(args.ev_load_dist, ev_cap_mw)

    return baseline_demand_hourly_mw, heating_load_hourly_mw, ev_avg_load_hourly_mw,  \
           onshore_pot_hourly, offshore_pot_hourly, solar_pot_hourly, fixed_hydro_hourly_mw, flex_hydro_daily_mwh


def get_raw_columns():

    # Define columns for raw results export
    columns = ['lct', 'nuclear_binary', 'h2_binary', 'hq-ch_cap',
               'add_heating_load', 'add_ev_load', 'total_onshore', 'total_offshore', 'total_solar',
               'total_new_gt_cap','total_battery_cap', 'total_battery_power', 'total_h2_cap', 'total_h2_power',
               'total_new_trans','total_hq_import', 'onshore_1', 'onshore_2', 'offshore_3', 'offshore_4',
               'solar_1', 'solar_2', 'solar_3', 'solar_4', 'new_gt_cap_1', 'new_gt_cap_2', 'new_gt_cap_3',
               'new_gt_cap_4', 'battery_cap_1', 'battery_cap_2', 'battery_cap_3', 'battery_cap_4', 'battery_power_1',
               'battery_power_2', 'battery_power_3', 'battery_power_4', 'battery_discharge_1', 'battery_discharge_2',
               'battery_discharge_3','battery_discharge_4', 'h2_cap_1', 'h2_cap_2', 'h2_cap_3', 'h2_cap_4',
               'h2_power_1', 'h2_power_2','h2_power_3', 'h2_power_4', 'h2_discharge_1', 'h2_discharge_2',
               'h2_discharge_3', 'h2_discharge_4', 'hq_import_1', 'hq_import_2', 'hq_import_3', 'hq_import_4',
               'total_trans_12', 'total_trans_23', 'total_trans_34', 'total_trans_21', 'total_trans_32',
               'total_trans_43']

    return  columns

def get_processed_columns():

    # Define columns for processed results export

    columns = ['RGT/LCT', 'RGT Binary', 'Nuclear Binary', 'H2 Binary', 'HQ-CH Addl. Cap.', 'Heating Load', 'EV Load',
                      'Onshore [MW]', 'Offshore [MW]', 'Solar [MW]', 'New GT [MW]', 'Battery Energy [MWh]',
                      'Battery Power [MW]', 'H2 Energy [MWh]', 'H2 Power [MW]', 'New Trans. [MW]',
                      'New Trans. [GW-Mi]', 'Avg. Existing HQ Imports [MW]', 'Avg. New HQ Imports [MW]', 'Curtailment',
                      'Existing Trans. + Cap. Cost  [$/MWh]', 'Total LCOE [$/MWh]']
    return  columns

def get_tx_tuples(args):
    tx_matrix_limits = pd.read_excel(os.path.join(args.data_dir, 'transmission_matrix_limits.xlsx'),
                                     header=0, index_col=0)
    tx_matrix_costs = pd.read_excel(os.path.join(args.data_dir, 'transmission_matrix_costs.xlsx'),
                                    header=0, index_col=0)

    tx_tuple_list = []

    for i in range(len(tx_matrix_limits)):
        for j in range(len(tx_matrix_limits.columns)):
            if tx_matrix_limits.iloc[i, j] > 0:
                tx_tuple_list.append(((i + 1, j + 1), tx_matrix_limits.iloc[i, j], tx_matrix_costs.iloc[i, j]))

    return tx_tuple_list


def load_gt_ramping_costs(args, results, results_ts):
    ramping_cost_mwh = args.gt_startup_cost_mw/2

    net_load_ramping_total_cost = np.zeros(results.shape[0])
    net_load_total_cost         = np.zeros(results.shape[0])

    for i in range(results.shape[0]):
        net_load = results_ts[i, :, 0:4]

        for l in range(net_load.shape[0] - 1):
            for m in range(4):
                net_load_ramping_total_cost[i] += abs(net_load[l + 1, m] - net_load[l, m]) * ramping_cost_mwh
                net_load_total_cost[i]         += net_load[l, m] * args.netload_cost_mwh[m]

    return net_load_total_cost, net_load_ramping_total_cost


def raw_results_retrieval(args, m, tx_tuple_list, heating_load_cap, ev_load_cap,
                          lct, nuclear_boolean, h2_boolean):
    T = args.num_years * 8760

    gen_batt_capacity_results = np.zeros((8, args.num_regions))
    for i in range(args.num_regions):
        gen_batt_capacity_results[0,i] = m.getVarByName('onshore_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[1,i] = m.getVarByName('offshore_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[2,i] = m.getVarByName('solar_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[3,i] = m.getVarByName('new_gt_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[4,i] = m.getVarByName('batt_energy_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[5,i] = m.getVarByName('batt_power_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[6,i] = m.getVarByName('h2_energy_cap_region_{}'.format(i + 1)).X
        gen_batt_capacity_results[7,i] = m.getVarByName('h2_power_cap_region_{}'.format(i + 1)).X


    timeseries_results = np.zeros((15, T, args.num_regions))
    for i in range(args.num_regions):
        for j in range(T):
            # timeseries_results[0, j, i]  = m.getVarByName('onshore_wind_util_region_{}[{}]'.format(i + 1, j)).X
            # timeseries_results[1, j, i]  = m.getVarByName('offshore_wind_util_region_{}[{}]'.format(i + 1, j)).X
            # timeseries_results[2, j, i]  = m.getVarByName('solar_util_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[3, j, i]  = m.getVarByName('flex_hydro_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[4, j, i]  = m.getVarByName('batt_charge_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[5, j, i]  = m.getVarByName('batt_discharge_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[6, j, i]  = m.getVarByName('h2_charge_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[7, j, i]  = m.getVarByName('h2_discharge_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[8, j, i]  = m.getVarByName('batt_level_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[9, j, i]  = m.getVarByName('h2_level_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[10, j, i] = m.getVarByName('hq_import_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[11, j, i] = m.getVarByName('netload_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[12, j, i] = m.getVarByName('netload_diff_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[13, j, i] = m.getVarByName('netload_abs_region_{}[{}]'.format(i + 1, j)).X
            timeseries_results[14, j, i] = m.getVarByName('ev_charging_region_{}[{}]'.format(i + 1, j)).X


    # Transmission result processing
    tx_cap_base_string = 'new_export_limits_{}_{}'
    tx_ts_base_string  = 'net_exports_ts_{}_to_{}[{}]'

    tx_new_cap_results   = np.zeros((len(tx_tuple_list)))
    tx_total_cap_results = np.zeros((len(tx_tuple_list)))
    tx_ts_results        = np.zeros((T, int(len(tx_tuple_list)/2)))
    export_ts_count     = 0

    for i, txt in enumerate(tx_tuple_list):

        tx_new_cap_results[i] = m.getVarByName(tx_cap_base_string.format(txt[0][0], txt[0][1])).X
        tx_total_cap_results[i] = m.getVarByName(tx_cap_base_string.format(txt[0][0], txt[0][1])).X + txt[1]

        if txt[0][1] > txt[0][0]:
            for j in range(T):
                tx_ts_results[j, export_ts_count] = m.getVarByName(tx_ts_base_string.format(txt[0][0], txt[0][1], j)).X
            export_ts_count += 1

    ## Export raw results
    results = np.zeros(62)
    results_ts = np.zeros((T, args.num_regions * 8))

    # Model run parameters
    results[0] = lct
    results[1] = int(nuclear_boolean)
    results[2] = int(h2_boolean)
    results[3] = int(args.hq_limit_mw[2])

    # Additional load parameters
    results[4] = heating_load_cap
    results[5] = ev_load_cap

    # Total new capacities + avg. hydro import
    results[6] = np.around(np.sum(gen_batt_capacity_results[0,:])) # total_onshore
    results[7] = np.around(np.sum(gen_batt_capacity_results[1,:])) # total_offshore
    results[8] = np.around(np.sum(gen_batt_capacity_results[2,:])) # total_solar
    results[9] = np.around(np.sum(gen_batt_capacity_results[3,:] * args.reserve_req)) # total_new_gt_cap
    results[10] = np.around(np.sum(gen_batt_capacity_results[4,:])) # total_battery_cap
    results[11] = np.around(np.sum(gen_batt_capacity_results[5,:])) # total_battery_power
    results[12] = np.around(np.sum(gen_batt_capacity_results[6,:])) # total_h2_cap
    results[13] = np.around(np.sum(gen_batt_capacity_results[7,:])) # total_h2_power
    results[14] = np.around(np.sum(tx_new_cap_results)) # total_new_trans
    results[15] = np.around(np.mean(timeseries_results[10, :, 0]) + np.mean(timeseries_results[10, :, 1]) +
                            np.mean(timeseries_results[10, :, 2]) + np.mean(timeseries_results[10, :, 3]))
                    # total_hq_import

    # Wind
    results[16] = np.around(gen_batt_capacity_results[0, 0]) # onshore_1
    results[17] = np.around(gen_batt_capacity_results[0, 1]) # onshore_2
    results[18] = np.around(gen_batt_capacity_results[1, 2]) # offshore_3
    results[19] = np.around(gen_batt_capacity_results[1, 3]) # offshore_4

    # Solar
    results[20] = np.around(gen_batt_capacity_results[2, 0]) # solar_1
    results[21] = np.around(gen_batt_capacity_results[2, 1]) # solar_2
    results[22] = np.around(gen_batt_capacity_results[2, 2]) # solar_3
    results[23] = np.around(gen_batt_capacity_results[2, 3]) # solar_4

    # GT
    results[24] = np.around(gen_batt_capacity_results[3, 0] * args.reserve_req) # new_gt_cap_1
    results[25] = np.around(gen_batt_capacity_results[3, 1] * args.reserve_req) # new_gt_cap_2
    results[26] = np.around(gen_batt_capacity_results[3, 2] * args.reserve_req) # new_gt_cap_3
    results[27] = np.around(gen_batt_capacity_results[3, 3] * args.reserve_req) # new_gt_cap_4

    # Battery energy, power, average discharge
    results[28] = np.around(gen_batt_capacity_results[4, 0]) # battery_cap_1
    results[29] = np.around(gen_batt_capacity_results[4, 1]) # battery_cap_2
    results[30] = np.around(gen_batt_capacity_results[4, 2]) # battery_cap_3
    results[31] = np.around(gen_batt_capacity_results[4, 3]) # battery_cap_4
    results[32] = np.around(gen_batt_capacity_results[5, 0]) # battery_power_1
    results[33] = np.around(gen_batt_capacity_results[5, 1]) # battery_power_2
    results[34] = np.around(gen_batt_capacity_results[5, 2]) # battery_power_3
    results[35] = np.around(gen_batt_capacity_results[5, 3]) # battery_power_4
    results[36] = np.around(np.mean(timeseries_results[5, j, 0])) # battery_discharge_1
    results[37] = np.around(np.mean(timeseries_results[5, j, 1])) # battery_discharge_2
    results[38] = np.around(np.mean(timeseries_results[5, j, 2])) # battery_discharge_3
    results[39] = np.around(np.mean(timeseries_results[5, j, 3])) # battery_discharge_4

    # H2 energy, power, average discharge
    results[40] = np.around(gen_batt_capacity_results[6, 0]) # h2_cap_1
    results[41] = np.around(gen_batt_capacity_results[6, 1]) # h2_cap_2
    results[42] = np.around(gen_batt_capacity_results[6, 2]) # h2_cap_3
    results[43] = np.around(gen_batt_capacity_results[6, 3]) # h2_cap_4
    results[44] = np.around(gen_batt_capacity_results[7, 0]) # h2_power_1
    results[45] = np.around(gen_batt_capacity_results[7, 1]) # h2_power_2
    results[46] = np.around(gen_batt_capacity_results[7, 2]) # h2_power_3
    results[47] = np.around(gen_batt_capacity_results[7, 3]) # h2_power_4
    results[48] = np.around(np.mean(timeseries_results[7, j, 0])) # h2_discharge_1
    results[49] = np.around(np.mean(timeseries_results[7, j, 1])) # h2_discharge_2
    results[50] = np.around(np.mean(timeseries_results[7, j, 2])) # h2_discharge_3
    results[51] = np.around(np.mean(timeseries_results[7, j, 3])) # h2_discharge_4

    # Avg. Imports from HQ
    results[52] = np.around(np.mean(timeseries_results[10, j, 0])) # hq_import_1
    results[53] = np.around(np.mean(timeseries_results[10, j, 1])) # hq_import_2
    results[54] = np.around(np.mean(timeseries_results[10, j, 2])) # hq_import_3
    results[55] = np.around(np.mean(timeseries_results[10, j, 3])) # hq_import_4

    # Total transmission capacity: WE results presented first, EW results follow
    results[56] = np.around(tx_total_cap_results[0]) # new_trans_12
    results[57] = np.around(tx_total_cap_results[2]) # new_trans_23
    results[58] = np.around(tx_total_cap_results[4]) # new_trans_34
    results[59] = np.around(tx_total_cap_results[1]) # new_trans_21
    results[60] = np.around(tx_total_cap_results[3]) # new_trans_32
    results[61] = np.around(tx_total_cap_results[5]) # new_trans_43

    # Time series results
    results_ts[:, 0:args.num_regions] = timeseries_results[11] # net load
    results_ts[:, args.num_regions    : args.num_regions * 2] = timeseries_results[8]  # battery level
    results_ts[:, args.num_regions * 2: args.num_regions * 3] = timeseries_results[9]  # h2 level
    results_ts[:, args.num_regions * 3: args.num_regions * 4] = timeseries_results[10] # hq import
    results_ts[:, args.num_regions * 4: args.num_regions * 5] = timeseries_results[3]  # flex hydro
    results_ts[:, args.num_regions * 5: args.num_regions * 6] = timeseries_results[14] # ev charging
    results_ts[:, args.num_regions * 6: args.num_regions * 6 + 3] = tx_ts_results # WE transmission flow

    return results, results_ts

def full_results_processing(args, results, results_ts, lct, nuclear_boolean, h2_boolean):
    # Retrieve necessary model parameters
    export_columns = get_processed_columns()
    T = args.num_years * 8760
    tx_tuple_list = get_tx_tuples(args)
    cap_ann = annualization_rate(args.i_rate, args.annualize_years_cap)

    # Potential generation time-series for curtailment calcs
    baseline_demand_hourly_mw = np.load(os.path.join(args.data_dir, 'baseline_demand_hourly_mw.npy'))
    onshore_pot_hourly = np.load(os.path.join(args.data_dir, 'onshore_pot_hourly.npy'))
    offshore_pot_hourly = np.load(os.path.join(args.data_dir, 'offshore_pot_hourly.npy'))
    solar_pot_hourly = np.load(os.path.join(args.data_dir, 'solar_pot_hourly.npy'))

    # Create arrays to store costs
    total_new_cap_cost          = np.zeros(results.shape[0])
    total_new_gt_cost           = np.zeros(results.shape[0])
    total_annual_gas_cost       = np.zeros(results.shape[0])
    total_annual_ramping_cost   = np.zeros(results.shape[0])
    total_annual_import_cost    = np.zeros(results.shape[0])
    total_cost_per_mwh          = np.zeros(results.shape[0])
    total_curtailment           = np.zeros(results.shape[0])
    total_supp_cost_array       = np.zeros(results.shape[0])

    data_for_export = np.zeros((results.shape[0], len(export_columns)))

    # Find additional load scenarios
    additional_load_domain = np.zeros(results.shape[0])
    for i in range(results.shape[0]):
        additional_load_domain[i] = results[i, 4] + results[i, 5]

    # Calculate demand for all scenario runs
    avg_baseline_demand = np.sum(np.mean(baseline_demand_hourly_mw[0:T], axis=0))
    avg_total_demand = [avg_baseline_demand + i for i in additional_load_domain]

    # Find uncurtailed capacity factors
    wind_uncurtailed_cf  = np.array((np.mean(onshore_pot_hourly[0:T, 0]),  np.mean(onshore_pot_hourly[0:T, 1]),
                                     np.mean(offshore_pot_hourly[0:T, 2]), np.mean(offshore_pot_hourly[0:T, 3])))

    solar_uncurtailed_cf = np.mean(solar_pot_hourly, axis = 0)

    # Hydro, nuclear, and netload costs
    total_annual_hydro_cost   = np.sum(args.hydro_gen_mw) * args.instate_hydro_cost_mwh * 8760
    total_annual_nuclear_cost = nuclear_boolean * np.sum(args.nuc_gen_mw) * args.instate_nuc_cost_mwh * 8760

    net_load_cost, net_load_ramping_cost = load_gt_ramping_costs(args, results, results_ts)

    # Calculate existing capacity and transmission cost
    total_cap_market_cost = np.sum([args.cap_market_cost_mw_yr[k] * (args.gt_current_cap_mw[k] +
                                                               nuclear_boolean * args.nuc_gen_mw[k] +
                                                               args.hydro_gen_mw[k]) for k in range(4)])
    total_existing_trans_cost = np.sum([float(args.existing_trans_load_mwh[k]) * args.existing_trans_cost_mwh[k]
                                        for k in range(3)])
    total_annual_supp_cost = total_existing_trans_cost + total_cap_market_cost

    # Calculate costs
    for i in range(results.shape[0]):

        total_new_tx_cost = ((results[i, 56] - tx_tuple_list[0][1]) * tx_tuple_list[0][2] +
                             (results[i, 57] - tx_tuple_list[2][1]) * tx_tuple_list[2][2] +
                             (results[i, 58] - tx_tuple_list[4][1]) * tx_tuple_list[4][2] +
                             (results[i, 59] - tx_tuple_list[1][1]) * tx_tuple_list[1][2] +
                             (results[i, 60] - tx_tuple_list[3][1]) * tx_tuple_list[3][2] +
                             (results[i, 61] - tx_tuple_list[5][1]) * tx_tuple_list[5][2])

        total_new_gt_cost[i] = (results[i, 24] * float(args.gt_cost_mw[0]) +
                                results[i, 25] * float(args.gt_cost_mw[1]) +
                                results[i, 26] * float(args.gt_cost_mw[2]) +
                                results[i, 27] * float(args.gt_cost_mw[3]))

        total_new_cap_cost[i] = cap_ann * (results[i, 6] * float(args.onshore_cost_mw) +
                                           results[i, 7] * float(args.offshore_cost_mw) +
                                           results[i, 8] * float(args.solar_cost_mw) +
                                           total_new_gt_cost[i] +
                                           total_new_tx_cost +
                                      2 * (results[i, 10] * float(args.battery_cost_mwh) +
                                           results[i, 11] * float(args.battery_cost_mw) +
                                           results[i, 12] * float(args.h2_cost_mwh) +
                                           results[i, 13] * float(args.h2_cost_mw)))

        total_annual_gas_cost[i]     = (net_load_cost[i] + net_load_ramping_cost[i]) / args.num_years
        total_annual_ramping_cost[i] =  net_load_ramping_cost[i] / args.num_years
        total_annual_import_cost[i]  = (results[i, 52] * args.hq_cost_mwh[0] +
                                        results[i, 53] * args.hq_cost_mwh[1] +
                                        results[i, 54] * args.hq_cost_mwh[2] +
                                        results[i, 55] * args.hq_cost_mwh[3]) * 8760

        total_imports = results[i, 15]
        demand_for_rgt = avg_total_demand[i] - total_imports

        # Find portion of energy met by renewables
        if nuclear_boolean:  # Nuclear
            rgt = lct[i] - np.sum(args.nuc_gen_mw) / avg_total_demand[j]
        else:
            rgt = lct[i]

        wind_solar_frac = rgt - np.sum(args.hydro_gen_mw) / demand_for_rgt

        gen_uncurtailed_energy = np.round(results[i, 16] * wind_uncurtailed_cf[0] +
                                          results[i, 17] * wind_uncurtailed_cf[1] +
                                          results[i, 18] * wind_uncurtailed_cf[2] +
                                          results[i, 19] * wind_uncurtailed_cf[3] +
                                          results[i, 20] * solar_uncurtailed_cf[0] +
                                          results[i, 21] * solar_uncurtailed_cf[1] +
                                          results[i, 22] * solar_uncurtailed_cf[2] +
                                          results[i, 23] * solar_uncurtailed_cf[3])

        # Find curtailment, supplemntary costs, and total LCOE
        total_curtailment[i] = (gen_uncurtailed_energy - (demand_for_rgt * wind_solar_frac)) / \
                                  gen_uncurtailed_energy

        total_supp_cost_array[i] = total_annual_supp_cost / (avg_total_demand[i] * 8760)

        total_cost_per_mwh[i] = (total_new_cap_cost[i] + total_annual_gas_cost[i] + total_annual_import_cost[i] +
                                 total_annual_supp_cost + total_annual_hydro_cost + total_annual_nuclear_cost) / \
                                (avg_total_demand[i] * 8760)

    ## Populate data_for_export
    data_for_export[:, 0] = np.multiply(lct, 100) # RGT/LCT
    data_for_export[:, 1] = int(args.rgt_boolean) # RGT Binary
    data_for_export[:, 2] = int(nuclear_boolean) # Nuclear Binary
    data_for_export[:, 3] = int(h2_boolean) # H2 Binary
    data_for_export[:, 4] = int(args.hq_limit_mw[2]) # HQ-CH Binary
    data_for_export[:, 5] = results[:, 4] # Heating Load
    data_for_export[:, 6] = results[:, 5] # EV Load
    data_for_export[:, 7] = results[:, 6] # Onshore [MW]
    data_for_export[:, 8] = results[:, 7] # Offshore [MW]
    data_for_export[:, 9] = results[:, 8] # Solar [MW]
    data_for_export[:, 10] = results[:, 9] # New GT [MW]
    data_for_export[:, 11] = results[:, 10] # Battery Energy [MWh]
    data_for_export[:, 12] = results[:, 11] # Battery Power [MW]
    data_for_export[:, 13] = results[:, 12] # H2 Energy [MWh]
    data_for_export[:, 14] = results[:, 13] # H2 Power [MW]
    data_for_export[:, 15] = results[:, 14] # New Trans. [MW]
    data_for_export[:, 16] = \
        np.round(((results[:, 56] + results[:, 59] - tx_tuple_list[0][1] - tx_tuple_list[1][1]) * 300 / 1000 +
                  (results[:, 57] + results[:, 60] - tx_tuple_list[2][1] - tx_tuple_list[3][1]) * 150 / 1000 +
                  (results[:, 58] + results[:, 61] - tx_tuple_list[4][1] - tx_tuple_list[5][1]) * 60 / 1000))
                # New Trans. [GW-Mi]
    data_for_export[:, 17] = results[:, 52] # Avg. Existing HQ Imports [MW]
    data_for_export[:, 18] = results[:, 54] # Avg. New HQ Imports [MW]
    data_for_export[:, 19] = total_curtailment # Curtailment
    data_for_export[:, 20] = total_supp_cost_array  # Curtailment
    data_for_export[:, 21] = total_cost_per_mwh # Total LCOE [$/MWh]

    results_df = pd.DataFrame(data_for_export, columns=export_columns)
    return results_df

