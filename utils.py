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
        description = 'Module for creating productive demand time series using irrigibility map')
    parser.add_argument('--params_filename',
                        type=str,
                        default='params.yaml',
                        help = 'Loads productive demand time series parameters')
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


def get_columns():

    # Define columns for results processing export
    columns = ['add_heating_load', 'add_ev_load', 'total_onshore', 'total_offshore', 'total_solar',
               'total_new_gt_cap','total_battery_cap', 'total_battery_power', 'total_h2_cap', 'total_h2_power',
               'total_new_trans','total_hq_import', 'onshore_1', 'onshore_2', 'offshore_3', 'offshore_4',
               'solar_1', 'solar_2', 'solar_3', 'solar_4', 'gt_cap_1', 'gt_cap_2', 'gt_cap_3', 'gt_cap_4',
               'battery_cap_1', 'battery_cap_2', 'battery_cap_3', 'battery_cap_4', 'battery_power_1',
               'battery_power_2', 'battery_power_3', 'battery_power_4', 'battery_discharge_1', 'battery_discharge_2',
               'battery_discharge_3','battery_discharge_4', 'h2_cap_1', 'h2_cap_2', 'h2_cap_3', 'h2_cap_4',
               'h2_power_1', 'h2_power_2','h2_power_3', 'h2_power_4', 'h2_discharge_1', 'h2_discharge_2',
               'h2_discharge_3', 'h2_discharge_4', 'hq_import_1', 'hq_import_2', 'hq_import_3', 'hq_import_4',
               'new_trans_12', 'new_trans_23', 'new_trans_34', 'new_trans_21', 'new_trans_32', 'new_trans_43']

    return  columns

def results_processing(args, allvars, heating_load_cap, ev_load_cap):

    ## This function takes the allvars object and reformats it into arrays
    T = args.num_years*8760

    import_export_cap_result  = np.zeros((4, args.num_regions))
    onshore_wind_cap_result   = np.zeros((args.num_regions))
    offshore_wind_cap_result  = np.zeros((args.num_regions))
    solar_cap_result          = np.zeros((args.num_regions))
    gt_cap_result             = np.zeros((args.num_regions))

    battery_cap_result        = np.zeros((args.num_regions))
    battery_power_result      = np.zeros((args.num_regions))
    h2_cap_result             = np.zeros((args.num_regions))
    h2_power_result           = np.zeros((args.num_regions))


    offshore_wind_util_result = np.zeros((T, args.num_regions))
    onshore_wind_util_result  = np.zeros((T, args.num_regions))
    solar_util_result         = np.zeros((T, args.num_regions))
    flex_hydro_result         = np.zeros((T, args.num_regions))
    batt_charge_result        = np.zeros((T, args.num_regions))
    batt_discharge_result     = np.zeros((T, args.num_regions))
    h2_charge_result          = np.zeros((T, args.num_regions))
    h2_discharge_result       = np.zeros((T, args.num_regions))

    exports_EW_result         = np.zeros((T, args.num_regions))
    imports_EW_result         = np.zeros((T, args.num_regions))
    exports_WE_result         = np.zeros((T, args.num_regions))
    imports_WE_result         = np.zeros((T, args.num_regions))

    hq_import_result          = np.zeros((T, args.num_regions))
    netload_diff_result       = np.zeros((T, args.num_regions))
    netload_abs_result        = np.zeros((T, args.num_regions))

    batt_level_result         = np.zeros((T, args.num_regions))
    h2_level_result           = np.zeros((T, args.num_regions))
    ev_charging_result        = np.zeros((T, args.num_regions))

    netload_result            = np.zeros((T, args.num_regions))

    capacity_var_offset = 12  # From import/export cap (4) & wind capacities (2) & solar cap (1) &
                              # ccgt cap (1) & battery/H2 cap (4)
    num_vars_per_region = 19 * T + capacity_var_offset # Num. of initialized time series vars. + single cap vars.

    for j in range(0, args.num_regions):
        for i in range(0, 4):
            import_export_cap_result[i, j]  = allvars[j * num_vars_per_region + i].X
        onshore_wind_cap_result[j]          = allvars[j * num_vars_per_region + 4].X
        offshore_wind_cap_result[j]         = allvars[j * num_vars_per_region + 5].X
        solar_cap_result[j]                 = allvars[j * num_vars_per_region + 6].X
        gt_cap_result[j]                    = allvars[j * num_vars_per_region + 7].X
        battery_cap_result[j]               = allvars[j * num_vars_per_region + 8].X
        battery_power_result[j]             = allvars[j * num_vars_per_region + 9].X
        h2_cap_result[j]                    = allvars[j * num_vars_per_region + 10].X
        h2_power_result[j]                  = allvars[j * num_vars_per_region + 11].X

    for j in range(0, args.num_regions):
        for i in range(0, T):
            ## Moves every time step back by an index -- hour 1 is now at hour 0
            onshore_wind_util_result[i, j]  = allvars[i + j * num_vars_per_region         + capacity_var_offset].X
            offshore_wind_util_result[i, j] = allvars[i + j * num_vars_per_region + T     + capacity_var_offset].X
            solar_util_result[i, j]         = allvars[i + j * num_vars_per_region + 2 * T + capacity_var_offset].X
            flex_hydro_result[i, j]         = allvars[i + j * num_vars_per_region + 3 * T + capacity_var_offset].X
            batt_charge_result[i, j]        = allvars[i + j * num_vars_per_region + 4 * T + capacity_var_offset].X
            batt_discharge_result[i, j]     = allvars[i + j * num_vars_per_region + 5 * T + capacity_var_offset].X
            h2_charge_result[i, j]          = allvars[i + j * num_vars_per_region + 6 * T + capacity_var_offset].X
            h2_discharge_result[i, j]       = allvars[i + j * num_vars_per_region + 7 * T + capacity_var_offset].X

            exports_EW_result[i, j]         = allvars[i + j * num_vars_per_region + 8 * T + capacity_var_offset].X
            imports_EW_result[i, j]         = allvars[i + j * num_vars_per_region + 9 * T + capacity_var_offset].X
            exports_WE_result[i, j]         = allvars[i + j * num_vars_per_region + 10 * T + capacity_var_offset].X
            imports_WE_result[i, j]         = allvars[i + j * num_vars_per_region + 11 * T + capacity_var_offset].X

            hq_import_result[i, j]          = allvars[i + j * num_vars_per_region + 12 * T + capacity_var_offset].X

            batt_level_result[i, j]         = allvars[i + j * num_vars_per_region + 13 * T + capacity_var_offset].X
            h2_level_result[i, j]           = allvars[i + j * num_vars_per_region + 14 * T + capacity_var_offset].X
            ev_charging_result[i, j]        = allvars[i + j * num_vars_per_region + 15 * T + capacity_var_offset].X

            netload_diff_result[i, j]       = allvars[i + j * num_vars_per_region + 16 * T + capacity_var_offset].X
            netload_abs_result[i, j]        = allvars[i + j * num_vars_per_region + 17 * T + capacity_var_offset].X
            netload_result[i, j]            = allvars[i + j * num_vars_per_region + 18 * T + capacity_var_offset].X


    results = np.zeros(58)
    results_ts = np.zeros((T, args.num_regions * 8))

    results[0] = heating_load_cap
    results[1] = ev_load_cap
    results[2] = np.sum(np.around(onshore_wind_cap_result))
    results[3] = np.sum(np.around(offshore_wind_cap_result))
    results[4] = np.sum(np.around(solar_cap_result))
    results[5] = np.around((np.sum(gt_cap_result) - np.sum(args.gt_current_cap_mw)/args.reserve_req) * args.reserve_req)
    results[6] = np.sum(np.around(battery_cap_result))
    results[7] = np.sum(np.around(battery_power_result))
    results[8] = np.sum(np.around(h2_cap_result))
    results[9] = np.sum(np.around(h2_power_result))
    results[10] = np.around(np.sum((import_export_cap_result[2, 0], import_export_cap_result[2, 1],
                                    import_export_cap_result[2, 2], import_export_cap_result[1, 0],
                                    import_export_cap_result[1, 1], import_export_cap_result[1, 2])) -
                            (np.sum(args.translimits_WE) + np.sum(args.translimits_EW)))
    results[11] = np.around(np.mean(hq_import_result[:, 0]) + np.mean(hq_import_result[:, 2]))
    results[12] = np.around(onshore_wind_cap_result[0])
    results[13] = np.around(onshore_wind_cap_result[1])
    results[14] = np.around(offshore_wind_cap_result[2])
    results[15] = np.around(offshore_wind_cap_result[3])
    results[16] = np.around(solar_cap_result[0])
    results[17] = np.around(solar_cap_result[1])
    results[18] = np.around(solar_cap_result[2])
    results[19] = np.around(solar_cap_result[3])
    results[20] = np.around(gt_cap_result[0] * args.reserve_req)
    results[21] = np.around(gt_cap_result[1] * args.reserve_req)
    results[22] = np.around(gt_cap_result[2] * args.reserve_req)
    results[23] = np.around(gt_cap_result[3] * args.reserve_req)
    results[24] = np.around(battery_cap_result[0])
    results[25] = np.around(battery_cap_result[1])
    results[26] = np.around(battery_cap_result[2])
    results[27] = np.around(battery_cap_result[3])
    results[28] = np.around(battery_power_result[0])
    results[29] = np.around(battery_power_result[1])
    results[30] = np.around(battery_power_result[2])
    results[31] = np.around(battery_power_result[3])
    results[32] = np.around(np.mean(batt_discharge_result[:, 0]))
    results[33] = np.around(np.mean(batt_discharge_result[:, 1]))
    results[34] = np.around(np.mean(batt_discharge_result[:, 2]))
    results[35] = np.around(np.mean(batt_discharge_result[:, 3]))
    results[36] = np.around(h2_cap_result[0])
    results[37] = np.around(h2_cap_result[1])
    results[38] = np.around(h2_cap_result[2])
    results[39] = np.around(h2_cap_result[3])
    results[40] = np.around(h2_power_result[0])
    results[41] = np.around(h2_power_result[1])
    results[42] = np.around(h2_power_result[2])
    results[43] = np.around(h2_power_result[3])
    results[44] = np.around(np.mean(h2_discharge_result[:, 0]))
    results[45] = np.around(np.mean(h2_discharge_result[:, 1]))
    results[46] = np.around(np.mean(h2_discharge_result[:, 2]))
    results[47] = np.around(np.mean(h2_discharge_result[:, 3]))
    results[48] = np.around(np.mean(hq_import_result[:, 0]))
    results[49] = np.around(np.mean(hq_import_result[:, 1]))
    results[50] = np.around(np.mean(hq_import_result[:, 2]))
    results[51] = np.around(np.mean(hq_import_result[:, 3]))
    results[52] = np.around(import_export_cap_result[2, 0])
    results[53] = np.around(import_export_cap_result[2, 1])
    results[54] = np.around(import_export_cap_result[2, 2])
    results[55] = np.around(import_export_cap_result[1, 0])
    results[56] = np.around(import_export_cap_result[1, 1])
    results[57] = np.around(import_export_cap_result[1, 2])

    results_ts[:, 0:args.num_regions] = netload_result
    results_ts[:, args.num_regions    : args.num_regions * 2] = batt_level_result
    results_ts[:, args.num_regions * 2: args.num_regions * 3] = h2_level_result
    results_ts[:, args.num_regions * 3: args.num_regions * 4] = hq_import_result
    results_ts[:, args.num_regions * 4: args.num_regions * 5] = flex_hydro_result
    results_ts[:, args.num_regions * 5: args.num_regions * 6] = ev_charging_result
    results_ts[:, args.num_regions * 6: args.num_regions * 7] = exports_WE_result
    results_ts[:, args.num_regions * 7: args.num_regions * 8] = exports_EW_result

    return results, results_ts



