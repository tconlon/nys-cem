import numpy as np
from utils import get_args
import os


def load_timeseries(args):
    # T = 8760 * args.num_years

    onshore_pot_hourly        = np.load(os.path.join(args.data_dir, 'onshore_pot_hourly.npy'))
    offshore_pot_hourly       = np.load(os.path.join(args.data_dir, 'offshore_pot_hourly.npy'))
    solar_pot_hourly          = np.load(os.path.join(args.data_dir, 'solar_pot_hourly.npy'))
    flex_hydro_daily_mwh      = np.load(os.path.join(args.data_dir, 'flex_hydro_daily_mwh.npy'))
    fixed_hydro_hourly_mw     = np.load(os.path.join(args.data_dir, 'fixed_hydro_hourly_mw.npy'))

    baseline_demand_hourly_mw = np.load(os.path.join(args.data_dir, 'baseline_demand_hourly_mw.npy'))
    nys_heating_mw            = np.load(os.path.join(args.data_dir, 'nys_heating_mw.npy'))



    for i in range(len(onshore_pot_hourly)):
        for j in range(4):
            if onshore_pot_hourly[i,j] <= 0.01:
                onshore_pot_hourly[i,j] = 0
            if solar_pot_hourly[i,j] <= 0.01:
                solar_pot_hourly[i,j] = 0
            if offshore_pot_hourly[i,j] <= 0.01:
                offshore_pot_hourly[i,j] = 0

    offshore_pot_hourly[:,3] = np.multiply(offshore_pot_hourly[:,3], 0.97)

    print(np.mean(onshore_pot_hourly, axis=0))
    print(np.mean(offshore_pot_hourly, axis=0))
    print(np.mean(solar_pot_hourly, axis=0))

    np.save(os.path.join(args.data_dir, 'onshore_pot_hourly.npy'), onshore_pot_hourly )
    np.save(os.path.join(args.data_dir, 'offshore_pot_hourly.npy'), offshore_pot_hourly )
    np.save(os.path.join(args.data_dir, 'solar_pot_hourly.npy'), solar_pot_hourly )

def load_results():
    results_ts = np.load('full_results_ts.npy')
    print(np.mean(results_ts[0, : , :], axis= 0))

if __name__ == '__main__':
    args = get_args()
    load_timeseries(args)
