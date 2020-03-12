import numpy as np
from model import create_model
from utils import *


if __name__ == '__main__':
    args    = get_args()
    raw_export_columns, ts_export_columns = get_raw_columns()


    # Define model config and set of heating, EV loads, and/or GHG reduction target appropriately
    model_config = 0

    # 0: LCT + Elec. specified, GHG returned
    if model_config == 0:
        perc_elec_load = [0.3] #[[0]0.389, 0.389, 0.550]
        lowc_targets   = [0.7] #, 1, 1]
        ghg_targets    = [np.nan]*len(perc_elec_load) # indeterminate

    # 1: LCT + GHG specified, Elec. returned
    elif model_config == 1:
        lowc_targets = [0.7, 0.7]
        ghg_targets  = [0.4, 0.45]
        perc_elec_load = [np.nan]*len(lowc_targets) # indeterminate

    # 2: Elec. + GHG specified, LCT returned.
    else: # model_config  == 2:
        ghg_targets    = [0.4, 0.45]
        perc_elec_load = [.389, .389]
        lowc_targets   = [np.nan]*len(ghg_targets) # indeterminate


    # Establish lists to store results
    results    = []
    results_ts = []

    for i in range(len(perc_elec_load)):
        # Initialize scenario parameters
        elec = perc_elec_load[i]
        lct  = lowc_targets[i]
        ghg  = ghg_targets[i]

        # Create the model
        m = create_model(args, model_config, elec, lct,  ghg)

        # Set model solver parameters
        m.setParam("FeasibilityTol", args.feasibility_tol)
        m.setParam("Method", 2)
        m.setParam("BarConvTol", 0)
        m.setParam("BarOrder", 0)
        m.setParam("Crossover", 0)

        # Solve the model
        m.optimize()

        # Retrieve the model solution
        allvars = m.getVars()

        # Process the model solution
        tx_tuple_list = get_tx_tuples(args)
        single_scen_results, single_scen_results_ts = raw_results_retrieval(args, model_config, m, tx_tuple_list)

        # Append single set of results to full results lists
        results.append(single_scen_results)
        results_ts.append(single_scen_results_ts)

    ## Save raw results
    df_results_raw = pd.DataFrame(np.array(results), columns=raw_export_columns)
    df_results_raw.to_excel(os.path.join(args.results_dir, 'raw_results_export.xlsx'))


    ts_writer = pd.ExcelWriter(os.path.join(args.results_dir, 'ts_results_export.xlsx'), engine= 'xlsxwriter')
    for i in range(len(results_ts)):
        df_results_ts = pd.DataFrame(np.array(results_ts[i]), columns=ts_export_columns)
        df_results_ts.to_excel(ts_writer, sheet_name = 'Sheet{}'.format(i+1))
    ts_writer.save()

    ## Save processed results
    df_results_processed = full_results_processing(args, np.array(results), np.array(results_ts))
    df_results_processed.to_excel(os.path.join(args.results_dir, 'processed_results_export.xlsx'))
