from model import create_model
from utils import *

if __name__ == '__main__':
    args    = get_args()
    raw_export_columns = get_raw_columns()

    ## Establish model run parameters
    nuclear_boolean = False
    h2_boolean = False

    # Define set of heating and EV loads for the model runs
    lowc_targets  = [0.7]
    heating_loads = [0] # Max is 8620 MW
    ev_loads      = [0] # Max is 6660 MW

    # Define number of tests to run, i.e. the length of the above lists
    num_test = len(lowc_targets)

    # Establish lists to store results
    results    = []
    results_ts = []

    for i in range(num_test):
        # Initialize scenario parameters
        heating_cap_mw  = heating_loads[i]
        ev_cap_mw       = ev_loads[i]
        lowc_target     = lowc_targets[i]

        # Create the model
        m = create_model(args, lowc_target, nuclear_boolean, h2_boolean, heating_cap_mw, ev_cap_mw)

        # Set model solver parameters
        m.setParam("FeasibilityTol", args.feasibility_tol)
        m.setParam("Method", args.solver_method)
        # m.setParam("Threads", 4)

        # Solve the model
        m.optimize()

        # Retrieve the model solution
        allvars = m.getVars()

        # Process the model solution
        tx_tuple_list = get_tx_tuples(args)
        single_scen_results, single_scen_results_ts = raw_results_retrieval(args, m, tx_tuple_list, heating_cap_mw,
                                                                            ev_cap_mw, lowc_target, nuclear_boolean,
                                                                            h2_boolean)

        # Append single set of results to full results lists
        results.append(single_scen_results)
        results_ts.append(single_scen_results_ts)

    ## Save raw results
    df_results_raw = pd.DataFrame(np.array(results), columns=raw_export_columns)
    df_results_raw.to_excel(os.path.join(args.results_dir, 'raw_results_export.xlsx'))
    np.save(os.path.join(args.results_dir,'raw_results_ts.npy'), np.array(results_ts))

    ## Save processed results
    df_results_processed = full_results_processing(args, np.array(results), np.array(results_ts), lowc_targets,
                                                   nuclear_boolean, h2_boolean)
    df_results_processed.to_excel(os.path.join(args.results_dir, 'processed_results_export.xlsx'))
