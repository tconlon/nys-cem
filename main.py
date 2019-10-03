from model import create_model
from utils import *

if __name__ == '__main__':
    args    = get_args()
    export_columns = get_columns()

    ## Establish model run parameters
    lowc_targets = [0.7]
    nuclear_boolean = False
    h2_boolean = False

    # Define set of heating and EV loads for the model runs
    heating_loads = [0, 2000, 4000, 6000]
    ev_loads      = [0, 2000, 4000, 6000]

    # Can be either len(lowc_target) or len(loads) depending on the tests you want to run
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
        single_scen_results, single_scen_results_ts =  results_processing(args, allvars, heating_cap_mw, ev_cap_mw)

        ## Append single set of results to full results lists
        results.append(single_scen_results)
        results_ts.append(single_scen_results_ts)

    ## Save full set of results
    df_results = pd.DataFrame(np.array(results), columns=export_columns)
    df_results.to_excel(os.path.join(args.results_dir, 'full_results_export.xlsx'))
    np.save(os.path.join(args.results_dir,'full_results_ts.npy'), np.array(results_ts))