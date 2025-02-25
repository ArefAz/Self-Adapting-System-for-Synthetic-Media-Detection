from datetime import datetime
import os
import pandas as pd
import pickle


def initialize_logger(base_log_dir):
    current_date = datetime.now()
    date_format = "%m-%d_%H:%M"
    log_dir = os.path.join(base_log_dir, current_date.strftime(date_format))
    os.makedirs(log_dir, exist_ok=True)

    visualizations_dir = os.path.join(log_dir, "visuals")
    os.makedirs(visualizations_dir, exist_ok=True)

    results_dir = os.path.join(log_dir, "states")
    os.makedirs(results_dir, exist_ok=True)

    run_dir = os.path.join(log_dir, "runs")
    os.makedirs(run_dir, exist_ok=True)

    return log_dir

def save_results(results_dict, log_dir):
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(log_dir, "results.csv"), index_label="Metric")


def save_state(open_set, ae, results_dict, emerging_source, emerging_label, log_dir):
    state = {
        "open_set": open_set,
        "ae": ae,
        "results": results_dict,
    }

    state_path = os.path.join(log_dir, "states", f"{emerging_label}_{emerging_source}")
    os.makedirs(state_path, exist_ok=True)
    with open(os.path.join(state_path, "state.pkl"), "wb") as f:
        pickle.dump(state, f)

def read_state(emerging_source, emerging_label, log_dir):
    state_path = os.path.join(log_dir, "states", f"{emerging_label}_{emerging_source}")
    with open(os.path.join(state_path, "state.pkl"), "rb") as f:
        state = pickle.load(f)
    return state

