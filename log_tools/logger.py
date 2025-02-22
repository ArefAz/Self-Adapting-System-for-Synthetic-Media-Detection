from datetime import datetime
import os


def initialize_logger(base_log_dir):
    current_date = datetime.now()
    date_format = "%m-%d_%H:%M"
    log_dir = os.path.join(base_log_dir, current_date.strftime(date_format))
    os.makedirs(log_dir, exist_ok=True)

    visualizations_dir = os.path.join(log_dir, "visuals")
    os.makedirs(visualizations_dir, exist_ok=True)

    results_dir = os.path.join(log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    run_dir = os.path.join(log_dir, "runs")
    os.makedirs(run_dir, exist_ok=True)

    return log_dir