"""thin cli for step3_process_results on mnist_eldr.

reads the step2_runner gathered results_all_cells.h5 (the per-pair raw files the
legacy process_results_main expected are no longer emitted by the runner).
"""
from ex.utils.results import process_results_gathered

CONFIG_PATH = 'ex/semisynth/mnist/config.yaml'

if __name__ == '__main__':
    process_results_gathered(CONFIG_PATH)
