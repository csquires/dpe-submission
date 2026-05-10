"""thin cli for step3_process_results on mnist_eldr."""
from experiments.utils.results import process_results_main

CONFIG_PATH = 'experiments/mnist/config.yaml'

if __name__ == '__main__':
    process_results_main(CONFIG_PATH)
