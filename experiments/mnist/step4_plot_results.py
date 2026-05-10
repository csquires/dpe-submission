"""thin cli for step4_plot_results on mnist_eldr."""
from experiments.utils.results import plot_results_main

CONFIG_PATH = 'experiments/mnist/config.yaml'

if __name__ == '__main__':
    plot_results_main(CONFIG_PATH)
