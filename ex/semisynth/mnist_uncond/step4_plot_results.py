"""Thin CLI for step4_plot_results on mnist_uncond.

Loads mae_summary.h5 and plots alpha vs MAE with error bands.
Delegates to experiments.utils.results.plot_results_main(config_path).
"""
from experiments.utils.results import plot_results_main


CONFIG_PATH = 'experiments/mnist_uncond/config.yaml'


if __name__ == '__main__':
    plot_results_main(CONFIG_PATH)
