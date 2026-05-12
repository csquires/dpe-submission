"""Thin CLI for step4_plot_results on mnist_uncond.

Loads mae_summary.h5 and plots alpha vs MAE with error bands.
Delegates to ex.utils.results.plot_results_main(config_path).
"""
from ex.utils.results import plot_results_main


CONFIG_PATH = 'ex/semisynth/mnist_uncond/config.yaml'


if __name__ == '__main__':
    plot_results_main(CONFIG_PATH)
