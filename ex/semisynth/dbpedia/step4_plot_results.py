"""thin cli for step4_plot_results on dbpedia_eldr."""
from ex.utils.results import plot_results_main

CONFIG_PATH = 'ex/semisynth/dbpedia/config.yaml'

if __name__ == '__main__':
    plot_results_main(CONFIG_PATH)
