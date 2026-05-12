"""thin cli for step3_process_results on dbpedia_eldr."""
from ex.utils.results import process_results_main

CONFIG_PATH = 'ex/semisynth/dbpedia/config.yaml'

if __name__ == '__main__':
    process_results_main(CONFIG_PATH)
