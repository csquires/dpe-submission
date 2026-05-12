"""thin cli for step3_process_results on mnist_uncond."""
from ex.utils.results import process_results_main

CONFIG_PATH = 'ex/semisynth/mnist_uncond/config.yaml'

if __name__ == '__main__':
    process_results_main(CONFIG_PATH)
