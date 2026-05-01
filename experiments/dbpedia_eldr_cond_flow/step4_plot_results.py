"""wrapper: delegate to old sibling's step4 with new config path."""
import sys
from experiments.mnist_eldr_estimation.step4_plot_results import main

if __name__ == '__main__':
    sys.argv = [
        sys.argv[0],
        '--config', 'experiments/dbpedia_eldr_cond_flow/config.yaml',
    ] + sys.argv[1:]
    main()
