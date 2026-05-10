"""wrapper: delegate to old sibling's diagnostic_mem_profile with new config path."""
import sys
from experiments.mnist_uncond.diagnostic_mem_profile import main

if __name__ == '__main__':
    sys.argv = [
        sys.argv[0],
        '--config', 'experiments/mnist/config.yaml',
    ] + sys.argv[1:]
    main()
