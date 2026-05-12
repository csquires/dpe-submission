"""wrapper: delegate to old sibling's diagnostic_mem_profile with new config path."""
import sys
from ex.semisynth.mnist_uncond.diagnostic_mem_profile import main

if __name__ == '__main__':
    sys.argv = [
        sys.argv[0],
        '--config', 'ex/semisynth/mnist/config.yaml',
    ] + sys.argv[1:]
    main()
