"""step4 plotting for mnist_eldr.

emits the per-alpha mae_vs_alpha line plot plus a pendulum-style family box plot
(base vs nested triangular variants per family, alpha encoded as box lightness).
the box plot covers every method in the gathered file, so all families are shown.
"""
import yaml

from ex.utils.results import plot_results_main, per_pair_mae_gathered
from ex.utils.family_boxplot import plot_family_boxplot

CONFIG_PATH = 'ex/semisynth/mnist/config.yaml'


def main():
    plot_results_main(CONFIG_PATH)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    per_pair, _kl, _na, _np = per_pair_mae_gathered(config, methods=None)
    plot_family_boxplot(
        per_pair, config['alphas'],
        sweep_name='alpha', ylabel='Pointwise LDR MAE',
        out_dir=config['figures_dir'], prefix='mnist_eldr_cond_flow_mae',
    )


if __name__ == '__main__':
    main()
