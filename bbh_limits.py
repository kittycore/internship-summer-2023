import matplotlib.pyplot as plt

import argparse

from common import *
import plot, preprocess, sample


def add_arguments(parser: argparse.ArgumentParser) -> None:
    '''Adds command-line arguments relevant to this module to an
    argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add
            the relevant command-line arguments to.
    '''

    parser.add_argument('-c',
        choices = [*JET_CASES, 'isotropic', 'uniform', 'fixed', 'all'],
        default = 'all',
        help = 'Which case of opening angle to plot. Defaults to "all".'
    )

    parser.add_argument('--hidden', action = 'store_true',
        help = 'Do not show the plots when sampling is complete.')

    group = parser.add_argument_group('sampler',
        description = 'Arguments relevant to the sampler.')
    sample.add_arguments(parser, group)


def main() -> None:
    '''Main entrypoint.'''

    parser = argparse.ArgumentParser(prog = 'limits',
        description = 'Estimates the gamma-ray upper limits of the binary ' \
            'black hole population.')
    add_arguments(parser)

    arguments = parser.parse_args()
    args = vars(arguments) # Shorthand for easier access!

    # If the `--preprocessor` flag is set, pass along to the preprocessor.
    if args['preprocessor']:
        preprocess.main(arguments)
        return

    model = args['m']
    case = args['c']

    models = JET_MODELS if model == 'all' else [model]
    cases = JET_CASES if case == 'all' else [case[0]]

    collector = sample.sample(arguments)

    for model in models:
        name = JET_MODELS_EXPANDED[model]
        print(f'Plotting model {name} ({model})...')

        for case in cases:
            print(f'Plotting case {JET_CASES_EXPANDED[case]}...')
            plot.plot(collector[model], model, case)
            sample.compute(collector[model], model, case)

        print(f'Finished plotting model {name} ({model}).')
        print()

    if not args['hidden']:
        plt.show()


if __name__ == '__main__':
    main()
