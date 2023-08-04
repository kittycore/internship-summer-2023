import matplotlib.pyplot as plt
import numpy as np

import argparse

from common import *
import plot, preprocess, sample


# Shorthand for the models of the relativistic jets sometimes produced
# during binary black hole mergers.
MODELS = ['QQ', 'NU', 'BZ', 'GW']

# Cases of opening angle for the relativistic jets sometimes produced
# during binary black hole mergers.
CASES = ['i', 'u', 'f']

# Default number of realisations.
DEFAULT_REALISATIONS = 1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    '''Adds command-line arguments relevant to this module to an
    argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add
            the relevant command-line arguments to.
    '''

    parser.add_argument('-c',
        choices = [*CASES, 'isotropic', 'uniform', 'fixed', 'all'],
        default = 'all', help = 'Which case to plot.')
    parser.add_argument('-m', choices = [*MODELS, 'all'], default = 'all',
        help = 'Which model to plot.')

    parser.add_argument('-p', action = 'store_true',
        help = 'Exclusively run the preprocessor.')
    parser.add_argument('-r', type = int, default = DEFAULT_REALISATIONS,
        help = 'The number of realisations to process.')
    parser.add_argument('-s', type = int, default = sample.DEFAULT_SEED,
        help = 'The seed used for the random number generator.')

    parser.add_argument('--silent', action = 'store_true',
        help = 'Do not show the plots when sampling is complete.')

    group = parser.add_argument_group('preprocessor',
        description = 'Arguments relevant to the preprocessor.')
    preprocess.add_arguments(parser, group)


def main() -> None:
    '''Main entrypoint.'''

    parser = argparse.ArgumentParser(prog = 'limits',
        description = 'Estimates the upper limits of the binary black hole ' \
            'population visible with Fermi-GBM.')
    add_arguments(parser)

    arguments = parser.parse_args()
    args = vars(arguments) # Shorthand for easier access!

    # If the `-p` flag is set, pass along to the preprocessor.
    if args['p']:
        preprocess.main(arguments)
        return

    model = args['m']
    case = args['c']
    realisations = args['r']

    events = preprocess.preprocess(arguments)

    # If 'all' is passed for the model or the case, substitute the
    # respective master list.
    models = MODELS if model == 'all' else [model]
    cases = CASES if case == 'all' else [case[0]]

    for m in models:
        name = MODELS_EXPANDED[m]
        print(f'Processing model {name} ({m})...')

        # Seed the random number generator.
        sample.random = np.random.default_rng(args['s'])

        samples = sample.process(events, m, realisations)

        for c in cases:
            print(f'Plotting case {CASES_EXPANDED[c]}...')
            plot.plot(samples, m, c)

            sample.compute(samples, m, c)

        print(f'Finished processing model {name} ({m}).')
        print()

    if not args['silent']:
        plt.show()


main()
