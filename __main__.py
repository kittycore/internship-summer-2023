import matplotlib.pyplot as plt
import numpy as np

import argparse

import preprocess
from preprocess import Event


# The different binary black hole merger models.
MODELS = ['QQ', 'NU', 'BZ', 'GW']

# More descriptive names for each model.
MODELS_EXPANDED = {
    'QQ': 'Charged Black Hole',
    'NU': 'Neutrino-Antineutrino Annihilation',
    'BZ': 'Blandford-Znajek',
    'GW': 'Gravitational Wave Energy Conversion',
}

# Default seed for the random number generator.
DEFAULT_SEED = 0x9B7DB742C51C67FF

# A random number generator accessible anywhere in this module.
random: np.random.Generator


class EventSample(np.ndarray):
    DTYPE = [
        ('predicted_QQ', np.float64),
        ('predicted_NU', np.float64),
        ('predicted_BZ', np.float64),
        ('predicted_GW', np.float64),
        ('visible_fixed', np.bool_),
        ('visible_uniform', np.bool_),
        ('detectable_QQ', np.bool_),
        ('detectable_NU', np.bool_),
        ('detectable_BZ', np.bool_),
        ('detectable_GW', np.bool_),
    ]


    def __new__(cls, size: int):
        shape = (size)
        return super().__new__(cls, shape, dtype = EventSample.DTYPE)


def process(events: dict[str, Event]) -> EventSample:
    sample = EventSample(len(events))

    # Collect all of the events into a single array.
    collector = np.concatenate([*events.values()], axis = -1)

    for model in MODELS:
        choices = random.choice(collector, size = sample.size, shuffle = False)
        sample[f'predicted_{model}'] = choices[f'flux_{model}']

    return sample


def plot(sample: EventSample) -> None:
    figure = plt.figure(figsize = [12, 12])

    for index, model in enumerate(MODELS):
        axes = figure.add_subplot(2, 2, index + 1)

        fluxes = sample[f'predicted_{model}']

        maximum = np.max(fluxes)
        minimum = np.min(fluxes)
        bins = np.logspace(np.log10(minimum), np.log10(maximum), 20)

        axes.hist(fluxes, bins, color = '#bcefb7')

        axes.set_title(f'{MODELS_EXPANDED[model]} ({model})')
        axes.set_xlabel('Flux (erg s⁻¹ cm⁻²)')
        axes.set_ylabel('Number')
        axes.set_xscale('log')

    figure.suptitle(f'Population Sample (Size: {sample.size})', fontsize = 14)
    figure.tight_layout(rect = (0, 0.03, 1, 0.975)) # type: ignore


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-p', action = 'store_true',
        help = 'Exclusively run the preprocessor.')

    group = parser.add_argument_group('preprocessor',
        description = 'Arguments relevant to the preprocessor.')
    preprocess.add_arguments(parser, group)


def main() -> None:
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

    # Seed the random number generator.
    global random
    random = np.random.default_rng(DEFAULT_SEED)

    events = preprocess.preprocess(arguments)
    sample = process(events)
    plot(sample)

    plt.show()


main()
