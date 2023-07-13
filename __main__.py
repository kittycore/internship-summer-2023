import matplotlib.pyplot as plt
import numpy as np

import argparse

import preprocess
from preprocess import Event


# The opening angle for the 'fixed' case, in radians.
FIXED_ANGLE = np.deg2rad(20)

# A right angle, in radians.
RIGHT_ANGLE = np.pi

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


def is_visible(
    inclination: np.ndarray,
    opening_angle: np.ndarray | None = None
) -> np.ndarray:
    # Wrap inclination angles outside the domain [0, np.pi].
    needs_wrapping = inclination > RIGHT_ANGLE
    wrapped = inclination - (2 * RIGHT_ANGLE)
    inclination = np.where(needs_wrapping, wrapped, inclination)

    angle = FIXED_ANGLE if opening_angle is None else opening_angle

    within_maximum = inclination <  angle
    within_minimum = inclination > -angle
    return within_maximum & within_minimum


def process(events: dict[str, Event]) -> EventSample:
    sample_size = len(events)
    sample = EventSample(sample_size)

    # Collect all of the events into a single array.
    collector = np.concatenate([*events.values()], axis = -1)

    choices = random.choice(collector, size = sample_size, shuffle = False)
    for model in MODELS:
        sample[f'predicted_{model}'] = choices[f'flux_{model}']

    sample['visible_fixed'] = is_visible(choices['inclination'])
    sample['visible_uniform'] = is_visible(
        choices['inclination'], choices['opening_angle'])

    return sample


def plot(sample: EventSample) -> None:
    figure = plt.figure(figsize = [12, 12])

    for index, model in enumerate(MODELS):
        axes = figure.add_subplot(2, 2, index + 1)

        fluxes = sample[f'predicted_{model}']
        uniform = fluxes[sample['visible_uniform']]
        fixed = fluxes[sample['visible_fixed']]

        percent_uniform = uniform.size / fluxes.size * 100
        percent_fixed = fixed.size / fluxes.size * 100

        maximum = np.max(fluxes)
        minimum = np.min(fluxes)
        bins = np.logspace(np.log10(minimum), np.log10(maximum), 20)

        axes.hist(fluxes, bins, color = '#bcefb7', label = 'Isotropic')
        axes.hist(uniform, bins, color = '#a9a9a9',
            label = f'Uniform ({uniform.size}, {percent_uniform:.0f}%)')
        axes.hist(fixed, bins, color = '#c9c9c9',
            label = f'Fixed ({fixed.size}, {percent_fixed:.0f}%)')

        axes.set_title(f'{MODELS_EXPANDED[model]} ({model})')
        axes.set_xlabel('Flux (erg s⁻¹ cm⁻²)')
        axes.set_ylabel('Number')
        axes.set_xscale('log')
        axes.legend()

    figure.suptitle(f'Population Sample (Size: {sample.size})', fontsize = 14)
    figure.tight_layout(rect = (0, 0.03, 1, 0.975)) # type: ignore


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-p', action = 'store_true',
        help = 'Exclusively run the preprocessor.')
    parser.add_argument('-s', type = int, default = DEFAULT_SEED,
        help = 'The seed used for the random number generator.')

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
    random = np.random.default_rng(args['s'])

    events = preprocess.preprocess(arguments)
    sample = process(events)
    plot(sample)

    plt.show()


main()
