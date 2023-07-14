# Third-party modules.
import matplotlib.pyplot as plt
import numpy as np

# Standard library modules.
import argparse

# First-party modules.
import preprocess
from preprocess import Event # To simplify type hints.


# The opening angle for the 'fixed' case (in radians).
FIXED_ANGLE = np.deg2rad(20)
# A right angle (in radians).
RIGHT_ANGLE = np.pi / 2

# Models of the relativistic jet produced during a BBH merger.
MODELS = ['QQ', 'NU', 'BZ', 'GW']
# More descriptive names for each model.
MODELS_EXPANDED = {
    'QQ': 'Charged Black Hole',
    'NU': 'Neutrino-Antineutrino Annihilation',
    'BZ': 'Blandford-Znajek',
    'GW': 'Gravitational Wave Energy Conversion',
}

# Default seed for the random number generator. The seed can be
# overridden at runtime using the `-s` command-line option.
DEFAULT_SEED = 0x9B7DB742C51C67FF
# A random number generator accessible anywhere in this module.
random: np.random.Generator

# Default number of bins for plotted histograms.
DEFAULT_BIN_COUNT = 20
# Default number of realisations.
DEFAULT_REALISATIONS = 1


class EventSample(np.ndarray):
    DTYPE = [
        ('predicted_QQ', np.float64),
        ('predicted_NU', np.float64),
        ('predicted_BZ', np.float64),
        ('predicted_GW', np.float64),
        ('visible_f', np.bool_),
        ('visible_u', np.bool_),
        ('detectable_QQ', np.bool_),
        ('detectable_NU', np.bool_),
        ('detectable_BZ', np.bool_),
        ('detectable_GW', np.bool_),
    ]


    def __new__(cls, size: int):
        shape = (size)
        return super().__new__(cls, shape, dtype = EventSample.DTYPE)


def is_visible(
    inclinations: np.ndarray,
    opening_angles: np.ndarray | None = None
) -> np.ndarray:
    # Wrap inclination angles outside the domain [-π/2, π/2], which
    # simplifies the rest of the function.
    needs_wrapping = inclinations > RIGHT_ANGLE
    wrapped = inclinations - (2 * RIGHT_ANGLE)
    inclinations = np.where(needs_wrapping, wrapped, inclinations)

    # If no `opening_angles` are specified, use the fixed angle stored
    # in the constant `FIXED_ANGLE`. Otherwise, use `opening_angles`.
    angle = FIXED_ANGLE if opening_angles is None else opening_angles

    within_maximum = inclinations <  angle
    within_minimum = inclinations > -angle
    return within_maximum & within_minimum


def is_detectable(fluxes: np.ndarray, upper_limits: np.ndarray) -> np.ndarray:
    return fluxes >= upper_limits


def realise(events: dict[str, Event]) -> EventSample:
    sample_size = len(events)
    sample = EventSample(sample_size)

    # Collect all of the events into a single array.
    collector = np.concatenate([*events.values()], axis = -1)

    # Randomly choose fluxes from the set of events and determine which
    # of these fluxes are potentially detectable.
    choices = random.choice(collector, size = sample_size, shuffle = False)
    for model in MODELS:
        fluxes = choices[f'flux_{model}']
        sample[f'predicted_{model}'] = fluxes
        sample[f'detectable_{model}'] = is_detectable(fluxes,
                                                      choices['upper_limit'])

    # Determine the visibility of the chosen fluxes.
    inclinations = choices['inclination']
    sample['visible_f'] = is_visible(inclinations)
    sample['visible_u'] = is_visible(inclinations, choices['opening_angle'])

    return sample


def process(events: dict[str, Event], realisations: int) -> list[EventSample]:
    collector = []

    # Repeatedly sample the set of events and collect the results.
    for realisation in range(0, realisations):
        print(f'Realising {realisation + 1:4d} of {realisations:4d}...')
        sample = realise(events)
        collector.append(sample)

    return collector


def plot(samples: list[EventSample]) -> None:
    figure = plt.figure(figsize = [12, 12])

    for index, model in enumerate(MODELS):
        key = f'predicted_{model}'

        # Find the maximum and minimum across all points of every
        # sample to determine a uniform set of histogram bins.
        net_maximum = np.max(samples[0][key])
        net_minimum = np.min(samples[0][key])
        for sample in samples:
            fluxes = sample[key]

            maximum = np.max(fluxes)
            minimum = np.min(fluxes)

            if maximum > net_maximum:
                net_maximum = maximum
            if minimum < net_minimum:
                net_minimum = minimum

        bins = np.logspace(np.log10(net_minimum), np.log10(net_maximum), 20)

        histograms_i = np.empty(shape = (bins.size - 1, 0))
        histograms_u = np.copy(histograms_i)
        histograms_f = np.copy(histograms_i)
        histograms_d = np.copy(histograms_i)

        # Compute the histogram of each sample for the isotropic,
        # uniform and fixed opening angle cases, as well as the
        # histogram for detectable fluxes.
        for sample in samples:
            fluxes = sample[key]
            histogram, _ = np.histogram(fluxes, bins)
            histograms_i = np.column_stack((histograms_i, histogram))

            fluxes_u = fluxes[sample['visible_u']]
            histogram, _ = np.histogram(fluxes_u, bins)
            histograms_u = np.column_stack((histograms_u, histogram))

            fluxes_f = fluxes[sample['visible_f']]
            histogram, _ = np.histogram(fluxes_f, bins)
            histograms_f = np.column_stack((histograms_f, histogram))

            # Only include detectable fluxes that are also visible.
            visible = sample['visible_f']
            fluxes_d = fluxes[sample[f'detectable_{model}'] & visible]
            histogram, _ = np.histogram(fluxes_d, bins)
            histograms_d = np.column_stack((histograms_d, histogram))

        median_i = np.empty(shape = bins.size - 1)
        median_u = np.copy(median_i)
        median_f = np.copy(median_i)
        median_d = np.copy(median_i)
        for b in range(bins.size - 1):
            median_i[b] = np.median(histograms_i[b])
            median_u[b] = np.median(histograms_u[b])
            median_f[b] = np.median(histograms_f[b])
            median_d[b] = np.median(histograms_d[b])

        # Plot the median histograms.
        axes = figure.add_subplot(2, 2, index + 1)
        axes.stairs(median_i, bins, fill = True, color = '#bcefb7',
            label = 'Isotropic')
        axes.stairs(median_u, bins, fill = True, color = '#a9a9a9',
            label = 'Uniform')
        axes.stairs(median_f, bins, fill = True, color = '#c9c9c9',
            label = 'Fixed')
        axes.stairs(median_d, bins, fill = True, color = '#eb3a2e',
            label = 'Detectable')

        # Configure the subplot.
        axes.set_title(f'{MODELS_EXPANDED[model]} ({model})')
        axes.set_xlabel('Flux (erg s⁻¹ cm⁻²)')
        axes.set_ylabel('Number')
        axes.set_xscale('log')
        axes.legend()

    # Configure the plot.
    figure.suptitle(f'Population Sample', fontsize = 14)
    figure.tight_layout(rect = (0, 0.03, 1, 0.975)) # type: ignore


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-p', action = 'store_true',
        help = 'Exclusively run the preprocessor.')
    parser.add_argument('-r', type = int, default = DEFAULT_REALISATIONS,
        help = 'The number of realisations to process.')
    parser.add_argument('-s', type = int, default = DEFAULT_SEED,
        help = 'The seed used for the random number generator.')

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

    # Seed the random number generator.
    global random
    random = np.random.default_rng(args['s'])

    events = preprocess.preprocess(arguments)
    samples = process(events, args['r'])
    plot(samples)

    plt.savefig('population.png')
    plt.show()


main()
