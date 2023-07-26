import matplotlib.pyplot as plt
import numpy as np

import argparse, os

import preprocess
from preprocess import Event # To simplify type hints.

from typing import cast


# The opening angle for the 'fixed' case, in radians.
FIXED_ANGLE = np.deg2rad(20)
# A right angle, in radians.
RIGHT_ANGLE = np.pi / 2

# Models for the relativistic jets sometimes produced during binary
# black hole mergers.
MODELS = ['QQ', 'NU', 'BZ', 'GW']
# More descriptive names for each of the above models.
MODELS_EXPANDED = {
    'QQ': 'Charged Black Hole',
    'NU': 'Neutrino-Antineutrino Annihilation',
    'BZ': 'Blandford-Znajek',
    'GW': 'Gravitational Wave Energy Conversion',
}

# Cases of opening angles for the relativistic jets sometimes produced
# during binary black hole mergers.
CASES = ['i', 'u', 'f']
# More descriptive names for each of the above cases.
CASES_EXPANDED = {
    'i': 'Isotropic',
    'u': 'Uniform',
    'f': 'Fixed',
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
# The maximum number of realisations to plot separately.
MAXIMUM_PLOTS = 5

# The upper percentile used for plotted confidence bands.
UPPER_PERCENTILE = 84
# The lower percentile used for plotted confidence bands.
LOWER_PERCENTILE = 16


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
    '''Determines the visibility of a set of events using the
    inclination and the opening angle of the relativistic jet
    associated with each event.

    Args:
        inclinations (np.ndarray): An array containing inclinations for
            the relativistic jets associated with a set of events.
        opening_angles (np.ndarray | None, optional): An array
            containing opening angles for the relativistic jets
            associated with a set of events. If None, then the angle
            specified by the constant `FIXED_ANGLE` is used instead.
            Defaults to None.

    Returns:
        np.ndarray: An array of booleans indicating which events are
            visible or not from Earth.
    '''

    # Wrap inclination angles outside the domain [-π/2, π/2] to
    # simplify the rest of the check.
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
    '''Determines which events from a set of events are detectable by
    Fermi-GBM using the the predicted fluxes of the relativistic jet
    associated with each event and the upper limits of Fermi-GBM.

    Args:
        fluxes (np.ndarray): An array containing the predicted fluxes
            of the relativistic jets associated with a set of events.
        upper_limits (np.ndarray): An array containing the upper limits
            of Fermi-GBM for each event.

    Returns:
        np.ndarray: An array of booleans indicating which events are
            detectable or not by Fermi-GBM.
    '''

    # Fermi-GBM's view of some events will be blocked by the Earth.
    # These are assigned a negative value by HealPy, so replace them
    # with a large positive value to simplify the check.
    blocked = upper_limits < 0
    upper_limits = np.where(blocked, np.finfo(np.float64).max, upper_limits)

    return fluxes >= upper_limits


def realise(events: dict[str, Event], model: str) -> EventSample:
    '''Samples a set of events for a given `model`.

    Args:
        events (dict[str, Event]): A set of events to sample.
        model (str): Which model of relativistic jet to sample.

    Returns:
        EventSample: A realisation of the set of events.
    '''

    sample_size = len(events)
    sample = EventSample(sample_size)

    # Collect all of the events into a single array.
    collector = np.concatenate([*events.values()], axis = -1)

    # Randomly choose fluxes from the set of events and determine which
    # of these fluxes are potentially detectable.
    choices = random.choice(collector, size = sample_size, shuffle = False)

    fluxes = choices[f'flux_{model}']
    sample[f'predicted_{model}'] = fluxes
    sample[f'detectable_{model}'] = is_detectable(fluxes,
                                                  choices['upper_limit'])

    # Determine the visibility of the chosen fluxes.
    inclinations = choices['inclination']
    sample['visible_f'] = is_visible(inclinations)
    sample['visible_u'] = is_visible(inclinations, choices['opening_angle'])

    return sample


def process(
    events: dict[str, Event], model: str, realisations: int
) -> list[EventSample]:
    '''Processes a set of events for a given `model` of relativistic
    jet, returning a set of samples of number `realisations`.

    Args:
        events (dict[str, Event]): A set of events to sample.
        model (str): Which model of relativistic jet to sample.
        realisations (int): The number of samples to produce.

    Returns:
        list[EventSample]: A set of samples of number `realisations`.
    '''

    collector = []

    # Repeatedly sample the set of events and collect the results.
    digits = int(np.log10(realisations)) + 1
    status = f'Realising {{0:{digits}d}} of {{1:{digits}d}}...'
    for realisation in range(1, realisations + 1):
        print(status.format(realisation, realisation))
        sample = realise(events, model)
        collector.append(sample)

    return collector


def plot_single(sample: EventSample, model: str, case: str) -> plt.Figure:
    '''Plots a histogram of a single `sample` for a given `model` of
    relativistic jet and `case` of opening angle.

    Args:
        sample (EventSample): The sample to plot.
        model (str): Which model of relativistic jet to plot.
        case (str): Which case of opening angle to plot; either
            isotropic, uniform or fixed.

    Returns:
        plt.Figure: The plotted figure.
    '''

    figure = cast(plt.Figure, plt.figure())

    anisotropic = case[0] != 'i'
    key_predicted = f'predicted_{model}'
    key_visible = f'visible_{case[0]}'

    # Pick out the predicted fluxes, visible fluxes (if applicable),
    # and detectable fluxes.
    fluxes = sample[key_predicted]
    visible = fluxes[sample[key_visible]] if anisotropic else None
    detected = None
    if anisotropic:
        detected = fluxes[sample[f'detectable_{model}'] & sample[key_visible]]
    else:
        detected = fluxes[sample[f'detectable_{model}']]

    # Determine the histogram binning based upon the minimum and
    # maximum fluxes within log-space.
    maximum = np.max(fluxes)
    minimum = np.min(fluxes)
    bins = np.logspace(np.log10(minimum), np.log10(maximum), DEFAULT_BIN_COUNT)

    axes = cast(plt.Axes, figure.subplots())
    axes.hist(fluxes, bins, color = '#bcefb7', label = 'Isotropic')

    # If `case` is not `isotropic`, plot the visible fluxes for `case`.
    if anisotropic:
        axes.hist(visible, bins, color = '#a9a9a9',
            label = f'Visible ({CASES_EXPANDED[case[0]]})')

    axes.hist(detected, bins, color = '#eb3a2e', label = 'Detectable')

    # Configure the subplot.
    axes.set_title(f'{MODELS_EXPANDED[model]} ({model})')
    axes.set_xlabel('Flux (erg s⁻¹ cm⁻²)')
    axes.set_ylabel('Number')
    axes.set_xscale('log') # type: ignore
    axes.legend()

    # Configure the plot.
    figure.suptitle('Population Sample', fontsize = 14)
    figure.tight_layout(rect = (0, 0.03, 1, 0.975)) # type: ignore

    return figure


def plot_median(samples: list[EventSample], model: str, case: str) -> plt.Figure:
    '''Plots a histogram of the median of `samples` for a given `model`
    of relativistic jet and `case` of opening angle.

    Args:
        samples (list[EventSample]): A list of samples to determine the
            median of.
        model (str): Which model of relativistic jet to plot.
        case (str): Which case of opening angle to plot; either
            isotropic, uniform or fixed.

    Returns:
        plt.Figure: The plotted figure.
    '''

    figure = cast(plt.Figure, plt.figure())

    anisotropic = case[0] != 'i'
    key_predicted = f'predicted_{model}'
    key_visible = f'visible_{case[0]}'

    # Find the maximum and minimum among the fluxes across every
    # sample to determine a uniform set of histogram bins.
    net_maximum = np.max(samples[0][key_predicted])
    net_minimum = np.min(samples[0][key_predicted])
    for sample in samples:
        fluxes = sample[key_predicted]

        maximum = np.max(fluxes)
        minimum = np.min(fluxes)

        if maximum > net_maximum:
            net_maximum = maximum
        if minimum < net_minimum:
            net_minimum = minimum

    bins = np.logspace(
        np.log10(net_minimum), np.log10(net_maximum), DEFAULT_BIN_COUNT)

    histograms_i = np.empty(shape = (bins.size - 1, 0))
    histograms_d = np.copy(histograms_i)
    histograms_v = np.copy(histograms_i)

    # Compute the histogram of each sample for the isotropic,
    # uniform and fixed opening angle cases, as well as the
    # histogram for detectable fluxes.
    fluxes_d = None
    for sample in samples:
        fluxes = sample[key_predicted]
        histogram, _ = np.histogram(fluxes, bins)
        histograms_i = np.column_stack((histograms_i, histogram))

        if anisotropic:
            fluxes_v = fluxes[sample[key_visible]]
            histogram, _ = np.histogram(fluxes_v, bins)
            histograms_v = np.column_stack((histograms_v, histogram))

            visible = sample[key_visible]
            fluxes_d = fluxes[sample[f'detectable_{model}'] & visible]
        else:
            fluxes_d = fluxes[sample[f'detectable_{model}']]

        histogram, _ = np.histogram(fluxes_d, bins)
        histograms_d = np.column_stack((histograms_d, histogram))

    median_i = np.empty(shape = bins.size - 1)
    median_d = np.copy(median_i)
    median_v = np.copy(median_i)

    upper_i = np.copy(median_i)
    lower_i = np.copy(median_i)

    for b in range(bins.size - 1):
        median_i[b] = np.median(histograms_i[b])
        median_d[b] = np.median(histograms_d[b])

        if anisotropic:
            median_v[b] = np.median(histograms_v[b])

        upper_i[b] = np.percentile(histograms_i[b], UPPER_PERCENTILE)
        lower_i[b] = np.percentile(histograms_i[b], LOWER_PERCENTILE)

    # Plot the median histograms.
    axes = cast(plt.Axes, figure.subplots())
    axes.stairs(median_i, bins, fill = True, color = '#bcefb7',
        label = 'Isotropic')

    # If `case` is not `isotropic`, plot the visible fluxes for `case`.
    if anisotropic:
        axes.stairs(median_v, bins, fill = True, color = '#a9a9a9',
            label = f'Visible ({CASES_EXPANDED[case[0]]})')

    axes.stairs(median_d, bins, fill = True, color = '#eb3a2e',
        label = 'Detectable')

    # Plot the confidence band.
    axes.stairs(upper_i, bins, color = '#78ab73')
    axes.stairs(lower_i, bins, color = '#78ab73')

    # Configure the subplot.
    axes.set_title(f'{MODELS_EXPANDED[model]} ({model})')
    axes.set_xlabel('Flux (erg s⁻¹ cm⁻²)')
    axes.set_ylabel('Number')
    axes.set_xscale('log') # type: ignore
    axes.legend()

    # Configure the plot.
    figure.suptitle('Population Sample', fontsize = 14)
    figure.tight_layout(rect = (0, 0.03, 1, 0.975)) # type: ignore

    return figure


def plot(samples: list[EventSample], model: str, case: str) -> None:
    '''Plots a histogram of the median of `samples` for a given `model`
    of relativistic jet and `case` of opening angle, along with
    individual histograms of several of the samples, and saves the
    results to the `./figures` folder.

    Args:
        samples (list[EventSample]): A list of samples to plot.
        model (str): Which model of relativistic jet to plot.
        case (str): Which case of opening angle to plot; either
            isotropic, uniform or fixed.
    '''

    # Create a folder to store the plots in if it doesn't exist.
    folder = os.path.join('.', 'figures')
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Plot a number of realisations separately, up to `MAXIMUM_PLOTS`.
    realisations = len(samples)
    if realisations > 1:
        plots = realisations if realisations < MAXIMUM_PLOTS else MAXIMUM_PLOTS
        for p in range(plots):
            figure = plot_single(samples[p], model, case)
            filename = os.path.join(folder,
                                   f'{model}_{case}_realisation_{p + 1}.png')
            plt.savefig(filename)
            plt.close(figure)

    plot_median(samples, model, case)
    suffix = '_median' if realisations > 1 else ''
    filename = os.path.join(folder, f'{model}_{case}{suffix}.png')
    plt.savefig(filename)


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

    model = args['m']
    case = args['c']
    realisations = args['r']

    events = preprocess.preprocess(arguments)

    models = MODELS if model == 'all' else [model]
    cases = CASES if case == 'all' else [case[0]]

    for m in models:
        name = MODELS_EXPANDED[m]
        print(f'--> Processing model {name} ({m})...')

        samples = process(events, m, realisations)
        print()

        for c in cases:
            print(f'-> Plotting {CASES_EXPANDED[c]}...')
            plot(samples, m, c)

        print()
        print(f'<-- Finished processing model {name} ({m}).')
        print()

    plt.show()


main()
