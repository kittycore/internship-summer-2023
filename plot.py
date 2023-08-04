import matplotlib.pyplot as plt
import numpy as np

import os

from common import *
from sample import EventSample

from typing import cast


# Default number of bins for plotted histograms.
DEFAULT_BIN_COUNT = 20
# The maximum number of realisations to plot separately.
MAXIMUM_PLOTS = 5


def plot_realisation(sample: EventSample, model: str, case: str) -> plt.Figure:
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
    fluxes_visible  = fluxes[sample[key_visible]] if anisotropic else None
    fluxes_detected = None
    if anisotropic:
        detectable = sample[f'detectable_{model}'] & sample[key_visible]
        fluxes_detected = fluxes[detectable]
    else:
        fluxes_detected = fluxes[sample[f'detectable_{model}']]

    # Determine the histogram binning based upon the minimum and
    # maximum fluxes within log-space.
    maximum = np.max(fluxes)
    minimum = np.min(fluxes)
    bins = np.logspace(np.log10(minimum), np.log10(maximum), DEFAULT_BIN_COUNT)

    # Plot each set of fluxes.
    axes = cast(plt.Axes, figure.subplots())
    axes.hist(fluxes, bins, color = '#bcefb7', label = 'Isotropic')

    # If `case` is not `isotropic`, plot the visible fluxes for `case`.
    if anisotropic:
        axes.hist(fluxes_visible, bins, color = '#a9a9a9',
            label = f'Visible ({CASES_EXPANDED[case[0]]})')

    axes.hist(fluxes_detected, bins, color = '#eb3a2e', label = 'Detectable')

    # Configure the subplot.
    axes.set_title(f'{MODELS_EXPANDED[model]} ({model})')
    axes.set_xlabel('Flux (erg s⁻¹ cm⁻²)')
    axes.set_ylabel('Number')
    axes.set_xscale('log') # type: ignore
    axes.legend()

    # Configure the plot.
    figure.suptitle('Population Sample (Single)')
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
            visible = sample[key_visible]

            fluxes_v = fluxes[visible]
            histogram, _ = np.histogram(fluxes_v, bins)
            histograms_v = np.column_stack((histograms_v, histogram))

            fluxes_d = fluxes[sample[f'detectable_{model}'] & visible]
        else:
            fluxes_d = fluxes[sample[f'detectable_{model}']]

        histogram, _ = np.histogram(fluxes_d, bins)
        histograms_d = np.column_stack((histograms_d, histogram))

    # Generate the median histogram and the confidence band.
    median_i = np.empty(shape = bins.size - 1)
    median_d = np.copy(median_i)
    median_v = np.copy(median_i)

    for b in range(bins.size - 1):
        median_i[b] = np.median(histograms_i[b])
        median_d[b] = np.median(histograms_d[b])

        if anisotropic:
            median_v[b] = np.median(histograms_v[b])

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

    # Configure the subplot.
    axes.set_title(f'{MODELS_EXPANDED[model]} ({model})')
    axes.set_xlabel('Flux (erg s⁻¹ cm⁻²)')
    axes.set_ylabel('Number')
    axes.set_xscale('log') # type: ignore
    axes.legend()

    # Configure the plot.
    figure.suptitle('Population Sample (Median)')
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
            figure = plot_realisation(samples[p], model, case)
            filename = os.path.join(folder,
                                   f'{model}_{case}_realisation_{p + 1}.png')
            plt.savefig(filename)
            plt.close(figure)
    # If there's only one realisation, don't plot a median histogram.
    else:
        plot_realisation(samples[0], model, case)
        filename = os.path.join(folder, f'{model}_{case}_realisation.png')
        plt.savefig(filename)
        return

    plot_median(samples, model, case)
    suffix = '_median' if realisations > 1 else ''
    filename = os.path.join(folder, f'{model}_{case}{suffix}.png')
    plt.savefig(filename)
