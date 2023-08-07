import numpy as np

import argparse, os

from common import *
import preprocess
from preprocess import Event


# The name format of the cache to store samples.
CACHE_FILE = 'sample_{model}_{realisations}.npz'

# The opening angle for the 'fixed' case, in radians.
FIXED_ANGLE = np.deg2rad(20)

RIGHT_ANGLE = np.pi / 2

DEFAULT_SEED = 0x9B7DB742C51C67FF
random: np.random.Generator = np.random.default_rng(DEFAULT_SEED)

# What percentile to consider 'confident' for the number of detections.
CONFIDENCE = 95

# Default number of realisations.
DEFAULT_REALISATIONS = 1


class EventSample(np.ndarray):
    DTYPE = [
        ('predicted_QQ', np.float64),
        ('predicted_NU', np.float64),
        ('predicted_BZ', np.float64),
        ('predicted_GW', np.float64),
        ('visible_u', np.bool_),
        ('visible_f', np.bool_),
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
    upper_limits: np.ndarray,
    opening_angles: np.ndarray | None = None,
) -> np.ndarray:
    '''Determines the visibility of a set of events using the
    inclination and the opening angle of the relativistic jet
    associated with each event and whether Earth blocks it.

    Args:
        inclinations (np.ndarray): An array containing inclinations for
            the relativistic jets associated with a set of events.
        upper_limits (np.ndarray): An array containing the upper limits
            of Fermi-GBM to check for the presence of the Earth.
        opening_angles (np.ndarray | None, optional): An array
            containing opening angles for the relativistic jets
            associated with a set of events. If None, then the angle
            specified by the constant `FIXED_ANGLE` is used instead.
            Defaults to None.

    Returns:
        np.ndarray: An array of booleans indicating which events are
            visible or not from Earth.
    '''

    # HealPy assigns a negative number to coordinates on the sky map
    # that are blocked by the Earth.
    unblocked = upper_limits > 0

    # Wrap inclination angles outside the domain [-π/2, π/2] to
    # simplify the rest of the check.
    needs_wrapping = inclinations > RIGHT_ANGLE
    wrapped = inclinations - (2 * RIGHT_ANGLE)
    inclinations = np.where(needs_wrapping, wrapped, inclinations)

    angle = FIXED_ANGLE if opening_angles is None else opening_angles

    within_maximum = inclinations <  angle
    within_minimum = inclinations > -angle
    return within_maximum & within_minimum & unblocked


def is_detectable(fluxes: np.ndarray, upper_limits: np.ndarray) -> np.ndarray:
    '''Determines which events from a set of events are detectable by
    Fermi-GBM using the the predicted fluxes of the relativistic jet
    associated with each event and the upper limits of Fermi-GBM.

    Args:
        fluxes (np.ndarray): An array containing the predicted fluxes
            of the relativistic jets associated with a set of events.
        upper_limits (np.ndarray): An array containing the upper limits
            of Fermi-GBM to check for the presence of the Earth.

    Returns:
        np.ndarray: An array of booleans indicating which events are
            detectable or not by Fermi-GBM.
    '''

    # HealPy assigns a negative number to coordinates on the sky map
    # that are blocked by the Earth. To simplify the comparison,
    # replace the negative numbers with large, positive numbers.
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

    collector = np.concatenate([*events.values()], axis = -1)
    choices = random.choice(collector, size = sample_size, shuffle = False)

    fluxes = choices[f'flux_{model}']
    sample[f'predicted_{model}'] = fluxes

    inclinations = choices['inclination']
    upper_limits = choices['upper_limit']
    sample['visible_f'] = is_visible(inclinations, upper_limits)
    sample['visible_u'] = is_visible(
        inclinations, upper_limits, choices['opening_angle'])

    sample[f'detectable_{model}'] = is_detectable(fluxes, upper_limits)

    return sample


def process(
    events: dict[str, Event], model: str, realisations: int
) -> list[EventSample]:
    '''Processes a set of events for a given `model`, returning a set
    of samples of number `realisations`.

    Args:
        events (dict[str, Event]): A set of events to sample.
        model (str): Which model of relativistic jet to sample.
        realisations (int): The number of samples to produce.

    Returns:
        list[EventSample]: A set of samples of number `realisations`.
    '''

    collector = []

    for _ in progress_bar(range(0, realisations), prefix = 'Realising: '):
        sample = realise(events, model)
        collector.append(sample)

    return collector


def compute(samples: list[EventSample], model: str, case: str) -> None:
    '''Computes the mean, median and 95% confidence for the number of
    detections for a given `model` and `case` of opening angle.

    Args:
        samples (list[EventSample]): A list of samples to compute with.
        model (str): The model of relativistic jet used for the sample.
        case (str): The case of opening angle to compute for.
    '''

    realisations = len(samples)
    detections = np.zeros(realisations, dtype = int)

    anisotropic = is_anisotropic(case)

    for index, sample in enumerate(samples):
        detectable = sample[f'detectable_{model}']
        if anisotropic:
            visible = sample[f'visible_{case[0]}']
            detectable &= visible
        detectable = detectable.astype(int)

        detections[index] = np.count_nonzero(detectable)

    mean = np.mean(detections)
    median = np.median(detections)
    confidence = np.percentile(detections, 100 - CONFIDENCE)

    events = realisations * samples[0].size
    print(f'{detections.sum()} detections from {events} events!',
        end = '\n\t')
    print(f'Mean: {mean:.3f}', end = ' | ')
    print(f'Median: {median:.3f}', end = ' | ')
    print(f'{CONFIDENCE}% confidence: {confidence:.3f}')


def is_cached(directory: str, model: str, realisations: int) -> bool:
    '''Returns True if a cache sample already exists. Each model of
    relativistic jet and number of realisations generates a unique
    cache sample.

    Args:
        directory (str): The directory to search for a cached sample.
        model (str): Which model to search for.
        realisations (int): The number of realisations to search for.

    Returns:
        bool: True is a cache exists, and False otherwise.
    '''

    path = os.path.join(directory,
        CACHE_FILE.format(model = model, realisations = realisations))
    return os.path.isfile(path)


def deserialise(directory: str, model: str, realisations: int) -> list[EventSample]:
    '''Deserialises a list of samples from a cache file.

    Args:
        directory (str): The directory where the cache is located.
        model (str): The model used for the samples.

    Returns:
        list[EventSample]: The deserialised list of samples.
    '''

    path = os.path.join(directory,
        CACHE_FILE.format(model = model, realisations = realisations))

    samples = None
    with np.load(path) as dataset:
        samples = list(dataset.values())

    return samples


def serialise(directory: str, samples: list[EventSample], model: str) -> None:
    '''Serialises a list of samples to a cache file.

    Args:
        directory (str): The directory to save the cache into.
        samples (list[EventSample]): The list of samples to cache.
        model (str): The model used for the samples.
    '''

    path = os.path.join(directory,
        CACHE_FILE.format(model = model, realisations = len(samples)))
    np.savez(path, *samples)


def sample(arguments: argparse.Namespace) -> dict[str, list[EventSample]]:
    '''Produces a set of samples.

    Args:
        arguments (argparse.Namespace): The main entrypoint should pass
            through the parsed arguments through this parameter.

    Returns:
        dict[str, list[EventSample]]: A dictionary containing the name
            of the sampled models (as the keys) and a list of samples
            (as the values).
    '''

    args = vars(arguments) # Shorthand for easier access!

    model = args['m']
    realisations = args['r']

    events = preprocess.preprocess(arguments)
    models = MODELS if model == 'all' else [model]

    collector = {}

    for model in models:
        name = MODELS_EXPANDED[model]
        print(f'Sampling model {name} ({model})...')

        if not args['force'] and is_cached(CACHE_DIRECTORY, model, realisations):
            print('A cached sample already exists! Loading from cache...')
            collector[model] = deserialise(CACHE_DIRECTORY, model, realisations)
        else:
            global random
            random = np.random.default_rng(args['s'])

            samples = process(events, model, realisations)
            serialise(CACHE_DIRECTORY, samples, model)
            collector[model] = samples

        print(f'Finished sampling model {name} ({model}).')
        print()

    return collector


def add_arguments(parser: argparse.ArgumentParser,
    group: argparse._ArgumentGroup | None = None) -> None:
    '''Adds command-line arguments relevant to the preprocessor to an
    argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add
            the relevant command-line arguments to.
        group (argparse._ArgumentGroup, optional): The argument group
            to add the arguments to. If unspecified, the arguments are
            added to the top-level group.
    '''

    # If a group is specified, re-route `add_argument` calls to it.
    target = parser
    if group:
        target = group

    target.add_argument('-m', choices = [*MODELS, 'all'], default = 'all',
        help = 'Which model to plot.')
    target.add_argument('-r', type = int, default = DEFAULT_REALISATIONS,
        help = 'The number of realisations to process.')
    target.add_argument('--seed', type = int, default = DEFAULT_SEED,
        help = 'The seed used for the random number generator.')

    group = parser.add_argument_group('preprocessor',
        description = 'Arguments relevant to the preprocessor.')
    preprocess.add_arguments(parser, group)


def main(arguments: argparse.Namespace) -> None:
    '''Main entrypoint.'''

    args = vars(arguments) # Shorthand for easier access!

    # If the `-p` flag is set, pass to the preprocessor.
    if args['p']:
        preprocess.main(arguments)
        return

    model = args['m']
    realisations = args['r']

    if not model == 'all':
        if not args['force'] and is_cached(CACHE_DIRECTORY, model, realisations):
            exit('A cache file already exists! Use `-f` or `--force` to ' \
                 'regenerate it.')

    events = preprocess.preprocess(arguments)
    models = MODELS if model == 'all' else list(model)

    for model in models:
        name = MODELS_EXPANDED[model]
        print(f'Sampling model {name} ({model})...')

        if not args['force'] and is_cached(CACHE_DIRECTORY, model, realisations):
            print('A cache file already exists! Use `-f` or `--force` to ' \
                  'regenerate it.')
        else:
            global random
            random = np.random.default_rng(args['seed'])

            samples = process(events, model, realisations)
            serialise(CACHE_DIRECTORY, samples, model)

        print(f'Finished sampling model {name} ({model}).')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'sample',
        description = 'Samples preprocessed events to be plotted.')
    add_arguments(parser)

    arguments = parser.parse_args()

    main(arguments)
