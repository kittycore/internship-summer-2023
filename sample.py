import numpy as np

from common import *
from preprocess import Event


# The opening angle for the 'fixed' case, in radians.
FIXED_ANGLE = np.deg2rad(20)
# A right angle, in radians.
RIGHT_ANGLE = np.pi / 2

# Default seed for the random number generator.
DEFAULT_SEED = 0x9B7DB742C51C67FF
# A random number generator accessible anywhere in this module.
random: np.random.Generator = np.random.default_rng(DEFAULT_SEED)

# What percentile to consider 'confident' for the number of detections.
CONFIDENCE = 95


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

    # If no `opening_angles` are specified, use the fixed angle stored
    # in the constant `FIXED_ANGLE`. Otherwise, use `opening_angles`.
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

    # Collect all of the events into a single array.
    collector = np.concatenate([*events.values()], axis = -1)

    # Randomly choose fluxes from the set of events and determine which
    # of these fluxes are potentially detectable.
    choices = random.choice(collector, size = sample_size, shuffle = False)

    fluxes = choices[f'flux_{model}']
    sample[f'predicted_{model}'] = fluxes

    # Determine the visibility of the chosen fluxes.
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

    # Repeatedly sample the set of events and collect the results.
    for _ in progress_bar(range(0, realisations), prefix = 'Realising: '):
        sample = realise(events, model)
        collector.append(sample)

    return collector


def compute(samples: list[EventSample], model: str, case: str) -> None:
    realisations = len(samples)
    detections = np.zeros(realisations, dtype = int)

    for index, sample in enumerate(samples):
        # Determine which of the events are detectable in this `case`.
        detectable = sample[f'detectable_{model}']
        if case[0] != 'i':
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
