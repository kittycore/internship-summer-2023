import healpy, h5py
import numpy as np

import glob, os, re

from typing import cast, Iterator


# The name of the cache file.
CACHE_FILE = 'cache.npz'

# Models found within the GWTC datasets.
MODELS = [
    'IMRPhenomXPHM_comoving', 'IMRPhenomXPHM', 'IMRPhenomPv3HM',
    'IMRPhenomPv2', 'NRSur7dq4', 'SEOBNRv4PHM', 'PrecessingSpinIMRHM',
    'SEOBNRv4P',
]

# The NumPy data type of the simulation models.
MODEL_DTYPE = [
    ('a_final', np.float64),
    ('mass_final[Msun]', np.float64),
    ('DL[Mpc]', np.float64),
    ('inclination[rad]', np.float64),
    ('OpeningAngleU10-40deg[rad]', np.float64),
    ('FQQ', np.float64),
    ('FNUNU', np.float64),
    ('FBZ', np.float64),
    ('FGW', np.float64),
]


class Event(np.ndarray):
    DTYPE = [
        ('inclination', np.float64),
        ('opening_angle', np.float64),
        ('flux_QQ', np.float64),
        ('flux_NU', np.float64),
        ('flux_BZ', np.float64),
        ('flux_GW', np.float64),
        ('right_ascension', np.float64),
        ('declination', np.float64),
        ('upper_limit', np.float64),
    ]

    def __new__(cls, simulations: int):
        shape = (simulations)
        return super().__new__(cls, shape, dtype = Event.DTYPE)


def wildcard(directory: str, extension: str) -> Iterator[str]:
    '''Searches for files ending with the given `extension` within
    `directory` and yields the file path of each matching file.

    Args:
        directory (str): The directory to search within for files.
        extension (str): The file extension to limit the search to.

    Yields:
        str: The file path of each matching file.
    '''

    files = glob.glob(os.path.join(directory, f'*.{extension}'))
    for path in files:
        yield path


def extract_identifier(path: str) -> str:
    '''Extracts an identifier (name) of a gravitational wave event
    from a file path if it is present.

    Args:
        path (str): The file path to extract an identifier from.

    Returns:
        str: The identifier (name) of the gravitational wave event,
            or 'Unknown' if no identifier is present.
    '''

    pattern = re.compile(r'GW\d+_?(?=\d)\d*')
    m = pattern.search(path)
    if m:
        return m.group(0)
    return 'Unknown'


def extract_model(path: str) -> str:
    '''Extract the model used for a set of simulations from a file
    path. If the model is not found within GWTC datasets, 'Unknown'
    is returned instead.

    Args:
        path (str): The file path to extract a model from.

    Returns:
        str: The extracted model, or 'Unknown' if the model is
            not found within GWTC datasets.
    '''

    for model in MODELS:
        if model in path:
            return model
    return 'Unknown'


def find_models(directory: str) -> dict[str, str]:
    '''Searches for models within a given `directory` and returns a
    dictionary containing the identifier, model, and file path of each.

    Args:
        directory (str): The directory to search for models within.

    Returns:
        dict[str, str]: A dictionary containing the identifier and
            model of each event (as the keys) and the file path to a
            text file containing the model data (as the values).
    '''

    models = {}

    paths = sorted(wildcard(directory, 'txt'))
    for path in paths:
        identifier = extract_identifier(path)
        model = extract_model(path)
        key = f'{identifier}_{model}'
        models[key] = path

    return models


def trim_models(models: dict[str, str]) -> dict[str, str]:
    '''Trims duplicate models from the dictionary generated by
    `find_models` so that each event has a single model associated
    with it.

    Args:
        models (dict[str, str]): A dictionary containing the identifier
            and model of each event (as the keys) and the file path to
            a text file containing the model data (as the values).

    Returns:
        dict[str, str]: A dictionary containing the identifier and
            model of each event (as the keys) and the file path to a
            text file containing the model data (as the values), with
            duplicate models for each event removed.
    '''

    keeps = []

    for key in sorted(models.keys()):
        identifier = extract_identifier(key)
        model = extract_model(key)

        # If the model is not found within GWTC datasets, skip it.
        if model == 'Unknown':
            continue

        skip = False
        for keep in keeps:
            # If an event with the same identifier has already been seen,
            # skip it, as we don't want any duplicates. IMRPhenom* models
            # will be preferred because of the alphabetisation by sorting.
            if identifier == extract_identifier(keep):
                skip = True
                break
        if skip:
            continue

        keeps.append(key)

    # Return a copy of the models dictionary with only the keys in `keeps`.
    return { key: models[key] for key in keeps }


def load_models(models: dict[str, str]) -> dict[str, Event]:
    '''Loads the models within the `models` dictionary generated by
    `find_models` into NumPy arrays.

    Args:
        models (dict[str, str]): A dictionary containing the identifier
            and model of each event (as the keys) and the file path to
            a text file containing the model data (as the values).

    Returns:
        dict[str, Event]: A dictionary containing the identifier and
            model of each event (as the keys) and a NumPy array with
            the data type `Event.DTYPE` with the data loaded from the
            model files (as the values).
    '''

    events = {}

    for key, path in models.items():
        data = np.loadtxt(path, dtype = MODEL_DTYPE)

        event = Event(data.size)
        event['inclination'] = data['inclination[rad]']
        event['opening_angle'] = data['OpeningAngleU10-40deg[rad]']
        event['flux_QQ'] = data['FQQ']
        event['flux_NU'] = data['FNUNU']
        event['flux_BZ'] = data['FBZ']
        event['flux_GW'] = data['FGW']

        # Initialise the rest of the array to NaN instead of garbage.
        event['right_ascension'] = np.nan
        event['declination'] = np.nan
        event['upper_limit'] = np.nan

        events[key] = event

        identifier = extract_identifier(key)
        model = extract_model(key)
        print(f'Read {identifier} ({model}) from {os.path.basename(path)}.')

    return events


def find_places(directory: str) -> dict[str, str]:
    '''Searches for GWTC datasets within a given `directory` and
    returns a dictionary containing the identifier of each event and
    the file path of its associated dataset.

    Args:
        directory (str): The directory to search for datasets within.

    Returns:
        dict[str, str]: A dictionary containing the identifier of each
            event (as the keys) and the file path to its associated
            GWTC-2, GWTC-2.1 or GWTC-3 dataset (as the values).
    '''

    places = {}

    paths = sorted(wildcard(directory, 'h5'))
    for path in paths:
        identifier = extract_identifier(path)
        places[identifier] = path

    return places


def load_places(places: dict[str, str], events: dict[str, Event]) \
    -> dict[str, Event]:
    '''Loads the coordinates located within the GWTC datasets pointed
    to by the dictionary `places` generated by `find_places` and
    returns an updated copy of `events` with the coordinates added.

    Args:
        places (dict[str, str]): A dictionary containing the identifier
            of each event (as the keys) and the file path to its
            associated GWTC dataset (as the values).
        events (dict[str, Event]): A dictionary containing the
            identifier and model of each event (as the keys) and a
            NumPy array with the data type `Event.DTYPE` (as the
            values).

    Returns:
        dict[str, Event]: A dictionary of the same structure as
            `events`, with the location of each event added. If the
            coordinates of an event could not be found, the event is
            excluded.
    '''

    loads = {}

    for key, event in events.items():
        identifier = extract_identifier(key)
        path = None

        # If there's an exact match for the identifier...
        if identifier in places:
            path = places[identifier]
        else:
            # Otherwise, search for a similar identifier.
            for i, p in places.items():
                if identifier in i:
                    path = p
                    break

        if path is None:
            print(f'No GWTC dataset was found for {identifier}!')
            continue

        with h5py.File(path) as f:
            model = extract_model(key)
            model_key = model

            # Determine if the model exists within the GWTC dataset and in
            # which format, as some datasets expect only the name of the model
            # and others use the format C01:{model}.
            if model_key not in f.keys():
                model_key = f'C01:{model}'
                if model_key not in f.keys():
                    print(f'{model} is not present in the GWTC dataset for ' \
                      f'{identifier}!')
                    continue

            data = f[model_key]['posterior_samples'] # type: ignore

            # Check if there is a shape mismatch.
            if event.shape != data['ra'].shape: # type: ignore
                print(f'Ignoring {identifier} ({model}) from {path} due to ' \
                       'shape mismatch.')
                continue

            loads[key] = np.copy(event, subok = True)
            loads[key]['right_ascension'] = cast(np.ndarray, data['ra']) # type: ignore
            loads[key]['declination'] = cast(np.ndarray, data['dec']) # type: ignore

        print(f'Read coordinates of {identifier} ({model}) from ' \
              f'{os.path.basename(path)}.')

    return loads


def find_limits(directory: str) -> dict[str, str]:
    '''Searches for NumPy binary files containing upper limit
    information within a given `directory` and returns a dictionary
    containing the identifier of each event and the file path of its
    associated NumPy binary file.

    Args:
        directory (str): The directory to search for NumPy binary files
            (*.npy) containing upper limit information within.

    Returns:
        dict[str, str]: A dictionary containing the identifier of each
            event (as the keys) and the file path to its associated
            binary NumPy file containing upper limits (as the values).
    '''

    limits = {}

    paths = sorted(wildcard(directory, 'npy'))
    for path in paths:
        identifier = extract_identifier(path)
        limits[identifier] = path

    return limits


def load_limits(limits: dict[str, str], events: dict[str, Event]) \
    -> dict[str, Event]:
    loads = {}

    for key, event in events.items():
        identifier = extract_identifier(key)

        # If there are any NaNs in the event's coordinates, then a location for
        # the event was unable to be determined and it has to be skipped.
        if np.isnan(event['declination']).any():
            print(f'Ignoring {identifier} due to uninitialised coordinates.')
            continue

        path = None

        # If there's an exact match for the identifier...
        if identifier in limits:
            path = limits[identifier]
        else:
            # Otherwise, search for a similar identifier.
            for i, p in limits.items():
                if identifier in i:
                    path = p
                    break

        if path is None:
            print(f'No upper limits were found for {identifier}!')
            continue

        # Convert equatorial coordinates (from the GWTC datasets) into
        # spherical coordinates (used in the files storing the upper limits).
        theta = np.pi / 2 - event['declination']
        phi = event['right_ascension']

        # Convert the spherical coordinates into array indices.
        pix = healpy.pixelfunc.ang2pix(16, theta, phi)

        healpix = np.load(path)
        loads[key] = np.copy(event, subok = True)
        loads[key]['upper_limit'] = healpix[pix]

        print(f'Read upper limits of {identifier} from {path}.')

    return loads


def is_cached(directory: str) -> bool:
    '''Returns True if a cache file already exists.

    Args:
        directory (str): The directory to search for a cache.

    Returns:
        bool: True is a cache exists, and False otherwise.
    '''

    path = os.path.join(directory, CACHE_FILE)
    return os.path.isfile(path)


def deserialise(directory: str) -> dict[str, Event]:
    '''Deserialises events from the cache file.

    Args:
        directory (str): The directory where the cache is located.

    Returns:
        dict[str, Event]: A dictionary containing the identifier and
            model of each event (as the keys) and a NumPy array with
            the data type `Event.DTYPE` containing information about
            the event (as the values).
    '''

    path = os.path.join(directory, CACHE_FILE)

    events = None
    with np.load(path) as dataset:
        events = dict(dataset)

    return events


def serialise(directory: str, events: dict[str, Event]) -> None:
    '''Serialises a set of events to a cache file.

    Args:
        directory (str): The directory to save the cache into.
        events (dict[str, Event]): A dictionary containing the
            identifier and model of each event (as the keys) and a
            NumPy array with the data type `Event.DTYPE` containing
            information about the event (as the values).
    '''

    path = os.path.join(directory, CACHE_FILE)
    np.savez(path, **events)


def preprocess() -> dict[str, Event]:
    print('--> Preprocessing...')
    print()

    # Find, trim and finally load the simulation models.
    directory = os.path.join('data', 'modelling')

    print(f'-> Searching for models in {directory}...')
    models = find_models(directory)
    print(f'<- Found {len(models)} models.')

    print()

    print('-> Trimming models...')
    models = trim_models(models)
    print(f'<- Trimmed down to {len(models)} models.')

    print()

    print('-> Loading models...')
    events = load_models(models)
    print(f'<- Successfully loaded {len(models)} models.')

    print()

    # Find and load coordinates for each event from the GWTC datasets.
    directory = os.path.join('data', 'GWTC')

    print(f'-> Searching for GWTC datasets in {directory}...')
    places = find_places(directory)
    print(f'<- Found {len(places)} datasets.')

    print()

    print(f'-> Loading coordinates from GWTC datasets...')
    events = load_places(places, events)
    print(f'<- Successfully loaded {len(events)} sets of coordinates.')

    print()

    # Find and load upper limits for each event.
    directory = os.path.join('data', 'upper_limits')

    print(f'-> Searching for upper limits in {directory}...')
    limits = find_limits(directory)
    print(f'<- Found {len(limits)} upper limits.')

    print()

    print(f'-> Loading upper limits...')
    events = load_limits(limits, events)
    print(f'<- Successfully loaded {len(events)} upper limits.')

    print()
    print('<-- Preprocessing complete!')

    return events


def main() -> None:
    # Check if a cache already exists, and if it does, print a notice and exit.
    directory = 'data'
    if is_cached(directory):
        exit('A cache file already exists!')

    events = preprocess()

    # Serialise the events to a cache file.
    directory = 'data'
    serialise(directory, events)


if __name__ == '__main__':
    main()
