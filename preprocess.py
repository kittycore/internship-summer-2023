import glob, os, re

from typing import Iterator


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

    for model in ['IMRPhenomXPHM', 'SEOBNRv4PHM']:
        if model in path:
            return model
    return 'Unknown'


def main() -> None:
    directory = os.path.join('data', 'modelling')
    for path in wildcard(directory, 'txt'):
        identifier = extract_identifier(path)
        model = extract_model(path)

        print(f'{identifier} ({model})')


if __name__ == '__main__':
    main()
