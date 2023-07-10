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
    pattern = re.compile(r'GW\d+_?(?=\d)\d*')
    m = pattern.search(path)
    if m:
        return m.group(0)
    return 'Unknown'


def main() -> None:
    directory = os.path.join('data', 'modelling')
    for path in wildcard(directory, 'txt'):
        identifier = extract_identifier(path)
        print(identifier)


if __name__ == '__main__':
    main()
