import glob, os

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


def main() -> None:
    for path in wildcard('.', '*'):
        print(path)


if __name__ == '__main__':
    main()
