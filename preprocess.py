import glob, os

from typing import Iterator


def wildcard(directory: str, extension: str) -> Iterator[str]:
    files = glob.glob(os.path.join(directory, f'*.{extension}'))
    for path in files:
        yield path


def main() -> None:
    for path in wildcard('.', '*'):
        print(path)


if __name__ == '__main__':
    main()
