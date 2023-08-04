# More descriptive names for each of the models.
MODELS_EXPANDED = {
    'QQ': 'Charged Black Hole',
    'NU': 'Neutrino-Antineutrino Annihilation',
    'BZ': 'Blandford-Znajek',
    'GW': 'Gravitational Wave Energy Conversion',
}

# More descriptive names for each of the cases.
CASES_EXPANDED = {
    'i': 'Isotropic',
    'u': 'Uniform',
    'f': 'Fixed',
}


def is_anisotropic(case: str) -> bool:
    '''Returns True if `case` is an anisotropic case of opening angle.

    Args:
        case (str): The case of opening angle to test.

    Returns:
        bool: True if `case` is anisotropic.
    '''

    return case[0] != 'i'


def progress_bar(iterator, prefix = '', length: int = 60):
    '''Wraps an iterable and displays a progress bar that updates as
    the iterable's items are yielded.

    Args:
        iterator (iterable): An iterable to wrap.
        prefix (str, optional): Displayed before the progress bar.
            Defaults to ''.
        length (int, optional): Length of the progress bar, including
            the prefix. Defaults to 60.

    Yields:
        Items from the iterable.
    '''

    length = length - len(prefix)
    size = len(iterator)

    def update(index: int):
        '''Updates the progress bar.

        Args:
            index (int): The index of the current iterator item.
        '''

        fill = int(length * index / size)
        print(f'{prefix}|{u"█" * fill}{"." * (length - fill)}| {index}/{size}',
            end = '\r', flush = True)

    update(0)
    for index, item in enumerate(iterator):
        yield item
        update(index + 1)
    print(flush = True)
