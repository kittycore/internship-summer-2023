import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(prog = 'limits',
        description = 'Estimates the upper limits of the binary black hole ' \
            'population visible with Fermi-GBM.')
    add_arguments(parser)

    arguments = parser.parse_args()


main()
