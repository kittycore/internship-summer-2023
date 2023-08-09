# Constraining the Population of Binary Black Holes

This programme samples simulated gravitational wave events and determines how
many of the sampled events exceed the upper limits of Fermi-GBM.

## Requirements

To use this programme, the following datasets are required:

- Simulations of different relativistic jet models for each gravitational wave
event in a text format. Supplied by Dr Peter Veres.
- The [GWTC-2](https://dcc.ligo.org/LIGO-P2000223/public),
[GWTC-2.1 v1](https://zenodo.org/record/5117703),
[GWTC-2.1 v2](https://zenodo.org/record/6513631), and/or
[GWTC-3](https://zenodo.org/record/5546663) catalogues
from the LIGO Scientific Collaboration, VIRGO Collaboration and KAGRA
Collaboration.
- The upper limits of Fermi-GBM for each gravitational wave event in a binary
NumPy format.

The default folder structure for this datasets is as follows:

```
data/
    GWTC/           | The GWTC datasets go here.
    modelling/      | Peter's models go here.
    upper_limits/   | The upper limits go here.
```

## Installation

It is recommended that you create a virtual environment to isolate packages
required by this programme. To create a virtual environment, activate it and
automatically install the required dependencies, execute the following commands
in this directory.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

To generate a plot of sample for a particular model and case of opening angle,
use `python . -m [model] -c [case]`. The resulting plot(s) will be located in
the `figures` directory.

If no model is specified, then all models will be sampled. If no case is
specified, then all cases will be sampled. Both can be left unspecified to
sample all cases of every model.

To generate plots of several samples (or realisations), use the `-r` option
and specify the number of realisations. For example, to generate plots of a
hundred realisations, use `python . -r 100`.

This programme creates caches of events and samples to improve speed. If you
want to regenerate these caches (e.g., to use a different seed for the random
number generator) then use the `-f` or `--force` command-line flags.

For other command-line options and usage information, use the `-h` flag.

## Credits

This programme was written as an internship project for NASA under the
mentorship of Dr Joshua Wood. The datasets used for this project were provided
by Dr Peter Veres and the LIGO Scientific Collaboration, VIRGO Collaboration,
and KAGRA Collaboration. Special thanks go to the Fermi-GBM team for their
kindness and support.

## License

This project is licensed under the [GNU General Public License v3](./COPYING).
