# Branching Process Model

[![Run Unit Tests on multiple OS](https://github.com/SABS-R3-Epidemiology/branchpro/actions/workflows/os-unittests.yml/badge.svg)](https://github.com/SABS-R3-Epidemiology/branchpro/actions/workflows/os-unittests.yml)
[![Run Unit Tests on multiple python versions](https://github.com/SABS-R3-Epidemiology/branchpro/actions/workflows/python-version-unittests.yml/badge.svg)](https://github.com/SABS-R3-Epidemiology/branchpro/actions/workflows/python-version-unittests.yml)
[![Documentation Status](https://readthedocs.org/projects/branchpro/badge/?version=latest)](https://branchpro.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SABS-R3-Epidemiology/branchpro/branch/main/graph/badge.svg?token=UBJG0AICF9)](https://codecov.io/gh/SABS-R3-Epidemiology/branchpro/)

In this package, we use branching processes to model the time-dependent reproduction number (the number of cases each infected individual will subsequently cause) of an infectious disease.

All features of our software are described in detail in our
[full API documentation](https://branchpro.readthedocs.io/en/latest/).

A web app for performing inference for branching process models can be found [here](https://sabs-r3-epidemiology.github.io/branchpro/).

More details on branching process models and inference can be found in these
papers:

## References
[1]
Cori A, Ferguson NM, Fraser C, Cauchemez S. (2013). A new framework and
software to estimate time-varying reproduction numbers during epidemics.
American Journal of Epidemiology 178(9): 1505-12.

[2]
Thompson RN, Stockwin JE, van Gaalen RD, Polonsky JA, Kamvar ZN, Demarsh PA,
Dahlqwist E, Li S, Miguel E, Jombart T, Lessler J. (2019). Improved inference of
time-varying reproduction numbers during infectious disease outbreaks.
Epidemics 29: 100356.

## Installation procedure
***
One way to install the module is to download the repositiory to your machine of choice and type the following commands in the terminal.
```bash
git clone https://github.com/SABS-R3-Epidemiology/branchpro.git
cd ../path/to/the/file
```

A different method to install this is using `pip`:

```bash
pip install -e .
```

## Usage

```python
import branchpro
import numpy as np

# create a simple branching process model with prescribed initial R and serial interval
branchpro.BranchProModel(initial_r=0.5, serial_interval=[0, 0.15, 0.52, 0.3, 0.01])

# create branching process model with local and imported cases with prescribed initial R
# and serial interval
# set imported cases data
libr_model_1 = branchpro.LocImpBranchProModel(
  initial_r=2, serial_interval=np.array([1, 2, 3, 2, 1]), epsilon=1)
libr_model_1.set_imported_cases(times=[1, 2.0, 4, 8], cases=[5, 10, 9, 2])

# create the posterior of a branching process model for multiple daily serial intervals
# and incidence data contained in the dataframe df; prior distribution is Gamma with
# parameters alpha and beta (shape, rate)
branchpro.BranchProPosteriorMultSI(
  inc_data=df, daily_serial_intervals=[[1, 2], [0, 1]], alpha=1, beta=0.2)
```

More examples on how to use the classes and features included in this repository can be found [here](https://github.com/SABS-R3-Epidemiology/branchpro/tree/main/examples).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
# exepiest
