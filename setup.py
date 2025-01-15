#
# branchpro setuptools script
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#
from setuptools import setup, find_packages, Extension
import os

ext = Extension(
    'fast_posterior', [os.path.join('branchpro', 'fast_posterior.c')])


def get_version():
    """
    Get version number from the branchpro module.
    The easiest way would be to just ``import branchpro ``, but note that this may  # noqa
    fail if the dependencies have not been installed yet. Instead, we've put
    the version number in a simple version_info module, that we'll import here
    by temporarily adding the oxrse directory to the pythonpath using sys.path.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('branchpro'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_readme():
    """
    Load README.md text for use as description.
    """
    with open('README.md', encoding='utf-8') as f:
        return f.read()


setup(
    # Module name (lowercase)
    name='branchpro',

    # Version
    version=get_version(),

    description='This is a one-week project in which we are using branching processes to estimate the time-dependent reproduction number of a disease.',  # noqa

    long_description=get_readme(),

    license='BSD 3-Clause "New" or "Revised" License',

    # author='',

    # author_email='',

    maintainer='',

    maintainer_email='',

    url='https://github.com/SABS-R3-Epidemiology/branchpro.git',

    # Packages to include
    packages=find_packages(include=('branchpro', 'branchpro.*')),
    include_package_data=True,

    # List of dependencies
    setup_requires=[
        'setuptools = 68.2.2'],
    install_requires=[
        # Dependencies go here!
        'setuptools==69.5.1',
        'matplotlib',
        'numpy<1.23.0,>=1.16.5',
        'dash>=2.0',
        'dash_bootstrap_components>=0.12',
        'dash_daq',
        'dash_defer_js_import',
        'pandas<2.0.0',
        'plotly',
        'scipy>=1.6',
        'pints',
        'numexpr',
        'diskcache',
        'multiprocess',
        'psutil',
        'fast_poibin',
        'pystan',
        'arviz',
        'seaborn'
    ],
    python_requires='>3.9',
    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
        ],
    },
    ext_modules=[ext],
)
