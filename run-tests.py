# This file contains code for running all tests.
#
# This file is part of BRANCHPRO
# (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is released
# under the BSD 3-clause license. See accompanying LICENSE.md for copyright
# notice and full license details.
#

import unittest
import os
import sys
import argparse


def run_unit_tests():
    """
    This function runs our unit tests.
    """
    tests = os.path.join('branchpro', 'tests')
    tests_suite = unittest.defaultTestLoader.discover(tests,
                                                      pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests_suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run unit test for branchpro',
        epilog='To run individual unit tests, use e.g.'
               ' $ python3 branchpro/tests/test_dummy.py',
    )
    # Unit tests
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Run all unit tests using `python` interpretor.',
    )

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False
    # Unit tests
    if args.unit:
        has_run = True
        run_unit_tests()

    if not has_run:
        parser.print_help()
