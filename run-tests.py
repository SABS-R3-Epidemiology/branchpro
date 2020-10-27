# This file contains code for running all tests.
#
# This file is part of BRANCHPRO (https://github.com/SABS-R3-Epidemiology/branchpro.git) which is  # noqa
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
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

# This function is from the Pints library
# https://github.com/pints-team/pints/blob/master/run-tests.py
def run_copyright_checks():
    """
    Checks that the copyright year in LICENSE.md is up-to-date and that each
    file contains the copyright header
    """
    print('\nChecking that copyright is up-to-date and complete.')

    year_check = True
    current_year = str(datetime.datetime.now().year)

    with open('LICENSE.md', 'r') as license_file:
        license_text = license_file.read()
        if 'Copyright (c) 2017-' + current_year in license_text:
            print("Copyright notice in LICENSE.md is up-to-date.")
        else:
            print('Copyright notice in LICENSE.md is NOT up-to-date.')
            year_check = False

    # Recursively walk the pints directory and check copyright header is in
    # each checked file type
    header_check = True
    checked_file_types = ['.py']
    copyright_header = """#
# This file is part of branchpro (https://github.com/SABS-R3-Epidemiology/branchpro/)
# which is released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#"""

    for dirname, subdir_list, file_list in os.walk('pints'):
        for f_name in file_list:
            if any([f_name.endswith(x) for x in checked_file_types]):
                path = os.path.join(dirname, f_name)
                with open(path, 'r') as f:
                    if copyright_header not in f.read():
                        print('Copyright blurb missing from ' + path)
                        header_check = False

    if header_check:
        print('All files contain copyright header.')

    if not year_check or not header_check:
        print('FAILED')
        sys.exit(1)

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
    
    parser.add_argument(
        '--copyright',
        action='store_true',
        help='Check that copyright license info is up to date.',
    )

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False
    # Unit tests
    if args.unit:
        has_run = True
        run_unit_tests()
    if args.copyright:
        has_run = True
        run_copyright_checks()

    if not has_run:
        parser.print_help()
