"""
This script is used to compare two files so that they are put side by
side.

Usage:
      cat file1 file2 | tensorial.py -c col1 [col2 [col3]] ...



Examples:
  (a) Join two files using the label in column 1 as a reference
        cat f1.txt f2.txt | tensorial.py -c 1
"""
import argparse
import sys


def entry_point():
    argParser = argparse.ArgumentParser(description=__doc__,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)  # for verbatim

    argParser.add_argument('--column', '-c', metavar='<int>', type=int, nargs='+',
                           help='Column that contains the reference label used to join the lines. '
                                'This option is repeatable and, if not present, defaults to 1. '
                                'Columns are expressed as 1-based.')

    args = argParser.parse_args()

    # Retrieve the columns that will be used to make the index
    if len(args.column) == 0:
        idxes = [0]
    else:
        idxes = [v - 1 for v in args.column]

    lines = {}
    for line in sys.stdin:
        values = line.split()

        # Build the index
        try:
            index = ' '.join([values[i] for i in idxes])
        except IndexError:
            sys.stderr.write("FATAL  : Not enough columns to build the index in the current line\n    %s\n" % line)
            sys.exit(1)

        # Store the line based on the index, or print it
        if index in lines:
            sys.stdout.write("{0} {1}\n".format(lines[index], line[:-1]))
        else:
            lines[index] = line[:-1]
