"""
Program to perform various columnar operations on inputs

All indicators have this format

 'x0'

where 'x' can be one of the following

- 'c' - Select column
- 'd' - diff column relative to the previous value
- 'f' - diff column relative to the first value of the column
- 'm' - Compute the minutes elapsed since the first value (divide column by 60,
        as it assumes that the values are in seconds))
- 'h' - Compute the hours elapsed since the first value (divide column by 3600,
        as it assumes that the values are in seconds))

and '0' is the column number (1 based)

Examples:

(a) Select columns with the indicated order (first output 5th column and then the
    first column)
        cat file.txt | cl c5 c1

(b) Select 6th column and output 1st column relative to the first one
        cat file.txt | cl c6 f1

(c) Make a diff of the third column relative to the first value
        cat file.txt | cl f3
"""
import argparse
import sys


class ColumnProcess:
    """
    Class that manages the processing of a set of fields based on some criteria
    """

    def __init__(self, colprocstr):
        """
        Class initialization. This method receives a string defining the type
        of operation to be performed
        """

        if len(colprocstr) < 2:
            raise ValueError(f"Do not know how to interpret [ {colprocstr} ], "
                             "column selector should be of the form 'n0', with "
                             "'n' being a character and '0' a column number")

        self.process_type = colprocstr[0]

        # Obtain the column number, taking into account that the indices must
        # be translated from 1-based to 0-based
        self.process_column = int(colprocstr[1:]) - 1

        self.previous_value = None
        self.first_value = None

    def process(self, fields):
        """
        Process a set of fields. Raise an exception if
        """

        if self.process_column >= len(fields):
            raise IndexError(f"Unable to fecth column [ {self.process_column + 1} ] (1-based) "
                             f"in line with [ {len(fields)} ] fields. "
                             f"Offending line [ {' '.join(fields)} ]\n")

        column_value = fields[self.process_column]

        if self.process_type == 'c':

            return column_value

        elif self.process_type == 'f' or self.process_type == 'm' or self.process_type == 'h':

            incoming_value = float(column_value)

            if self.first_value is None:
                self.first_value = incoming_value

            value = incoming_value - self.first_value

            if self.process_type == 'm':
                value = value / 60.0
            elif self.process_type == 'h':
                value = value / 3600.0

            return str(value)

        elif self.process_type == 'd':

            incoming_value = float(column_value)

            if self.previous_value is None:
                self.previous_value = incoming_value

            value = incoming_value - self.previous_value

            # Update internal value only if the process method is the difference
            # relative to the previous value
            if self.process_type == 'd':
                self.previous_value = incoming_value

            return str(value)

        else:
            raise ValueError("Do not know what process type is '%c'" % self.process_type)

    def __str__(self):

        return self.__repr__()

    def __repr__(self):

        return "Process type [ %s ], process column [ %d ]\n" % (self.process_type, self.process_column)


def entry_point():

    # Process the options of the executable

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('columns', metavar='<selector>', type=str, nargs='+',
                        help="Set of column selectors and operators")

    args = parser.parse_args()

    # Make an array of objects that will take care of processing the fields
    colprocs = [ColumnProcess(colproc) for colproc in args.columns]

    for line in sys.stdin:

        # If line is empty, print an empty line
        if len(line.strip()) == 0:
            sys.stdout.write("\n")
            continue

        fields = line.strip().split()

        # Process each column
        newfields = [cp.process(fields) for cp in colprocs]

        sys.stdout.write(" ".join(newfields) + "\n")
