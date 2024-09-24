import argparse
import sys
from typing import Iterable

import numpy as np


def cdf_cli():
    argParser = argparse.ArgumentParser(description=__doc__,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)  # verbatim

    argParser.add_argument('--n-bins', '-n', metavar='<int>', type=int,
                           help='Number of bins', default=10)

    args = argParser.parse_args()

    samples = []

    for sample in sys.stdin:
        try:
            _ = float(sample)
        except ValueError:
            continue

        samples.append(float(sample))

    pdf, edges = np.histogram(samples, bins=args.n_bins, density=True)
    binwidth = edges[1] - edges[0]

    pdf = np.array(pdf) * binwidth
    cdf = np.cumsum(pdf)

    for i in range(args.n_bins):
        print(f"{edges[i]} {pdf[i]} {cdf[i]}")


def rms(values: Iterable) -> float:
    """
    Compute the Root Mean Square of an array of values

    >>> array = [1, 2, 3, 4, 5]
    >>> rms(array)
    3.3166247903554
    """
    return np.sqrt(np.mean(np.square(values)))
