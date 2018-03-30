#!/usr/bin/env python3

"""
Creates a language model for generating prose.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import random
import sys

import numpy as np


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Creates a language model for generating prose.")
	result.add_argument("infiles", metavar="FILE", nargs='+',
						help="The text files to process.")
	result.add_argument("-e", "--encoding", metavar="CODEC", default="utf-8",
						help="The input file encoding.")
	result.add_argument("-s", "--random-seed", dest="random_seed", metavar="SEED", type=int, default=7,
						help="The random seed to use.")
	return result


def __main(args):
	random_seed = args.random_seed
	print("Setting random seed to {}.".format(random_seed), file=sys.stderr)
	# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
	# fix random seed for reproducibility
	random.seed(random_seed)
	np.random.seed(random_seed)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
