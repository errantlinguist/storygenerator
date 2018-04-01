#!/usr/bin/env python3

"""
Creates a language model for generating prose.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import os
import random
import re
import sys
from typing import Callable, Iterable, Iterator, Tuple

import magic
import numpy as np

from storygenerator.io import TextChapterReader

INPUT_FILE_MIMETYPE = "text/plain"
INPUT_FILENAME_PATTERN = re.compile("(\d+)\s+([^\.]+)\..+")


class MimetypeFileWalker(object):

	def __init__(self, mimetype_matcher: Callable[[str], bool]):
		self.mimetype_matcher = mimetype_matcher
		self.__mime = magic.Magic(mime=True)

	def __call__(self, inpaths: Iterable[str]) -> Iterator[str]:
		for inpath in inpaths:
			if os.path.isdir(inpath):
				for root, _, files in os.walk(inpath, followlinks=True):
					for file in files:
						filepath = os.path.join(root, file)
						mimetype = self.__mime.from_file(filepath)
						if self.mimetype_matcher(mimetype):
							yield filepath
			else:
				mimetype = self.__mime.from_file(inpath)
				if self.mimetype_matcher(mimetype):
					yield inpath


def parse_book_filename(inpath: str) -> Tuple[int, str]:
	filename = os.path.basename(inpath)
	m = INPUT_FILENAME_PATTERN.match(filename)
	if m:
		ordinality = int(m.group(1))
		title = m.group(2)
	else:
		raise ValueError("Could not parse filename for path \"{}\".".format(inpath))
	return ordinality, title


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Creates a language model for generating prose.")
	result.add_argument("inpaths", metavar="FILE", nargs='+',
						help="The text file(s) and/or directory path(s) to process.")
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
	file_walker = MimetypeFileWalker(lambda mimetype: mimetype == INPUT_FILE_MIMETYPE)
	inpaths = args.inpaths
	print("Looking for input text files underneath {}.".format(inpaths), file=sys.stderr)
	infiles = file_walker(inpaths)
	reader = TextChapterReader()

	for infile in infiles:
		print("Reading \"{}\".".format(infile), file=sys.stderr)
		book_ordinality, book_title = parse_book_filename(infile)
		chaps = reader(infile)
		print("Read {} chapter(s) for book {}, titled \"{}\".".format(len(chaps), book_ordinality, book_title),
			  file=sys.stderr)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
