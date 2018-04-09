#!/usr/bin/env python3

"""
Extracts features from a given set of text files for use in training a language model.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import datetime
import os
from typing import Iterable, Iterator, Tuple

import numpy as np
import tzlocal

import storygenerator.io

INPUT_FILE_MIMETYPE = "text/plain"
OUTPUT_FEATURE_DIRNAME = "features"


class NPZFeatureWriter(object):
	OUTPUT_FILE_DELIMITER = '\t'
	OUTPUT_FILE_EXTENSION = ".npz"

	def __init__(self, outdir: str):
		self.outdir = outdir

	def __call__(self, infile: str, features: np.array, book: storygenerator.Book):
		infile_name_base = os.path.splitext(os.path.basename(infile))[0]
		outpath = os.path.join(self.outdir, infile_name_base + self.OUTPUT_FILE_EXTENSION)
		print("Writing features extracted from \"{}\" to \"{}\".".format(infile, outpath))
		np.savez_compressed(outpath, **{infile_name_base: features})


class TextFeatureWriter(object):
	OUTPUT_FILE_DELIMITER = '\t'
	OUTPUT_FILE_EXTENSION = ".features.gz"

	def __init__(self, outdir: str):
		self.outdir = outdir

	def __call__(self, infile: str, features: np.array, book: storygenerator.Book):
		infile_name_base = os.path.splitext(os.path.basename(infile))[0]
		outpath = os.path.join(self.outdir, infile_name_base + self.OUTPUT_FILE_EXTENSION)
		print("Writing features extracted from \"{}\" to \"{}\".".format(infile, outpath))
		timezone = tzlocal.get_localzone()
		timestamp = datetime.datetime.now(timezone).isoformat()
		header = "{}; \"{}\"; {}; Shape: {}".format(infile_name_base, book.title, timestamp, features.shape)
		np.savetxt(outpath, features, header=header, delimiter=self.OUTPUT_FILE_DELIMITER)


def read_books(infiles: Iterable[str]) -> Iterator[Tuple[str, storygenerator.io.Book]]:
	reader = storygenerator.io.TextChapterReader()

	for infile in infiles:
		print("Reading \"{}\".".format(infile))
		book_ordinality, book_title = storygenerator.io.parse_book_filename(infile)
		chaps = reader(infile)
		print("Read {} chapter(s) for book {}, titled \"{}\".".format(len(chaps), book_ordinality, book_title))
		yield infile, storygenerator.io.Book(book_ordinality, book_title, chaps)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Extracts features from a given set of text files for use in training a language model.")
	result.add_argument("inpaths", metavar="FILE", nargs='+',
						help="The text file(s) and/or directory path(s) to process.")
	result.add_argument("-o", "--outdir", metavar="DIR", required=True,
						help="The directory to write the output files to.")
	result.add_argument("-t", "--text", action="store_true",
						help="Writes the feature arrays in tabular text format.")
	return result


def __main(args):
	file_walker = storygenerator.io.MimetypeFileWalker(lambda mimetype: mimetype == INPUT_FILE_MIMETYPE)
	outdir = args.outdir
	print("Will write output files to \"{}\".".format(outdir))

	inpaths = args.inpaths
	print("Looking for input text files underneath {}.".format(inpaths))
	infiles = file_walker(inpaths)
	infile_books = tuple(read_books(infiles))
	print("Read {} book(s).".format(len(infile_books)))

	vocab = sorted(
		frozenset(char for (_, book) in infile_books for chap in book.chaps for par in chap.pars for char in par))
	print("Vocab size: {}".format(len(vocab)))
	os.makedirs(outdir, exist_ok=True)
	storygenerator.io.write_vocab(vocab, outdir)

	feature_dirpath = os.path.join(outdir, OUTPUT_FEATURE_DIRNAME)
	try:
		os.mkdir(feature_dirpath)
	except FileExistsError:
		pass
	feature_extractor = storygenerator.io.FeatureExtractor(vocab)
	writer = TextFeatureWriter(feature_dirpath) if args.text else NPZFeatureWriter(feature_dirpath)
	for infile, book in infile_books:
		features = feature_extractor(book)
		writer(infile, features, book)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
