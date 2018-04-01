#!/usr/bin/env python3

"""
Creates a language model for generating prose.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import os
from typing import Iterable, Iterator, Tuple

import numpy as np

import storygenerator.io

INPUT_FILE_MIMETYPE = "text/plain"


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
		description="Creates a language model for generating prose.")
	result.add_argument("inpaths", metavar="FILE", nargs='+',
						help="The text file(s) and/or directory path(s) to process.")
	result.add_argument("-o", "--outdir", metavar="DIR", required=True,
						help="The directory to write the output files to.")
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

	feature_dirpath = os.path.join(outdir, storygenerator.io.OUTPUT_FEATURE_DIRNAME)
	os.mkdir(feature_dirpath)
	feature_extractor = storygenerator.io.FeatureExtractor(vocab)
	for infile, book in infile_books:
		features = feature_extractor(book)
		book_filename_base = os.path.splitext(os.path.basename(infile))[0]
		outpath = os.path.join(feature_dirpath, book_filename_base + ".npz")
		print("Writing features extracted from \"{}\" to \"{}\".".format(infile, outpath))
		np.savez_compressed(outpath, features)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
