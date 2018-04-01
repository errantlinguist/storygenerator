#!/usr/bin/env python3

"""
Creates a language model for generating prose.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import os
import re
from typing import Callable, Iterable, Iterator, Sequence, Tuple

import magic
import numpy as np

from storygenerator import Chapter
from storygenerator.io import TextChapterReader

INPUT_FILE_MIMETYPE = "text/plain"
INPUT_FILENAME_PATTERN = re.compile("(\d+)\s+([^.]+)\..+")
OUTPUT_FEATURE_DIRNAME = "features"
OUTPUT_VOCAB_FILENAME = "vocab.tsv"


class Book(object):
	def __init__(self, ordinality: int, title: str, chaps: Sequence[Chapter]):
		self.ordinality = ordinality
		self.title = title
		self.chaps = chaps


class FeatureExtractor(object):

	def __init__(self, vocab: Sequence[str]):
		self.__vocab = vocab
		self.__vocab_idxs = dict((char, idx) for idx, char in enumerate(vocab))
		# Includes features representing possible actual characters as well as features representing ends of paragraphs, chapters and books.
		self.__feature_count = len(self.__vocab) + 3
		self.__par_end_feature_idx = len(self.__vocab)
		self.__chapter_end_feature_idx = len(self.__vocab) + 1
		self.__book_end_feature_idx = len(self.__vocab) + 2

	def __call__(self, book: Book) -> np.array:
		chap_arrs = tuple(self.__extract_chap_features(chap) for chap in book.chaps)
		result = np.concatenate(chap_arrs, axis=0)
		result[-1, self.__book_end_feature_idx] = 1
		return result

	def __extract_chap_features(self, chap: Chapter) -> np.array:
		par_arrs = tuple(self.__extract_par_features(par) for par in chap.pars)
		result = np.concatenate(par_arrs, axis=0)
		result[-1, self.__chapter_end_feature_idx] = 1
		return result

	def __extract_par_features(self, par: str) -> np.array:
		"""
		Creates a 2D numpy array representing a paragraph using each of its characters as a single datapoint.
		Includes features representing possible actual characters as well as features representing ends of paragraphs and chapters.

		:param par: The paragraph to extract features from.
		:return: A new numpy array representing the paragraph.
		"""
		result = np.zeros((len(par), self.__feature_count))
		for idx, char in enumerate(par):
			feature_idx = self.__vocab_idxs[char]
			result[idx, feature_idx] = 1

		result[-1, self.__par_end_feature_idx] = 1
		return result


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


def read_books(infiles: Iterable[str]) -> Iterator[Tuple[str, Book]]:
	reader = TextChapterReader()

	for infile in infiles:
		print("Reading \"{}\".".format(infile))
		book_ordinality, book_title = parse_book_filename(infile)
		chaps = reader(infile)
		print("Read {} chapter(s) for book {}, titled \"{}\".".format(len(chaps), book_ordinality, book_title))
		yield infile, Book(book_ordinality, book_title, chaps)


def write_vocab(vocab: Iterable[str], outdir: str):
	vocab_outfile_path = os.path.join(outdir, OUTPUT_VOCAB_FILENAME)
	print("Writing vocab to \"{}\".".format(vocab_outfile_path))
	with open(vocab_outfile_path, 'w') as vocab_outf:
		vocab_writer = csv.writer(vocab_outf, dialect=csv.excel_tab)
		vocab_writer.writerow(vocab)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Creates a language model for generating prose.")
	result.add_argument("inpaths", metavar="FILE", nargs='+',
						help="The text file(s) and/or directory path(s) to process.")
	result.add_argument("-o", "--outdir", metavar="DIR", required=True,
						help="The directory to write the output files to.")
	return result


def __main(args):
	file_walker = MimetypeFileWalker(lambda mimetype: mimetype == INPUT_FILE_MIMETYPE)
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
	write_vocab(vocab, outdir)

	feature_dirpath = os.path.join(outdir, OUTPUT_FEATURE_DIRNAME)
	os.mkdir(feature_dirpath)
	feature_extractor = FeatureExtractor(vocab)
	for infile, book in infile_books:
		features = feature_extractor(book)
		book_filename_base = os.path.splitext(os.path.basename(infile))[0]
		outpath = os.path.join(feature_dirpath, book_filename_base + ".npz")
		print("Writing features extracted from \"{}\" to \"{}\".".format(infile, outpath))
		np.savez_compressed(outpath, features)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
