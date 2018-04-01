"""
Functionalities for reading in literature in the format written by the "storygenerator-preprocessing" package.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import csv
import os
import re
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import magic
import numpy as np

from . import Book, Chapter

CHAPTER_DELIM_PATTERN = re.compile("=+")
DEFAULT_PART_NAME_ORDINALITIES = {"PROLOGUE": -1, "CHAPTER": 0, "EPILOGUE": 1}
ORDERED_CHAPTER_PATTERN = re.compile("^(\w+)\s+(\d+)")

INPUT_FILENAME_PATTERN = re.compile("(\d+)\s+([^.]+)\..+")
OUTPUT_FEATURE_DIRNAME = "features"
OUTPUT_VOCAB_FILENAME = "vocab.tsv"
VOCAB_FILE_CSV_DIALECT = csv.excel_tab


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


class TextChapterReader(object):

	def __init__(self, part_name_ordinality_mapper: Optional[Callable[[str], int]] = None):
		self.part_name_ordinality_mapper = part_name_ordinality_mapper if part_name_ordinality_mapper is not None else lambda \
				part_desc: DEFAULT_PART_NAME_ORDINALITIES[part_desc]

	def __call__(self, inpath: str):
		result = []

		with open(inpath, 'r') as inf:
			chap = Chapter(-2)
			parsed_chap_header = False

			for line in inf:
				line = line.strip()
				if line:
					chap_end_m = CHAPTER_DELIM_PATTERN.match(line)
					if chap_end_m:
						result.append(chap)
						chap = Chapter(-2)
						parsed_chap_header = False
					elif not parsed_chap_header:
						part, seq, chap_title = self.parse_chapter_header(line)
						chap.part = part
						chap.seq = seq
						chap.title = chap_title
						parsed_chap_header = True
					else:
						# Each line denotes a single paragraph
						chap.pars.append(line)

			# Add the final chapter if it is not empty
			if chap.pars:
				result.append(chap)

		return result

	def parse_chapter_header(self, line: str) -> Tuple[int, Optional[int], str]:

		sep_idx = line.find(":")
		if sep_idx > 0:
			chap_seq = line[:sep_idx].strip()
			chap_title = line[sep_idx + 1:].strip()
		else:
			chap_seq = line
			chap_title = ""

		ordered_chap_m = ORDERED_CHAPTER_PATTERN.match(chap_seq)
		if ordered_chap_m:
			part_desc = ordered_chap_m.group(1)
			part = self.part_name_ordinality_mapper(part_desc)
			seq = int(ordered_chap_m.group(2))
		else:
			part_desc = chap_seq
			part = self.part_name_ordinality_mapper(part_desc)
			seq = None

		return part, seq, chap_title


def parse_book_filename(inpath: str) -> Tuple[int, str]:
	filename = os.path.basename(inpath)
	m = INPUT_FILENAME_PATTERN.match(filename)
	if m:
		ordinality = int(m.group(1))
		title = m.group(2)
	else:
		raise ValueError("Could not parse filename for path \"{}\".".format(inpath))
	return ordinality, title


def read_vocab(infile: str) -> List[str]:
	with open(infile, 'r') as inf:
		reader = csv.reader(inf, dialect=VOCAB_FILE_CSV_DIALECT)
		return next(reader)


def write_vocab(vocab: Iterable[str], outdir: str):
	vocab_outfile_path = os.path.join(outdir, OUTPUT_VOCAB_FILENAME)
	print("Writing vocab to \"{}\".".format(vocab_outfile_path))
	with open(vocab_outfile_path, 'w') as vocab_outf:
		vocab_writer = csv.writer(vocab_outf, dialect=VOCAB_FILE_CSV_DIALECT)
		vocab_writer.writerow(vocab)
