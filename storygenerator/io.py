"""
Functionalities for reading in literature in the format written by the "storygenerator-preprocessing" package.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import re
from typing import Callable, Optional, Tuple

from . import Chapter

CHAPTER_DELIM_PATTERN = re.compile("=+")
DEFAULT_PART_NAME_ORDINALITIES = {"PROLOGUE": -1, "CHAPTER": 0, "EPILOGUE": 1}
ORDERED_CHAPTER_PATTERN = re.compile("^(\w+)\s+(\d+)")


class TextChapterReader(object):

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
