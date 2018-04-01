"""
Functionalities modelling fictional language and creating new data based on it.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

from typing import Optional, Sequence


class Chapter(object):

	def __init__(self, part: int, seq: Optional[int] = None, title: Optional[str] = None,
				 pars: Optional[Sequence[str]] = None):
		"""
		:param part: -1 for prologue, 0 for "normal" chapters and 1 for epilogue. Higher values can be used for appendices, etc.
		:param seq: The chapter number.
		:param title: The chapter title.
		:param pars A sequence of strings, each representing a single paragraph.
		"""
		self.part = part
		self.seq = seq
		self.title = title if title is not None else ""
		self.pars = [] if pars is None else pars

	@property
	def __key(self):
		return self.seq, self.title, self.pars

	def __bool__(self):
		return bool(self.seq) or bool(self.title) or bool(self.pars)

	def __eq__(self, other):
		return self is other or (isinstance(other, type(self)) and self.__key == other.__key)

	def __hash__(self):
		return hash(self.__key)

	def __ne__(self, other):
		return not (self == other)

	def __repr__(self):
		fields = (
		"{part=", str(self.part), ", seq=", str(self.seq), ", title=", str(self.title), ", pars=", str(self.pars), "}")
		field_repr = "".join(fields)
		return self.__class__.__name__ + field_repr
