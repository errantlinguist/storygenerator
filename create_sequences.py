#!/usr/bin/env python3

"""
Creates sequences of datapoints for use in training a sequential language model.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import os
from typing import Iterable, List, Tuple

import numpy as np

from extract_features import OUTPUT_FEATURE_DIRNAME
from storygenerator.io import NPZFileWalker

OUTPUT_SEQUENCE_DIRNAME = "sequences"


class NPZSequenceWriter(object):
	OUTPUT_FILE_DELIMITER = '\t'
	OUTPUT_FILE_EXTENSION = ".npz"

	def __init__(self, outdir: str):
		self.outdir = outdir

	def __call__(self, infile: str, x: np.array, y: np.array):
		infile_name_base = os.path.splitext(os.path.basename(infile))[0]
		outpath = os.path.join(self.outdir, infile_name_base + self.OUTPUT_FILE_EXTENSION)
		print("Writing features extracted from \"{}\" to \"{}\".".format(infile, outpath))
		np.savez_compressed(outpath, x=x, y=y)


def create_sequences(features: np.array, maxlen: int, sampling_rate: int) -> Tuple[
	List[np.array], List[np.array]]:
	# cut the text in semi-redundant sequences of maxlen characters
	obs_seqs = []
	next_chars = []
	for i in range(0, len(features) - maxlen, sampling_rate):
		obs_seqs.append(features[i: i + maxlen])
		next_chars.append(features[i + maxlen])
	print('nb sequences:', len(obs_seqs))
	return obs_seqs, next_chars


def read_file(infile_path: str, maxlen: int, sampling_rate: int) -> Tuple[List[np.array], List[np.array]]:
	print("Loading data from \"{}\".".format(infile_path))
	x = []
	y = []
	with np.load(infile_path, mmap_mode=True) as archive:
		for (_, arr) in archive.iteritems():
			obs_seqs, next_chars = create_sequences(arr, maxlen, sampling_rate)
			x.extend(obs_seqs)
			y.extend(next_chars)

	return x, y


def read_files(infile_paths: Iterable[str], maxlen: int, sampling_rate: int) -> Tuple[List[np.array], List[np.array]]:
	x = []
	y = []
	for infile_path in infile_paths:
		obs_seqs, next_chars = read_file(infile_path, maxlen, sampling_rate)
		x.extend(obs_seqs)
		y.extend(next_chars)
	return x, y


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Creates a language model for generating prose.")
	result.add_argument("indir", metavar="INDIR",
						help="The directory containing the vocabulary and feature data to read.")
	result.add_argument("outdir", metavar="OUTDIR", help="The directory to write sequence files under.")
	result.add_argument("-l", "--max-length", dest="max_length", metavar="LENGTH", type=int, default=40,
						help="The maximum sequence length to create.")
	result.add_argument("-s", "--sampling-rate", dest="sampling_rate", metavar="RATE", type=int, default=3,
						help="The sequence sampling rate to use.")
	return result


def __main(args):
	indir = args.indir
	print("Will read data from \"{}\".".format(indir))
	outdir = args.outdir
	print("Will save sequence data to \"{}\".".format(outdir))
	maxlen = args.max_length
	sampling_rate = args.sampling_rate
	print("Maximum length: {}; Sampling rate: {}".format(maxlen, sampling_rate))

	file_walker = NPZFileWalker()
	feature_dir = os.path.join(indir, OUTPUT_FEATURE_DIRNAME)
	print("Reading feature files under \"{}\".".format(feature_dir))
	infiles = tuple(file_walker(feature_dir))
	print("Found {} feature file(s).".format(len(infiles)))
	seq_outdir = os.path.join(outdir, OUTPUT_SEQUENCE_DIRNAME)
	os.makedirs(seq_outdir, exist_ok=True)
	print("Writing sequence data to \"{}\".".format(seq_outdir))
	writer = NPZSequenceWriter(seq_outdir)
	for infile in infiles:
		x, y = read_file(infile, maxlen, sampling_rate)
		writer(infile, x, y)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
