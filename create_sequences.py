#!/usr/bin/env python3

"""
Creates sequences of datapoints for use in training a sequential language model.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import csv
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np

import extract_features
from storygenerator.io import MatchingFileWalker

DEFAULT_MAX_LENGTH = 60
DEFAULT_BATCH_MEMORY = 1024
DEFAULT_SAMPLING_RATE = 3

FEATURE_FILE_SUFFIX = ".npz"
SEQUENCE_DIR_NAME = "sequences"


class MetadataWriter(object):
	OUTPUT_FILENAME = "metadata.tsv"
	OUTPUT_CSV_DIALECT = csv.excel_tab

	def __init__(self, metadata: Dict[str, Any]):
		self.metadata = metadata

	def __call__(self, outdir: str):
		outpath = os.path.join(outdir, self.OUTPUT_FILENAME)
		print("Writing sequence metadata to \"{}\".".format(outpath))
		sorted_items = sorted(self.metadata.items(), key=lambda item: item[0])
		with open(outpath, 'w') as outf:
			writer = csv.writer(outf, dialect=self.OUTPUT_CSV_DIALECT)
			for key, value in sorted_items:
				writer.writerow((key, value))


class NPZSequenceWriter(object):
	OUTPUT_FILE_EXTENSION = ".npz"

	def __init__(self, outdir: str, batch_size: int):
		self.outdir = outdir
		self.batch_size = float(batch_size)

	def __call__(self, infile: str, x: np.array, y: np.array):
		# Don't touch this: The shape of the two arrays is likely different in dimensionality
		assert x.shape[0] == y.shape[0]
		assert x.shape[-1] == y.shape[-1]

		infile_name_base = os.path.splitext(os.path.basename(infile))[0]

		total_size = x.nbytes + y.nbytes
		# print("Total size in MB: {}".format(total_size / 1024 / 1024))
		batch_count = math.ceil(total_size / self.batch_size)
		print("Splitting data from \"{}\" into {} batch(es).".format(infile, batch_count))
		rows_per_batch = math.ceil(x.shape[0] / batch_count)
		remaining_row_count = x.shape[0]
		next_batch_start_idx = 0
		next_batch_id = 1
		while remaining_row_count > 0:
			next_batch_size = min(rows_per_batch, remaining_row_count)
			next_batch_end_idx = next_batch_start_idx + next_batch_size
			batch_x = x[next_batch_start_idx: next_batch_end_idx]
			batch_y = y[next_batch_start_idx: next_batch_end_idx]

			outpath = os.path.join(self.outdir,
								   infile_name_base + "-" + str(next_batch_id) + self.OUTPUT_FILE_EXTENSION)
			print(
				"Writing batch {} of features extracted from \"{}\" to \"{}\" (start idx: {}; end idx: {}; batch rows: {}).".format(
					next_batch_id, infile, outpath, next_batch_start_idx, next_batch_end_idx, next_batch_size))
			np.savez_compressed(outpath, x=batch_x, y=batch_y)

			remaining_row_count -= next_batch_size
			next_batch_start_idx = next_batch_end_idx
			next_batch_id += 1


def create_sequences(features: np.array, max_length: int, sampling_rate: int) -> Tuple[
	List[np.array], List[np.array]]:
	# cut the text in semi-redundant sequences of max_length characters
	obs_seqs = []
	next_chars = []
	for i in range(0, len(features) - max_length, sampling_rate):
		obs_seqs.append(features[i: i + max_length])
		next_chars.append(features[i + max_length])
	print("Created {} sequence(s).".format(len(obs_seqs)))
	return obs_seqs, next_chars


def read_file(infile_path: str, max_length: int, sampling_rate: int) -> Tuple[List[np.array], List[np.array]]:
	print("Loading data from \"{}\".".format(infile_path))
	x = []
	y = []
	with np.load(infile_path) as archive:
		for (_, arr) in archive.iteritems():
			obs_seqs, next_chars = create_sequences(arr, max_length, sampling_rate)
			x.extend(obs_seqs)
			y.extend(next_chars)

	return x, y


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Creates a language model for generating prose.")
	result.add_argument("indir", metavar="INDIR",
						help="The directory containing the vocabulary and feature data to read.")
	result.add_argument("outdir", metavar="OUTDIR", help="The directory to write sequence files under.")
	result.add_argument("-l", "--max-length", dest="max_length", metavar="LENGTH", type=int, default=DEFAULT_MAX_LENGTH,
						help="The maximum sequence length to create.")
	result.add_argument("-s", "--sampling-rate", dest="sampling_rate", metavar="RATE", type=int,
						default=DEFAULT_SAMPLING_RATE,
						help="The sequence sampling rate to use.")
	result.add_argument("-m", "--memory", metavar="MBYTES", type=int, default=DEFAULT_BATCH_MEMORY,
						help="The maximum size of a sequence training batch in megabytes.")
	return result


def __main(args):
	indir = args.indir
	print("Will read data from \"{}\".".format(indir))
	outdir = args.outdir
	print("Will save sequence data to \"{}\".".format(outdir))
	max_length = args.max_length
	sampling_rate = args.sampling_rate
	batch_size = args.memory
	print("Maximum length: {}; Sampling rate: {}; Batch size (in MB): {}".format(max_length, sampling_rate, batch_size))

	file_walker = MatchingFileWalker(lambda filepath: filepath.endswith(FEATURE_FILE_SUFFIX))
	feature_dir = os.path.join(indir, extract_features.FEATURE_DIR_NAME)
	print("Reading feature files under \"{}\".".format(feature_dir))
	infiles = tuple(file_walker(feature_dir))
	print("Found {} feature file(s).".format(len(infiles)))
	seq_outdir = os.path.join(outdir, SEQUENCE_DIR_NAME)
	os.makedirs(seq_outdir, exist_ok=True)
	print("Writing sequence data to \"{}\".".format(seq_outdir))
	metadata = {"max_length": max_length, "sampling_rate": sampling_rate, "batch_size": batch_size}
	metadata_writer = MetadataWriter(metadata)
	metadata_writer(seq_outdir)
	writer = NPZSequenceWriter(seq_outdir, batch_size * 1024 * 1024)
	for infile in infiles:
		x, y = read_file(infile, max_length, sampling_rate)
		writer(infile, np.asarray(x), np.asarray(y))


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
