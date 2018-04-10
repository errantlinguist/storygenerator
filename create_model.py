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
import random
import tempfile
from typing import Any, Callable, Dict, Iterable, Iterator, Sequence, Tuple

import keras.preprocessing.sequence
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

import create_sequences
from storygenerator.io import OUTPUT_VOCAB_FILENAME, FeatureExtractor, NPZFileWalker, read_vocab

MODEL_CHECKPOINT_DIRNAME = "models"


class CachingFileReader(object):

	def __init__(self, cache_dirpath: str):
		self.cache_dirpath = cache_dirpath

	def __call__(self, infile_path: str) -> Tuple[np.array, np.array]:
		common_path = os.path.commonpath((self.cache_dirpath, infile_path))
		relative_path = os.path.relpath(infile_path, common_path)
		cached_filepath_base = os.path.join(self.cache_dirpath, relative_path)
		os.makedirs(os.path.dirname(cached_filepath_base), exist_ok=True)
		cached_filepath_x = cached_filepath_base + ".x"
		cached_filepath_y = cached_filepath_base + ".y"
		try:
			x = np.load(cached_filepath_x, mmap_mode='r')
			y = np.load(cached_filepath_y, mmap_mode='r')
		except FileNotFoundError:
			x, y = read_file(infile_path)
			np.save(cached_filepath_x, x)
			np.save(cached_filepath_y, y)

		return x, y


class FileLoadingDataGenerator(keras.utils.Sequence):
	"""
	WARNING: There is some sort of memory leak when using this class
	"""

	def __init__(self, infile_paths: Sequence[str], file_reader: Callable[[str], Tuple[np.array, np.array]]):
		self.infile_paths = list(infile_paths)
		self.file_reader = file_reader

	def __len__(self) -> int:
		return len(self.infile_paths)

	def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
		infile_path = self.infile_paths[idx]
		x, y = self.file_reader(infile_path)
		return x, y

	def on_epoch_end(self):
		random.shuffle(self.infile_paths)


def create_model(maxlen: int, feature_count: int) -> Sequential:
	result = Sequential()
	result.add(LSTM(128, input_shape=(maxlen, feature_count)))
	result.add(Dense(feature_count))
	# result.add(LSTM(128, input_shape=(maxlen, feature_count), return_sequences=True))
	# result.add(TimeDistributed(Dense(feature_count)))
	result.add(Activation('softmax'))
	optimizer = RMSprop(lr=0.01)
	result.compile(loss='categorical_crossentropy', optimizer=optimizer)
	return result


def read_file(infile_path: str) -> Tuple[np.array, np.array]:
	print("Loading data from \"{}\".".format(infile_path))
	with np.load(infile_path) as archive:
		x = archive["x"]
		assert x.size > 0
		y = archive["y"]
		# Don't touch this: The shape of the two arrays is likely different in dimensionality
		assert x.shape[0] == y.shape[0]
		assert x.shape[-1] == y.shape[-1]
		return x, y


def read_files(infile_paths: Iterable[str]) -> Iterator[Tuple[np.array, np.array]]:
	for infile_path in infile_paths:
		x, y = read_file(infile_path)
		yield x, y


def read_seq_metadata(seq_dir: str) -> Dict[str, Any]:
	result = {}
	infile_path = os.path.join(seq_dir, create_sequences.MetadataWriter.OUTPUT_FILENAME)
	print("Reading sequence metadata from \"{}\".".format(infile_path))
	with open(infile_path, 'r') as inf:
		reader = csv.reader(inf, dialect=create_sequences.MetadataWriter.OUTPUT_CSV_DIALECT)
		for row in reader:
			assert len(row) == 2
			key = row[0]
			value = row[1]
			result[key] = value

	return result


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Creates a language model for generating prose.")
	result.add_argument("indir", metavar="INDIR",
						help="The directory containing the vocabulary and feature data to read.")
	result.add_argument("outdir", metavar="OUTDIR", help="The directory to store model files under.")
	result.add_argument("-s", "--random-seed", dest="random_seed", metavar="SEED", type=int, default=7,
						help="The random seed to use.")
	result.add_argument("-e", "--epochs", metavar="EPOCHS", type=int, default=60,
						help="The number of epochs use for training.")
	return result


def __train_generator(model: Sequential, seq_files: Sequence[str],
					  file_reader: Callable[[str], Tuple[np.array, np.array]], epochs: int,
					  model_checkpoint_outdir: str):
	filepath = os.path.join(model_checkpoint_outdir, "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
	# define the checkpoint
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	data_generator = FileLoadingDataGenerator(seq_files, file_reader)
	# workers = max(multiprocessing.cpu_count() // 2, 1)
	workers = 1
	max_queue_size = 1
	print("Using {} worker thread(s) with a max queue size of {}.".format(workers, max_queue_size))
	training_history = model.fit_generator(data_generator, epochs=epochs, verbose=1, use_multiprocessing=False,
										   workers=workers, max_queue_size=max_queue_size,
										   callbacks=callbacks_list)


def __train_iteratively(model: Sequential, seq_files: Sequence[str],
						file_reader: Callable[[str], Tuple[np.array, np.array]], epochs: int,
						model_checkpoint_outdir: str):
	filepath = os.path.join(model_checkpoint_outdir, "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
	# define the checkpoint
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	seq_files = list(seq_files)
	for epoch_id in range(0, epochs):
		for seq_file in seq_files:
			x, y = file_reader(seq_file)
			training_history = model.fit(x, y, initial_epoch=epoch_id, epochs=epoch_id, callbacks=callbacks_list)
		random.shuffle(seq_files)


def __main(args):
	random_seed = args.random_seed
	print("Setting random seed to {}.".format(random_seed))
	# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
	# fix random seed for reproducibility
	random.seed(random_seed)
	np.random.seed(random_seed)

	indir = args.indir
	print("Will read data from \"{}\".".format(indir))
	outdir = args.outdir
	print("Will save model data to \"{}\".".format(outdir))
	os.makedirs(outdir, exist_ok=True)

	vocab_filepath = os.path.join(indir, OUTPUT_VOCAB_FILENAME)
	vocab = read_vocab(vocab_filepath)
	print("Read vocabulary of size {}.".format(len(vocab)))

	seq_dir = os.path.join(indir, create_sequences.OUTPUT_SEQUENCE_DIRNAME)
	seq_metadata = read_seq_metadata(seq_dir)
	assert seq_metadata

	file_walker = NPZFileWalker()
	seq_files = tuple(file_walker(seq_dir))
	print("Found {} sequence file(s).".format(len(seq_files)))

	model_checkpoint_outdir = os.path.join(outdir, MODEL_CHECKPOINT_DIRNAME)
	print("Will save model checkpoints to \"{}\".".format(model_checkpoint_outdir))
	os.makedirs(model_checkpoint_outdir, exist_ok=True)

	max_length = int(seq_metadata["max_length"])
	feature_count = FeatureExtractor.feature_count(vocab)
	# https://stackoverflow.com/a/43472000/1391325
	with keras.backend.get_session():
		# build the model: a single LSTM
		print('Compiling model.')
		model = create_model(max_length, feature_count)
		print(model.summary())
		# epoch_end_hook = EpochEndHook(model, raw_text, chars, char_indices, indices_char, maxlen)
		# print_callback = LambdaCallback(on_epoch_end=epoch_end_hook)

		# train LSTM
		epochs = args.epochs
		print("Training model using {} epoch(s).".format(epochs))
		# training_history = model.fit(x, y,
		#		  #batch_seq_count=128,
		#		  epochs=epochs,
		#		  callbacks=callbacks_list)

		with tempfile.TemporaryDirectory() as tmpdir_path:
			print("Will cache array data to \"{}\".".format(tmpdir_path))
			file_reader = CachingFileReader(tmpdir_path)
			# __train_generator(model, seq_files, file_reader, epochs, model_checkpoint_outdir)
			__train_iteratively(model, seq_files, file_reader, epochs, model_checkpoint_outdir)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
