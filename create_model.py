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
from typing import Iterator, Sequence, Tuple

import keras.preprocessing.sequence
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, LSTM, TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop

from storygenerator.io import OUTPUT_FEATURE_DIRNAME, OUTPUT_VOCAB_FILENAME, FeatureExtractor, read_vocab

MODEL_CHECKPOINT_DIRNAME = "models"


class FileLoadingDataGenerator(keras.utils.Sequence):

	def __init__(self, infile_paths: Sequence[str], maxlen: int):
		self.infile_paths = infile_paths
		self.maxlen = maxlen

	def __len__(self) -> int:
		return len(self.infile_paths)

	def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
		infile_path = self.infile_paths[idx]
		print("Loading data from \"{}\".".format(infile_path))
		with np.load(infile_path) as archive:
			# print(archive.files)
			# help(archive)
			arrs = tuple(arr for (_, arr) in archive.iteritems())
			# print(arrs)
			arr = np.concatenate(arrs, axis=0)
			step = 3
			seqs = []
			next_chars = []
			for i in range(0, arr.shape[0] - self.maxlen, step):
				seqs.append(arr[i: i + self.maxlen])
				next_chars.append(arr[i + self.maxlen])
			print('nb sequences:', len(seqs))
			return np.asarray(seqs), np.asarray(next_chars)

	def on_epoch_end(self):
		pass


class NPZFileWalker(object):
	FILE_EXTENSION_PATTERN = re.compile("\.npz", re.IGNORECASE)

	@classmethod
	def is_file(cls, path: str) -> bool:
		ext = os.path.splitext(path)[1]
		match = cls.FILE_EXTENSION_PATTERN.match(ext)
		return bool(match)

	def __call__(self, indir: str) -> Iterator[str]:
		for root, dirs, files in os.walk(indir, followlinks=True):
			for file in files:
				filepath = os.path.join(root, file)
				if self.is_file(filepath):
					yield filepath


def create_model(maxlen: int, feature_count: int) -> Sequential:
	result = Sequential()
	result.add(LSTM(128, input_shape=(maxlen, feature_count), return_sequences=True))
	result.add(TimeDistributed(Dense(feature_count)))
	result.add(Activation('softmax'))
	optimizer = RMSprop(lr=0.01)
	result.compile(loss='categorical_crossentropy', optimizer=optimizer)
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
	print("Read vocabulary of size {}".format(len(vocab)))

	file_walker = NPZFileWalker()
	feature_dir = os.path.join(indir, OUTPUT_FEATURE_DIRNAME)
	print("Reading feature files under \"{}\".".format(feature_dir))
	feature_files = tuple(file_walker(feature_dir))
	print("Found {} feature file(s).".format(len(feature_files)))
	maxlen = 40
	data_generator = FileLoadingDataGenerator(feature_files, maxlen)
	feature_count = FeatureExtractor.feature_count(vocab)

	model_checkpoint_outdir = os.path.join(outdir, MODEL_CHECKPOINT_DIRNAME)
	print("Will save model checkpoints to \"{}\".".format(model_checkpoint_outdir))

	# https://stackoverflow.com/a/43472000/1391325
	with keras.backend.get_session():
		# build the model: a single LSTM
		print('Build model...')
		model = create_model(maxlen, feature_count)
		# epoch_end_hook = EpochEndHook(model, raw_text, chars, char_indices, indices_char, maxlen)
		# print_callback = LambdaCallback(on_epoch_end=epoch_end_hook)
		# define the checkpoint
		filepath = os.path.join(model_checkpoint_outdir, "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5")
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]
		# train LSTM
		epochs = args.epochs
		print("Training model using {} epoch(s).".format(epochs), file=sys.stderr)
		# workers = max(multiprocessing.cpu_count() // 2, 1)
		workers = 1
		print("Using {} worker thread(s).".format(workers), file=sys.stderr)
		training_history = model.fit_generator(data_generator, epochs=epochs, verbose=0, use_multiprocessing=False,
											   workers=workers, callbacks=callbacks_list)


# model.fit(x, y,
#		  batch_size=128,
#		  epochs=60,
#		  callbacks=callbacks_list)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
