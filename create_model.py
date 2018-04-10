#!/usr/bin/env python3

"""
Creates a language model for generating prose.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import multiprocessing
import os
import random
from typing import Iterable, List, Sequence, Tuple

import keras.preprocessing.sequence
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

from extract_features import OUTPUT_FEATURE_DIRNAME
from storygenerator.io import OUTPUT_VOCAB_FILENAME, FeatureExtractor, NPZFileWalker, read_vocab

MODEL_CHECKPOINT_DIRNAME = "models"


class FileLoadingDataGenerator(keras.utils.Sequence):

	def __init__(self, infile_paths: Sequence[str], maxlen: int, feature_count: int, sampling_rate: int):
		self.infile_paths = infile_paths
		self.maxlen = maxlen
		self.feature_count = feature_count
		self.sampling_rate = sampling_rate

	def __len__(self) -> int:
		return len(self.infile_paths)

	def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
		infile_path = self.infile_paths[idx]
		x, y = read_file(infile_path, self.maxlen, self.sampling_rate)
		return np.asarray(x), np.asarray(y)

	def on_epoch_end(self):
		pass


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
	with np.load(infile_path) as archive:
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
	print("Read vocabulary of size {}.".format(len(vocab)))

	file_walker = NPZFileWalker()
	feature_dir = os.path.join(indir, OUTPUT_FEATURE_DIRNAME)
	print("Reading feature files under \"{}\".".format(feature_dir))
	feature_files = tuple(file_walker(feature_dir))
	print("Found {} feature file(s).".format(len(feature_files)))
	feature_count = FeatureExtractor.feature_count(vocab)
	maxlen = 40
	sampling_rate = 3
	# x, y = read_files(feature_files, maxlen, sampling_rate)
	# x = np.asarray(x)
	# print(x.shape)
	# y = np.asarray(y)
	# print(y.shape)
	# x = keras.preprocessing.sequence.pad_sequences(x, maxlen=maxlen, dtype='bool')
	# y = keras.preprocessing.sequence.pad_sequences(y, maxlen=maxlen, dtype='bool')
	data_generator = FileLoadingDataGenerator(feature_files, maxlen, feature_count, sampling_rate)

	model_checkpoint_outdir = os.path.join(outdir, MODEL_CHECKPOINT_DIRNAME)
	print("Will save model checkpoints to \"{}\".".format(model_checkpoint_outdir))
	os.makedirs(model_checkpoint_outdir, exist_ok=True)

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
		print("Training model using {} epoch(s).".format(epochs))
		# training_history = model.fit(x, y,
		#		  #batch_seq_count=128,
		#		  epochs=epochs,
		#		  callbacks=callbacks_list)
		workers = max(multiprocessing.cpu_count() // 2, 1)
		workers = 1
		max_queue_size = 1
		print("Using {} worker thread(s) with a max queue size of {}.".format(workers, max_queue_size))
		training_history = model.fit_generator(data_generator, epochs=epochs, verbose=1, use_multiprocessing=False,
											   workers=workers, max_queue_size=max_queue_size, callbacks=callbacks_list)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
